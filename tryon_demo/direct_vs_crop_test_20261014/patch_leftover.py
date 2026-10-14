#!/usr/bin/env python3
"""B 路线贴回后处理：把「旧剪影残留」用最近真实背景像素精确填充，再重贴新人。

残留 = 旧人剪影（alpha>0.05）中未被新剪影覆盖的部分 —— 直接贴只会盖住新剪影，
旧毛衣/旧裤子的边缘条带会露出来。填充源严格限定在旧剪影之外的纯背景像素，
用距离变换做「最近源填充」（不留迭代深度死角）。
用法（脚本内置两组路径）：大尺寸 full / 小尺寸 small。
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import proc  # noqa: E402

CASES = [
    {   # 大尺寸版（1024x1536 输入）
        "person": os.path.join(HERE, "00_inputs/person_complex_bg.png"),
        "gen": os.path.join(HERE, "03_B_crop_paste/B1_gen_crop.png"),
        "bbox_json": (os.path.join(HERE, "01_cutout/cutout_meta.json"), "person_bbox_xyxy"),
        "out_png": os.path.join(HERE, "03_B_crop_paste/B3p_pasteback_inpaint.png"),
        "out_mask": os.path.join(HERE, "03_B_crop_paste/B3p_leftover_mask.png"),
        "out_meta": os.path.join(HERE, "03_B_crop_paste/B3p_meta.json"),
    },
    {   # 小尺寸版（683x1024 输入，用 s1 成功生成图）
        "person": os.path.join(HERE, "small_input_683x1024/00_inputs/person_683x1024.png"),
        "gen": os.path.join(HERE, "small_input_683x1024/03_B/B_gen_crop_s1.png"),
        "bbox_json": (os.path.join(HERE, "small_input_683x1024/timing_small.json"),
                      "B_crop_paste.person_bbox_xyxy"),
        "out_png": os.path.join(HERE, "small_input_683x1024/03_B/B_pasteback_s1_fixed.png"),
        "out_mask": os.path.join(HERE, "small_input_683x1024/03_B/leftover_mask.png"),
        "out_meta": os.path.join(HERE, "small_input_683x1024/03_B/B_fixed_meta.json"),
    },
]


def _bbox(path, keypath):
    d = json.load(open(path, encoding="utf-8"))
    for k in keypath.split("."):
        d = d[k]
    return [int(v) for v in d]


def fix_case(c):
    person = cv2.imread(c["person"], cv2.IMREAD_COLOR)
    gen = cv2.imread(c["gen"], cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = _bbox(*c["bbox_json"])
    gw, gh = x1 - x0, y1 - y0

    t0 = time.time()
    _, p_a = proc.remove_bg(person)
    g_fg, g_a = proc.remove_bg(gen)
    t_mat = round(time.time() - t0, 1)
    if gen.shape[1] != gw or gen.shape[0] != gh:
        g_fg = cv2.resize(g_fg, (gw, gh))
        g_a = cv2.resize(g_a, (gw, gh))

    pa_sub = p_a[y0:y1, x0:x1].astype(np.float32)
    new_sil = g_a > 0.5
    new_prot = cv2.dilate(new_sil.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    old_soft = cv2.morphologyEx((pa_sub > 0.02).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8)).astype(bool)
    old_soft = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    old_guard = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    shell = old_soft & (~new_prot)          # 需要按背景重填的残留
    source = (~old_guard) & (~new_sil)      # 填充源：旧剪影外、且不在新剪影内的真实背景

    # 最近源填充（距离变换，精确到像素，无迭代深度限制）
    t0 = time.time()
    src_img = np.where(source, 0, 1).astype(np.uint8)      # 0 = 源
    dt, labels = cv2.distanceTransformWithLabels(src_img, cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.where(src_img == 0)
    lab = labels[zy, zx]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[lab] = np.arange(len(lab))
    nearest = lut[labels]                                   # 每个像素的最近源下标
    colors = person[y0:y1, x0:x1][zy, zx]                   # 源像素颜色
    fill = colors[nearest]
    base = person.copy()
    region = base[y0:y1, x0:x1].copy()
    region[shell] = fill[shell]
    base[y0:y1, x0:x1] = region
    t_fill = round(time.time() - t0, 2)

    # 重贴新人（与 B3 相同数学）
    t0 = time.time()
    ga = cv2.GaussianBlur(g_a, (7, 7), 0)
    a3 = np.repeat(ga[..., None], 3, 2)
    out = base.copy()
    out[y0:y1, x0:x1] = np.clip(
        out[y0:y1, x0:x1].astype(np.float32) * (1 - a3) + g_fg.astype(np.float32) * a3, 0, 255
    ).astype(np.uint8)
    t_paste = round(time.time() - t0, 2)

    cv2.imwrite(c["out_png"], out)
    cv2.imwrite(c["out_mask"], (shell * 255).astype(np.uint8))
    with open(c["out_meta"], "w", encoding="utf-8") as f:
        json.dump({"matting_s": t_mat, "fill_s": t_fill, "paste_s": t_paste,
                   "leftover_px": int(shell.sum())}, f, ensure_ascii=False, indent=2)
    # 橙色像素统计（诊断）
    b, g, r = out[..., 0].astype(int), out[..., 1].astype(int), out[..., 2].astype(int)
    n_orange = int(((r > 140) & (r - g > 60) & (g - b > 10)).sum())
    print("%-60s 残留=%d px 填充=%ss 重贴=%ss 余橙色=%d px" % (
        os.path.relpath(c["out_png"], HERE), int(shell.sum()), t_fill, t_paste, n_orange), flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for c in CASES:
        if only and only not in c["out_png"]:
            continue
        fix_case(c)
