#!/usr/bin/env python3
"""小尺寸输入版对照实验（用户指定：输入图缩小到不溢出的尺寸）。

输入缩放到 683x1024（长边 1024，0.70MP）——与 0 点前安全运行的量级一致。
A：缩小后的原图直接生成（resolution=0，画布≈672x1024）
B：rembg 抠图 → 小画布生成（≈288x928）→ 抠出 → 贴回小尺寸原人像背景
全部中间结果存到本目录 small_input_683x1024/。
"""
import base64
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.dirname(HERE)
ROOT = os.path.dirname(TD)
sys.path.insert(0, HERE)
sys.path.insert(0, TD)
import run_test as RT  # noqa: E402
import proc  # noqa: E402

# 复用 run_test 的保存函数，但指向本子目录
SUB = os.path.join(HERE, "small_input_683x1024")


def save(sub, name, img):
    d = os.path.join(SUB, sub)
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, name)
    cv2.imwrite(fp, img)
    return fp


def save_json(sub, name, obj):
    d = os.path.join(SUB, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    person = cv2.imread(RT.PERSON_SRC, cv2.IMREAD_COLOR)
    clothes = cv2.imread(RT.CLOTHES_SRC, cv2.IMREAD_COLOR)
    ps = cv2.resize(person, (683, 1024), interpolation=cv2.INTER_AREA)
    cs = cv2.resize(clothes, (683, 1024), interpolation=cv2.INTER_AREA)
    save("00_inputs", "person_683x1024.png", ps)
    save("00_inputs", "clothes_683x1024.png", cs)
    RT.log("小尺寸输入 %dx%d / %dx%d" % (ps.shape[1], ps.shape[0], cs.shape[1], cs.shape[0]))
    T = {"input": {"person": [683, 1024], "clothes": [683, 1024]}}

    # 预热（模型刚被 /free 卸载）
    RT.log("== 预热（不计入）==")
    t0 = time.time()
    proc.remove_bg(cv2.resize(ps, (256, 384)))
    RT.run_edit(cv2.resize(ps, (171, 256)), cv2.resize(cs, (256, 384)), 256, "W_small")
    RT.log("   预热 %.1fs" % (time.time() - t0))

    # -------- A：缩小的原图直接生成 --------
    RT.log("== A：直接生成（resolution=0，缩小原图）==")
    a_img, a_wall, a_comfy, raw = RT.run_edit(ps, cs, None, "A_small_direct")
    save("02_A", "A_direct_res0.png", a_img)
    if raw:
        shutil.copy(raw, os.path.join(SUB, "02_A", "A_direct_res0_comfy_raw.png"))
    T["A_direct"] = {"wall_s": a_wall, "comfy_s": a_comfy, "canvas": [a_img.shape[1], a_img.shape[0]]}
    RT.log("   A wall=%.1fs comfy=%s canvas=%dx%d" % (a_wall, a_comfy or -1, a_img.shape[1], a_img.shape[0]))

    # -------- B：抠图小画布生成 + 贴回 --------
    RT.log("== B：抠图 ==")
    t0 = time.time()
    p_fg, p_a = proc.remove_bg(ps)
    b_mat_p = round(time.time() - t0, 1)
    t0 = time.time()
    c_fg, c_a = proc.remove_bg(cs)
    b_mat_c = round(time.time() - t0, 1)
    p_crop, pbox = RT.cutout_white(p_fg, p_a)
    c_crop, cbox = RT.cutout_white(c_fg, c_a)
    save("01_cutout", "person_fg_rgba.png", RT.rgba(p_fg, p_a))
    save("01_cutout", "person_crop_white.png", p_crop)
    save("01_cutout", "clothes_crop_white.png", c_crop)
    RT.log("   人像 bbox=%s 裁切=%dx%d；衣服裁切=%dx%d；抠图 %ss+%ss" % (
        list(map(int, pbox)), p_crop.shape[1], p_crop.shape[0], c_crop.shape[1], c_crop.shape[0], b_mat_p, b_mat_c))

    RT.log("== B：小画布生成 ==")
    B1, b_wall, b_comfy, raw = RT.run_edit(p_crop, c_crop, None, "B_small_gen")
    save("03_B", "B_gen_crop.png", B1)
    if raw:
        shutil.copy(raw, os.path.join(SUB, "03_B", "B_gen_crop_comfy_raw.png"))

    RT.log("== B：贴回 ==")
    t0 = time.time()
    g_fg, g_a = proc.remove_bg(B1)
    b_mat_g = round(time.time() - t0, 1)
    save("03_B", "B_gen_cutout_rgba.png", RT.rgba(g_fg, g_a))
    t0 = time.time()
    x0, y0, x1, y1 = pbox
    gw, gh = x1 - x0, y1 - y0
    g_alpha = g_a
    if B1.shape[1] != gw or B1.shape[0] != gh:
        g_fg = cv2.resize(g_fg, (gw, gh))
        g_alpha = cv2.resize(g_alpha, (gw, gh))
    g_alpha = cv2.GaussianBlur(g_alpha, (7, 7), 0)
    a3 = np.repeat(g_alpha[..., None], 3, 2)
    B3 = ps.copy()
    B3[y0:y1, x0:x1] = np.clip(
        B3[y0:y1, x0:x1].astype(np.float32) * (1 - a3) + g_fg.astype(np.float32) * a3, 0, 255).astype(np.uint8)
    b_paste = round(time.time() - t0, 2)
    save("03_B", "B_pasteback.png", B3)

    b_total = round(b_mat_p + b_mat_c + b_wall + b_mat_g + b_paste, 1)
    T["B_crop_paste"] = {"matting_person_s": b_mat_p, "matting_clothes_s": b_mat_c,
                         "gen_wall_s": b_wall, "gen_comfy_s": b_comfy,
                         "gen_canvas": [B1.shape[1], B1.shape[0]],
                         "matting_gen_s": b_mat_g, "paste_s": b_paste, "total_s": b_total,
                         "person_bbox_xyxy": list(map(int, pbox))}
    RT.log("   B wall=%.1fs comfy=%s canvas=%dx%d；合计 %.1fs" % (
        b_wall, b_comfy or -1, B1.shape[1], B1.shape[0], b_total))
    save_json(".", "timing_small.json", T)
    RT.log(json.dumps(T, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
