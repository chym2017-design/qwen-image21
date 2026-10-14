#!/usr/bin/env python3
"""本地模型试装对照实验：整图直接生成(A) vs 抠图生成+贴回原背景(B)。

输入：
  - 人像：tryon_demo/sample_model3.png（街道背景，较复杂）
  - 衣服：下载.png（绿裙模特图）
路线：
  A 原图直接进 ComfyUI(Qwen-Image-2.1 turbo) 编辑，模型自己重绘背景；
  B rembg 抠人像/衣服 → 白底裁切小图进模型 → 生成结果再抠图 → 按 bbox 贴回原人像（背景像素原样保留）。
产出：本目录下 00_inputs ~ 04_compare + timing.json，全程中间结果落盘。
"""
import base64
import json
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.dirname(HERE)                       # tryon_demo
ROOT = os.path.dirname(TD)                       # ~/qwen-image21
for _p in (TD, os.path.join(ROOT, "simple_ui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import proc                                      # noqa: E402
import server as SUI                             # noqa: E402

PERSON_SRC = os.path.join(TD, "sample_model3.png")
CLOTHES_SRC = os.path.join(ROOT, "下载.png")
PROMPT = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
COMFY_LOG = os.path.join(ROOT, "comfyui_new.log")


def log(*a):
    print(*a, flush=True)


def save_img(sub, name, img):
    d = os.path.join(HERE, sub)
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, name)
    cv2.imwrite(fp, img)
    return fp


def save_json(sub, name, obj):
    d = os.path.join(HERE, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def comfy_log_seconds(mark, timeout=30):
    """从日志 mark 偏移处解析 ComfyUI 自报的 'Prompt executed in X seconds'。"""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with open(COMFY_LOG, encoding="utf-8", errors="ignore") as f:
                f.seek(mark)
                tail = f.read()
            m = re.findall(r"Prompt executed in ([0-9.]+) seconds", tail)
            if m:
                return float(m[-1])
        except OSError:
            pass
        time.sleep(1)
    return None


def run_edit(a_img, b_img, res, tag):
    """提交一次本地编辑任务。res=None → resolution=0（保持原生尺寸）。"""
    def b64(im):
        ok, buf = cv2.imencode(".png", im)
        return base64.b64encode(buf).decode()

    data = {"mode": "edit", "quality": "turbo", "prompt": PROMPT,
            "refs": [{"b64": b64(a_img), "ext": "png"}, {"b64": b64(b_img), "ext": "png"}],
            "resolution": 1024 if res is None else int(res)}
    wf = SUI.build_edit(data, "turbo", PROMPT, "")
    if res is None:
        for n in wf.values():
            if n.get("class_type") == "TextEncodeQwenImage21":
                n["inputs"]["resolution"] = 0

    mark = os.path.getsize(COMFY_LOG)
    t0 = time.time()
    resp = SUI.comfy_post("/prompt", {"prompt": wf})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError("%s 提交失败: %r" % (tag, resp))
    got = None
    raw = None
    while time.time() - t0 < 1800:
        time.sleep(1)
        h = SUI.comfy_get("/history/" + pid).get(pid)
        if not h:
            continue
        st = h.get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError(tag + " ComfyUI 报错")
        if st.get("completed") or st.get("status_str") == "success":
            for o in h.get("outputs", {}).values():
                for im in o.get("images", []):
                    fp = os.path.join(SUI.COMFY_OUTPUT, im.get("subfolder", ""), im["filename"])
                    g = cv2.imread(fp)
                    if g is not None:
                        got, raw = g, fp
            break
    if got is None:
        raise RuntimeError(tag + " 等待超时")
    wall = time.time() - t0
    comfy_s = comfy_log_seconds(mark)
    log("   [%s] wall=%.1fs comfy=%.1fs canvas=%dx%d" % (tag, wall, comfy_s or -1, got.shape[1], got.shape[0]))
    return got, round(wall, 1), comfy_s, raw


def cutout_white(fg_bgr, alpha):
    """与 proc._rembg_crop_rgb 相同数学：前景 bbox 裁切 + 白底合成。"""
    b = proc._alpha_bbox(alpha)
    if b is None:
        raise RuntimeError("抠图 alpha 为空")
    x0, y0, x1, y1 = b
    fgc = fg_bgr[y0:y1, x0:x1].astype(np.float32)
    ac = alpha[y0:y1, x0:x1]
    a3 = np.repeat(ac[..., None], 3, 2)
    return (fgc * a3 + 255.0 * (1 - a3)).astype(np.uint8), b


def rgba(fg_bgr, alpha):
    return np.dstack([fg_bgr, (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])


def main():
    timing = {}

    # ---------------- 输入 ----------------
    person = cv2.imread(PERSON_SRC, cv2.IMREAD_COLOR)
    clothes = cv2.imread(CLOTHES_SRC, cv2.IMREAD_COLOR)
    assert person is not None and clothes is not None, "输入图读取失败"
    save_img("00_inputs", "person_complex_bg.png", person)
    save_img("00_inputs", "clothes.png", clothes)
    log("输入：人像 %dx%d（%s）；衣服 %dx%d（%s）" % (
        person.shape[1], person.shape[0], os.path.basename(PERSON_SRC),
        clothes.shape[1], clothes.shape[0], os.path.basename(CLOTHES_SRC)))
    timing["input"] = {"person": [person.shape[1], person.shape[0]],
                       "clothes": [clothes.shape[1], clothes.shape[0]]}

    # ---------------- 预热（不计入对比）：模型进内存 + rembg 加载 ----------------
    log("== 预热（不计入对比）==")
    t0 = time.time()
    proc.remove_bg(cv2.resize(person, (256, 384)))
    small_p = cv2.resize(person, (171, 256))
    small_c = cv2.resize(clothes, (256, 384))
    w_img, w_wall, _, w_raw = run_edit(small_p, small_c, 256, "W0_warmup")
    save_img("00_inputs", "warmup_not_counted.png", w_img)
    log("   预热合计 %.1fs（含模型加载）" % (time.time() - t0))

    # ---------------- 路线 A：整图直接生成 ----------------
    a_runs = [(None, "A1_native_res0", "原生尺寸(resolution=0，与输入同比例 1536 级)"),
              (1024, "A2_res1024", "1024 像素预算"),
              (512, "A3_res512", "512 像素预算（与 B 抠图画布同量级）")]
    for res, tag, note in a_runs:
        log("== 路线A %s（%s）==" % (tag, note))
        img, wall, comfy_s, raw = run_edit(person, clothes, res, tag)
        save_img("02_A_direct", tag + ".png", img)
        if raw:
            shutil.copy(raw, os.path.join(HERE, "02_A_direct", tag + "_comfy_raw.png"))
        timing[tag] = {"note": note, "wall_s": wall, "comfy_s": comfy_s,
                       "canvas": [img.shape[1], img.shape[0]]}

    # ---------------- 路线 B：抠图 → 生成 → 贴回 ----------------
    log("== 路线B：rembg 抠图 ==")
    b_t = {}
    t0 = time.time()
    p_fg, p_a = proc.remove_bg(person)
    b_t["matting_person_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    c_fg, c_a = proc.remove_bg(clothes)
    b_t["matting_clothes_s"] = round(time.time() - t0, 1)
    p_crop, pbox = cutout_white(p_fg, p_a)
    c_crop, cbox = cutout_white(c_fg, c_a)
    save_img("01_cutout", "person_fg_rgba.png", rgba(p_fg, p_a))
    save_img("01_cutout", "person_crop_white.png", p_crop)
    save_img("01_cutout", "clothes_fg_rgba.png", rgba(c_fg, c_a))
    save_img("01_cutout", "clothes_crop_white.png", c_crop)
    save_json("01_cutout", "cutout_meta.json", {
        "person_bbox_xyxy": list(map(int, pbox)), "person_crop_wh": [p_crop.shape[1], p_crop.shape[0]],
        "clothes_bbox_xyxy": list(map(int, cbox)), "clothes_crop_wh": [c_crop.shape[1], c_crop.shape[0]],
        "timing": b_t})
    log("   人像 bbox=%s 裁切=%dx%d；衣服裁切=%dx%d；抠图耗时 %ss + %ss" % (
        list(map(int, pbox)), p_crop.shape[1], p_crop.shape[0], c_crop.shape[1], c_crop.shape[0],
        b_t["matting_person_s"], b_t["matting_clothes_s"]))

    log("== 路线B：抠图画布生成 ==")
    B1, gen_wall, gen_comfy, raw = run_edit(p_crop, c_crop, None, "B1_gen_crop")
    save_img("03_B_crop_paste", "B1_gen_crop.png", B1)
    if raw:
        shutil.copy(raw, os.path.join(HERE, "03_B_crop_paste", "B1_gen_crop_comfy_raw.png"))

    log("== 路线B：生成结果抠图 + 贴回原背景 ==")
    t0 = time.time()
    g_fg, g_a = proc.remove_bg(B1)
    b_t["matting_gen_result_s"] = round(time.time() - t0, 1)
    save_img("03_B_crop_paste", "B2_gen_cutout_rgba.png", rgba(g_fg, g_a))
    t0 = time.time()
    x0, y0, x1, y1 = pbox
    gw, gh = x1 - x0, y1 - y0
    g_rgb, g_alpha = g_fg, g_a
    if B1.shape[1] != gw or B1.shape[0] != gh:
        g_rgb = cv2.resize(g_rgb, (gw, gh))
        g_alpha = cv2.resize(g_alpha, (gw, gh))
    g_alpha = cv2.GaussianBlur(g_alpha, (7, 7), 0)
    a3 = np.repeat(g_alpha[..., None], 3, 2)
    B3 = person.copy()
    B3[y0:y1, x0:x1] = np.clip(
        B3[y0:y1, x0:x1].astype(np.float32) * (1 - a3) + g_rgb.astype(np.float32) * a3, 0, 255
    ).astype(np.uint8)
    b_t["pasteback_s"] = round(time.time() - t0, 2)
    save_img("03_B_crop_paste", "B3_pasteback.png", B3)

    b_total = (b_t["matting_person_s"] + b_t["matting_clothes_s"] + gen_wall
               + b_t["matting_gen_result_s"] + b_t["pasteback_s"])
    timing["B_crop_paste"] = {
        "matting_person_s": b_t["matting_person_s"], "matting_clothes_s": b_t["matting_clothes_s"],
        "gen_wall_s": gen_wall, "gen_comfy_s": gen_comfy,
        "gen_canvas": [B1.shape[1], B1.shape[0]],
        "matting_gen_result_s": b_t["matting_gen_result_s"], "pasteback_s": b_t["pasteback_s"],
        "total_s": round(b_total, 1),
        "total_excl_matting_s": round(gen_wall + b_t["matting_gen_result_s"] + b_t["pasteback_s"], 1),
    }
    log("   B 合计 %.1fs（其中生成 %.1fs，抠图/贴图 %.1fs）" % (
        b_total, gen_wall, b_total - gen_wall))
    save_json("03_B_crop_paste", "B_meta.json", timing["B_crop_paste"])

    # ---------------- 客观指标：背景保真 ----------------
    log("== 对比指标 ==")
    def bg_drift(res_img):
        r = res_img if res_img.shape[:2] == person.shape[:2] else cv2.resize(res_img, (person.shape[1], person.shape[0]))
        outside = p_a <= 0.05
        d = np.abs(r.astype(np.float32) - person.astype(np.float32)).mean(2)
        return round(float(d[outside].mean()), 2)

    metrics = {"bg_drift_outside_person(A/B 越小=原背景保留越好)": {
        "A1_native": bg_drift(cv2.imread(os.path.join(HERE, "02_A_direct", "A1_native_res0.png"))),
        "A2_1024": bg_drift(cv2.imread(os.path.join(HERE, "02_A_direct", "A2_res1024.png"))),
        "A3_512": bg_drift(cv2.imread(os.path.join(HERE, "02_A_direct", "A3_res512.png"))),
        "B3_pasteback": bg_drift(B3)}}
    for tag, fp in [("A1_native", "02_A_direct/A1_native_res0.png"),
                    ("A2_1024", "02_A_direct/A2_res1024.png"),
                    ("B3_pasteback", "03_B_crop_paste/B3_pasteback.png")]:
        s = proc.identity_similarity(cv2.imread(os.path.join(HERE, fp)), person)
        if s is not None:
            metrics.setdefault("身份相似度(与原人像)", {})[tag] = s
    timing["metrics"] = metrics
    log("   背景漂移 %s" % metrics["bg_drift_outside_person(A/B 越小=原背景保留越好)"])
    log("   身份相似度 %s" % metrics.get("身份相似度(与原人像)"))

    # ---------------- 对比图 ----------------
    H = 1000
    def panel(img, label):
        sc = H / img.shape[0]
        p_ = cv2.resize(img, (max(1, int(img.shape[1] * sc)), H))
        bar = np.zeros((44, p_.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, p_])

    row = [panel(person, "input person (complex bg)"),
           panel(cv2.imread(os.path.join(HERE, "01_cutout", "person_crop_white.png")), "B: person cutout %dx%d" % (p_crop.shape[1], p_crop.shape[0])),
           panel(cv2.imread(os.path.join(HERE, "02_A_direct", "A2_res1024.png")), "A: direct gen 1024"),
           panel(B3, "B: crop gen + pasteback")]
    grid = np.hstack(row)
    if grid.shape[1] > 5600:
        sc = 5600 / grid.shape[1]
        grid = cv2.resize(grid, (int(grid.shape[1] * sc), int(grid.shape[0] * sc)))
    save_img("04_compare", "compare_grid.png", grid)

    save_json(".", "timing.json", timing)
    log("== 完成 ==")
    log(json.dumps(timing, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
