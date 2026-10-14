#!/usr/bin/env python3
"""AI 换装流水线：抠图 → 小画布生成 → 抠出 → 贴回原背景（+ 残留背景填充修复）。

结论来源：tryon_demo/direct_vs_crop_test_20261014/报告.md（速度/画质/内存对照实验）。
要点：
- 输入缩放到长边 ≤1024（32GB 机型内存安全线，直接整图/大图会触发 swap 慢数倍）
- rembg 抠人像/衣服 → 前景 bbox 裁切贴白底 → resolution=0 生成（画布 = 裁切尺寸，图1比例）
- 换装失败自动检测（人像剪影内平均色差 < 25 → 判为没换装）→ 自动重试，最多 3 次
  （实测失败样本 8~12，成功样本 75+，阈值 25 有巨大余量）
- 生成结果抠图 → 旧剪影残留区域用最近真实背景像素填充（距离变换）→ 重贴新人物
"""
import time

import cv2
import numpy as np

PROMPT = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
MAX_SIDE = 1024          # 输入缩放长边（内存安全线）
MAX_ATTEMPTS = 3         # 生成尝试次数上限
SUCCESS_DIFF = 25.0      # 剪影内平均色差阈值：≥ 视为换装成功


# ---------------------------------------------------------------- 编解码
def encode_png_b64(img_bgr):
    import base64
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return base64.b64encode(buf).decode()


def decode_png_file(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("读取图片失败: " + path)
    return img


def decode_png_b64(b64):
    import base64
    raw = base64.b64decode(b64.split(",", 1)[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("图片数据无法解码")
    return img


def save_png(path, img_bgr):
    if not cv2.imwrite(path, img_bgr):
        raise RuntimeError("写图失败: " + path)
    return path


# ---------------------------------------------------------------- 抠图
def remove_bg(img_bgr):
    """rembg 抠图，返回 (前景BGR, alpha float32[0,1])；失败时返回原图与全 1 alpha。"""
    try:
        from io import BytesIO

        from PIL import Image
        from rembg import remove
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        rgba = np.array(Image.open(BytesIO(remove(buf.getvalue()))).convert("RGBA"))
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3].astype(np.float32) / 255.0
    except Exception:
        return img_bgr, np.ones(img_bgr.shape[:2], np.float32)


def _alpha_bbox(a):
    ys, xs = np.where(a > 0.05)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _crop_white(fg_bgr, alpha):
    """前景 bbox 裁切 + 白底合成（rembg 抠图 → 送模型的干净小图）。"""
    b = _alpha_bbox(alpha)
    if b is None:
        raise RuntimeError("抠图失败：未找到前景")
    x0, y0, x1, y1 = b
    fgc = fg_bgr[y0:y1, x0:x1].astype(np.float32)
    a = alpha[y0:y1, x0:x1]
    a3 = np.repeat(a[..., None], 3, 2)
    return (fgc * a3 + 255.0 * (1 - a3)).astype(np.uint8), b


def _fit(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA), s


# ---------------------------------------------------------------- 换装成功检测
def change_score(person_crop_white, gen_crop):
    """人像剪影（白底裁切中的非白区域）内的平均绝对色差。失败 8~12，成功 75+。"""
    if gen_crop.shape[:2] != person_crop_white.shape[:2]:
        gen_crop = cv2.resize(gen_crop, (person_crop_white.shape[1], person_crop_white.shape[0]))
    sil = np.abs(person_crop_white.astype(np.int16) - 255).sum(2) > 60
    if sil.sum() < 500:                      # 兜底：剪影异常时用全图
        sil = np.ones(person_crop_white.shape[:2], bool)
    d = np.abs(gen_crop.astype(np.float32) - person_crop_white.astype(np.float32)).mean(2)
    return float(d[sil].mean())


# ---------------------------------------------------------------- 贴回 + 残留修复
def _paste_with_fill(person, gen, p_a, box):
    """把生成人物贴回原图：先按「旧剪影残留 → 最近真实背景像素填充」，再羽化贴回。"""
    x0, y0, x1, y1 = box
    gw, gh = x1 - x0, y1 - y0
    g_fg, g_a = remove_bg(gen)
    if gen.shape[1] != gw or gen.shape[0] != gh:
        g_fg = cv2.resize(g_fg, (gw, gh))
        g_a = cv2.resize(g_a, (gw, gh))

    pa_sub = p_a[y0:y1, x0:x1].astype(np.float32)
    new_sil = g_a > 0.5
    new_prot = cv2.dilate(new_sil.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    old_soft = cv2.morphologyEx((pa_sub > 0.02).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8)).astype(bool)
    old_soft = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    shell = old_soft & (~new_prot)                          # 旧衣残留（需按背景重填）
    old_guard = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    source = (~old_guard) & (~new_sil)                      # 填充源：旧剪影外的真实背景

    base = person.copy()
    region = base[y0:y1, x0:x1].copy()
    if shell.any() and source.any():
        # 扩散式填充（归一化模糊迭代）：比“最近源填充”更稳——
        # 大面积区域的填色是周围真实背景的平滑过渡，不会把远处不同色的背景块硬搬过来。
        vals = region.astype(np.float32)
        known = source.astype(np.float32)
        to_fill = shell & (~source)
        for _ in range(12):
            num = cv2.blur(vals * known[..., None], (21, 21))
            den = cv2.blur(known, (21, 21))
            den3 = np.repeat(den[..., None], 3, 2)
            upd = to_fill & (den > 0.02)
            if not upd.any():
                break
            vals[upd] = (num[upd] / den3[upd])
            known[upd] = 1.0
        region[to_fill] = np.clip(vals, 0, 255).astype(np.uint8)[to_fill]
        # 填充边界做轻微羽化，避免硬边
        m = cv2.GaussianBlur(to_fill.astype(np.float32), (9, 9), 0)[..., None]
        orig = base[y0:y1, x0:x1].astype(np.float32)
        region = (region.astype(np.float32) * m + orig * (1 - m)).astype(np.uint8)
    base[y0:y1, x0:x1] = region

    ga = cv2.GaussianBlur(g_a, (7, 7), 0)
    a3 = np.repeat(ga[..., None], 3, 2)
    out = base.copy()
    out[y0:y1, x0:x1] = np.clip(
        out[y0:y1, x0:x1].astype(np.float32) * (1 - a3) + g_fg.astype(np.float32) * a3, 0, 255
    ).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 主流程
def run_tryon(person_bgr, clothes_bgr, generate_fn, log=None):
    """完整换装。generate_fn(a_img, b_img) -> BGR 生成结果（由调用方提交 ComfyUI）。

    返回 {out, intermediates:[(label, img)], attempts, score, warn, timings, meta}
    """
    log = log or (lambda *a: None)
    t = {}
    warn = ""

    t0 = time.time()
    person, s_p = _fit(person_bgr)
    clothes, _ = _fit(clothes_bgr)
    t["fit_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    p_fg, p_a = remove_bg(person)
    t["matting_person_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    c_fg, c_a = remove_bg(clothes)
    t["matting_clothes_s"] = round(time.time() - t0, 1)
    p_crop, pbox = _crop_white(p_fg, p_a)
    c_crop, _ = _crop_white(c_fg, c_a)
    log("抠图完成：人像 bbox=%s 裁切=%dx%d，衣服裁切=%dx%d" % (
        list(map(int, pbox)), p_crop.shape[1], p_crop.shape[0], c_crop.shape[1], c_crop.shape[0]))

    gen, score, attempts = None, 0.0, 0
    t["gen_s"] = []
    for i in range(MAX_ATTEMPTS):
        t0 = time.time()
        g = generate_fn(p_crop, c_crop)
        t["gen_s"].append(round(time.time() - t0, 1))
        attempts += 1
        score = change_score(p_crop, g)
        gen = g
        log("第 %d 次生成：画布 %dx%d，换装检测分 %.1f（阈值 %.0f）" % (
            attempts, g.shape[1], g.shape[0], score, SUCCESS_DIFF))
        if score >= SUCCESS_DIFF:
            break
        if i < MAX_ATTEMPTS - 1:
            log("未检测到换装变化，自动重试…")
    if score < SUCCESS_DIFF:
        warn = "已尝试 %d 次仍未检测到明显换装变化（检测分 %.1f），结果可能未成功，可重试或更换衣服图" % (attempts, score)

    t0 = time.time()
    out = _paste_with_fill(person, gen, p_a, pbox)
    t["paste_s"] = round(time.time() - t0, 2)

    meta = {
        "person_bbox": list(map(int, pbox)),
        "person_crop_wh": [p_crop.shape[1], p_crop.shape[0]],
        "gen_canvas_wh": [gen.shape[1], gen.shape[0]],
        "out_wh": [out.shape[1], out.shape[0]],
        "downscale": round(s_p, 3),
    }
    return {"out": out,
            "intermediates": [("人像（抠图裁切）", p_crop), ("衣服（抠图裁切）", c_crop), ("生成结果", gen)],
            "attempts": attempts, "score": round(score, 1), "warn": warn,
            "timings": t, "meta": meta}
