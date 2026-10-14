#!/usr/bin/env python3
"""AI 换装流水线：抠图 → 小画布生成 → 抠出 → 贴回原背景（+ 残留背景填充修复）。

结论来源：tryon_demo/direct_vs_crop_test_20261014/报告.md（速度/画质/内存对照实验）。
要点：
- 输入缩放到长边 ≤1024 做抠图/生成（32GB 机型内存安全线，直接整图会触发 swap 慢数倍）
- rembg 抠人像/衣服 → 前景 bbox 裁切贴白底 → resolution=0 生成（画布 = 裁切尺寸，图1比例）
- 单次生成，不做自动重试；用「剪影内平均色差」检测是否换装成功（<25 给提示，由用户决定是否重试）
  （实测失败样本 8~12，成功样本 75+，阈值 25 有巨大余量）
- 贴回：生成结果抠图 → 旧剪影残留区域用扩散式背景填充 → 羽化贴回；
  upscale=True 时把生成人物放大到【原图分辨率】后再贴回（背景保持原图清晰度）
"""
import time

import cv2
import numpy as np

PROMPT = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
MAX_SIDE = 1024          # 输入缩放长边（内存安全线）
UPSCALE_MAX_SIDE = 4096  # 放大贴回时输出长边上限（超大原图自动压缩后贴回）
SUCCESS_DIFF = 25.0      # 剪影内平均色差阈值：< 则提示"可能未成功换装"


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
def _paste_with_fill(person, gen, p_a, box, scale=1.0):
    """把生成人物贴回原图并修复旧剪影残留。

    scale>1：把新人物以【底边中点】为锚点放大（默认 1.03），让新剪影盖住旧衣边缘的
    残留条带；锚点在底边可保持脚部落点不变。
    """
    x0, y0, x1, y1 = map(int, box)
    gw, gh = x1 - x0, y1 - y0
    g_fg, g_a = remove_bg(gen)

    # 目标框：以底边中点为中心放缩到 scale 倍
    nw, nh = max(1, int(round(gw * scale))), max(1, int(round(gh * scale)))
    cx = (x0 + x1) // 2
    nx0, ny0 = cx - nw // 2, y1 - nh

    H, W = person.shape[:2]
    wx0, wy0 = max(0, min(x0, nx0)), max(0, min(y0, ny0))
    wx1, wy1 = min(W, max(x1, nx0 + nw)), min(H, max(y1, ny0 + nh))
    ww, wh = wx1 - wx0, wy1 - wy0
    if ww <= 0 or wh <= 0:
        return person

    # 缩放后的新人物（前景 + alpha）放到工作区画布上（自动裁剪越界部分）
    fg_r = cv2.resize(g_fg, (nw, nh), interpolation=cv2.INTER_CUBIC)
    a_r = cv2.resize(g_a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    G_fg = np.zeros((wh, ww, 3), np.uint8)
    G_a = np.zeros((wh, ww), np.float32)
    sx0, sy0 = nx0 - wx0, ny0 - wy0
    dx0, dy0 = max(0, sx0), max(0, sy0)
    px0, py0 = max(0, -sx0), max(0, -sy0)
    dw, dh = min(nw - px0, ww - dx0), min(nh - py0, wh - dy0)
    if dw > 0 and dh > 0:
        G_fg[dy0:dy0 + dh, dx0:dx0 + dw] = fg_r[py0:py0 + dh, px0:px0 + dw]
        G_a[dy0:dy0 + dh, dx0:dx0 + dw] = a_r[py0:py0 + dh, px0:px0 + dw]

    pa_sub = p_a[wy0:wy1, wx0:wx1].astype(np.float32)
    new_sil = G_a > 0.5
    new_prot = cv2.dilate(new_sil.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    old_soft = cv2.morphologyEx((pa_sub > 0.02).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8)).astype(bool)
    old_soft = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    shell = old_soft & (~new_prot)                          # 旧衣残留（需按背景重填）
    old_guard = cv2.dilate(old_soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    source = (~old_guard) & (~new_sil)                      # 填充源：旧剪影外的真实背景

    base = person.copy()
    region = base[wy0:wy1, wx0:wx1].copy()
    if shell.any() and source.any():
        # 扩散式填充（归一化模糊迭代）：大面积区域的填色是周围真实背景的平滑过渡，
        # 不会把远处不同色的背景块硬搬过来。
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
        orig = base[wy0:wy1, wx0:wx1].astype(np.float32)
        region = (region.astype(np.float32) * m + orig * (1 - m)).astype(np.uint8)
    base[wy0:wy1, wx0:wx1] = region

    ga = cv2.GaussianBlur(G_a, (7, 7), 0)
    a3 = np.repeat(ga[..., None], 3, 2)
    reg2 = base[wy0:wy1, wx0:wx1].astype(np.float32)
    out = base.copy()
    out[wy0:wy1, wx0:wx1] = np.clip(
        reg2 * (1 - a3) + G_fg.astype(np.float32) * a3, 0, 255
    ).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 主流程
def run_tryon(person_bgr, clothes_bgr, generate_fn, prompt=None, upscale=False, scale=1.0, log=None):
    """完整换装。generate_fn(a_img, b_img, prompt) -> BGR 生成结果（由调用方提交 ComfyUI）。

    upscale=True：输出保持【原人像分辨率】（生成人物按比例放大回原图再贴回，背景更清晰）。
    scale：新人物贴回时的放大比例（默认 1.0；>1 时以底边中点放大，用于盖住旧衣边缘残留）。
    返回 {out, intermediates:[(label, img)], attempts, score, warn, timings, meta}
    """
    log = log or (lambda *a: None)
    prompt = (prompt or "").strip() or PROMPT
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

    t0 = time.time()
    gen = generate_fn(p_crop, c_crop, prompt)
    t["gen_s"] = round(time.time() - t0, 1)
    score = change_score(p_crop, gen)
    log("生成完成：画布 %dx%d，换装检测分 %.1f（阈值 %.0f）" % (
        gen.shape[1], gen.shape[0], score, SUCCESS_DIFF))
    if score < SUCCESS_DIFF:
        warn = "检测分 %.1f：本次可能未成功换装，可修改提示词后再试一次" % score

    t0 = time.time()
    if upscale and s_p < 1.0:
        h0, w0 = person_bgr.shape[:2]
        f2 = 1.0
        if max(h0, w0) > UPSCALE_MAX_SIDE:       # 超大原图：压缩后贴回，避免大矩阵爆内存
            f2 = UPSCALE_MAX_SIDE / max(h0, w0)
            base_img = cv2.resize(person_bgr, (max(8, int(round(w0 * f2))), max(8, int(round(h0 * f2)))),
                                  interpolation=cv2.INTER_AREA)
            log("原图 %dx%d 超过 %d，压缩后贴回" % (w0, h0, UPSCALE_MAX_SIDE))
        else:
            base_img = person_bgr
        hb, wb = base_img.shape[:2]
        p_a_full = cv2.resize(p_a, (wb, hb), interpolation=cv2.INTER_LINEAR)
        box_full = [int(round(v / s_p * f2)) for v in pbox]
        box_full[2] = min(box_full[2], wb)
        box_full[3] = min(box_full[3], hb)
        out = _paste_with_fill(base_img, gen, p_a_full, tuple(box_full), scale=scale)
        log("贴回 %dx%d：新人物放大 %.0f%%" % (wb, hb, scale * 100))
    else:
        out = _paste_with_fill(person, gen, p_a, pbox, scale=scale)
    t["paste_s"] = round(time.time() - t0, 2)

    meta = {
        "person_bbox": list(map(int, pbox)),
        "person_crop_wh": [p_crop.shape[1], p_crop.shape[0]],
        "gen_canvas_wh": [gen.shape[1], gen.shape[0]],
        "out_wh": [out.shape[1], out.shape[0]],
        "downscale": round(s_p, 3),
        "upscale": bool(upscale and s_p < 1.0),
        "scale": round(float(scale), 3),
        "prompt": prompt,
    }
    return {"out": out,
            "intermediates": [("人像（抠图裁切）", p_crop), ("衣服（抠图裁切）", c_crop), ("生成结果", gen)],
            "attempts": 1, "score": round(score, 1), "warn": warn,
            "timings": t, "meta": meta}
