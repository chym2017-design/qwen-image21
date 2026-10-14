"""换装/换脸对照实验 · 图像处理（带选项 + 客观指标）。

非生成式（纯 OpenCV/insightface/rembg/gfpgan）路线。每个 process 返回 (结果b64, 说明, 指标dict)。

选项：
- tryon: blend=feather|poisson|poisson_mixed, hair=on/off
- face_swap: restore=on/off
- composite: seg=rembg|grabcut, restore=on/off
"""
import base64
import math
import os
import sys
import time
import types

import cv2
import numpy as np

# gfpgan/basicsr 兼容垫片（老代码引用已移除的 torchvision.transforms.functional_tensor）
try:
    import torchvision.transforms.functional as _F
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        _m = types.ModuleType("torchvision.transforms.functional_tensor")
        _m.rgb_to_grayscale = _F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _m
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# ---------------------------------------------------------------- YuNet
_YUNET = None
def detector():
    global _YUNET
    if _YUNET is None:
        _YUNET = cv2.FaceDetectorYN.create(os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"), "", (320, 320))
    return _YUNET

def detect_face(img):
    h, w = img.shape[:2]
    d = detector(); d.setInputSize((w, h))
    _, faces = d.detect(img)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    box = [int(best[0]), int(best[1]), int(best[2]), int(best[3])]
    lms = [(float(best[4 + 2 * i]), float(best[5 + 2 * i])) for i in range(5)]
    return box, lms, float(best[14])

def _guess_face(img):
    h, w = img.shape[:2]
    fw, fh = int(w * 0.42), int(h * 0.26)
    x, y = int(w * 0.5 - fw / 2), int(h * 0.06)
    lms = [(x + fw * 0.35, y + fh * 0.42), (x + fw * 0.65, y + fh * 0.42),
           (x + fw * 0.5, y + fh * 0.62), (x + fw * 0.4, y + fh * 0.82), (x + fw * 0.6, y + fh * 0.82)]
    return [x, y, fw, fh], lms

# ---------------------------------------------------------------- insightface / inswapper / gfpgan
_APP = None; _SW = None; _GFPGAN = None; _GFPGAN_TRIED = False

def _insf_app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l")
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP

def _inswapper():
    global _SW
    if _SW is None:
        import insightface
        p = os.path.join(MODELS, "inswapper_128.onnx")
        if os.path.exists(p):
            _SW = insightface.model_zoo.get_model(p, providers=["CPUExecutionProvider"])
    return _SW

def _face_area(f):
    b = f.bbox; return (b[2] - b[0]) * (b[3] - b[1])

def _gfpgan():
    global _GFPGAN, _GFPGAN_TRIED
    if _GFPGAN is None and not _GFPGAN_TRIED:
        _GFPGAN_TRIED = True
        p = os.path.join(MODELS, "GFPGANv1.4.pth")
        if os.path.exists(p):
            try:
                from gfpgan import GFPGANer
                _GFPGAN = GFPGANer(model_path=p, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
            except Exception:
                _GFPGAN = None
    return _GFPGAN

def restore_faces(img):
    g = _gfpgan()
    if g is None:
        return img
    try:
        _, _, out = g.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        return out if out is not None else img
    except Exception:
        return img

def remove_bg(img_bgr):
    try:
        from io import BytesIO
        from PIL import Image
        from rembg import remove
        buf = BytesIO(); Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        rgba = np.array(Image.open(BytesIO(remove(buf.getvalue()))).convert("RGBA"))
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3].astype(np.float32) / 255.0
    except Exception:
        return img_bgr, np.ones(img_bgr.shape[:2], np.float32)

# ---------------------------------------------------------------- 融合 / 贴图
def _seamless(work, piece, mask, cx, cy, flag=cv2.NORMAL_CLONE):
    ph, pw = piece.shape[:2]
    if ph < 8 or pw < 8 or ph >= work.shape[0] or pw >= work.shape[1] or int(mask.max()) == 0:
        return None
    cxp = min(max(int(cx), pw // 2), work.shape[1] - pw // 2 - 1)
    cyp = min(max(int(cy), ph // 2), work.shape[0] - ph // 2 - 1)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    try:
        return cv2.seamlessClone(piece, work, m, (cxp, cyp), flag)
    except Exception:
        return None

def _feather(work, piece, mask, box):
    dx, dy, dw, dh = box
    dx0, dy0 = max(0, dx), max(0, dy)
    dx1, dy1 = min(work.shape[1], dx + dw), min(work.shape[0], dy + dh)
    if dx1 <= dx0 or dy1 <= dy0:
        return work
    dw, dh = dx1 - dx0, dy1 - dy0
    p = cv2.resize(piece, (dw, dh)); m = cv2.resize(mask, (dw, dh)).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (5, 5), 0)
    blend = work.astype(np.float32)
    for c in range(3):
        blend[dy0:dy1, dx0:dx1, c] = blend[dy0:dy1, dx0:dx1, c] * (1 - m) + p[..., c] * m
    return np.clip(blend, 0, 255).astype(np.uint8)

def _blend(work, piece, mask, center, box, blend):
    """按融合方式分发。"""
    if blend == "feather":
        return _feather(work, piece, mask, box), "羽化贴"
    flag = cv2.MIXED_CLONE if blend == "poisson_mixed" else cv2.NORMAL_CLONE
    r = _seamless(work, piece, mask, center[0], center[1], flag)
    if r is not None:
        return r, "泊松(MIXED)" if blend == "poisson_mixed" else "泊松(NORMAL)"
    return _feather(work, piece, mask, box), "泊松失败回退羽化"

# ---------------------------------------------------------------- crude 贴脸（兜底）
def _align_matrix(slms, dlms):
    s_r, s_l = slms[0], slms[1]; d_r, d_l = dlms[0], dlms[1]
    s_c = ((s_r[0] + s_l[0]) / 2, (s_r[1] + s_l[1]) / 2); d_c = ((d_r[0] + d_l[0]) / 2, (d_r[1] + d_l[1]) / 2)
    angle = math.degrees(math.atan2(d_l[1] - d_r[1], d_l[0] - d_r[0]) - math.atan2(s_l[1] - s_r[1], s_l[0] - s_r[0]))
    scale = math.hypot(d_l[0] - d_r[0], d_l[1] - d_r[1]) / max(math.hypot(s_l[0] - s_r[0], s_l[1] - s_r[1]), 1e-6)
    M = cv2.getRotationMatrix2D(s_c, angle, scale); M[0, 2] += d_c[0] - s_c[0]; M[1, 2] += d_c[1] - s_c[1]
    return M

def _face_mask(shape, box, feather):
    mask = np.zeros(shape[:2], np.float32); x, y, w, h = box
    cv2.ellipse(mask, (int(x + w / 2), int(y + h / 2)), (int(w * 0.46), int(h * 0.62)), 0, 0, 360, 1.0, -1)
    k = max(5, feather * 2 + 1); k += (k % 2 == 0)
    return cv2.GaussianBlur(mask, (k, k), 0)

def _face_swap_crude(person_img, model_img):
    out = model_img.copy()
    dst = detect_face(model_img)
    box, lms = (dst[0], dst[1]) if dst else _guess_face(model_img)
    src = detect_face(person_img)
    if src is None: return out, "人像图未检测到脸"
    warped = cv2.warpAffine(person_img, _align_matrix(src[1], lms), (out.shape[1], out.shape[0]))
    x, y, w, h = box; m = int(min(w, h) * 0.55)
    x0, y0 = max(0, x - m), max(0, y - m); x1, y1 = min(out.shape[1], x + w + m), min(out.shape[0], y + h + m)
    if x1 <= x0 or y1 <= y0: return out, "脸区异常"
    dst_f = out[y0:y1, x0:x1].astype(np.float32); src_f = warped[y0:y1, x0:x1].astype(np.float32)
    nb = (src_f.sum(2) > 30) & (dst_f.sum(2) > 30)
    if nb.sum() > 50:
        for c in range(3):
            sm, ss = src_f[..., c][nb].mean(), src_f[..., c][nb].std() + 1e-6
            dm, ds = dst_f[..., c][nb].mean(), dst_f[..., c][nb].std() + 1e-6
            src_f[..., c] = (src_f[..., c] - sm) / ss * ds + dm
    src_f = np.clip(src_f, 0, 255)
    mask = _face_mask(out.shape, box, max(6, int(min(w, h) * 0.12))); sub = mask[y0:y1, x0:x1]
    blend = out.astype(np.float32)
    for c in range(3):
        blend[y0:y1, x0:x1, c] = blend[y0:y1, x0:x1, c] * (1 - sub) + src_f[..., c] * sub
    return np.clip(blend, 0, 255).astype(np.uint8), "crude 贴脸"

# ---------------------------------------------------------------- 方案2：换脸
def face_swap(person_img, model_img, restore=True, hair=False):
    try:
        sw = _inswapper()
        if sw is not None:
            app = _insf_app()
            sfaces = app.get(person_img); tfaces = app.get(model_img)
            if sfaces and tfaces:
                sface = max(sfaces, key=_face_area); out = model_img.copy()
                for tface in sorted(tfaces, key=_face_area, reverse=True):
                    out = sw.get(out, tface, sface, paste_back=True)
                if restore:
                    out = restore_faces(out)
                if hair:
                    pdst = detect_face(person_img); mdst = detect_face(model_img)
                    if pdst and mdst:
                        out = _hair_onto(out, person_img, pdst[0], mdst[0])
                return out, "inswapper" + ("+GFPGAN" if restore else "") + ("+换发型" if hair else "")
    except Exception:
        pass
    return _face_swap_crude(person_img, model_img)

# ---------------------------------------------------------------- 方案1：换装+换发型（可配融合方式）
def ar_tryon(person_img, model_img, blend="feather", hair=True):
    """方案1：把模特脸下方「衣服区」和脸上方「头发区」贴在【人像】上（AR 贴纸式）。
    方向：以人像为主体，贴模特的衣服/头发。"""
    pdst = detect_face(person_img); mdst = detect_face(model_img)
    if pdst is None or mdst is None:
        return person_img, "需两人都有清晰正脸", None
    px, py, pw, ph = pdst[0]; mx, my, mw, mh = mdst[0]
    mfg, malpha = remove_bg(model_img)
    out = person_img.copy(); used = []
    # 衣服：模特脸下 0.9~3.8 脸高，宽 ±1.3 脸宽（含剪影，不贴模特背景）
    my0 = max(0, int(my + mh * 0.9)); my1 = min(model_img.shape[0], int(my + mh * 3.8))
    mx0 = max(0, int(mx + mw / 2 - mw * 1.3)); mx1 = min(model_img.shape[1], int(mx + mw / 2 + mw * 1.3))
    py0 = max(0, int(py + ph * 0.9)); py1 = min(person_img.shape[0], int(py + ph * 3.8))
    px0 = max(0, int(px + pw / 2 - pw * 1.3)); px1 = min(person_img.shape[1], int(px + pw / 2 + pw * 1.3))
    box = (px0, py0, px1 - px0, py1 - py0)
    cp = mfg[my0:my1, mx0:mx1]; cm = (np.clip(malpha[my0:my1, mx0:mx1], 0, 1) * 255).astype(np.uint8)
    if cp.size and int(cm.max()) > 0 and px1 > px0 and py1 > py0:
        tw, th = px1 - px0, py1 - py0
        cp2 = cv2.resize(cp, (tw, th)); cm2 = cv2.resize(cm, (tw, th), interpolation=cv2.INTER_NEAREST)
        out, tag = _blend(out, cp2, cm2, (px0 + tw // 2, py0 + th // 2), box, blend); used.append("衣:" + tag)
    # 发型：模特脸上方 0.55 脸高 → 人像脸上方
    if hair:
        sy0 = max(0, int(my - mh * 0.55)); sx0 = max(0, int(mx + mw / 2 - mw * 0.9)); sx1 = min(model_img.shape[1], int(mx + mw / 2 + mw * 0.9))
        dy0 = max(0, int(py - ph * 0.55)); dx0 = max(0, int(px + pw / 2 - pw * 0.9)); dx1 = min(person_img.shape[1], int(px + pw / 2 + pw * 0.9))
        hp = mfg[sy0:my, sx0:sx1]; hm = (np.clip(malpha[sy0:my, sx0:sx1], 0, 1) * 255).astype(np.uint8)
        if hp.size and int(hm.max()) > 0 and py > dy0 and dx1 > dx0:
            hw, hh = dx1 - dx0, py - dy0
            hp2 = cv2.resize(hp, (hw, hh)); hm2 = cv2.resize(hm, (hw, hh), interpolation=cv2.INTER_NEAREST)
            out, tag = _blend(out, hp2, hm2, (dx0 + hw // 2, dy0 + hh // 2), (dx0, dy0, hw, hh), blend); used.append("发:" + tag)
    return out, "以人像为主，贴模特 " + " ".join(used) + ("" if hair else "（不换发型）"), box

# ---------------------------------------------------------------- 方案2.1：换脸 + 抠图 + 贴背景
def composite(person_img, model_img, bg_img, seg="rembg", restore=True, hair=False):
    swapped, msg = face_swap(person_img, model_img, restore=restore, hair=hair)
    h, w = swapped.shape[:2]
    if seg == "grabcut":
        fdst = detect_face(swapped)
        rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))
        if fdst:
            bx, by, bw, bh = fdst[0]
            rect = (max(0, bx - int(bw * 0.3)), max(0, by - int(bh * 0.4)),
                    min(w, bx + bw + int(bw * 0.5)) - max(0, bx - int(bw * 0.3)),
                    min(h, by + bh + int(bh * 3.2)) - max(0, by - int(bh * 0.4)))
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(swapped, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        except Exception:
            pass
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        fg = swapped
        seg_tag = "GrabCut"
    else:
        fg, alpha = remove_bg(swapped); seg_tag = "rembg"
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    bg = cv2.resize(bg_img, (w, h)).astype(np.float32); a3 = np.repeat(alpha[..., None], 3, 2)
    out = fg.astype(np.float32) * a3 + bg * (1 - a3)
    return np.clip(out, 0, 255).astype(np.uint8), msg + "；" + seg_tag + "抠图贴背景"

# ---------------------------------------------------------------- 客观指标
def identity_similarity(a, b):
    try:
        app = _insf_app()
        fa = app.get(a); fb = app.get(b)
        if not fa or not fb: return None
        return round(float(np.dot(max(fa, key=_face_area).normed_embedding, max(fb, key=_face_area).normed_embedding)), 4)
    except Exception:
        return None

def seam_energy(img, box):
    if not box: return None
    x0, y0 = max(0, box[0]), max(0, box[1]); x1, y1 = min(img.shape[1], box[0] + box[2]), min(img.shape[0], box[1] + box[3])
    if x1 <= x0 or y1 <= y0: return None
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY); lap = cv2.Laplacian(g, cv2.CV_64F)
    band = 12
    m = np.zeros(lap.shape, bool); m[:band, :] = m[-band:, :] = m[:, :band] = m[:, -band:] = True
    return round(float(np.abs(lap[m]).mean()), 2)

# ---------------------------------------------------------------- 方案3/4/5
_BAILIAN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_BAILIAN_KEY = None
def _bailian_key():
    global _BAILIAN_KEY
    if _BAILIAN_KEY is None:
        _BAILIAN_KEY = os.environ.get("BAILIAN_API_KEY") or ""
        if not _BAILIAN_KEY:
            try:
                import json as _json
                with open(os.path.expanduser("~/qwen-image21/simple_ui/settings.json"), encoding="utf-8") as f:
                    _BAILIAN_KEY = _json.load(f).get("bailian_api_key") or ""
            except Exception:
                _BAILIAN_KEY = ""
    return _BAILIAN_KEY

def _bailian_edit(person_img, clothes_img, prompt, size_wh, model="qwen-image-2.1-turbo"):
    import urllib.request, json as _json, base64 as _b64
    key = _bailian_key()
    if not key:
        raise RuntimeError("未配置百炼 API Key")
    def dur(img):
        ok, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + _b64.b64encode(buf).decode()
    content = [{"image": dur(person_img)}, {"image": dur(clothes_img)}, {"text": prompt}]
    params = {"n": 1, "prompt_extend": False}
    if size_wh:
        params["size"] = "%d*%d" % size_wh
    body = {"model": model, "input": {"messages": [{"role": "user", "content": content}]}, "parameters": params}
    req = urllib.request.Request(_BAILIAN_URL, data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = _json.load(r)
    if resp.get("code"):
        raise RuntimeError("百炼报错 %s: %s" % (resp.get("code"), resp.get("message", "")))
    url = resp["output"]["choices"][0]["message"]["content"][0]["image"]
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)

def _alpha_bbox(alpha):
    ys, xs = np.where(alpha > 0.05)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

def _rembg_crop(img_bgr):
    fg, alpha = remove_bg(img_bgr)
    b = _alpha_bbox(alpha)
    if b is None:
        return None, None
    x0, y0, x1, y1 = b
    bgra = np.dstack([fg[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)])
    return bgra, b

def _rembg_crop_rgb(img_bgr):
    """rembg 抠图 + 裁到前景 bbox，前景贴到白底（RGB；避免 RGBA 让百炼把 alpha 当颜色）。返回 (rgb, bbox)。"""
    fg, alpha = remove_bg(img_bgr)
    b = _alpha_bbox(alpha)
    if b is None:
        return None, None
    x0, y0, x1, y1 = b
    fgc = fg[y0:y1, x0:x1].astype(np.float32); ac = alpha[y0:y1, x0:x1]
    a3 = np.repeat(ac[..., None], 3, 2)
    rgb = (fgc * a3 + 255.0 * (1 - a3)).astype(np.uint8)
    return rgb, b

def scheme3(person_img, clothes_img):
    """方案3：百炼 qwen-image-2.1 生成式换装（用整图；抠孤立小图会让百炼乱调色/拼图）。"""
    h, w = person_img.shape[:2]
    prompt = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
    gen = _bailian_edit(person_img, clothes_img, prompt, (w, h))
    out = gen if gen.shape[2] == 3 else cv2.cvtColor(gen, cv2.COLOR_BGRA2BGR)
    return out, "百炼生成式换装（整图输入）"

def _hair_mask(img, face_box):
    """person 的头发掩膜 = 剪影 − 脸椭圆（限定在头部区域）。"""
    fx, fy, fw, fh = face_box
    fg, alpha = remove_bg(img)
    sil = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    fe = np.zeros(sil.shape, np.uint8)
    cv2.ellipse(fe, (int(fx + fw / 2), int(fy + fh / 2)), (int(fw * 0.52), int(fh * 0.62)), 0, 0, 360, 255, -1)
    hair = cv2.bitwise_and(sil, cv2.bitwise_not(fe))
    region = np.zeros(sil.shape, np.uint8)
    cv2.rectangle(region, (max(0, fx - int(fw * 0.9)), max(0, fy - int(fh * 1.1))),
                  (min(img.shape[1], fx + fw + int(fw * 0.9)), min(img.shape[0], fy + int(fh * 0.4))), 255, -1)
    return fg, cv2.bitwise_and(hair, region)

def _hair_onto(base, person_img, person_face_box, target_face_box):
    """把 person 的头发（剪影−脸椭圆）按脸框对齐缩放贴到 base 上（换发型）。"""
    pf = person_face_box; mf = target_face_box
    fg, hair = _hair_mask(person_img, pf)
    scale = mf[2] / max(pf[2], 1)
    pcx, pcy = pf[0] + pf[2] / 2, pf[1] + pf[3] / 2
    mcx, mcy = mf[0] + mf[2] / 2, mf[1] + mf[3] / 2
    M = np.float32([[scale, 0, mcx - scale * pcx], [0, scale, mcy - scale * pcy]])
    wf = cv2.warpAffine(fg, M, (base.shape[1], base.shape[0]))
    wh = cv2.warpAffine(hair, M, (base.shape[1], base.shape[0]))
    wh = cv2.GaussianBlur(wh, (7, 7), 0).astype(np.float32) / 255.0
    a3 = np.repeat(wh[..., None], 3, 2)
    return np.clip(base.astype(np.float32) * (1 - a3) + wf.astype(np.float32) * a3, 0, 255).astype(np.uint8)

def scheme4(person_img, clothes_img):
    """方案4：以衣服图为底 + inswapper换脸 + 贴person头发(换发型)。"""
    return face_swap(person_img, clothes_img, restore=True, hair=True)

def scheme5(person_img, clothes_img, bg_img):
    """方案5：方案4 + rembg抠图贴背景。"""
    sw4, msg = scheme4(person_img, clothes_img)
    fg, alpha = remove_bg(sw4)
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    h, w = sw4.shape[:2]
    bg = cv2.resize(bg_img, (w, h)).astype(np.float32); a3 = np.repeat(alpha[..., None], 3, 2)
    out = fg.astype(np.float32) * a3 + bg * (1 - a3)
    return np.clip(out, 0, 255).astype(np.uint8), msg + "；rembg 抠图贴背景"

# ---------------------------------------------------------------- 生成式换装（本地 ComfyUI / 云端百炼）
def _local_edit(person_img, clothes_img, prompt, res=1024):
    """用本机 ComfyUI(Qwen-Image-2.1) 做生成式换装（慢，约 1-2 分钟）。"""
    import base64 as _b64
    sd = os.path.expanduser("~/qwen-image21/simple_ui")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import server as SUI
    def b64(img):
        ok, buf = cv2.imencode(".png", img); return _b64.b64encode(buf).decode()
    data = {"mode": "edit", "quality": "turbo", "prompt": prompt,
            "refs": [{"b64": b64(person_img), "ext": "png"}, {"b64": b64(clothes_img), "ext": "png"}],
            "resolution": int(res)}
    wf = SUI.build_edit(data, "turbo", prompt, "")
    resp = SUI.comfy_post("/prompt", {"prompt": wf})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError("ComfyUI 未返回任务")
    for _ in range(320):
        time.sleep(2)
        h = SUI.comfy_get("/history/" + pid).get(pid)
        if not h:
            continue
        st = h.get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError("本地出图失败")
        if st.get("completed") or st.get("status_str") == "success":
            for out in h.get("outputs", {}).values():
                for im in out.get("images", []):
                    p = os.path.join(SUI.COMFY_OUTPUT, im.get("subfolder", ""), im["filename"])
                    g = cv2.imread(p)
                    if g is not None:
                        return g
            raise RuntimeError("本地无输出图")
    raise RuntimeError("本地出图超时（>5 分钟）")

def _ar_paste_clothes(base_img, other_img, base_face, other_face, blend="feather"):
    """把 other(衣服图) 脸下方的衣服区贴到 base 身上（带剪影掩膜）。"""
    px, py, pw, ph = base_face; mx, my, mw, mh = other_face
    mfg, malpha = remove_bg(other_img)
    my0 = max(0, int(my + mh * 0.9)); my1 = min(other_img.shape[0], int(my + mh * 3.8))
    mx0 = max(0, int(mx + mw / 2 - mw * 1.3)); mx1 = min(other_img.shape[1], int(mx + mw / 2 + mw * 1.3))
    py0 = max(0, int(py + ph * 0.9)); py1 = min(base_img.shape[0], int(py + ph * 3.8))
    px0 = max(0, int(px + pw / 2 - pw * 1.3)); px1 = min(base_img.shape[1], int(px + pw / 2 + pw * 1.3))
    box = (px0, py0, px1 - px0, py1 - py0)
    cp = mfg[my0:my1, mx0:mx1]; cm = (np.clip(malpha[my0:my1, mx0:mx1], 0, 1) * 255).astype(np.uint8)
    if not (cp.size and int(cm.max()) > 0 and px1 > px0 and py1 > py0):
        return base_img
    tw, th = px1 - px0, py1 - py0
    cp2 = cv2.resize(cp, (tw, th)); cm2 = cv2.resize(cm, (tw, th), interpolation=cv2.INTER_NEAREST)
    r, _ = _blend(base_img, cp2, cm2, (px0 + tw // 2, py0 + th // 2), box, blend)
    return r

def _gen_swap_crop(person_img, clothes_img, gen="local", res=None):
    """抠图式生成换装：rembg 抠人+抠衣成小图 → 生成（输出≈小图尺寸，更快）→ 抠出生成结果 → 按 bbox 贴回【原始人像背景】。"""
    P = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
    p_crop, pbox = _rembg_crop_rgb(person_img)
    c_crop, _ = _rembg_crop_rgb(clothes_img)
    if p_crop is None or c_crop is None:
        raise RuntimeError("抠图失败")
    r = res or int((p_crop.shape[0] * p_crop.shape[1]) ** 0.5)
    r = max(256, min(2048, int(r)))
    if gen == "local":
        g = _local_edit(p_crop, c_crop, P, r)
    else:
        g = _bailian_edit(p_crop, c_crop, P, (p_crop.shape[1], p_crop.shape[0]))
        g = g if g.shape[2] == 3 else cv2.cvtColor(g, cv2.COLOR_BGRA2BGR)
    g_rgb, g_a = remove_bg(g)
    x0, y0, x1, y1 = pbox; gw, gh = x1 - x0, y1 - y0
    if g_rgb.shape[1] != gw or g_rgb.shape[0] != gh:
        g_rgb = cv2.resize(g_rgb, (gw, gh)); g_a = cv2.resize(g_a, (gw, gh))
    g_a = cv2.GaussianBlur(g_a, (7, 7), 0)
    a3 = np.repeat(g_a[..., None], 3, 2)
    out = person_img.copy()
    out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1].astype(np.float32) * (1 - a3) + g_rgb.astype(np.float32) * a3, 0, 255).astype(np.uint8)
    return out

# ---------------------------------------------------------------- 统一组合入口（按选项穷举，不用方案编号）
def compose(person_img, clothes_img, bg_img, opts=None):
    """opts: base=person|clothes, face=bool, hair=bool, bg=keep|input|other,
    blend=feather|poisson|poisson_mixed, gen=off|local|cloud。"""
    opts = opts or {}
    base_side = opts.get("base", "person")
    do_face = bool(opts.get("face"))
    do_hair = bool(opts.get("hair"))
    bgmode = opts.get("bg", "keep")
    blend = opts.get("blend", "feather")
    gen = opts.get("gen", "off")
    base = (person_img if base_side == "person" else clothes_img).copy()
    other = clothes_img if base_side == "person" else person_img
    used = []
    fb = detect_face(base); fo = detect_face(other)
    bf = fb[0] if fb else None; of = fo[0] if fo else None
    # 1) 换装（主体=人像：贴/生成衣服图上的衣服到人像）
    if base_side == "person":
        if gen in ("local", "cloud"):
            if opts.get("crop"):
                base = _gen_swap_crop(person_img, clothes_img, gen, opts.get("res")); used.append("抠图生成+贴回原背景")
            else:
                prompt = "将<image2>的衣服穿到<image1>的人物身上，保持人物面部、姿态和背景不变"
                if gen == "local":
                    base = _local_edit(person_img, clothes_img, prompt, opts.get("res", 1024)); used.append("整图生成(本地)")
                else:
                    g = _bailian_edit(person_img, clothes_img, prompt, (person_img.shape[1], person_img.shape[0]))
                    base = g if g.shape[2] == 3 else cv2.cvtColor(g, cv2.COLOR_BGRA2BGR); used.append("整图生成(云端)")
        elif bf and of:
            base = _ar_paste_clothes(base, clothes_img, bf, of, blend); used.append("贴衣服")
    # 2) 换脸：把 other 的脸换到 base
    if do_face and bf and of:
        base = face_swap(other, base, restore=True, hair=False)[0]; used.append("换脸")
    # 3) 换发型：把 other 的头发贴到 base
    if do_hair:
        fb2 = detect_face(base); fo2 = detect_face(other)
        if fb2 and fo2:
            base = _hair_onto(base, other, fo2[0], fb2[0]); used.append("换发型")
    # 4) 换背景
    if bgmode in ("input", "other") and bg_img is not None:
        fg, alpha = remove_bg(base); alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
        h, w = base.shape[:2]; bg = cv2.resize(bg_img, (w, h)).astype(np.float32)
        a3 = np.repeat(alpha[..., None], 3, 2)
        base = np.clip(fg.astype(np.float32) * a3 + bg * (1 - a3), 0, 255).astype(np.uint8); used.append("换背景")
    label = "主体=%s 换脸=%s 换发型=%s 背景=%s 融合=%s 生成=%s" % (
        "人像" if base_side == "person" else "衣服图", "是" if do_face else "否",
        "是" if do_hair else "否", bgmode, blend, gen)
    if used:
        label += "  →  " + "+".join(used)
    return base, label

# ---------------------------------------------------------------- 编解码 + 派发
def _decode(b64):
    if not b64: return None
    if "," in b64[:60]: b64 = b64.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

def _encode(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else None

def process(method, person_b64, model_b64, bg_b64, options=None):
    options = options or {}
    person = _decode(person_b64); model = _decode(model_b64); bg = _decode(bg_b64)
    t0 = time.time()
    metrics = {}
    try:
        if method == "compose":
            if person is None or model is None: return None, "缺人像图或衣服图", {}
            r, label = compose(person, model, bg, options)
            metrics["用时"] = "%.1fs" % (time.time() - t0)
            if options.get("base") == "clothes":
                s = identity_similarity(r, person)
                if s is not None: metrics["身份相似度(与原人像)"] = s
            return _encode(r), label, metrics
        if method == "face_swap":
            if person is None or model is None: return None, "缺人像图或模特图", {}
            r, m = face_swap(person, model, restore=options.get("restore", True), hair=options.get("hair", False))
            s = identity_similarity(r, person)
            if s is not None: metrics["身份相似度"] = s
        elif method == "tryon":
            if person is None or model is None: return None, "缺人像图或模特图", {}
            r, m, box = ar_tryon(person, model, blend=options.get("blend", "poisson"), hair=options.get("hair", True))
            metrics["接缝能量"] = seam_energy(r, box)
        elif method == "composite":
            if person is None or model is None or bg is None: return None, "缺人像/模特/背景图", {}
            r, m = composite(person, model, bg, seg=options.get("seg", "rembg"), restore=options.get("restore", True), hair=options.get("hair", False))
            s = identity_similarity(r, person)
            if s is not None: metrics["身份相似度"] = s
        elif method == "scheme3":
            if person is None or model is None: return None, "缺人像图或模特图", {}
            r, m = scheme3(person, model)
            s = identity_similarity(r, person)
            if s is not None: metrics["身份相似度"] = s
        elif method == "scheme4":
            if person is None or model is None: return None, "缺人像图或模特图", {}
            r, m = scheme4(person, model)
            s = identity_similarity(r, person)
            if s is not None: metrics["身份相似度"] = s
        elif method == "scheme5":
            if person is None or model is None or bg is None: return None, "缺人像/模特/背景图", {}
            r, m = scheme5(person, model, bg)
            s = identity_similarity(r, person)
            if s is not None: metrics["身份相似度"] = s
        else:
            return None, "未知方案", {}
        metrics["用时"] = "%.1fs" % (time.time() - t0)
        return _encode(r), m, metrics
    except Exception as e:
        return None, "处理异常：" + repr(e), {}
