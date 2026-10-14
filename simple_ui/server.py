#!/usr/bin/env python3
"""Qwen-Image-2.1 简易出图界面 v2（文生图 / 多图编辑 / 画圈标注 / 蒙版局部重绘）。

- 只依赖 Python 标准库；推理由本机 ComfyUI 后端（127.0.0.1:8188）执行。
- 启动：  ~/qwen-image21/venv/bin/python ~/qwen-image21/simple_ui/server.py
- 访问：  http://127.0.0.1:8190
"""
import base64
import json
import os
import random
import secrets
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

COMFY = "http://127.0.0.1:8188"
ROOT = os.path.expanduser("~/qwen-image21")
WF_DIR = os.path.join(ROOT, "workflows")
COMFY_INPUT = os.path.join(ROOT, "ComfyUI", "input")
COMFY_OUTPUT = os.path.join(ROOT, "ComfyUI", "output")
SETTINGS_FILE = os.path.join(ROOT, "simple_ui", "settings.json")
PORT = 8190

try:  # 蒙版模式需要等比缩小源图；PIL 随 torch 一起装在 venv 里
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def normalize_dir(raw):
    """展开 ~ 并转绝对路径；空字符串返回空。"""
    d = (raw or "").strip()
    return os.path.abspath(os.path.expanduser(d)) if d else ""


# ---------------------------------------------------------------- 访问控制
MAX_BODY = 60 * 1024 * 1024      # 单次请求体上限（多图上传）
MAX_QUEUE = 2                    # 队列里超过这个数量就拒绝新任务（防滥用）
MIN_INTERVAL = 2.0               # 同一实例两次提交的最小间隔（秒）
_last_submit = 0.0


def access_token():
    """首次运行时生成访问口令，存进 settings.json；可用 QWEN_UI_TOKEN 环境变量覆盖。"""
    env = os.environ.get("QWEN_UI_TOKEN")
    if env:
        return env
    s = load_settings()
    tok = s.get("access_token")
    if not tok:
        tok = secrets.token_urlsafe(9)
        s["access_token"] = tok
        save_settings(s)
    return tok


def home_confined(path):
    """导出目录必须位于用户主目录内（防止公网访问时往系统任意路径写文件）。"""
    home = os.path.expanduser("~")
    return os.path.realpath(path).startswith(os.path.realpath(home) + os.sep)


def export_images(imgs):
    """把生成结果复制到用户设定的导出目录，返回 (导出目录, 错误信息)。"""
    raw = load_settings().get("export_dir", "")
    if not raw:
        return "", ""
    target = normalize_dir(raw)
    try:
        if os.path.realpath(target) == os.path.realpath(COMFY_OUTPUT):
            return "", ""
        os.makedirs(target, exist_ok=True)
        for im in imgs:
            src = os.path.join(COMFY_OUTPUT, im.get("subfolder", ""), im["name"])
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target, im["name"]))
        return target, ""
    except Exception as e:
        return "", str(e)


# ---------------------------------------------------------------- ComfyUI 交互
def load_wf(name):
    with open(os.path.join(WF_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def comfy_post(path, payload):
    req = urllib.request.Request(
        COMFY + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def comfy_get(path):
    with urllib.request.urlopen(COMFY + path, timeout=15) as r:
        return json.load(r)


def _patch_common(wf, prompt, negative, seed, prefix):
    for n in wf.values():
        t = n["class_type"]
        if t == "TextEncodeQwenImage21":
            n["inputs"]["prompt"] = prompt
            if negative:
                n["inputs"]["negative_prompt"] = negative
        elif t == "KSampler":
            n["inputs"]["seed"] = seed
        elif t == "RandomNoise":
            n["inputs"]["noise_seed"] = seed
        elif t == "SaveImage":
            n["inputs"]["filename_prefix"] = prefix


def _pick_seed(data):
    seed = data.get("seed")
    return int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)


def _clean_ext(raw):
    ext = (raw or "png").lower().lstrip(".")
    return ext if ext in ("png", "jpg", "jpeg", "webp") else "png"


def _apply_resolution(wf, resolution):
    """编辑模式的清晰度：参考图缩放目标 + 输出画幅（跟随图1比例）。"""
    try:
        r = int(resolution)
    except (TypeError, ValueError):
        return
    if r <= 0:
        return
    r = max(256, min(4096, r))
    for n in wf.values():
        if n["class_type"] == "TextEncodeQwenImage21":
            n["inputs"]["resolution"] = r


def build_t2i(data, quality, prompt, negative):
    wf = load_wf("turbo_t2i_api.json" if quality == "turbo" else "t2i_api.json")
    w = int(data.get("width") or 1328)
    h = int(data.get("height") or 1328)
    for n in wf.values():
        if n["class_type"] in ("EmptySD3LatentImage", "EmptyLatentImage"):
            n["inputs"]["width"] = w
            n["inputs"]["height"] = h
    _apply_resolution(wf, data.get("resolution"))
    _patch_common(wf, prompt, negative, _pick_seed(data), "qwen21_ui")
    return wf


def _fit_mask_source(ext, resolution):
    """蒙版模式：把源图与蒙版等比缩到清晰度档位以内（只缩不放），保持像素对齐。

    缩小后的源图另存为 PNG（避免 JPG 二次压缩损失），返回新文件名；无需缩小时返回 None。
    """
    if _PILImage is None:
        return None
    src = os.path.join(COMFY_INPUT, f"ui_ref_1.{ext}")
    msk = os.path.join(COMFY_INPUT, "ui_mask.png")
    try:
        im = _PILImage.open(src)
        mk = _PILImage.open(msk).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if w * h <= resolution * resolution:
        return None
    scale = (resolution * resolution / (w * h)) ** 0.5
    nw = max(8, round(w * scale / 8) * 8)
    nh = max(8, round(h * scale / 8) * 8)
    out_name = "ui_ref_1_fit.png"
    im.resize((nw, nh), _PILImage.LANCZOS).save(os.path.join(COMFY_INPUT, out_name))
    mk.resize((nw, nh), _PILImage.LANCZOS).save(msk)
    return out_name


def build_edit(data, quality, prompt, negative):
    refs = data.get("refs") or []
    if not refs:
        raise ValueError("请先添加至少一张参考图")
    if len(refs) > 10:
        raise ValueError("最多支持 10 张参考图")
    mask_b64 = data.get("mask_b64") or ""
    inpaint = bool(mask_b64)

    if inpaint:
        name = "turbo_inpaint_api.json" if quality == "turbo" else "inpaint_api.json"
    else:
        name = "turbo_edit_api.json" if quality == "turbo" else "edit_api.json"
    wf = load_wf(name)

    # 保存参考图并按顺序接线（第 1 张 = <image1>）
    for i, ref in enumerate(refs, 1):
        fname = f"ui_ref_{i}.{_clean_ext(ref.get('ext'))}"
        with open(os.path.join(COMFY_INPUT, fname), "wb") as f:
            f.write(base64.b64decode(ref["b64"]))
        if i == 1:
            wf["10"]["inputs"]["image"] = fname
        else:
            nid = str(200 + i)
            wf[nid] = {"class_type": "LoadImage", "inputs": {"image": fname}}
            wf["5"]["inputs"][f"images.image_{i}"] = [nid, 0]

    ins = wf["5"]["inputs"]
    for k in list(ins.keys()):
        if k.startswith("image") and not k.startswith("images."):
            ins.pop(k)
    ins["images.image_1"] = ["10", 0]

    if inpaint:
        with open(os.path.join(COMFY_INPUT, "ui_mask.png"), "wb") as f:
            f.write(base64.b64decode(mask_b64))
        for nid, n in wf.items():
            if n["class_type"] == "LoadImage" and nid != "10":
                n["inputs"]["image"] = "ui_mask.png"
                break
        # 蒙版模式的采样分辨率 = 源图尺寸，因此按清晰度档位等比缩小（源图与蒙版同步）
        try:
            res = int(data.get("resolution") or 0)
        except (TypeError, ValueError):
            res = 0
        if res > 0:
            fit = _fit_mask_source(_clean_ext(refs[0].get("ext")), max(256, min(4096, res)))
            if fit:
                wf["10"]["inputs"]["image"] = fit   # 指向缩小后的 PNG 源图

    _patch_common(wf, prompt, negative, _pick_seed(data), "qwen21_ui")
    _apply_resolution(wf, data.get("resolution"))
    return wf


def build_job(data):
    mode = data.get("mode", "t2i")
    quality = data.get("quality", "turbo")
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("请先填写提示词")
    negative = (data.get("negative") or "").strip()
    if mode == "t2i":
        return build_t2i(data, quality, prompt, negative)
    return build_edit(data, quality, prompt, negative)


# ---------------------------------------------------------------- 云端（阿里云百炼 qwen-image-2.1）
BAILIAN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
CLOUD_JOBS = {}   # job_id -> {state, images, error, ...}（存内存，进程重启即丢）
TRYON_JOBS = {}   # AI换装（抠图→生成→贴回）任务，同上


def bailian_key():
    """百炼 API Key：环境变量 BAILIAN_API_KEY 优先，否则读 settings.json 的 bailian_api_key。"""
    env = os.environ.get("BAILIAN_API_KEY")
    if env:
        return env
    return (load_settings().get("bailian_api_key") or "").strip()


def _bailian_model(quality):
    return "qwen-image-2.1-pro" if quality == "standard" else "qwen-image-2.1-turbo"


def bailian_generate(data, api_key, model):
    """把任务参数映射到百炼 DashScope 原生接口（multimodal-generation/generation），返回 (图片字节, usage)。
    文生图：parameters.size；图像编辑：content 里放 1–10 张 image(data URI) + text。size 用星号分隔。
    百炼 API 不支持 mask 蒙版——调用方需在上层拦截。"""
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("请先填写提示词")
    refs = data.get("refs") or []
    content = []
    for r in refs[:10]:
        content.append({"image": "data:image/%s;base64,%s" % (_clean_ext(r.get("ext")), r["b64"])})
    content.append({"text": prompt})
    parameters = {"n": 1, "prompt_extend": False}
    if not refs:
        w = int(data.get("width") or 1328)
        h = int(data.get("height") or 1328)
        w = max(512, min(2048, w))
        h = max(512, min(2048, h))
        parameters["size"] = "%d*%d" % (w, h)   # 原生接口 size 用星号分隔
    seed = str(data.get("seed") or "").strip()
    if seed:
        try:
            parameters["seed"] = int(seed)
        except ValueError:
            pass
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }
    req = urllib.request.Request(
        BAILIAN_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError("百炼返回 %d: %s" % (e.code, detail))
    except Exception as e:
        raise RuntimeError("调用百炼失败: %s" % e)
    if resp.get("code"):
        raise RuntimeError("百炼报错 %s: %s" % (resp.get("code"), resp.get("message", "")))
    try:
        url = resp["output"]["choices"][0]["message"]["content"][0]["image"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("百炼未返回图片: %s" % json.dumps(resp, ensure_ascii=False)[:400])
    with urllib.request.urlopen(url, timeout=120) as r:
        img = r.read()
    return img, resp.get("usage", {})


def _image_size(img_bytes):
    if _PILImage is None:
        return 0, 0
    try:
        return _PILImage.open(BytesIO(img_bytes)).size
    except Exception:
        return 0, 0


def _run_cloud_job(job_id, data, api_key, uid=None, cost=0):
    """后台线程：调百炼，下载结果图存到 ComfyUI/output（复用 /api/image 展示与下载）。
    失败时退还积分。"""
    try:
        os.makedirs(COMFY_OUTPUT, exist_ok=True)
        model = _bailian_model(data.get("quality", "turbo"))
        img, usage = bailian_generate(data, api_key, model)
        fname = "qwen21_cloud_%s.png" % job_id.replace("cloud-", "")
        with open(os.path.join(COMFY_OUTPUT, fname), "wb") as f:
            f.write(img)
        qs = urllib.parse.urlencode({"filename": fname, "subfolder": "", "type": "output"})
        CLOUD_JOBS[job_id] = {
            "state": "done",
            "images": [{"url": "/api/image?" + qs, "name": fname}],
            "error": "", "usage": usage, "model": model,
        }
    except Exception as e:
        if uid and cost:
            db.grant_credits(uid, cost, "云端失败退还")
        CLOUD_JOBS[job_id] = {"state": "error", "images": [], "error": str(e)}


# ---------------------------------------------------------------- AI 换装（抠图→生成→贴回原背景）
def _img_url(name):
    return "/api/image?" + urllib.parse.urlencode({"filename": name, "subfolder": "", "type": "output"})


def _tryon_generate(a_img, b_img, prompt, timeout=1500):
    """提交一次换装小画布生成（resolution=0，画布=图1裁切尺寸）并等待，返回 BGR 图。"""
    import tryon as _tryon        # 延迟导入（依赖 cv2/rembg/PIL）
    data = {"mode": "edit", "quality": "turbo", "prompt": prompt,
            "refs": [{"b64": _tryon.encode_png_b64(a_img), "ext": "png"},
                     {"b64": _tryon.encode_png_b64(b_img), "ext": "png"}],
            "resolution": 1024}
    wf = build_edit(data, "turbo", prompt, "")
    for n in wf.values():
        if n.get("class_type") == "TextEncodeQwenImage21":
            n["inputs"]["resolution"] = 0     # 保持裁切小图原生尺寸（不做缩放，快且省内存）
    resp = comfy_post("/prompt", {"prompt": wf})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError("ComfyUI 未返回任务: %r" % resp)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        h = comfy_get("/history/" + pid).get(pid)
        if not h:
            continue
        st = h.get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError("本地生成失败")
        if st.get("completed") or st.get("status_str") == "success":
            for out in h.get("outputs", {}).values():
                for im in out.get("images", []):
                    fp = os.path.join(COMFY_OUTPUT, im.get("subfolder", ""), im["filename"])
                    return _tryon.decode_png_file(fp)
            raise RuntimeError("本地无输出图")
    raise RuntimeError("本地生成超时")


def _run_tryon_job(job_id, person_b64, clothes_b64, prompt, upscale, uid=None, cost=0):
    """后台线程：抠图 → 小画布生成（单次）→ 贴回原背景（可选放大贴回），结果存 ComfyUI/output。"""
    import tryon as _tryon
    try:
        os.makedirs(COMFY_OUTPUT, exist_ok=True)
        tag = job_id.replace("tryon-", "")

        def log(*a):
            print("[tryon %s] %s" % (tag, " ".join(str(x) for x in a)), flush=True)

        person = _tryon.decode_png_b64(person_b64)
        clothes = _tryon.decode_png_b64(clothes_b64)
        log("提示词：%s ｜ 放大贴回=%s" % (prompt or _tryon.PROMPT, bool(upscale)))
        r = _tryon.run_tryon(person, clothes, _tryon_generate, prompt=prompt, upscale=upscale, log=log)

        fname = "qwen21_tryon_%s.png" % tag
        _tryon.save_png(os.path.join(COMFY_OUTPUT, fname), r["out"])
        stages = []
        for i, (label, img) in enumerate(r["intermediates"]):
            sn = "qwen21_tryon_%s_i%d.png" % (tag, i)
            _tryon.save_png(os.path.join(COMFY_OUTPUT, sn), img)
            stages.append({"label": label, "url": _img_url(sn), "name": sn})
        log("完成：尝试 %d 次，检测分 %.1f，用时 %s" % (r["attempts"], r["score"], r["timings"]))
        TRYON_JOBS[job_id] = {"state": "done",
                              "images": [{"url": _img_url(fname), "name": fname}],
                              "stages": stages, "warn": r["warn"], "score": r["score"],
                              "attempts": r["attempts"], "meta": r["meta"]}
    except Exception as e:
        if uid and cost:
            db.grant_credits(uid, cost, "换装失败退还")
        TRYON_JOBS[job_id] = {"state": "error", "images": [], "error": str(e)}


# ---------------------------------------------------------------- HTTP 服务
SIMPLE_UI_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_file(path):
    with open(path, "rb") as f:
        return f.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _redirect(self, to):
        self.send_response(302)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cookie(self, name):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return ""

    def _current_user(self):
        return db.get_user_by_session(self._cookie("qz_session"))

    def _set_session_cookie(self, sid):
        self.send_header("Set-Cookie",
                         "qz_session=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d" % (sid, db.SESSION_TTL))

    def _clear_session_cookie(self):
        self.send_header("Set-Cookie", "qz_session=; Path=/; HttpOnly; Max-Age=0")

    def _serve_comfy_image(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        want_dl = qs.pop("download", ["0"])[0] in ("1", "true")
        fname = (qs.get("filename") or ["image.png"])[0]
        subfolder = (qs.get("subfolder") or [""])[0]
        ftype = (qs.get("type") or ["output"])[0]
        base = COMFY_INPUT if ftype == "input" else COMFY_OUTPUT
        fpath = os.path.join(base, subfolder, fname)
        if not os.path.realpath(fpath).startswith(os.path.realpath(base) + os.sep):
            self._send(403, json.dumps({"error": "非法路径"})); return
        try:
            with open(fpath, "rb") as f:
                data = f.read()
        except Exception as e:
            self._send(404, json.dumps({"error": "取图失败: " + str(e)})); return
        low = fname.lower()
        ctype = "image/png" if low.endswith(".png") else ("image/jpeg" if low.endswith((".jpg", ".jpeg")) else "image/webp")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        if want_dl:
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % fname)
        self.end_headers()
        self.wfile.write(data)

    def _serve_lib_image(self, user):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            iid = int((qs.get("item") or ["0"])[0])
        except ValueError:
            self._send(400, json.dumps({"error": "item 无效"})); return
        it = db.get_item(iid)
        if not it or it["user_id"] != user["id"]:
            self._send(404, json.dumps({"error": "图片不存在"})); return
        fpath = os.path.join(db.DATA_DIR, "users", str(user["id"]), it["kind"], it["filename"])
        try:
            with open(fpath, "rb") as f:
                data = f.read()
        except Exception:
            self._send(404, json.dumps({"error": "取图失败"})); return
        low = it["filename"].lower()
        ctype = "image/png" if low.endswith(".png") else ("image/jpeg" if low.endswith((".jpg", ".jpeg")) else "image/webp")
        dlname = it["orig_name"] or it["filename"]
        want_dl = qs.get("download", ["0"])[0] in ("1", "true")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        if want_dl:
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % dlname)
        self.end_headers()
        self.wfile.write(data)

    def _status(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        pid = (qs.get("id") or [""])[0]
        if pid.startswith("cloud-"):
            job = CLOUD_JOBS.get(pid)
            if not job:
                self._send(200, json.dumps({"state": "error", "message": "云任务不存在（服务可能已重启）"}))
            elif job["state"] == "done":
                self._send(200, json.dumps({"state": "done", "images": job["images"]}))
            elif job["state"] == "error":
                self._send(200, json.dumps({"state": "error", "message": job["error"]}))
            else:
                self._send(200, json.dumps({"state": "running"}))
            return
        if pid.startswith("tryon-"):
            job = TRYON_JOBS.get(pid)
            if not job:
                self._send(200, json.dumps({"state": "error", "message": "换装任务不存在（服务可能已重启）"}))
            elif job["state"] == "done":
                self._send(200, json.dumps({"state": "done", "images": job["images"],
                                            "stages": job.get("stages", []),
                                            "warn": job.get("warn", ""),
                                            "attempts": job.get("attempts", 1)}))
            elif job["state"] == "error":
                self._send(200, json.dumps({"state": "error", "message": job["error"]}))
            else:
                self._send(200, json.dumps({"state": "running"}))
            return
        try:
            hist = comfy_get("/history/" + pid)
        except Exception as e:
            self._send(200, json.dumps({"state": "error", "message": str(e)})); return
        h = hist.get(pid)
        if not h:
            try:
                q = comfy_get("/queue")
                running = any(it[1] == pid for it in q.get("queue_running", []))
            except Exception:
                running = False
            self._send(200, json.dumps({"state": "running" if running else "queued"})); return
        st = h.get("status", {})
        if st.get("status_str") == "error":
            msg = ""
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    msg = m[1].get("exception_message", "")
            self._send(200, json.dumps({"state": "error", "message": msg})); return
        if not st.get("completed") and st.get("status_str") != "success":
            self._send(200, json.dumps({"state": "running"})); return
        imgs = []
        for out in h.get("outputs", {}).values():
            for im in out.get("images", []):
                qs2 = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                imgs.append({"url": "/api/image?" + qs2, "name": im["filename"], "subfolder": im.get("subfolder", "")})
        self._send(200, json.dumps({"state": "done", "images": imgs}))

    def _handle_signup(self, raw):
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        cfg = db.config_all()
        if cfg["allow_open_signup"] != "1":
            if not secrets.compare_digest((data.get("code") or "").strip(), cfg["signup_code"]):
                self._send(200, json.dumps({"error": "注册码不正确"})); return
        try:
            u = db.create_user((data.get("username") or "").strip(), data.get("password") or "", int(cfg["signup_bonus"]))
        except ValueError as e:
            self._send(200, json.dumps({"error": str(e)})); return
        sid = db.create_session(u["id"])
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_session_cookie(sid)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def _handle_login(self, raw):
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        u = db.verify_user(data.get("username") or "", data.get("password") or "")
        if not u:
            self._send(200, json.dumps({"error": "用户名或密码错误"})); return
        sid = db.create_session(u["id"])
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_session_cookie(sid)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def _handle_generate(self, raw, user):
        global _last_submit
        now = time.time()
        if now - _last_submit < MIN_INTERVAL:
            self._send(200, json.dumps({"error": "提交过于频繁，请稍候几秒"})); return
        try:
            data = json.loads(raw or b"{}")
        except Exception as e:
            self._send(200, json.dumps({"error": "请求解析失败: " + str(e)})); return
        backend = data.get("backend", "local")
        if backend == "cloud":
            if not user["allow_cloud"]:
                self._send(200, json.dumps({"error": "管理员已关闭你的云端出图权限"})); return
        elif not user["allow_local"]:
            self._send(200, json.dumps({"error": "管理员已关闭你的本地出图权限"})); return
        quality = data.get("quality", "turbo")
        cfg = db.config_all()
        cost = int(cfg["cost_standard"]) if quality == "standard" else int(cfg["cost_turbo"])
        reason = "出图-" + ("高质量" if quality == "standard" else "turbo") + ("(云端)" if backend == "cloud" else "(本地)")
        if not db.spend_credits(user["id"], cost, reason):
            self._send(200, json.dumps({"error": "积分不足（需要 %d，当前 %d）" % (cost, user["credits"])})); return
        if backend == "cloud":
            key = bailian_key()
            if not key:
                db.grant_credits(user["id"], cost, "退还：未配置云端Key")
                self._send(200, json.dumps({"error": "未配置百炼 API Key"})); return
            if data.get("mask_b64"):
                db.grant_credits(user["id"], cost, "退还：云端不支持蒙版")
                self._send(200, json.dumps({"error": "云端模式暂不支持蒙版局部重绘，请切到「本地」模式"})); return
            job_id = "cloud-" + secrets.token_hex(8)
            CLOUD_JOBS[job_id] = {"state": "running", "images": [], "error": ""}
            threading.Thread(target=_run_cloud_job, args=(job_id, data, key, user["id"], cost), daemon=True).start()
            _last_submit = now
            self._send(200, json.dumps({"prompt_id": job_id, "cost": cost})); return
        try:
            q = comfy_get("/queue")
            backlog = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
            if backlog >= MAX_QUEUE:
                db.grant_credits(user["id"], cost, "退还：队列繁忙")
                self._send(200, json.dumps({"error": "服务器繁忙（队列已有 %d 个任务），请稍后再试" % backlog})); return
        except Exception:
            pass
        try:
            wf = build_job(data)
            _last_submit = now
            resp = comfy_post("/prompt", {"prompt": wf})
            self._send(200, json.dumps({"prompt_id": resp.get("prompt_id"), "cost": cost}))
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:800]
            db.grant_credits(user["id"], cost, "退还：ComfyUI拒绝")
            self._send(200, json.dumps({"error": "ComfyUI 拒绝任务: " + detail}))
        except Exception as e:
            db.grant_credits(user["id"], cost, "退还：提交异常")
            self._send(200, json.dumps({"error": str(e)}))

    def _handle_tryon(self, raw, user):
        """AI 换装：抠图→小画布生成（失败自动重试）→贴回原背景。仅本地。"""
        global _last_submit
        now = time.time()
        if now - _last_submit < MIN_INTERVAL:
            self._send(200, json.dumps({"error": "提交过于频繁，请稍候几秒"})); return
        try:
            data = json.loads(raw or b"{}")
        except Exception as e:
            self._send(200, json.dumps({"error": "请求解析失败: " + str(e)})); return
        if not user["allow_local"]:
            self._send(200, json.dumps({"error": "管理员已关闭你的本地出图权限"})); return
        person_b64 = ((data.get("person") or {}).get("b64") or "").strip()
        clothes_b64 = ((data.get("clothes") or {}).get("b64") or "").strip()
        if not person_b64 or not clothes_b64:
            self._send(200, json.dumps({"error": "请先选择人像图和衣服图"})); return
        prompt = (data.get("prompt") or "").strip()[:600]
        upscale = bool(data.get("upscale"))
        cost = int(db.config_all()["cost_turbo"])
        if not db.spend_credits(user["id"], cost, "AI换装(本地)"):
            self._send(200, json.dumps({"error": "积分不足（需要 %d，当前 %d）" % (cost, user["credits"])})); return
        try:
            q = comfy_get("/queue")
            backlog = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
            if backlog >= MAX_QUEUE:
                db.grant_credits(user["id"], cost, "退还：队列繁忙")
                self._send(200, json.dumps({"error": "服务器繁忙（队列已有 %d 个任务），请稍后再试" % backlog})); return
        except Exception:
            pass
        job_id = "tryon-" + secrets.token_hex(8)
        TRYON_JOBS[job_id] = {"state": "running", "images": [], "error": ""}
        threading.Thread(target=_run_tryon_job,
                         args=(job_id, person_b64, clothes_b64, prompt, upscale, user["id"], cost),
                         daemon=True).start()
        _last_submit = now
        self._send(200, json.dumps({"prompt_id": job_id, "cost": cost}))

    def _handle_item_upload(self, raw, user):
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        kind = data.get("kind")
        if kind not in db.KINDS:
            self._send(400, json.dumps({"error": "kind 无效"})); return
        b64 = data.get("b64") or ""
        if not b64:
            self._send(400, json.dumps({"error": "缺少图片数据"})); return
        try:
            img_bytes = base64.b64decode(b64)
        except Exception:
            self._send(400, json.dumps({"error": "base64 解码失败"})); return
        w, h = _image_size(img_bytes)
        ext = _clean_ext(data.get("ext"))
        fname = "%d_%d.%s" % (user["id"], int(time.time() * 1000), ext)
        with open(os.path.join(db.user_dir(user["id"], kind), fname), "wb") as f:
            f.write(img_bytes)
        iid = db.add_item(user["id"], kind, fname, data.get("orig_name") or "", w, h)
        db.set_item_tags(iid, data.get("tags") or [])
        self._send(200, json.dumps({"ok": True, "id": iid}))

    def _handle_item_fromfile(self, raw, user):
        """把刚生成的结果图（在 ComfyUI/output）存入个人仓库。"""
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        kind = data.get("kind")
        if kind not in db.KINDS:
            self._send(400, json.dumps({"error": "kind 无效"})); return
        fname = data.get("filename")
        if not fname:
            self._send(400, json.dumps({"error": "缺少 filename"})); return
        subfolder = data.get("subfolder", "")
        ftype = data.get("type", "output")
        base = COMFY_INPUT if ftype == "input" else COMFY_OUTPUT
        src = os.path.join(base, subfolder, fname)
        if not os.path.realpath(src).startswith(os.path.realpath(base) + os.sep):
            self._send(403, json.dumps({"error": "非法路径"})); return
        try:
            with open(src, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            self._send(404, json.dumps({"error": "源图不存在: " + str(e)})); return
        w, h = _image_size(img_bytes)
        ext = _clean_ext(fname.rsplit(".", 1)[-1] if "." in fname else "png")
        new_fname = "%d_%d.%s" % (user["id"], int(time.time() * 1000), ext)
        with open(os.path.join(db.user_dir(user["id"], kind), new_fname), "wb") as f:
            f.write(img_bytes)
        iid = db.add_item(user["id"], kind, new_fname, fname, w, h)
        db.set_item_tags(iid, data.get("tags") or [])
        self._send(200, json.dumps({"ok": True, "id": iid}))

    def _handle_tag_add(self, raw, user):
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        try:
            tid = db.add_tag(user["id"], (data.get("name") or "").strip(), data.get("kind") or "自定义")
        except ValueError as e:
            self._send(200, json.dumps({"error": str(e)})); return
        self._send(200, json.dumps({"ok": True, "id": tid}))

    def _handle_item_tags(self, path, raw, user):
        try:
            iid = int(path.split("/")[3])
        except (ValueError, IndexError):
            self._send(400, json.dumps({"error": "id 无效"})); return
        it = db.get_item(iid)
        if not it or it["user_id"] != user["id"]:
            self._send(404, json.dumps({"error": "图片不存在"})); return
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        db.set_item_tags(iid, data.get("tag_ids") or [])
        self._send(200, json.dumps({"ok": True}))

    def _handle_item_rename(self, path, raw, user):
        try:
            iid = int(path.split("/")[3])
        except (ValueError, IndexError):
            self._send(400, json.dumps({"error": "id 无效"})); return
        it = db.get_item(iid)
        if not it or it["user_id"] != user["id"]:
            self._send(404, json.dumps({"error": "图片不存在"})); return
        try:
            data = json.loads(raw or b"{}")
            db.rename_item(iid, user["id"], data.get("name") or "")
        except ValueError as e:
            self._send(200, json.dumps({"error": str(e)})); return
        self._send(200, json.dumps({"ok": True}))

    def _handle_admin_user(self, raw, user):
        if user["role"] != "admin":
            self._send(403, json.dumps({"error": "需要管理员权限"})); return
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        uid = int(data.get("id") or 0)
        if not db.get_user(uid):
            self._send(404, json.dumps({"error": "用户不存在"})); return
        fields = {}
        for k in ("role", "credits", "is_member", "allow_local", "allow_cloud"):
            if k in data:
                fields[k] = data[k]
        db.update_user(uid, **fields)
        self._send(200, json.dumps({"ok": True}))

    def _handle_admin_config(self, raw, user):
        if user["role"] != "admin":
            self._send(403, json.dumps({"error": "需要管理员权限"})); return
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "请求格式错误"})); return
        for k in ("signup_code", "allow_open_signup", "cost_turbo", "cost_standard",
                  "signup_bonus", "member_daily", "member_daily_cap"):
            if k in data:
                db.config_set(k, str(data[k]))
        self._send(200, json.dumps({"ok": True, "config": db.config_all()}))

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/login":
            self._send(200, _read_file(os.path.join(SIMPLE_UI_DIR, "login.html")), "text/html; charset=utf-8")
            return
        user = self._current_user()
        if not user:
            if path.startswith("/api/"):
                self._send(401, json.dumps({"error": "未登录"}))
            else:
                self._redirect("/login")
            return
        if path in ("/", "/index.html"):
            self._send(200, _read_file(os.path.join(SIMPLE_UI_DIR, "app.html")), "text/html; charset=utf-8")
        elif path == "/api/me":
            self._send(200, json.dumps({"user": {
                "id": user["id"], "username": user["username"], "role": user["role"],
                "credits": user["credits"], "is_member": user["is_member"],
                "allow_local": user["allow_local"], "allow_cloud": user["allow_cloud"]}}))
        elif path == "/api/settings":
            s = load_settings()
            cfg = db.config_all()
            self._send(200, json.dumps({
                "export_dir": s.get("export_dir", ""), "normalized": normalize_dir(s.get("export_dir", "")),
                "cloud_key_set": bool(bailian_key()), "default_backend": s.get("default_backend", "local"),
                "cost_turbo": int(cfg["cost_turbo"]), "cost_standard": int(cfg["cost_standard"])}))
        elif path == "/api/health":
            try:
                comfy_get("/system_stats"); self._send(200, json.dumps({"comfy": True}))
            except Exception:
                self._send(200, json.dumps({"comfy": False}))
        elif path == "/api/image":
            self._serve_comfy_image()
        elif path == "/api/lib":
            self._serve_lib_image(user)
        elif path == "/api/mem":
            try:
                d = comfy_get("/system_stats")["system"]
                self._send(200, json.dumps({"total_gb": round(d["ram_total"] / 1e9, 1),
                                            "used_gb": round((d["ram_total"] - d["ram_free"]) / 1e9, 1)}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif path.startswith("/api/status"):
            self._status()
        elif path == "/api/items":
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            kind = (qs.get("kind") or [""])[0]
            if kind not in db.KINDS:
                self._send(400, json.dumps({"error": "kind 无效"})); return
            self._send(200, json.dumps({"items": db.items_with_tags(user["id"], kind)}))
        elif path == "/api/tags":
            self._send(200, json.dumps({"tags": db.list_tags(user["id"])}))
        elif path == "/api/admin/users":
            if user["role"] != "admin":
                self._send(403, json.dumps({"error": "需要管理员权限"})); return
            self._send(200, json.dumps({"users": db.list_users()}))
        elif path == "/api/admin/config":
            if user["role"] != "admin":
                self._send(403, json.dumps({"error": "需要管理员权限"})); return
            self._send(200, json.dumps({"config": db.config_all()}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            self._send(413, json.dumps({"error": "请求体过大（上限 %dMB）" % (MAX_BODY // 1024 // 1024)})); return
        raw = self.rfile.read(n) or b"{}"
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/signup":
            self._handle_signup(raw); return
        if path == "/api/login":
            self._handle_login(raw); return
        if path == "/api/logout":
            db.delete_session(self._cookie("qz_session"))
            self.send_response(200); self._clear_session_cookie()
            self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}"); return
        user = self._current_user()
        if not user:
            self._send(401, json.dumps({"error": "未登录"})); return
        if path == "/api/free":
            try:
                req = urllib.request.Request(COMFY + "/free",
                    data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=60)
                self._send(200, json.dumps({"ok": True}))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}))
            return
        if path == "/api/settings":
            try:
                data = json.loads(raw or b"{}")
                d = (data.get("export_dir") or "").strip()
                norm = normalize_dir(d)
                if norm and not home_confined(norm):
                    self._send(200, json.dumps({"ok": False, "error": "导出目录必须位于用户主目录内"})); return
                if norm: os.makedirs(norm, exist_ok=True)
                s = load_settings(); s["export_dir"] = d; save_settings(s)
                self._send(200, json.dumps({"ok": True, "export_dir": d, "normalized": norm}))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}))
            return
        if path == "/api/items":
            self._handle_item_upload(raw, user); return
        if path == "/api/items/fromfile":
            self._handle_item_fromfile(raw, user); return
        if path == "/api/tags":
            self._handle_tag_add(raw, user); return
        if path == "/api/tryon":
            self._handle_tryon(raw, user); return
        if path.startswith("/api/items/") and path.endswith("/tags"):
            self._handle_item_tags(path, raw, user); return
        if path.startswith("/api/items/") and path.endswith("/rename"):
            self._handle_item_rename(path, raw, user); return
        if path == "/api/admin/user":
            self._handle_admin_user(raw, user); return
        if path == "/api/admin/config":
            self._handle_admin_config(raw, user); return
        if path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"})); return
        self._handle_generate(raw, user)

    def do_DELETE(self):
        user = self._current_user()
        if not user:
            self._send(401, json.dumps({"error": "未登录"})); return
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/items/"):
            try:
                iid = int(path.split("/")[3])
            except (ValueError, IndexError):
                self._send(400, json.dumps({"error": "id 无效"})); return
            if not db.delete_item(iid, user["id"]):
                self._send(404, json.dumps({"error": "图片不存在"})); return
            self._send(200, json.dumps({"ok": True})); return
        if path.startswith("/api/tags/"):
            try:
                tid = int(path.split("/")[3])
            except (ValueError, IndexError):
                self._send(400, json.dumps({"error": "id 无效"})); return
            db.delete_tag(user["id"], tid)
            self._send(200, json.dumps({"ok": True})); return
        self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    db.init_db()
    if not db.config_get("signup_code"):
        db.config_set("signup_code", access_token())   # 默认注册码 = 旧访问口令
    print("简易出图界面 v3: http://127.0.0.1:%d" % PORT)
    print("首个注册用户自动成为管理员；注册码默认为旧访问口令（见 settings.json 的 access_token）。")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
