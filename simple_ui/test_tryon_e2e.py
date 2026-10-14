#!/usr/bin/env python3
# 用法：先启动 8190（./start_all.sh），再执行本脚本：
#   ./venv/bin/python simple_ui/test_tryon_e2e.py
# 作用：直接建会话（跳过登录）→ 提交一次 AI 换装 → 轮询 → 把结果与中间图下载到 /tmp/tryon_e2e
"""AI换装 E2E 测试：登录（直接建会话）→ POST /api/tryon → 轮询 → 下载结果与中间图。"""
import base64
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.expanduser("~/qwen-image21")
sys.path.insert(0, os.path.join(ROOT, "simple_ui"))
import db  # noqa: E402

BASE = "http://127.0.0.1:8190"
OUT = "/tmp/tryon_e2e"


def api(path, data=None, sid=""):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Cookie": "qz_session=" + sid})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    db.init_db()
    users = db.list_users()
    print("现有用户:", [(u["id"], u["username"]) for u in users])
    uid = users[0]["id"]
    sid = db.create_session(uid)
    print("已为用户 %s 创建会话 %s..." % (uid, sid[:8]))

    me = api("/api/me", sid=sid)["user"]
    print("身份确认:", me["username"], "积分:", me["credits"])

    person = open(os.path.join(ROOT, "tryon_demo/direct_vs_crop_test_20261014/00_inputs/person_complex_bg.png"), "rb").read()
    clothes = open(os.path.join(ROOT, "下载.png"), "rb").read()
    body = {"person": {"b64": base64.b64encode(person).decode(), "ext": "png"},
            "clothes": {"b64": base64.b64encode(clothes).decode(), "ext": "png"}}
    t0 = time.time()
    resp = api("/api/tryon", body, sid=sid)
    print("提交响应:", resp)
    if resp.get("error"):
        return
    jid = resp["prompt_id"]
    while True:
        time.sleep(5)
        st = api("/api/status?id=" + jid, sid=sid)
        el = int(time.time() - t0)
        if st["state"] == "running":
            print("  [%3ds] 运行中…" % el)
            continue
        print("  [%3ds] 结束: state=%s" % (el, st["state"]))
        if st["state"] == "error":
            print("错误:", st.get("message"))
            return
        print("尝试次数:", st.get("attempts"), "警告:", repr(st.get("warn") or ""))
        os.makedirs(OUT, exist_ok=True)
        for i, im in enumerate(st.get("images", [])):
            req2 = urllib.request.Request(BASE + im["url"], headers={"Cookie": "qz_session=" + sid})
            with urllib.request.urlopen(req2, timeout=60) as r:
                data = r.read()
            p = os.path.join(OUT, "result_%d_%s" % (i, im["name"]))
            open(p, "wb").write(data)
            print("结果已存:", p, len(data), "bytes")
        for s in st.get("stages", []):
            req3 = urllib.request.Request(BASE + s["url"], headers={"Cookie": "qz_session=" + sid})
            with urllib.request.urlopen(req3, timeout=60) as r:
                data = r.read()
            p = os.path.join(OUT, "stage_%s" % s["name"])
            open(p, "wb").write(data)
            print("中间图:", s["label"], "->", p, len(data), "bytes")
        print("总用时 %ds" % el)
        break


if __name__ == "__main__":
    main()
