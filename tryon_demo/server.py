#!/usr/bin/env python3
"""换装/换脸三方案对照实验 · 本地 demo 服务（端口 8191）。

不触碰 simple_ui(8190)。启动： ~/qwen-image21/venv/bin/python tryon_demo/server.py
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/qwen-image21")
PORT = 8191

import proc  # noqa: E402

SAMPLES = {
    "person": os.path.join(HERE, "sample_person.png"),
    "model": os.path.join(HERE, "sample_model3.png"),
    "bg": os.path.join(ROOT, "ComfyUI", "input", "input_shop.png"),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif p == "/api/sample":
            q = parse_qs(urlparse(self.path).query)
            name = (q.get("name") or ["person"])[0]
            path = SAMPLES.get(name)
            if not path or not os.path.exists(path):
                self._send(404, json.dumps({"error": "无此样图"})); return
            self._send(200, open(path, "rb").read(), "image/png")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        from urllib.parse import urlparse
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        if p == "/api/proc":
            try:
                data = json.loads(raw)
            except Exception:
                self._send(400, json.dumps({"error": "请求格式错误"})); return
            b64, msg, metrics = proc.process(data.get("method"), data.get("person"), data.get("model"), data.get("bg"), data.get("options"))
            if b64 is None:
                self._send(200, json.dumps({"error": msg})); return
            self._send(200, json.dumps({"result": b64, "msg": msg, "metrics": metrics}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print("换装/换脸对照实验: http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
