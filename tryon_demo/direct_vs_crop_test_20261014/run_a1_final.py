#!/usr/bin/env python3
"""A1 原生尺寸最终复测：/free → 预热 → 计时（与 A2 复测同样的干净条件）。"""
import json
import os
import shutil
import sys
import time
import urllib.request

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_test as RT  # noqa: E402


def free_mem():
    req = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def main():
    person = cv2.imread(RT.PERSON_SRC, cv2.IMREAD_COLOR)
    clothes = cv2.imread(RT.CLOTHES_SRC, cv2.IMREAD_COLOR)
    free_mem()
    time.sleep(12)
    small_p = cv2.resize(person, (171, 256))
    small_c = cv2.resize(clothes, (256, 384))
    RT.run_edit(small_p, small_c, 256, "W_A1f")
    RT.log("预热完成，开始计时")
    img, wall, comfy_s, raw = RT.run_edit(person, clothes, None, "A1f_native_final")
    RT.save_img("02_A_direct", "A1f_native_final.png", img)
    if raw:
        shutil.copy(raw, os.path.join(RT.HERE, "02_A_direct", "A1f_native_final_comfy_raw.png"))
    out = {"A1f_native_final": {"wall_s": wall, "comfy_s": comfy_s,
                                "canvas": [img.shape[1], img.shape[0]]}}
    RT.save_json(".", "timing_a1f.json", out)
    RT.log(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
