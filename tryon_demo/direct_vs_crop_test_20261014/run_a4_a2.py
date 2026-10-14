#!/usr/bin/env python3
"""补充测量：A4@759（与 B 抠图画布 0.57MP 同像素预算的直接生成）+ A2@1024 复测。

顺序由轻到重（0.56MP → 1.04MP），每个配置前 /free + 预热（不计时），避免上一轮
“先重后轻”导致的 swap 污染。结果追加到 timing.json 的补充字段。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_test as RT  # noqa: E402

CFG = [
    ("A4_res759", 759, "759 像素预算（0.56MP，与 B 抠图画布 0.57MP 同量级）"),
    ("A2_res1024_rerun", 1024, "1024 像素预算（App 默认档位）复测"),
]


def free_mem():
    req = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def main():
    person = cv2.imread(RT.PERSON_SRC, cv2.IMREAD_COLOR)
    clothes = cv2.imread(RT.CLOTHES_SRC, cv2.IMREAD_COLOR)
    out = {}
    for tag, res, note in CFG:
        RT.log("== %s（%s）==" % (tag, note))
        free_mem()
        time.sleep(12)
        small_p = cv2.resize(person, (171, 256))
        small_c = cv2.resize(clothes, (256, 384))
        RT.run_edit(small_p, small_c, 256, "W_" + tag)
        RT.log("   预热完成，开始计时")
        img, wall, comfy_s, raw = RT.run_edit(person, clothes, res, tag)
        RT.save_img("02_A_direct", tag + ".png", img)
        if raw:
            shutil.copy(raw, os.path.join(RT.HERE, "02_A_direct", tag + "_comfy_raw.png"))
        out[tag] = {"note": note, "wall_s": wall, "comfy_s": comfy_s, "canvas": [img.shape[1], img.shape[0]]}
        RT.log("   [%s] wall=%.1fs comfy=%s canvas=%dx%d" % (tag, wall, comfy_s or -1, img.shape[1], img.shape[0]))
    p = os.path.join(RT.HERE, "timing_supplement.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    RT.log("已写 timing_supplement.json")


if __name__ == "__main__":
    main()
