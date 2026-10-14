#!/usr/bin/env python3
"""干净状态复测：每次生成前 /free 释放 ComfyUI 内存 + 小图预热（不计时）再测。

目的：第一轮 A2 的 702.9s 明显被 swap 污染（首步 251s）。本脚本对 A1/A2/A4 各自
在“内存已复位 + 模型已热身”的相同条件下重新计时，输出 timing_clean.json。
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
    ("A1c_native_clean", None, "原生尺寸(resolution=0，1024x1536)"),
    ("A2c_res1024_clean", 1024, "1024 像素预算（App 默认档位）"),
    ("A4c_res759_clean", 759, "759 像素预算（与 B 抠图画布 0.57MP 同量级）"),
]


def free_mem():
    req = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def swap_line():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


def main():
    person = cv2.imread(RT.PERSON_SRC, cv2.IMREAD_COLOR)
    clothes = cv2.imread(RT.CLOTHES_SRC, cv2.IMREAD_COLOR)
    timing = {}
    for tag, res, note in CFG:
        RT.log("== %s（%s）==" % (tag, note))
        free_mem()
        time.sleep(15)
        RT.log("   释放后 %s" % swap_line())
        small_p = cv2.resize(person, (171, 256))
        small_c = cv2.resize(clothes, (256, 384))
        RT.run_edit(small_p, small_c, 256, "W_" + tag)   # 预热加载，不计时
        RT.log("   预热完成，开始计时")
        img, wall, comfy_s, raw = RT.run_edit(person, clothes, res, tag)
        RT.save_img("02_A_direct", tag + ".png", img)
        if raw:
            shutil.copy(raw, os.path.join(RT.HERE, "02_A_direct", tag + "_comfy_raw.png"))
        timing[tag] = {"note": note, "wall_s": wall, "comfy_s": comfy_s,
                       "canvas": [img.shape[1], img.shape[0]], "swap_at_end": swap_line()}
        RT.log("   [%s] wall=%.1fs comfy=%s canvas=%dx%d" % (
            tag, wall, comfy_s or -1, img.shape[1], img.shape[0]))
    RT.save_json(".", "timing_clean.json", timing)
    RT.log(json.dumps(timing, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
