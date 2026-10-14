#!/usr/bin/env python3
"""百炼 qwen-image-2.1 云端出图自测（复用 server.bailian_generate，单一事实来源）。

用法：
  ~/qwen-image21/venv/bin/python ~/qwen-image21/simple_ui/test_cloud.py [model] [prompt]
  model 默认 qwen-image-2.1-turbo；prompt 默认一张橘猫窗台图。

Key 读取顺序：环境变量 BAILIAN_API_KEY → settings.json 的 bailian_api_key。
成功后图片保存到 ~/qwen-image21/ComfyUI/output/cloud_test.png。
"""
import os
import sys

ROOT = os.path.expanduser("~/qwen-image21")
sys.path.insert(0, os.path.join(ROOT, "simple_ui"))
from server import bailian_generate, bailian_key, COMFY_OUTPUT  # noqa: E402


def main():
    key = bailian_key()
    if not key:
        print("未找到百炼 API Key。请在 simple_ui/settings.json 加入")
        print('  "bailian_api_key": "sk-你的百炼key"')
        print("或设置环境变量 BAILIAN_API_KEY。")
        sys.exit(1)
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen-image-2.1-turbo"
    prompt = (sys.argv[2] if len(sys.argv) > 2
              else "一只橘猫趴在洒满午后阳光的窗台上，窗外绿树，写实摄影，浅景深")
    print(f"调用百炼 {model} …\n  prompt={prompt}")
    try:
        img, usage = bailian_generate(
            {"prompt": prompt, "width": 1024, "height": 1024}, key, model)
    except Exception as e:
        print(f"\n✖ 失败: {e}")
        sys.exit(2)
    os.makedirs(COMFY_OUTPUT, exist_ok=True)
    path = os.path.join(COMFY_OUTPUT, "cloud_test.png")
    with open(path, "wb") as f:
        f.write(img)
    print(f"\n✔ 成功，{len(img)} 字节，已保存到 {path}")
    print(f"  usage: {usage}")


if __name__ == "__main__":
    main()
