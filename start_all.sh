#!/bin/zsh
# 一键启动：ComfyUI 推理后端（8188）+ 简易出图界面（8190）
set -u
ROOT=~/qwen-image21

if ! curl -s --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo "启动 ComfyUI 推理后端..."
  (cd "$ROOT/ComfyUI" && nohup "$ROOT/venv/bin/python" main.py --listen 127.0.0.1 --port 8188 > "$ROOT/comfyui.log" 2>&1 &)
  sleep 8
fi

if ! curl -s --max-time 3 http://127.0.0.1:8190/api/health >/dev/null 2>&1; then
  echo "启动简易出图界面..."
  (nohup "$ROOT/venv/bin/python" "$ROOT/simple_ui/server.py" > "$ROOT/simple_ui.log" 2>&1 &)
  sleep 2
fi

echo ""
echo "  简易出图界面（推荐日常使用）: http://127.0.0.1:8190"
echo "  ComfyUI 画布（高级编排时用）: http://127.0.0.1:8188"
