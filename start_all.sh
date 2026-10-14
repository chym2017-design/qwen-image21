#!/bin/zsh
# 一键启动：ComfyUI 推理后端(8188) + 简易出图界面(8190) + 换装·试装实验台(8191)
# 日志统一写到 logs/ 下。重复执行只补缺：已在跑的不会被重启。
set -u
ROOT=~/qwen-image21
PY="$ROOT/venv/bin/python"
mkdir -p "$ROOT/logs"

start_one() {
  local name="$1" port="$2" workdir="$3" log="$4"; shift 4
  if curl -s --max-time 3 "http://127.0.0.1:$port/" -o /dev/null 2>/dev/null; then
    echo "✓ $name 已在运行（端口 $port）"
    return
  fi
  echo "启动 $name ..."
  (cd "$workdir" && nohup "$PY" -u "$@" > "$log" 2>&1 &)
  local i
  for i in {1..30}; do
    sleep 1
    curl -s --max-time 2 "http://127.0.0.1:$port/" -o /dev/null 2>/dev/null && { echo "✓ $name 就绪（端口 $port）"; return; }
  done
  echo "⚠ $name 启动后 30 秒内未就绪，看日志：$log"
}

start_one "ComfyUI 推理后端"  8188 "$ROOT/ComfyUI" "$ROOT/logs/comfyui.log"    "$ROOT/ComfyUI/main.py" --listen 127.0.0.1 --port 8188
start_one "简易出图界面"      8190 "$ROOT"         "$ROOT/logs/simple_ui.log"  "$ROOT/simple_ui/server.py"
start_one "换装·试装实验台"   8191 "$ROOT"         "$ROOT/logs/tryon_demo.log" "$ROOT/tryon_demo/server.py"

echo ""
echo "  简易出图界面（日常推荐）  : http://127.0.0.1:8190"
echo "  换装·试装实验台           : http://127.0.0.1:8191"
echo "  ComfyUI 画布（高级编排）  : http://127.0.0.1:8188  ← 无鉴权，不要暴露公网"
echo ""
echo "  停止全部：~/qwen-image21/stop_all.sh      日志在：~/qwen-image21/logs/"
