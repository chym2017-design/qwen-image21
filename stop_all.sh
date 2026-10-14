#!/bin/zsh
# 一键停止：ComfyUI(8188) + 简易出图界面(8190) + 换装·试装实验台(8191)（以及可选公网隧道）
set -u
killed=0

stop_pat() {
  local name="$1" pat="$2"
  local pids
  pids=$(pgrep -f "$pat" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "停止 $name（pid: $(echo $pids | tr '\n' ' ')）"
    kill $pids 2>/dev/null || true
    killed=1
  else
    echo "· $name 未在运行"
  fi
}

stop_pat "ComfyUI 推理后端"   "main.py --listen 127.0.0.1 --port 8188"
stop_pat "简易出图界面"       "simple_ui/server.py"
stop_pat "换装·试装实验台"    "tryon_demo/server.py"

# 公网隧道（如果开着就一起关；不想关就注释掉下面两行）
stop_pat "公网隧道 cloudflared" "cloudflared tunnel --url http://127.0.0.1:8190"

[ "$killed" = 1 ] && echo "" && echo "已停止。重新启动：~/qwen-image21/start_all.sh"
exit 0
