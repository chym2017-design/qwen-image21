#!/bin/zsh
# 启动公网隧道：把本机 8190（简易出图界面）映射到一个临时公网域名（免注册）
#
# 用法：
#   ./start_tunnel.sh          # 启动，屏幕上会打印公网地址（https://xxxx.trycloudflare.com）
#   Ctrl+C 或 pkill cloudflared  # 停止对外暴露
#
# 注意：
# 1. 域名是随机且临时的，隧道重启后会变
# 2. 只暴露 8190（带口令保护），绝不要暴露 8188（ComfyUI 后端，无鉴权）
# 3. 本机不能休眠，否则外部访问会断（可另开终端跑 caffeinate -i）
set -u
CF="/opt/homebrew/opt/cloudflared/bin/cloudflared"
[ -x "$CF" ] || { echo "未找到 cloudflared：brew install cloudflared"; exit 1; }

# 确认界面服务在跑
curl -s --max-time 3 -o /dev/null http://127.0.0.1:8190/login || {
  echo "简易界面未启动，先执行：~/qwen-image21/start_all.sh"; exit 1;
}

echo "启动隧道（公网地址见下方 https://...trycloudflare.com）..."
exec "$CF" tunnel --url http://127.0.0.1:8190 --no-autoupdate
