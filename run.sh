#!/bin/zsh
# 用法: ./run.sh <t2i|edit|rgba|turbo-t2i|turbo-edit|turbo-rgba|turbo-t2i-bf16> [seed]
# 依赖 ComfyUI 已在 127.0.0.1:8188 运行，且模型已就位
set -eu
CASE="${1:-t2i}"
SEED="${2:-42}"
W=~/qwen-image21/workflows
API="http://127.0.0.1:8188"

case $CASE in
  t2i)        WF=$W/t2i_api.json ;;
  edit)       WF=$W/edit_api.json ;;
  rgba)       WF=$W/rgba_api.json ;;
  turbo-t2i)  WF=$W/turbo_t2i_api.json ;;
  turbo-edit) WF=$W/turbo_edit_api.json ;;
  turbo-rgba) WF=$W/turbo_rgba_api.json ;;
  turbo-t2i-bf16) WF=$W/turbo_t2i_bf16_api.json ;;
  *) echo "用法: ./run.sh <t2i|edit|rgba|turbo-t2i|turbo-edit|turbo-rgba|turbo-t2i-bf16> [seed]"; exit 1 ;;
esac

# 注入 seed 并生成 API 请求体
PAYLOAD=$(python3 - "$WF" "$SEED" <<'EOF'
import json, sys
wf = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
for n in wf.values():
    if n["class_type"] == "KSampler":
        n["inputs"]["seed"] = seed
    if n["class_type"] == "RandomNoise":
        n["inputs"]["noise_seed"] = seed
print(json.dumps({"prompt": wf}))
EOF
)

echo "提交工作流: $(basename $WF) (seed=$SEED)"
RESP=$(curl -s -X POST -H "Content-Type: application/json" -d "$PAYLOAD" $API/prompt)
PID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('prompt_id','') or '')" 2>/dev/null || true)
if [ -z "$PID" ]; then echo "提交失败: $RESP"; exit 1; fi
echo "prompt_id: $PID"

# 轮询等待完成
while true; do
  OUT=$(curl -s "$API/history/$PID" | python3 -c "
import json,sys
d=json.load(sys.stdin)
h=d.get('$PID',{})
st=h.get('status',{})
if st.get('completed') or st.get('status_str')=='success':
    outs=h.get('outputs',{})
    for nid,out in outs.items():
        for img in out.get('images',[]):
            print('OUTPUT:', img.get('subfolder','')+'/'+img['filename'])
    print('STATE: done')
elif st.get('status_str')=='error':
    print('STATE: error')
    print(json.dumps(st)[:600])
else:
    print('STATE: running')
" 2>/dev/null || echo "STATE: waiting")
  if echo "$OUT" | grep -q "STATE: done"; then echo "$OUT" | grep OUTPUT; echo "输出目录: ~/qwen-image21/ComfyUI/output/"; break; fi
  if echo "$OUT" | grep -q "STATE: error"; then echo "$OUT"; exit 1; fi
  sleep 10
done
