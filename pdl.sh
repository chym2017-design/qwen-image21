#!/bin/zsh
# 分段并行下载器：32MB 分片，16 并发，断点续传（分片完整则跳过）
set -u
# worker 模式：pdl.sh --worker URL DEST i s e
if [[ "${1:-}" == "--worker" ]]; then
  URL="$2"; i="$4"; s="$5"; e="$6"; f=$(printf "%05d" $i)
  want=$((e - s + 1))
  while true; do
    if [[ -f $f && $(stat -f%z $f) -eq $want ]]; then exit 0; fi
    curl -sL --max-time 600 -r ${s}-${e} -o $f "$URL"
    if [[ -f $f && $(stat -f%z $f) -eq $want ]]; then exit 0; fi
    rm -f $f; sleep 5
  done
fi
URL="$1"; DEST="$2"; SIZE="$3"; WORKERS="${4:-16}"
CHUNK=$((32*1024*1024))
PARTS="${DEST}.parts"
mkdir -p "$PARTS"
cd "$PARTS"

if [[ -f "$DEST" && $(stat -f%z "$DEST") -eq $SIZE ]]; then
  echo "skip: $DEST complete"; exit 0
fi

n_chunks=$(( (SIZE + CHUNK - 1) / CHUNK ))
echo "[$(date +%H:%M:%S)] start $(basename $DEST): $((SIZE/1024/1024))MB in $n_chunks chunks"

typeset -a lines
for ((i=0; i<n_chunks; i++)); do
  s=$((i*CHUNK)); e=$((s+CHUNK-1)); (( e >= SIZE )) && e=$((SIZE-1))
  lines+=("$i $s $e")
done

print -rl -- $lines | xargs -P $WORKERS -n 3 $0 --worker "$URL" "$DEST"

cat $(printf "%05d " $(seq 0 $((n_chunks-1)))) > "$DEST"
actual=$(stat -f%z "$DEST")
if [[ $actual -eq $SIZE ]]; then
  rm -rf "$PARTS"; echo "[$(date +%H:%M:%S)] DONE $DEST"
else
  echo "SIZE MISMATCH: $actual != $SIZE"; exit 1
fi
