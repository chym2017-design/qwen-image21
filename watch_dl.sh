#!/bin/zsh
# 等待大文件下载完成后，校验三个文件的精确大小；文本编码器曾用错误大小下载，需删掉重下（分片可复用）
B=https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main
M=~/qwen-image21/ComfyUI/models

# 等待主下载任务结束
while pgrep -f "pdl.sh https" >/dev/null 2>&1; do sleep 60; done

# 校验 transformer
D=$M/diffusion_models/qwen_image_2.1_int8_convrot.safetensors
if [ "$(stat -f%z $D 2>/dev/null || echo 0)" != "7256783064" ]; then
  echo "transformer bad, redownload"
  ~/qwen-image21/pdl.sh "$B/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" $D 7256783064 16
fi

# 文本编码器：第一次用了错误大小 9350000000，会产生截断文件，删除重下
T=$M/text_encoders/qwen3vl_8b_int8_convrot.safetensors
if [ "$(stat -f%z $T 2>/dev/null || echo 0)" != "9350798360" ]; then
  rm -f $T
  ~/qwen-image21/pdl.sh "$B/text_encoders/qwen3vl_8b_int8_convrot.safetensors" $T 9350798360 16
fi

echo "FINAL CHECK:"
for f in $D $T $M/vae/qwen_image_2.1_vae_bf16.safetensors; do
  echo "$(stat -f%z $f 2>/dev/null || echo MISSING)  $f"
done
