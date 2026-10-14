# 模型说明

**本仓库不含任何模型权重**（约 31GB，全部在 `.gitignore` 中排除）。本文件说明每个模型的
用途、精确大小与放置路径，按表在本地放好后即可运行（下载脚本 `pdl.sh` 支持 32MB 分片、
16 并发、断点续传；`watch_dl.sh` 带按字节数校验）。

## 放置路径总览

| 用途 | 放置目录 |
|---|---|
| Qwen-Image-2.1 出图 / 编辑（ComfyUI，端口 8188） | `ComfyUI/models/` 下对应子目录 |
| 试装 / 换脸（tryon_demo，端口 8191） | `tryon_demo/models/` |
| GFPGAN 依赖权重 | `gfpgan/weights/` |
| 首次运行自动下载（无需手动） | `~/.insightface/`、`~/.u2net/` |

## 1. Qwen-Image-2.1 全家桶（核心出图/编辑模型）

来源：**Comfy-Org/Qwen-Image-2.1**（HuggingFace；`watch_dl.sh` 里用的是 hf-mirror 镜像）。
工作流默认使用 **int8 量化版**（`workflows/turbo_*.json` 指定），bf16 原版仅作备用。

| 文件 | 大小(字节) | 放置路径 | 说明 |
|---|---|---|---|
| `qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `ComfyUI/models/diffusion_models/` | 主模型（int8 量化，工作流默认） |
| `qwen_image_2.1_bf16.safetensors` | 14,230,280,616 | `ComfyUI/models/diffusion_models/` | 全精度版（可选，更慢更占内存） |
| `qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 | `ComfyUI/models/text_encoders/` | 文本/视觉编码器（Qwen3-VL-8B 量化版） |
| `qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `ComfyUI/models/vae/` | VAE |
| `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors` | 679,604,800 | `ComfyUI/models/loras/` | 6 步加速 LoRA（工作流默认） |
| `Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors` | 339,832,808 | `ComfyUI/models/loras/` | 4 步加速 LoRA（可选） |

下载示例（BASE 按需换成官方地址或镜像；字节数用于校验完整性）：

```zsh
BASE=https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main
M=~/qwen-image21/ComfyUI/models

./pdl.sh $BASE/diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
    $M/diffusion_models/qwen_image_2.1_int8_convrot.safetensors 7256783064
./pdl.sh $BASE/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
    $M/text_encoders/qwen3vl_8b_int8_convrot.safetensors 9350798360
./pdl.sh $BASE/vae/qwen_image_2.1_vae_bf16.safetensors \
    $M/vae/qwen_image_2.1_vae_bf16.safetensors 675509688
```

> 两个 viggle-turbo LoRA 与仓库内自带的 `ComfyUI/custom_nodes/viggle_turbo/` 节点配套使用
> （工作流里的 `ViggleTurboLora` / `ViggleTurboSigmas` 由该节点提供）。LoRA 权重来源以你
> 下载时的发行渠道为准，放入 `ComfyUI/models/loras/` 即可。

## 2. 试装 / 换脸（tryon_demo 8191）

| 文件 | 大小(字节) | 放置路径 | 说明 |
|---|---|---|---|
| `inswapper_128.onnx` | 554,253,681 | `tryon_demo/models/` | 换脸模型（insightface 生态发布，按你获取的渠道重新下载） |
| `GFPGANv1.4.pth` | 348,632,874 | `tryon_demo/models/` | 人脸修复（TencentARC/GFPGAN 官方发布） |
| `face_detection_yunet_2023mar.onnx` | 232,589 | `tryon_demo/models/` | 人脸检测（OpenCV Zoo / YuNet） |
| `haarcascade_frontalface_default.xml` | 930,127 | `tryon_demo/models/` | OpenCV 自带数据文件，可从 OpenCV 安装包复制 |

## 3. GFPGAN 依赖权重（`gfpgan/weights/`）

| 文件 | 大小(字节) | 说明 |
|---|---|---|
| `detection_Resnet50_Final.pth` | 109,497,761 | GFPGAN 的人脸检测依赖（TencentARC/GFPGAN 官方发布） |
| `parsing_parsenet.pth` | 85,331,193 | GFPGAN 的人脸解析依赖（同上） |

## 4. 首次运行自动下载（无需手动放置）

| 模型 | 位置 | 大小 | 用途 |
|---|---|---|---|
| insightface `buffalo_l` | `~/.insightface/models/buffalo_l/` | ~320MB | 人脸识别/身份相似度（insightface 自动下载） |
| `u2net.onnx` | `~/.u2net/u2net.onnx` | ~176MB | rembg 抠图（首次调用自动下载） |

## 5. 未纳入本仓库的其他内容

- **ComfyUI 本体**：第三方项目，请按官方仓库自行安装；本仓库只自带 `workflows/`（API 工作流）
  与自装节点 `ComfyUI/custom_nodes/viggle_turbo/`。
- **ollama-models/**（约 17GB）：另一条本地 LLM 线（qwen3.8:27b 等），与本项目无关，已忽略。
- **simple_ui/settings.json**：含 API Key 与访问口令，已忽略；部署时复制
  `simple_ui/settings.example.json` 改为 `settings.json` 并填入自己的 Key。
