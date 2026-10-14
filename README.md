# qwen-image21

基于 **Qwen-Image-2.1** 的本地出图 / 图像编辑 / 试装换装工具（Mac mini M6 32GB 实测）。
本仓库只含代码、工作流与实验产物；**模型权重不入库**，部署前请按 [MODELS.md](MODELS.md) 放好模型。

## 快速开始

```bash
cd ~/qwen-image21
./start_all.sh        # 一键启动三个服务（已在跑的不会重启）
```

启动后打开：

| 页面 | 地址 | 用途 |
|---|---|---|
| 简易出图界面 | http://127.0.0.1:8190 | 日常使用：文生图 / 图生图编辑 / **AI换装**（抠图→生成→贴回原背景）；带账号/积分/仓库/云端切换 |
| 换装·试装实验台 | http://127.0.0.1:8191 | 抠图生成+贴回、换脸等试装实验 |
| ComfyUI 画布 | http://127.0.0.1:8188 | 高级编排工作流（推理后端，**无鉴权，勿暴露公网**） |

停止：`./stop_all.sh` ｜ 运行日志：`logs/` ｜ 公网分享（可选）：`./start_tunnel.sh`（只暴露带口令登录的 8190）

单独启动某个服务：

```bash
~/qwen-image21/venv/bin/python ~/qwen-image21/simple_ui/server.py    # 8190
~/qwen-image21/venv/bin/python ~/qwen-image21/tryon_demo/server.py   # 8191
cd ~/qwen-image21/ComfyUI && ~/qwen-image21/venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

## 目录结构

```
qwen-image21/
├── start_all.sh / stop_all.sh      # 一键启动 / 停止（8188 + 8190 + 8191）
├── workflows/                      # ComfyUI API 工作流 JSON
├── simple_ui/                      # 8190 简易出图界面（settings.example.json 为配置模板）
├── tryon_demo/                     # 8191 试装实验台
│   ├── direct_vs_crop_test_20261014/   # 换装对照实验（含报告.md）
│   └── archive_20261013/               # 更早的实验结果
├── ComfyUI/                        # 推理框架（第三方；models/ 已 gitignore）
├── logs/                           # 运行日志（archive/ 为历史日志）
├── assets/                         # 杂项素材
├── MODELS.md                       # 模型清单：文件名/大小/路径/下载方式
└── 使用指南.md                     # 详细使用教程
```

## 文档

- [使用指南.md](使用指南.md) — 部署、启动、使用案例、公网分享方案
- [MODELS.md](MODELS.md) — 所有模型的文件名、精确大小、放置路径、下载命令
- [tryon_demo/direct_vs_crop_test_20261014/报告.md](tryon_demo/direct_vs_crop_test_20261014/报告.md) — 换装「直接生成 vs 抠图生成+贴回」对照实验（速度/画质/内存结论）

## 注意事项

- **8188 无鉴权**：只在本机使用；公网分享请走 `start_tunnel.sh`（仅暴露带口令的 8190）。
- 32GB 内存机型：直出时输入长边建议 ≤1024（1536 级会触发 swap，慢数倍），详见实验报告。
- 部署配置：复制 `simple_ui/settings.example.json` 为 `simple_ui/settings.json`，填入访问口令与（可选的）百炼 API Key；该文件已 gitignore，不会入库。
