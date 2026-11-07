# Multimodal-AI4S：化学实验多模态监测

目标是记录容器内可见状态、温度、pH、质量及仪器读数，并逐步实现变化检测。
当前阶段聚焦视觉物态 baseline 的部署与推理验证。

## 项目导航

- [算法架构与实验运行方式](docs/算法架构.md)：模块边界、配置入口、追溯与 Git 工作方式。

- [开发与进度记录](DEVELOP.md)：当前完成情况、下一步和工作约定。
- [目录职责与扩展方式](docs/仓库结构.md)：代码、数据、标注和实验产物放在哪里。
- [实验索引](experiments/README.md)：已有实验、结果和证据路径。
- [实验记录模板](experiments/TEMPLATE.md)：新实验的目的、参数、结果和局限。
- [需求范围](docs/采集范围与MOF基线匹配.md)：五项采集需求及 baseline 覆盖范围。

大体积数据、权重、上游源码与运行输出留在本机；实验结论和复现信息保存在文档中。

首批采用 CV-HTE-Tutorial 的容器定位 + 物态检测，支持本地图片和视频。
目前是逐帧检测 baseline，尚未实现持续对象跟踪、变化报警或化学原因解释。

已完成作者378张用户研究图片的本地固定阈值验证，完整两级流程物态框micro-F1为87.30%，
固体框召回率为72.04%。这是自定义PR/F1验证，不是论文mAP表格复现。
详见[作者数据验证报告](docs/作者数据验证与需求匹配报告.md)和[采集范围](docs/采集范围与MOF基线匹配.md)。

## 本机运行

在项目目录 PowerShell 中执行，无需激活环境：

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py "data/videos/example.mp4" --output "outputs/example" --sample-seconds 1
```

视频可放入 `data/videos/`，也可以传入其他目录的绝对路径。默认每秒采样一次，
`--max-frames 10` 可限制试跑规模；`--device cpu` 可切换 CPU。
输出目录必须为空，避免混合不同运行的结果。

输出包含：
- `metadata.json`：视频来源、帧率、模型实际类别、运行设备。
- `detections.jsonl`：帧号、时间、容器框、物态框、类别与置信度。未检测到容器的帧也会记录。
- `frame_*.jpg`：采样帧的检测标注图。
- `metrics.json`：模型准备及处理耗时、处理帧数与 PyTorch 显存峰值。处理耗时包括解码、首次推理预热及结果写盘；显存峰值不包括其他进程、CUDA 上下文或全部驱动开销。

时间为帧号 / 解码器报告帧率，适用于恒定帧率视频；可变帧率视频需要进一步按真实 PTS 对齐。
`vial_index` 仅为当前帧编号，不是跨帧身份。检测输出可包含多个物态区域。
类别从权重读取，不手工假设类别编号。可见固体不能直接证明 MOF 结晶成功。

## 环境

Python 3.10.9；本地 `.venv` 使用 `--system-site-packages`，复用本机
`D:/Python` 的 torch 2.5.1+cu121 与 torchvision 0.20.1+cu121。
新增依赖安装在项目环境，未更改全局依赖。此环境依赖本机基础环境，不是独立可搬移环境。
运行依赖见 `requirements-baseline.txt`，环境快照见 `docs/environment-baseline.txt`。

初次尝试下载上游指定的 torch 2.2.0 CUDA 包时，C 盘临时空间不足，
已改为复用通过 CUDA 验证的现有版本。上游完整训练依赖没有全部安装。

## 来源与权重

- 教程：https://github.com/AccelerationConsortium/CV-HTE-Tutorial
- 本地源码：`external/CV-HTE-Tutorial`，提交 `df7f0a2fafb699880d79de90244e8a626b3fa877`
- YOLOv5：https://github.com/ultralytics/yolov5 ，v7.0，提交 `915bbf294bb74c859f0b41f1c23bc395014ea679`
- Ultralytics：8.3.96
- 下载日期：2026-11-06

权重通过官方仓库 Git LFS 媒体地址下载，SHA-256 与仓库指针一致：

| 本地文件 | 上游文件 | 字节数 | SHA-256 |
|---|---|---:|---|
| models/phase.pt | models/combined/best.pt | 51996235 | a7797ba5f428dac95eb2d5dca89a898df62a17be050cfeebaec652722019bc24 |
| models/vial.pt | models/vial/best_nocap.pt | 167630971 | c5d217f10ac82cdc879e17af88136b3c1cddc06804a5eaa5e8efdd142f5f17fe |

下载 URL 格式：`https://media.githubusercontent.com/media/AccelerationConsortium/CV-HTE-Tutorial/main/<上游文件>`。
上游仓库中的 `.pt` 仍为 LFS 指针，实际权重位于本项目 `models/`。

本项目新增入口按原始分辨率处理画面，使用标准 OpenCV BGR 输入传入 Ultralytics，
逐容器推理以控制显存。它不是上游 test.py 的逐字复现；没有更改或重新训练模型权重。

## 验证范围

`outputs/smoke-image` 使用上游已标注的示例图验证两级模型加载和推理。
`data/smoke_fixture.mp4` 是两张上游示例图和一张黑帧组成的 3 帧测试视频，
仅用于验证视频解码、时间记录、结果写入及无容器帧处理，不代表真实实验或性能评估。
已对首个全桌面视频进行了抽帧、全帧和人工指定区域的探索测试；因缺少可判读的物态真值，
未给出准确率。跨场景泛化和变化检测效果尚待正式验证。

网上收集的视频建议同时记录来源链接、实验说明、是否剪辑及使用许可，
后续将其作为探索样例与正式评测数据区分保存。
