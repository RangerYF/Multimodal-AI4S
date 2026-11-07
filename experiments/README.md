# 实验索引

此处保存实验说明，原始预测和可视化仍在 `outputs/`。以下为 2026-11-07 对既有运行的补登记；
历史目录保持原名，具体参数与计数以原始元数据及报告为准。

| 实验 | 数据及目的 | 输出目录（相对根目录） | 结论与边界 |
|---|---|---|---|
| 图片/视频冒烟检查 | 作者示例图与合成的 3 帧视频；检查流程 | `outputs/smoke-image`、`outputs/smoke-video` | 模型加载及输出流程验证，不作性能证据 |
| 作者训练包诊断 | 168 张图；流程检查 | `outputs/author-training-diagnostic` | 属于训练来源数据，不作独立泛化结论 |
| 作者用户研究集评测 | 378 张图、1053 个物态真值框 | `outputs/author-user-study` | 自动模式 micro-F1 87.30%，Solid 框召回率 72.04%；不是论文 mAP 复现 |
| 首个视频抽帧 | 每 5 秒一次，共 19 帧 | `outputs/first-video` | 探索测试，无可判读物态真值 |
| 首个视频 1 fps | 共 95 帧 | `outputs/first-video-1fps` | 检查更密采样表现，未计算准确率 |
| 首个视频全帧 | 15 fps，共 1425 帧 | `outputs/first-video-allframes` | 处理约 78.7 秒；更多帧未消除误检 |
| 指定容器区域诊断 | 7 个时刻 × 2 个区域 | `outputs/manual-roi-first-video` | 助手辅助框；内部物态不可可靠判读，不作准确率评测 |

首个视频为 `data/videos/飞书20261106-111753.mp4`；区域清单为
`annotations/first-video-manual.json`。`outputs/video-inspect` 是检查素材，不是单独评测结果。

## 详细证据

- [作者数据验证与需求匹配报告](../docs/作者数据验证与需求匹配报告.md)：数据来源、协议、分类别指标和失败案例。
- [作者评测原始指标](../outputs/author-user-study/metrics.json)：本机存在；不会随代码版本保存。
- [首次人工容器框对照](../docs/首次人工容器框对照.md)：区域诊断结果和标注限制。
- [模型来源、哈希与环境](../README.md)：权重及上游源码版本；环境快照在 `docs/environment-baseline.txt`。

## 新实验

- [20261107-runner-smoke-001](20261107-runner-smoke-001.md)：配置驱动运行器集成验证，3 帧成功完成，基础检查通过。

复制 [TEMPLATE.md](TEMPLATE.md) 为 `YYYYMMDD-主题-序号.md`，开始前填写目的、数据和评测协议，
完成后补结果及局限，再登记到本索引。失败或无提升的实验也保留。
历史完整报告沿用原位置，不为统一格式复制一套易失同步的报告。
