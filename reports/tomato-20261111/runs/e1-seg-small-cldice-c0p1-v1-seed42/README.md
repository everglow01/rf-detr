**E1：果梗 soft-clDice 首轮对照**

用户于 2026-11-11 授权启动。实际状态与 job ID 见同目录 `launch_status.json`；不能仅以此说明文件判断任务仍在运行。

模型与训练协议：Seg Small / 384 / 100 epochs / BF16 / 2 GPUs / batch 8 per GPU / accumulation 1 / seed 42。
使用相同官方预训练权重，lr=1e-4、encoder lr=1.5e-4、lr_drop=70，不从 E0 续训。
新增 stem=1 的最终 decoder soft-clDice，系数 0.1、迭代 10。其余训练超参数与 E0 相同，继续按整体 EMA mask AP 选模。

本地模块尚未提交。此次将当前 `src/`、项目配置和 E1 配置冻结为 `source.tar.gz`，在远端实验目录的 `source/` 下解包运行。
`source_manifest.json` 记录基准 Git 提交和逐文件 SHA-256，`source_changes.patch` 记录已跟踪源码差异；新增文件包含在完整源码归档内。
启动命令固定 manifest 的 SHA-256，脚本启动前逐个核对源码文件。后续本地修改不会改变本次训练代码。
本次不执行 commit、push 或 pull；远端 Git 主工作区保留 E0 版本。后续仍可在本地审阅提交，通过 Git 更新远端。

远端实验根目录：`/usr/local/rfdetr26/output/output1/e1-seg-small-cldice-c0p1-v1-seed42`。

`training.yaml` 从 E0 的实际启动配置复制，仅新增三个 clDice 选项和独立训练/计时输出目录。
`config-comparison.json` 记录差异；`benchmark_callbacks.py` 与 E0 文件逐字节相同。
`run_e1.py` 沿用 E0 的日志、计时和 GPU 采样方式，拒绝版本/源码校验不一致、已有输出目录或已有 GPU 计算进程。
通过面板长期任务启动，并使用 `flock` 防止重复运行。正式训练目录是 `training/`；短验证目录是 `preflight/`。

`preflight.yaml` 使用相同双卡、batch 8 和 BF16，执行两个训练 step 及两个验证 step。
`preflight_callbacks.py` 验证 NCCL、clDice 启用和日志、有限总损失/梯度，并保存每卡峰值显存。
两个 rank 均通过后才启动正式训练；短验证权重不用于正式初始化。

保留正式最佳 EMA、最终 EMA、恢复检查点和训练日志。训练后用与 E0 补充评估相同的 batch=1、GPU BF16、完整验证集、Vernier、max_dets=500 设置重评估最佳/最终权重。
报告三类 mask AP50:95、果梗 AP50/AP75、固定阈值召回/精确率、条件 hard clDice 和未匹配计0的 clDice，以及耗时和显存。
现有 baseline `evaluate_stems.py` 固定了 E0 路径，E1 评估需切换到本次冻结源码、E1 配置/权重及独立结果目录。
新增项改变总损失定义，不以 E1 总 loss 比 E0 小作为成功依据。微小收益需后续多种子配对重复验证。
