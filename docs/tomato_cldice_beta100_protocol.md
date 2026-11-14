# clDice beta=1 / beta=2：100 轮串行训练

2026-11-14，用户明确授权两组各训练 100 轮，先 beta1、后 beta2，任务面板持续显示日志。确认启动后结束本轮操作，由用户在训练结束后通知，再进行结果分析。

- 入口：`bash scripts/train_tomato_beta100.sh <verified-full-git-sha>`。由 GPU 面板后台作业执行。
- 输出固定为 `/usr/local/rfdetr26/output/output1/cldice-beta100-v1-seed42/` 下的 `beta1/`、`beta2/`。根目录必须不存在，不覆盖旧结果。
- 两组从相同官方 SegSmall 预训练权重重新开始。各占用两张 L2，per-GPU batch 8、累积 1、BF16、种子 42、K25、系数 0.1、chunk4、果梗 class 1。
- 模型、数据、增强、学习率与 EMA 延续上一轮对照；唯一训练变量为 beta=1 / 2。
- 实际训练 100 轮，无 `StopAfterTen`，early stopping 关闭。第 70 轮学习率衰减。原生每轮验证、最佳/最终 EMA、每轮 latest 恢复检查点及每 20 轮归档保留。
- 日志开启原生 tqdm（每 5 个批次刷新），将回车转换为换行，同时送往任务面板 stdout 和每组 `train.log`。每轮时间、验证指标和显存记录仍由已有 BenchmarkTiming 回调保存。
- 使用共同的 `tomato-experiment.lock` 防止本项目重复排队执行。检查版本、工作区、GPU 空闲和新输出目录；第一组非零退出或未完成 100 轮时停止队列，不继续第二组。
- 两组结束后写入完成标记；此入口不自动执行离线评估或结果分析。

代码通过正式仓库提交、GitHub 推送、服务器 `git pull --ff-only` 同步。provenance 保存实际提交与配置、预训练、数据标注、回调和启动脚本哈希；source.tar.gz 保存已提交源码快照。
