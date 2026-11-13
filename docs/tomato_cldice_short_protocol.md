# K25 分块版与原版的十轮短对照

2026-11-13，用户授权完成原 OOM 批次梯度补测、两组十轮短训及结果比较。

- 入口：`python -m scripts.tomato_short_ablation run --output <新目录> --expected-commit <完整SHA>`。目录必须不存在，源码通过 GitHub 同步后使用。
- 先在旧 E1 最佳 EMA 权重上重放 batch 2（504 尺度、8 张图），分别计算原生总损失、区域分割损失和加权 clDice 的共享参数梯度。K=25、系数 0.1、分块4、果梗类别1。检查有限值、共享梯度连接和参数哈希不变；负余弦单独记录，不据此认定实现错误。
- 双卡顺序运行原版 baseline 和 clDice，两组都从同一官方 Seg Small 预训练权重开始，seed 42、每卡 batch8、累积1、BF16。除 clDice 系数（0 / 0.1）及输出位置外配置一致；K25、类别1、分块4在 baseline 中不生效。其他新增模块关闭。
- 保留 `train_config.epochs=100` 和 trainer 的100轮上限。`StopAfterTen` 在第10轮完整训练和验证结束后设置停止标记，不缩短学习率/drop-path计划。每个 rank 记录初始化参数哈希、每轮首批增强数据哈希、学习率及更新数，两组结束后核对一致。
- 复用 Git 中保存的 `BenchmarkTiming` 记录每个rank的训练、验证、整轮耗时和峰值显存，另采样设备占用。没有更改模型源码、损失公式或原训练归档。
- 主比较固定为第10轮EMA，补充比较各自前10轮内最佳EMA。复用原果梗评估器，对完整199张验证图计算果梗mask AP、confidence=0.3 / mask IoU=0.5下的召回率、漏检、条件clDice、漏检计零clDice及断裂情况。原日志中的逐类框AP不当作mask AP。
- 每组保存原生最佳/最后权重及恢复检查点，便于复核；输出统一置于新的 `output/output1/` 子目录。十轮仅判断早期趋势，不能作为最终收敛结果或多种子结论。
