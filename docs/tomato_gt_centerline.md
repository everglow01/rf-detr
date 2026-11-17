# GT 中心线覆盖变体：冻结诊断

本变体只保留 clDice 的 GT 骨架覆盖项。记 S 为增强后 GT 的软骨架，p 为预测果梗概率：

`L_gt_centerline = 1 - (sum(S * p) + 1) / (sum(S) + 1)`

这是独立实验变体，不是完整双侧 clDice 的等价优化。它不计算预测骨架，梯度只要求提高 GT 中心线位置的概率； 自身不惩罚中心线之外的假阳性，也不直接监督实例分类置信度。继续保留原生分类、mask CE 和 Dice。 原生损失、匹配、模型参数及推理路径不变。生产损失不中断共享特征的梯度。

## 开关

```yaml
mask_cldice_mode: gt_centerline
mask_cldice_loss_coef: 0.1
mask_cldice_class_ids: [1]
mask_cldice_iterations: 25
mask_cldice_chunk_size: 4
```

默认 `mask_cldice_mode: cldice`、系数 `0.0`，完全关闭。仅指定模式不会启用。 两个模式互斥；共享已有配置、类别筛选、padding 对齐、GT 骨架去重、分块及全局匹配计数归一化。 `mask_cldice_recall_beta` 只在 cldice 模式生效。新模式使用独立日志键 `loss_mask_gt_centerline`，只计算最终 decoder 层。 类别通过 GT 选择；番茄实验显式使用 stem=1，不改其他类别监督。K=25 只控制 GT 骨架化。 空骨架没有梯度，极小/较粗目标及预测饱和仍可能造成弱监督；不声明一定改善召回。

## 本轮比较

使用 `scripts/diagnose_tomato_decoder_gradients.py --gt-centerline`，冻结 baseline、β1、β2 的最佳 EMA 权重， 沿用此前 8 个困难 GT 与 8 个对照、batch=8、13 组、504 像素、BF16，以及三个原生 mask 采样种子。 每份权重都比较完整 clDice β1/β2 和 GT 中心线变体，系数均为 0.1。报告同一共享 decoder 特征处的梯度； 相同系数不代表相同梯度强度。保存原始向量，按每个 GT 合并 13 组，避免伪独立样本。

性能比较包含 GT 骨架生成、分块插值、损失前向和到共享特征的反向；每种模式预热一次，测量五次并轮换执行顺序。 显存同时记录开始值、峰值 allocated 及增量，计时前后 CUDA 同步。不含网络主干前向、完整参数反向、优化器、DDP或数据加载， 不能换算成整轮训练耗时或训练显存。只运行诊断，不创建优化器、不保存训练权重、不启动训练。
