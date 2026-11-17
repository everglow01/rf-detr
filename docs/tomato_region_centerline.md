# 区域精度与 GT 中心线覆盖：冻结诊断

保持原生 mask CE=5、Dice=5、分类=1，以及匹配和模型结构不变。 新增独立 `region_centerline` 模式；它不是原始 clDice 的等价实现，也不保证改善检测召回。

对有效 GT 网格上的概率 p、GT 区域 Y、GT 骨架 S：

```
P = (sum(p * Y) + 1) / (sum(p) + 1)
R = (sum(p * S) + 1) / (sum(S) + 1)
L = 1 - 2 * P * R / (P + R)
```

P 惩罚 GT 外的概率质量，R 监督中心线覆盖。两者都不需要预测骨架。 直接对 mask logits 求导时，GT 外梯度非负（梯度下降压低概率），中心线内梯度非正（提高概率）。 共享 decoder 特征会耦合多个像素，不能由此推断特征上的两种方向也一定一致。 空骨架仍保留区域精度监督；空 GT 实例由既有 adapter 排除。

```yaml
mask_cldice_mode: region_centerline
mask_cldice_loss_coef: 0.1
mask_cldice_class_ids: [1]
mask_cldice_iterations: 25
mask_cldice_chunk_size: 4
```

默认模式仍为 cldice，系数仍为 0；选择模式本身不会启用损失。 复用 GT 类别筛选、骨架去重、padding 对齐、分块及全局计数归一化；只增加最终层 `loss_mask_region_centerline`。beta 仅在原始 cldice 模式生效。本变体不新增参数、不修改推理。

使用 `scripts/diagnose_tomato_decoder_gradients.py --region-centerline --split-mask`： 冻结 baseline/β1/β2 最佳 EMA，同一 8 个困难 GT + 8 个对照、两批 batch=8、Group DETR=13、 504 像素、BF16、三个原生 mask 采样种子。每组权重同时比较原 clDice β1/β2、GT-only 和本变体， 额外系数均为 0.1。保存 CE、Dice 区域归因和共享 decoder 梯度，13 组按 GT 合并。 除总梯度与原生损失的夹角、强度外，也检查新梯度与 GT-only 的夹角；相同系数并非相同强度。

性能包含 GT 骨架准备、分块插值、损失前向和共享特征反向；每变体预热一次、轮换顺序测五次， CUDA 同步计时，记录 allocated 基线、峰值和增量。范围不含主干前向、完整参数反向、优化器或 DDP， 不得换算成整轮训练的加速比。无优化器、不更新权重、不启动训练。

## 降低区域精度相对权重（冻结扫描）

新增 `mask_region_precision_weight: a`，仅作用于 region_centerline，范围[0,1]，默认1复现上版。 保持外部系数0.1、原生CE/Dice权重5/5，调整内部调和平均：

```
L_a = 1 - (1+a) * P * R / (P + a*R)
```

这是按精度:覆盖 = a:1 加权的调和平均。a=0完全退化为GT-only，a=1保留原F1算术。 在同一P/R下，两个偏导的绝对值之比为 a\*(R/P)^2；降低a降低区域精度的相对梯度贡献。 归一化使目标仍在同一0至1尺度，但相同外系数不意味着相同梯度范数。旧clDice的beta独立且不受影响。

`scripts/diagnose_tomato_decoder_gradients.py --region-precision-sweep` 固定扫描1、0.5、0.25、0.1、0.05、0， 沿用同一冻结权重、样本、增强和native mask采样。0以既有gt_centerline为对照；同时保留原clDice β1/β2。 此外直接对当前漏分中心线的平均概率缺口、GT外平均概率、当前GT外假阳性平均概率求共享decoder梯度。 误差区域在冻结前向下以0.5阈值固定，排除padding；空区域梯度为0，不算作改善。 这些probe只用于诊断，不计入训练损失。与其梯度内积为正代表局部下降有利于降低对应错误， 必须同时报告中心线、外部区域和实际FP三个方向，不能只用与native Dice的相似度判断成功。 不创建优化器、不更新模型、不启动训练；记录损失计算的耗时和allocated显存供复核。
