# 果梗召回诊断与非对称 clDice

2026-11-13。目标是在误检可控的前提下减少整根果梗漏检，同时保持掩膜精度。骨架覆盖率与实例检出率分别统计，不能互相替代。

## 保留基线与原版 clDice

- `mask_cldice_loss_coef=0`：关闭拓扑项，保持原版 RF-DETR。
- `mask_cldice_loss_coef=0.1, mask_cldice_recall_beta=1`：保留目前对称 clDice 的原始算术表达式。
- `mask_cldice_loss_coef=0.1, mask_cldice_recall_beta=2`：待实验的覆盖率优先版本；无新增网络参数。
- 三者的原始分类、框、BCE、Dice 损失不变；新增项仅作用于果梗 class 1、最终解码层。仍保留 K=25、chunk=4、GT 去重、Group DETR 平均和共享特征梯度。

令 P 为预测骨架落在 GT 掩膜内的比例，R 为 GT 骨架被预测概率图覆盖的比例，二者沿用原平滑定义：

```text
L_topology(beta) = 1 - (1 + beta^2) * P * R / (beta^2 * P + R)
```

beta > 1 时更重视 R；beta=1 时完全恢复现有 clDice。实现对系数先归一化，避免大 beta 的平方溢出；beta=1 单独走原表达式以保持数值兼容。该 beta 只调整拓扑项内部的平衡，外部损失系数仍为 0.1，不代表分类召回权重。

调整中心线精确率和灵敏度平衡已有相关研究，不能将一个 beta 参数宣称为全新损失。本项目的待验证内容是它在 RF-DETR 果梗实例分割、匹配与共享特征中的作用。参考：[clDice 原论文](https://arxiv.org/abs/2003.07311)；[稀疏标记下双结构约束脑血管分割网络](https://www.jcad.cn/cn/article/pdf/preview/10.3724/SP.J.1089.2023.19585.pdf)。

## 全置信度诊断

使用 `scripts/diagnose_tomato_recall.py`，只读取既有两个第 10 轮 EMA 权重，不创建优化器。

```bash
python -m scripts.diagnose_tomato_recall \
    --pair-root /usr/local/rfdetr26/output/output1/e1-k25-short-pair-20261113-1523 \
    --output /usr/local/rfdetr26/output/output1/<new-audit-dir> \
    --expected-commit <full-git-sha>
```

原生 PR 曲线保留模型的 top-k=100，在其全部不同置信度阈值上做一对一匹配；没有果梗的图像也计入误检。另行检查全部 100 个查询的掩膜、框和类别分数，避免将原生 top-k 之外的查询误称为“完全没有预测”。扩展查询只用于诊断，不混入原生 PR 指标。开始分析前，断言置信度 0.3 的 TP、预测数及漏检 GT 集合与归档评估一致。相同分数成组进入曲线，组内沿用原生输出顺序。

2026-11-13 的两组各 199 张图、121 个果梗诊断已完成：baseline 的 18 个漏检中，16 个存在低于 0.3 的合格几何候选；clDice 的 13 个漏检中为 12 个。对剩余 clDice 的 1 个实例，全部查询的 mask 与 box IoU 均不到 0.5。低分几何候选是排查线索，并不等于模型已做出可靠检出。

以 baseline 在阈值 0.3 时的精确率 69.59% 为下限，在各自曲线上寻找可达到的最高召回，结果为 baseline 85.95%、clDice 89.26%。这是同一验证集上的描述性操作点比较，不是独立测试成绩；两条 PR 曲线并非处处由 clDice 占优。

因此不能假设仅增强 R 就一定解决漏检。下一次短对照需同时观察：合格候选的果梗分数与排序、固定阈值实例召回、同等精确率下的召回、AP75 和 mask IoU。不可将降低阈值或扩大掩膜产生的收益混为结构监督收益。

## 验证与下一步

- 默认 beta=1 与 Git 旧实现比较 K=25 的损失及 logits 梯度，覆盖 FP32/FP64、缓存/未缓存 GT 骨架。
- beta=2 验证：漏掉中心线像素的补偿梯度增强；错误骨架仍有压制梯度；数值梯度正确；分块和 Group DETR 平均保持等价；非果梗类别不受该项直接监督。
- 复用 `scripts.tomato_short_ablation probe --recall-beta 2 --output <new-dir>` 在旧 E1 固定权重上测试原 OOM 批次，检查共享梯度、有限值、显存与参数哈希。此参数只允许用于 probe，不会悄悄修改已有成对训练入口。
- 后续训练只改变 beta，保留 K、外部系数、分块、种子、预训练、batch、学习率和调度跨度；与原版 clDice、原生 baseline 分开比较。beta=2 是首个待验证候选，尚未证明优于 beta=1。
- 这次修改不省略预测骨架计算，不能期待自动解决约三倍训练耗时。效率优化作为后续独立消融。

原始记录和结论保存于 `reports/tomato-20261113-cldice-recall/` 的本地结果副本及远端 `output/output1/tomato-recall-audit-20261113-1645/`。本步骤没有启动新的短训或长训。
