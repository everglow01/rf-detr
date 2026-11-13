# 非对称 clDice：beta=1 / beta=2 十轮对照

2026-11-13，用户授权开始下一步短对照。仅增加实验入口，不修改模型或损失实现。

## 固定条件

- 两组分别命名为 `beta1`、`beta2`，从同一官方 SegSmall 预训练权重重新开始，不接着旧 E1 权重训练。
- 基础种子 42，2 张 L2，每卡 batch 8，累积 1，BF16。数据集、增强、学习率、EMA 及其他模块开关沿用 `seg_small_baseline.yaml`。
- 两组 clDice 均为 class 1、系数 0.1、K=25、chunk=4，仅改变 `mask_cldice_recall_beta` 的 1.0 / 2.0。
- 均保留 100 轮调度跨度，第 10 轮完整验证后停止；按 beta1、beta2 顺序执行，各占用两张 GPU。
- 核对初始化、每轮首批输入指纹、学习率、更新次数；每轮首批一致不等于逐个核验全部训练批次。
- 主比较固定第 10 轮 EMA；次比较为前 10 轮按整体 mask AP 选出的最佳 EMA。新输出目录不得覆盖历史结果。

## 评估口径与预先固定的判据

在相同 199 张验证图、121 个果梗上，记录果梗 AP50:95、AP75、固定阈值 0.3 的召回/精确率/误检、匹配 IoU 和 hard-clDice，以及训练耗时与单卡显存。

原生 top-k PR 沿用现有一对一匹配。固定精确率下限为 **0.6959459459459459**，来源是此前原生 baseline 在阈值 0.3 的精确率；不随本轮结果调整。完整曲线也保留，不以一个操作点代替整体评价。

全部 100 查询用于诊断几何覆盖与候选排序，不混入原生 PR。预先关注上一阶段对称 clDice 的 12 个低分漏检 GT（按 image_id / target_index 对齐，不把不同模型的 query 编号视为同一实体）。对每个 GT 比较：是否仍有 mask IoU >= 0.5 的查询、其中最高果梗分数、该候选在全部果梗查询中的分数排名、以及原生输出是否真正找回该 GT。排名按 `1 + 分数严格更高的果梗查询数` 统计，同分并列。

若仅改善骨架或阈值下召回，同时误检、AP75 恶化，不能直接宣称成功。单种子 10 轮只用于筛选方向，不证明稳定泛化提升，不自动启动 100 轮或新增分类模块。

## 已有入口与执行方式

```bash
python -m scripts.tomato_short_ablation run \
    --comparison recall-beta --output <fresh-output> --expected-commit <sha>

python -m scripts.diagnose_tomato_recall \
    --pair-root <fresh-output> --output <fresh-audit-output> \
    --arms beta1 beta2 --precision-floor 0.6959459459459459 --expected-commit <sha>
```

训练完成后如只需补评估，使用已有 `evaluate --output ... --expected-commit ...`。该动作从 provenance 读取组别，不重新训练。原有 `run` 不带 `--comparison` 时仍做 baseline / clDice；`--recall-beta` 仍仅供冻结探针使用。

源代码在正式 `rfdetr26` 仓库修改并通过 GitHub 同步，实验归档记录提交、配置、数据和权重哈希。训练结果统一放到 `output/output1`，完成后保存正式仓库中的报告副本。
