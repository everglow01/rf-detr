# 番茄模块实现准备与消融边界

## 当前状态

2026-11-11，用户授权启动 E0 的 100 轮正式训练，并在训练正常后准备模块代码。 E0 作业为 `20261111-172658-8eff`，原版提交为 `a459eeade6387927c599a50f287a319a19d7ae82`。 结果根目录为 `/usr/local/rfdetr26/output/output1/baseline-seg-small-v1-seed42`， 权重和指标位于其 `training/` 子目录。正式权重保留。

E0 固定 Seg Small、384 分辨率、双卡每卡 batch 8、累积 1、BF16、seed 42， 100 轮、lr=1e-4、lr_encoder=1.5e-4、step 调度的 lr_drop=70、关闭早停。 每轮验证并更新最佳 EMA 权重与 last.ckpt，每 20 轮保存周期检查点。 源码归档、正式配置和版本哈希保存在该输出目录。

E0 已完成 100 轮，训练日志在 2026-11-11 18:42:55 记录最佳权重保存，面板作业退出码为 0。 核对 176 个服务器源码文件均与 E0 启动归档一致，Git 工作区干净；模块代码在本地实验分支开发。 接入检查使用 `output/output1/soft-cldice-integration-20261111` 中的独立源码副本， `CUDA_VISIBLE_DEVICES=''`，没有改写训练目录或安装新依赖。 同步仍由用户审阅后 commit/push、服务器 pull 完成，当前没有提交或推送。

soft-clDice 已接入模型配置、训练输出和 criterion，默认关闭，仅在显式配置的 GT 类别上启用。 E1 已于 2026-11-12 08:52:59（上海时间）启动 100 轮正式训练，作业 `20261112-085259-9638`。 双卡 BF16/NCCL 短验证已通过；最终效果与稳定训练耗时仍需根据完整运行评估。 本次使用本地未提交源码的哈希冻结副本，位于远端 E1 输出目录的 `source/`，没有 commit/push/pull。 启动、修复和验证记录见 `reports/tomato-20261111/runs/e1-seg-small-cldice-c0p1-v1-seed42/runtime_validation.md`。 原版 E0 配置和服务器 Git 管理源码未改动。

## E1：类别选择性 soft-clDice

保留原版分类、框、mask BCE/Dice、匹配器和推理路径，在匹配后的果梗实例上补充拓扑损失。 算法参考 [clDice，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) 和[作者实现](https://github.com/jocpae/clDice)。soft-clDice 不能替代区域损失， 也不能保证修复遮挡或错误标注造成的所有断裂。

已经实现的内核位于 `src/rfdetr/models/soft_cldice.py`：

- 输入同一有效网格上的 `(N,H,W)` 预测 logits 与二值 GT，输出 `(N,)`。
- 每个实例独立计算，不把不同实例的骨架合并为一个比值。
- 半精度输入在 FP32 中计算骨架和重叠；FP64 保留供梯度检查使用。
- GT 骨架不参与梯度；空匹配集合返回与预测图连接的空损失向量。
- `iterations=10`、`smooth=1.0` 是函数起点，尚未经过番茄 E1 超参数实验。

接入范围：

| 文件                                     | 职责                                                             |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `config.py` 的 `SegmentationTrainConfig` | 损失权重、目标类别和骨架迭代数；默认关闭                         |
| `_namespace.py`、`models/_types.py`      | 按库的配置桥接规则传递新增字段                                   |
| `models/lwdetr.py`                       | 构建开关与损失权重；仅启用后的训练/验证损失前向携带 padding      |
| `models/criterion.py`                    | 使用已有匹配索引筛选 GT 类别；对齐索引设备、mask 网格和归一化    |
| `training/module_model.py`               | 多尺度输入缩放前保留 GT 对应 padding，仅供启用的 clDice 损失对齐 |
| `models/soft_cldice.py`                  | 独立数学内核；已实现并测试                                       |
| `configs/tomato/`                        | E0 与 E1 独立配置和独立输出目录                                  |
| `tests/models/`                          | 默认关闭等价性、类别筛选、mask 格式及梯度测试                    |

使用 `mask_cldice_loss_coef=0.0` 作为默认关闭开关；类别列表默认空，E1 配置显式选择 当前数据的 `stem=1`。启用时要求类别合法，权重为零时完全不调用拓扑分支、 不新增 loss key、不分配额外预测张量、不消耗随机数。 第一版只在最终 decoder 输出计算拓扑项；辅助层和 encoder 暂时沿用原版损失， 避免构建 weight_dict 时自动复制出未计划的拓扑项。

计算路径：

1. 原版 mask BCE/Dice 使用随机采样点，soft-clDice 必须使用完整二维网格。
2. 同时支持密集 `pred_masks` 和 `spatial_features/query_features/bias` 延迟表示。 延迟表示只重建匹配到的果梗 logits，不物化所有 query 的高分辨率 masks。
3. 使用 GT 对应的 padding 画布，将选中 logits 以 FP32 双线性插值对齐到完整监督网格。 原版 per-batch 多尺度仅缩放输入、保留 GT 网格，因此启用 clDice 时在缩放前保留 padding， 并在 training_step 中替换该损失使用的 `mask_padding`；每批清空，关闭时不替换输出。 无 per-batch 缩放时沿用模型实际输入画布。 再裁到该图 GT 网格；padding 对应概率置零，GT 同步屏蔽。不会下采样细梗标注。 按图计算，反向时通过非重入梯度检查点重算骨架，减少完整分辨率激活的保存量。
4. 筛除 padding 后为空的 GT；每个有效匹配单独计数，包含 Group DETR 的重复匹配。 损失分子与计数同时增加，因此复制相同 query 组不会放大损失。
5. DDP 所有 rank 都归约有效计数。分母为 `max(global_count, 1) / world_size`， 先钳制全局计数再除以卡数；仅一张卡有一个果梗时也不会被错误减半。 空选择仍连接预测计算图，双进程测试覆盖单侧为空和两侧都为空。
6. 原始 mask BCE/Dice 数值与随机采样过程保持不变；类别筛选用 GT 类别，不用预测类别。

配置入口是 `SegmentationTrainConfig`，E1 文件为 `configs/tomato/seg_small_cldice.yaml`：

```yaml
mask_cldice_loss_coef: 0.1
mask_cldice_class_ids: [1]
mask_cldice_iterations: 10
```

`0.1` 是初步消融起点，不代表已调优。E1 其余训练参数与 E0 一致，独立输出到 `output/output1/e1-seg-small-cldice-c0p1-v1-seed42/training`。 新增损失日志键为 `loss_mask_cldice`；总损失在库原有加权路径中乘以该系数。 关闭时不注册此键。推理路径不计算骨架，不新增模型权重或 state_dict 键。 后续更换系数、种子或重复运行时必须另选 output_dir；不要覆盖 E0 或既有 E1 结果。

第一版范围：支持原版分割自动优化路径、按微批计算的梯度累积及 DDP； 仅支持 `sum_group_losses=False` 的组平均方式。暂不与 GroupPose 关键点的手动优化路径组合， 也不接受 criterion 的 `num_boxes` 外部覆盖，因为果梗需要独立计数归一化。 原版 `pad_targets_to` 固定目标数量填充仍不支持分割，与图像空间 padding 是两回事。 这些组合会明确报错，避免默默改变损失尺度。

CPU 验收覆盖默认关闭、原始损失与随机数不变、非果梗无直接拓扑梯度、非顺序匹配、 空匹配/空果梗/空 GT、dense/lazy 一致、非方形和贴边实例、BF16、实际 mask head 反传， 以及真实双进程 Gloo 的归一化和梯度。另有 8 组与 E0 归档 criterion 的逐位对照， 覆盖 train/eval、dense/lazy、默认/显式 box 分母，损失、梯度和 RNG 状态完全一致。 新增/内核测试共 59 项通过（其中 3 项为双进程 CPU/Gloo），相关原版配置、namespace、 分割前向和 criterion 回归测试共 403 项通过；另有 2 项 XLA 依赖条件跳过，44 项 GPU 用例未选择。 Ruff 0.17.3 检查与格式检查通过，E1 配置通过官方 CLI `--print_config` 解析。 初始测试 XML、日志和 `baseline-parity.json` 保存在独立验证目录。 2026-11-12 启动前补充并通过 4 个 GPU 索引用例、4 个多尺度网格用例、7 个原版训练模块回归， 以及真实双卡 BF16/NCCL 的两个训练和验证 step。GPU 检查发现的索引设备和多尺度网格问题已修复。 没有运行完整上游套件；新增证据保存在 E1 输出目录及项目报告副本中。

## E2 当前准备状态

2026-11-12，Query MLP 初稿已归回正式仓库，保持独立开关、默认关闭，E2 配置中 clDice 为零。 本版尚待当前 Git 版本的动态验收，未启动训练；详细接口和未完成项见 [tomato_e2_implementation.md](tomato_e2_implementation.md)。 后续代码只在正式本地仓库开发，经用户审核提交、推送后由服务器 pull，见 [tomato_git_workflow.md](tomato_git_workflow.md)。已有 E1 运行副本保持原样。

## 后续模块顺序

| 实验 | 内容                     | 与 E0 的主要区别                 |
| ---- | ------------------------ | -------------------------------- |
| E0   | 原版分割 baseline        | 100 轮已完成，源码与结果保持独立 |
| E1   | 果梗 soft-clDice         | 只加可关闭损失                   |
| E2   | Query MLP 采摘点基线     | 新增可关闭坐标头（首版）         |
| E3   | Query 与空间特征融合热图 | 与 MLP 做独立对照                |
| E4   | 拓扑损失与热图联合       | 组合已验证模块                   |
| E5   | GT 骨架距离场引导        | 只在训练时提供结构监督           |
| E6   | 关键点匹配代价           | 后期可选探索                     |

E2 不通过开启现有 GroupPose 分支代替计划中的 MLP。 当前数据为整串采摘的总果梗候选切割点，1033 个点已全部人工认可； 只使用既定 train/valid，不新增测试集，也不重写用户已接受的点。

评估需区分整体 mask AP、果梗单类 mask AP、实例 clDice 与关键点误差。 原版日志的逐类 `val/ema_AP/*` 是框 AP，不能当作果梗 mask AP。 关键点评估沿用果梗 GT 框对角线归一化，并分别报告条件 PCK 与漏检计失败的端到端 PCK。

在新模块训练前固定数据、轮数、有效 batch、增强和模型选择规则。 新增损失的权重与分辨率允许随预实验修订，每次保留独立配置和结果， 不将初步设想或未完成的实验写成最终方法与提升。

E2 实施前计划见 [tomato_e2_query_mlp_plan.md](tomato_e2_query_mlp_plan.md)。原始点均为 v=2，首版先回归坐标， 独立比较 E0/E1 固定点提取与 E2 Query MLP；可见性头和 E1+E2 组合后续单独评估。
