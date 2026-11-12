**E1 启动与运行时验证记录（2026-11-12）**

正式作业：`20261112-085259-9638`，上海时间 `2026-11-12 08:52:59` 启动。
最近观察已完成 5 轮，状态为 `running`；这是时点记录，当前状态应查询面板。
结果目录：`/usr/local/rfdetr26/output/output1/e1-seg-small-cldice-c0p1-v1-seed42/training`。

训练参数保持 E0：Seg Small、384、100 轮、双卡每卡 batch 8、累积 1、BF16、seed 42、lr=1e-4、encoder lr=1.5e-4、lr_drop=70。
唯一实验变量为最终 decoder 的果梗 soft-clDice：类别 [1]、系数 0.1、骨架迭代 10。官方预训练初始化，不从 E0 续训。
权重和日志保留，不覆盖 E0。

源码为本地未提交版本的冻结副本，基准提交 `a459eeade6387927c599a50f287a319a19d7ae82`。
当前源码清单 SHA-256：`1f56cda2bd9d2b3ad5febcc08564f7176e2cc54887492f57d6a770d90033b4bd`。`source.tar.gz` 包含完整 src、项目配置、E1 配置和两个新回归文件，逐项哈希已校验。
`source_changes.patch` 记录已跟踪源码差异；新增文件保存在归档内。没有执行 commit、push 或 pull。
`provenance.json` 保存数据注释文件与预训练权重的 SHA-256。冻结副本通过 PYTHONPATH 加载，远端 Git 工作区仍是 E0 版本。

启动前发现并修复两个运行时问题：

- 匹配器返回 CPU 索引，新增类别筛选在 CUDA 上。`criterion.py` 仅在 clDice 分支将两个匹配索引移到标签设备；新增 4 个 GPU 用例验证 dense/lazy、有/无果梗时的损失与梯度。
- 原版 per-batch 多尺度仅缩放输入，不缩放 GT 掩码。`module_model.py` 在启用 clDice 时保留缩放前 padding，training_step 将其交给 clDice 对齐 GT 网格；每批重置，关闭时不覆盖输出。新增 4 个用例覆盖缩小、放大、默认关闭及损失/梯度对照。

验证结果：索引修复后的相关 CPU/Gloo 集成测试 59 passed；最终源码的 8 个运行时回归全部通过；原版训练模块相关用例 7 passed（263 deselected）。
初次原版训练回归因网络不可用被 DINOv2 预热 fixture 跳过，设置测试用的 `RFDETR_SKIP_DINOV2_PREWARM=1` 后实际执行并通过。
测试源码路径显式固定，`gpu-regression-red.log` 确认修复前 4 个 CUDA 设备错误，`multiscale-red.log` 确认修复前 2 failed / 2 passed。
Ruff 与 git diff --check 通过；没有运行完整上游或全部 pre-commit 检查。

双卡 BF16/NCCL 短验证作业 `20261112-085139-00f5` 正常退出，两卡均完成 2 个 backward step 和 2 个验证 step。
检查了新增损失日志、有限总损失和梯度；两卡峰值 allocated 约 8.45/8.50 GiB、reserved 约 9.27/9.39 GiB。
DDP 各 rank 的 clDice 按全局匹配数归一化，局部值可以大于 1；应结合跨 rank 平均值解释，不能把局部值范围误认为计算异常。

此前三次短验证分别因 checkpoint 回调冲突、CPU/CUDA 索引、多尺度 GT 网格失败，配置、日志和作业号保留。
第一项只修正短验证配置；正式训练配置的超参数未修改。失败短验证权重不用于正式训练。

首轮指标、轮耗时和显存见 `training/metrics.csv`、`training/timing.rank0.json`、`training/timing.rank1.json`。
这些只是早期运行证据，不代表 E1 最终效果；最终应按既定 E0/E1 评估协议比较最佳及最终 EMA 权重。
