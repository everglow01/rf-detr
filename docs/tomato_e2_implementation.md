# E2 Query MLP：实现初稿与验收状态

状态：2026-11-12 已归回正式 `rfdetr26` 仓库；当前版本已通过本机 CPU 单元测试、关联原版回归和类型检查，真实模型/GPU 验收仍待完成。尚未启动 E2 训练。 设计依据见 `tomato_e2_query_mlp_plan.md`；版本同步见 `tomato_git_workflow.md`。

## 当前实现

| 配置                                            | 默认值              | E2 配置       |
| ----------------------------------------------- | ------------------- | ------------- |
| `ModelConfig.query_point_head`                  | `false`             | `true`        |
| `ModelConfig.query_point_class_ids`             | `[]`                | `[1]`（果梗） |
| `SegmentationTrainConfig.query_point_loss_coef` | `1.0`，仅开头时生效 | `1.0`         |
| `mask_cldice_loss_coef`                         | `0.0`               | `0.0`         |
| `use_grouppose_keypoints`                       | `false`             | `false`       |
| `freeze_encoder`                                | `false`             | `false`       |

配置文件为 `configs/tomato/seg_small_query_point.yaml`，其训练设置沿用 E0：Seg Small、384、100 轮、 双卡每卡 batch 8、累积 1、BF16、seed 42、lr=1e-4、lr_encoder=1.5e-4。 输出目录为 `output/output1/e2-seg-small-query-mlp-v1-seed42/training`；当前仅准备配置，不据此启动训练。

最终 decoder query 接 `hidden → hidden → hidden → 2` 的 MLP，中间 ReLU、末端 sigmoid。 输出键为 `pred_pick_points`，与 GroupPose 的 `pred_keypoints` 分开。初始化使用独立 RNG 上下文， 默认关闭时不创建新头。代码中没有 detach；点损失直接连接 query、decoder 和骨干的可训练参数。 真实模型上的非零共享梯度以及梯度量级仍需运行验收，不能仅凭连接形式宣布通过。

`loss_query_point` 复用最终层原有 Hungarian 匹配，只监督所选 GT 类别且 `v>0` 的单个采摘点。 单点损失为 `(|dx|+|dy|)/2`。分母按全局有效匹配点数归约，包含 Group DETR 重复匹配， 并在除以 DDP 卡数前钳制至至少 1。空点仍保留连接计算图的零值。 不增加匹配代价、辅助层损失或 encoder 点损失；总损失沿用原版加权与自动优化。 权重设为 0 时仍创建点头并计算未加权日志，完全移除模块应使用 `query_point_head=false`。

COCO 读取开关与 GroupPose 架构解耦，复用已有点/掩码几何变换。 单个点不需要左右索引交换，因此保留原分割流水线的水平翻转。 当前训练入口明确限制 torchvision 和累积 1；其他增强后端尚未验收。

后处理使用与 mask 相同的 top-k query 和阈值筛选，返回像素坐标 `pick_points` 和布尔 `pick_point_valid`。 公共 `predict()` 仍返回分割 `Detections`，对应字段保存在 `detections.data`。 `pick_point_valid` 只表示该行预测类别是否属于启用点头的类别，不是可见性或点置信度； 非果梗行的候选坐标不应使用。E2 导出暂不支持，明确抛出异常；关闭模块时沿用原导出。

预训练加载允许完整新增点头缺失，拒绝点头部分缺失、其他未解释的参数缺失或额外参数。 既有类别头对齐规则继续由库处理。官方 Seg Small 权重的实际加载仍待当前版本动态确认。

## 验证证据与待办

先前 `green1` 独立副本：20 项通过，包括 10 项当时的 E2 测试和 10 项 E1 回归。 之后增加了数据增强、公共推理、双进程归一化、CUDA 索引和权重加载测试，并修正训练入口的增强枚举比较、 保留单点翻转以及导出拒绝前的状态修改。这些后续改动尚未在服务器运行。 历史归位检查见 `reports/tomato-20261112-code-review/README.md`，其结果保留为当时状态。

本轮已在正式仓库使用 Windows、Python 3.11.15 和 CPU 版 PyTorch 2.14.0 运行新增模块及相关原版配置、namespace、COCO、权重和公共推理回归（含 doctest）：806 passed、18 skipped、19 deselected。GPU 测试按仓库 CPU CI 规则通过 `-m "not gpu"` 排除；Windows 符号链接和双进程 Gloo 测试使用仓库现有平台标记。完整包与脚本的 mypy 检查通过，E0/E1/E2 三份 YAML 均通过官方 CLI 的 `fit --print_config`。这不等于真实骨干共享梯度或双卡训练已验收。完整提交检查及遗留限制见 `reports/tomato-20261112-git-submission/README.md`。

Git 同步后的必要验收：

- 在 Linux 服务器复跑新增测试和相关原版回归，特别是本机因 Windows 平台限制跳过的符号链接和双进程 Gloo 测试；按仓库要求运行完整 pre-commit 检查。
- 使用真实样本确认点头、query、decoder、backbone 的点损失梯度非零且有限；验证总梯度的加权和。
- 核对默认关闭与归档 E0 的损失、输出、随机状态；确认 per-batch 多尺度下坐标不被重复缩放。
- GPU 空闲后，确认官方预训练加载和双卡 BF16/NCCL 短验证，包括无点批次。
- 补全点误差/PCK 评估与 E0 框中心、mask 质心、骨架中点参考。它们尚未实现，不能宣布 E2 完成。

服务器 CPU 验证命令（在 pull 后的正式仓库运行）：

```bash
cd /usr/local/rfdetr26
CUDA_VISIBLE_DEVICES='' RFDETR_SKIP_DINOV2_PREWARM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -m pytest -o pythonpath=src -o addopts='' -m "not gpu" --doctest-plus --tb=short -q \
    tests/models/test_query_points.py tests/models/test_query_points_distributed.py \
    tests/datasets/test_query_points.py tests/inference/prediction/test_query_points.py \
    tests/models/test_lwdetr_cldice.py tests/training/test_cldice_multiscale.py
```
