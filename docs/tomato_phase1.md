# 番茄实验：数据 v1 与原版分割基线

## 当前完成范围

2026-11-11，用户确认：“我全部复核了一遍，没有问题，可以加入数据集进行下一步工作”。 本次将全量采摘点作为人工复核通过的标注接入 COCO 数据集，并准备 E0 原版分割基线配置。 数据接入阶段未修改模型。随后已完成单卡与双卡 batch 4/8/16 的短训， E0 的 100 轮训练已完成；E1 已于 2026-11-12 启动正式训练。 模块状态和接入边界见 [tomato_modules.md](tomato_modules.md)。

| 划分  | 图片 | 分割实例 | 采摘点 |
| ----- | ---: | -------: | -----: |
| train | 1395 |    16965 |    912 |
| valid |  199 |     2348 |    121 |
| 合计  | 1594 |    19313 |   1033 |

不创建测试集。没有果梗的 578 张图片仍保留全部果实分割监督。 1033 条标注对应 425 个独立果梗组；其中 608 条属于同一划分内、几何完全相同的增强副本复用， 不应在论文中称为 1033 个独立采摘点样本。

## 数据版本与坐标约定

服务器项目根目录：`/usr/local/rfdetr26`。

- 原始数据：`data/0/TOMATO.v7i.yolov11`。
- 正式数据：`data/0/TOMATO.v7i.rfdetr-keypoints-v1`。
- 原复核包：`output/output1/keypoint-full-v1-20261111/reviewed-v1`，保持原样。
- 本次 CPU 检查和配置快照：`output/output1/phase1-dataset-20261111`。

正式数据包含 `train/_annotations.coco.json`、`valid/_annotations.coco.json`、 `keypoints.approved.json`、`approval.json`、`release.json`。 各划分的 `images` 是指向原图目录的相对符号链接；移动或清理数据时必须保留原始数据目录。

类别索引保持 `0=ripe, 1=stem, 2=unripe`。只为 stem 定义一个 `whole_truss_candidate_cut`， 关键点数为 `[0, 1, 0]`。stem 使用原图连续像素坐标 `[x, y, 2]`；果实的关键点列表为空、 `num_keypoints=0`，由上游加载器补齐不可见的零值，保留果实的框与掩膜。 `v=2` 表示图中采摘点可见，不代表总果梗的两个语义端点都完整可见。

全部 1033 点均参与监督，包括此前带 `endpoint_uncertain` 的 538 条。 端点不确定、人工描线及增强副本来源等信息继续保存，不能据此删除已认可的点。 接受记录以用户本次全量复核声明为依据；坐标没有二次修改。

YOLO 多边形使用当前上游的 float32 坐标处理、裁边和 PIL 栅格化函数，再编码为 COCO RLE， 避免直接改用 COCO 多边形栅格化后改变细果梗边界。COCO `area` 为掩膜像素面积。 每个实例保留原标签行号及哈希，每张图片保留原图、标签文件哈希。

复核 JSON 的 SHA-256： `1a158285fe4a5dccc577ba5bfb5967a2250c1299fa0903d8826e30cba58d1107`。 `release.json` 记录正式文件及生成脚本的哈希；生成时脚本快照保存在 `output/output1/phase1-dataset-20261111/dataset-build-source.zip`。 重新生成时需使用一个不存在的新版本目录，工具拒绝覆盖已有数据。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python scripts/prepare_tomato_keypoints.py \
    --source data/0/TOMATO.v7i.yolov11 \
    --annotations output/output1/keypoint-full-v1-20261111/reviewed-v1/annotations.json \
    --approval data/0/TOMATO.v7i.rfdetr-keypoints-v1/approval.json \
    --destination data/0/TOMATO.v7i.rfdetr-keypoints-v2
```

## E0 基线配置

配置：`configs/tomato/seg_small_baseline.yaml`。原版提交为 `a459eeade6387927c599a50f287a319a19d7ae82`，实验分支为 `codex/tomato-topology-keypoints`。

正式 E0 已于 2026-11-11 17:26:58 启动，作业 `20261111-172658-8eff`。 采用官方 RF-DETR-Seg Small 权重、384 分辨率、100 epochs、双卡每卡 batch 8、 梯度累积 1、有效 batch 16、BF16、seed 42。保持 torchvision 增强、多尺度、 AdamW、lr=1e-4、lr_encoder=1.5e-4、EMA；lr_drop=70，关闭早停。 关键点和自定义拓扑分支不参与 E0，`run_test=false`。

结果根目录为 `output/output1/baseline-seg-small-v1-seed42`， 权重与指标位于 `training/` 子目录；根目录保存源码归档、配置与哈希。 每轮验证并更新 best EMA 和 last.ckpt，每 20 轮另存周期检查点；正式权重保留。 100 轮已完成，面板作业正常退出（exit_code=0）；训练日志在 2026-11-11 18:42:55 记录最佳 EMA 权重保存。模块接入期间核对了 176 个源码文件，均与启动归档一致。

复现实验时先将配置的 output_dir 改为新的独立目录，再从服务器项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 --module rfdetr fit \
    --config configs/tomato/seg_small_baseline.yaml
```

运行中的服务器 Git 管理源码保持原样，模块代码只在本地实验分支开发。 代码同步仍采用本地审阅、用户 commit/push、服务器 `git pull --ff-only`； E0 完成前不拉取模块改动。CPU 模块测试使用输出目录下的独立源码副本。

## 校验与下一阶段

定向测试覆盖复核文件哈希、原图/标签/标签行校验、缺失或重复关键点、越界及掩膜外点、 禁止覆盖已有版本、没有果梗的图片，以及上游翻转、缩放、裁剪的数据对齐。 全量 CPU 检查逐张比较原生 YOLO 掩膜与 COCO 加载结果，并比较纯分割和带关键点加载的 类别、框及掩膜。证据位于上述输出目录的 `pytest.xml`、`validation.json`、 `baseline-resolved.json` 和 `baseline-cli-resolved.yaml`。 数据接入时的 CPU 检查未执行完整上游测试套件或模型前向/反向； 随后完成的四组 5 轮短训已验证原版训练与验证流程，日志另存于 output/output1。

原版 E0 与模块开发保持隔离。果梗 soft-clDice 已完成默认关闭的接入和 CPU 验证， 独立 E1 配置见 `configs/tomato/seg_small_cldice.yaml`；双卡短验证通过后已启动正式训练。 实现范围、默认关闭等价性与 DDP 验证见 `docs/tomato_modules.md`。 后续采摘点预测模块仍需单独实现和消融。 新模块默认关闭，零权重或关闭时必须回到 E0 行为；不能用新增模块改写基线。 现有 COCO 转换器已经支持同时读取掩膜和关键点，但上游训练数据开关绑定 `use_grouppose_keypoints`。后续自定义 query-MLP 头需要独立的开关及数据接入， 不能直接开启现有 GroupPose 模型分支来冒充所计划的 MLP 实验。 方案和超参数允许随实验调整，每次调整需另存配置和输出目录。
