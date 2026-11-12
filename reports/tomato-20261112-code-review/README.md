# E1/E2 工作区归位与提交审核（2026-11-12）

本轮完成代码归位及审核材料整理，E2 仍为开发初稿，尚未完成当前版本动态验收或正式训练。
正式仓库：`F:\OneDrive\Desktop\工作资料\rfdetr26`。
分支：`codex/tomato-topology-keypoints`；HEAD 仍为 `a459eeade6387927c599a50f287a319a19d7ae82`。
本轮没有暂存、commit、push、远端 pull 或源码上传。

## 改动范围

| 内容 | 主要文件 | 状态 |
| --- | --- | --- |
| E1 soft-clDice | `models/soft_cldice.py`、criterion、模型构建、训练模块 | 此前已实现并验证，仍未提交 |
| E2 Query MLP | config/namespace/types、lwdetr、criterion | 默认关闭的坐标头及有效匹配点 L1 初稿 |
| E2 数据/推理 | datasets/coco、_aug_utils、postprocess、inference、_prediction/core、weights | 单点读取、保留翻转、联合输出及预训练校验初稿 |
| 实验配置 | `configs/tomato/` | E0、E1、E2 独立配置；E2 clDice=0、骨干未冻结 |
| 测试 | `tests/**/test_query_points*.py` 等 | E1 已有验证；E2 后续新增用例待运行 |
| 已有数据与报告 | `scripts/prepare_tomato_keypoints.py`、`docs/`、`reports/tomato-20261111/` | 关键点准备、baseline 分析、E1 运行归档 |
| 流程与当前记录 | `AGENTS.md`、`docs/tomato_git_workflow.md`、本目录 | 唯一开发目录、Git 同步和真实验收状态 |

从临时副本归回 17 个文件，覆盖前逐项匹配原始 SHA256；已有文件归位前备份保留在对话工作目录的
`e2-work/before-return/`。随后修改均直接发生在正式仓库。
独立的 [E2 增量](e2-only.patch) 相对于归位前的 E1 工作区生成，不把已有 E1 改动混入 E2。
该 patch 用于审核，不需要再次应用。它覆盖本轮 17 个 E2 代码/配置/测试文件，文档另外在 Git 中审核。
既有 E1 报告目录还保留了 6 个历史源码压缩包，未删除或改写；审核暂存内容时请区分这些历史证据与当前源码。
文件哈希见 [e2-files.json](e2-files.json)，完整当前未提交文件清单见 [working-tree-files.json](working-tree-files.json)。

## 当前验证

- 16 个相关 Python 文件语法解析通过。
- Ruff 检查和格式检查通过。
- `git diff --check` 通过；E2 patch 的反向 dry-run 校验通过，未实际应用补丁。
- YAML 解析和与 E0 的逐字段比较通过：除点头开关、类别、系数及独立输出目录外，训练参数一致；明确写出的 `freeze_encoder=false` 和 `mask_cldice_loss_coef=0` 与原默认值一致。
- 本机未确认可复用的 RF-DETR 运行环境；默认 Python 为 3.9，未为本轮新装训练依赖。完整 pre-commit 尚未运行（本机未安装）。
- 当前版本的模型、数据增强、双进程与 GPU 动态验收尚未运行，等待用户审核提交推送后，在远端已有环境中通过 Git 同步开展。

结构化结果见 [checks.json](checks.json)。

先前的 [green1 测试日志](previous-green1-tests.log) 记录 **20 passed**：当时的 10 项 E2 用例和 10 项 E1 回归。
这是历史副本结果。此后新增/修改了单点翻转、推理包装、训练入口、导出拒绝行为及更多测试，不能把该结果标成当前版本全部通过。
差异文件列于 `checks.json` 的 `files_changed_since_green1`。
E1 更完整的 CPU/GPU 验证记录保留在既有 `reports/tomato-20261111/` 中。

## 提交审核与后续

[提交标题及正文草稿](commit-message.txt) 针对当前 E1/E2、配置、数据准备和报告的整体未提交状态，明确 E2 为初稿。
若只选择部分文件提交，应对应调整草稿，避免提交说明与暂存内容不一致。

在正式仓库审阅并按需要暂存后，可使用：

```powershell
git diff --cached --stat
git diff --cached --check
git commit -F reports/tomato-20261112-code-review/commit-message.txt
git push origin codex/tomato-topology-keypoints
```

提交前仍需完成仓库要求的完整 pre-commit 检查；本记录没有把静态检查视为其替代。
这些命令未执行。按用户给出的提交偏好，由用户审核、提交、推送；不会代为执行。
随后服务器通过 `git pull --ff-only` 同步，记录新 SHA，再进行 E2 CPU、真实共享梯度及空闲 GPU 短验证。
启动 E2 训练前还需补全点评估和 E0 定位参考，见 [E2 当前实现与待办](../../docs/tomato_e2_implementation.md)。

E1 已运行任务的历史源码快照保持原样；后续 Git 提交不是其启动版本。
