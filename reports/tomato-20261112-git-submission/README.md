# 番茄实验分模块提交与本机验证记录

2026-11-12，按用户明确授权，在正式目录 `F:\OneDrive\Desktop\工作资料\rfdetr26` 分批提交到 `codex/tomato-topology-keypoints`。本报告所在提交保存实验历史与本轮验证证据；GitHub 推送状态以远端分支为准。

| 提交 | 内容批次 |
| --- | --- |
| 7f301649 | 01-repository |
| a1a77f32 | 02-data |
| 39e8af46 | 03-cldice |
| 22ed56c1 | 04-query-points |
| e1916536 | 05-protocols |

E0 基准仍是 `a459eeade6387927c599a50f287a319a19d7ae82`。数据工具、E1 soft-clDice、E2 query-point 模块分别独立提交；模块默认关闭。报告与配置单独提交。后续可按模块选择版本并通过配置做消融，避免把启用多个模块的结果当成单模块实验。

## 本次验证

- 新增模块及相关原版回归（含 doctest）：**806 passed，18 skipped，19 deselected**，耗时 74.09 秒。测试涵盖配置、namespace、COCO、损失、点/掩码后处理、权重兼容和公共推理。详细输出见 `pytest-cpu.txt`。
- CPU 环境：Windows、Python 3.11.15、PyTorch 2.14.0+cpu。使用 `-m "not gpu"` 排除 GPU 用例，符号链接和双进程 Gloo 用例使用仓库已有 Windows 平台标记。首次扩大回归范围时误选 8 项 CUDA 用例；按 CPU CI 选择规则重跑后全部适用用例通过。它们不代表 GPU 验收完成。
- 完整 `src/rfdetr/` 与 `scripts/` 的 mypy 通过。修正数据脚本容器类型，以及 clDice 图内零值的 Tensor 类型初始化；不改变损失定义。
- E0/E1/E2 三份 YAML 均通过官方 `fit --print_config`，未启动训练。CLI YAML 中的 `TV`、`PER_BATCH` 是枚举名称；Python/Pydantic 直接构造使用 `torchvision` 等枚举值。
- 每批提交前运行完整 pre-commit，未用 SKIP 屏蔽检查。使用已安装项目环境的 `UV_NO_SYNC=1`，避免重复解析未用到的导出后端依赖。前五批日志保存在本目录；完整任务日志位于对话工作区 `commit-work`。

## 保留的上游检查问题

`check-added-large-files --enforce-all` 对原有 `tests/export/tensorrt/test_tensorrt_int8.py` 报告 253 KB 超过 250 KB；该文件没有本轮改动，阈值和检查范围也未放宽。各批最终检查日志中，其余检查通过。

完整格式检查还会给原版 `CHANGELOG.md` 增加一个空行；检查后撤回了这一无关改动，保留上游内容。因此不能把本次结果描述为整个仓库在干净检出下无遗留检查问题。

## 归档和后续验收

历史报告与本轮开始时的备份逐字节核对；保留原有哈希与当时结论，不把旧记录改写成最新状态。六个源码 tar.gz、两张较大 PNG 及已有忽略规则覆盖的日志保留本地，未进入 Git；对应清单仍指向完整本地归档。数据集和训练权重不纳入提交。

E1 作业 `20261112-085259-9638` 使用此前的冻结源码快照，其清单 SHA256 为 `1f56cda2bd9d2b3ad5febcc08564f7176e2cc54887492f57d6a770d90033b4bd`。新提交不能反写成该作业的启动版本。本任务只做本地提交及 GitHub 推送，未修改服务器代码、训练进程或实验产物。

E2 仍是初稿：Linux 平台用例、真实样本在点头/query/decoder/backbone 上的非零有限梯度、双卡 BF16/NCCL、官方预训练实际加载及采摘点误差/PCK 评估仍待完成。后续经授权检查服务器工作区，再通过 `git pull --ff-only` 更新并记录实际提交 SHA。
