# 番茄实验代码与 Git 同步约定

2026-11-12，用户确认恢复以下流程，并进一步明确授权本轮按内容和模块提交、推送到 `codex/tomato-topology-keypoints`。本轮由代理执行已授权的提交和推送；后续仍以每次任务授权范围为准。

## 唯一开发目录

本地代码在 `F:\OneDrive\Desktop\工作资料\rfdetr26` 修改，分支为 `codex/tomato-topology-keypoints`。 对话工作目录中的 `e2-work/source` 是此前的临时副本，已停止编辑，不是另一个开发仓库。 2026-11-12 已按文件哈希核对，将其中 17 个 E2 文件归回正式仓库，并保留归位前备份。 后续以正式仓库的工作区和 Git 历史为准。

## 审核、提交、同步

1. 在正式仓库检查 `git status`、diff、测试和配置。本轮按数据准备、E1、E2、实验配置及报告分批提交；具体版本和检查结果见 `reports/tomato-20261112-git-submission/README.md`。

2. 通常提供提交标题/正文，由用户审核并手动提交、推送；获得明确指令时可代为执行。本轮用户已授权分批提交和推送。

3. 在服务器 `/usr/local/rfdetr26` 检查分支和工作区，确认没有待处理的本地改动后执行：

    ```bash
    cd /usr/local/rfdetr26
    git status --short
    git branch --show-current
    git pull --ff-only origin codex/tomato-topology-keypoints
    git rev-parse HEAD
    ```

4. 核对本地、GitHub 与远端提交一致，再用服务器已有 `.venv` 验证代码。记录完整提交 SHA、配置、数据/权重哈希和验证结果。

5. 新训练使用已验证的 Git 提交；运行归档从该提交生成。源码归档仅作为可复现证据，不再替代 Git 同步。

不再上传未提交源码包作为服务器的新执行版本。不强制 reset、不清理用户工作区、不覆盖已有实验输出。 测试发现问题时，回到同一本地仓库修改，重新审核、提交、推送、pull。

## 已有实验的历史来源

E0 原版提交为 `a459eeade6387927c599a50f287a319a19d7ae82`。 此前 E1 通过面板上传未提交源码包，并通过 `PYTHONPATH` 使用其 `source/src` 启动；这不是一次 Git pull。 E1 作业为 `20261112-085259-9638`，源码清单 SHA256 为 `1f56cda2bd9d2b3ad5febcc08564f7176e2cc54887492f57d6a770d90033b4bd`。 保留该运行副本、清单、日志和结果，后来的提交不能被写成它的启动提交。

E2 以前的 CPU 检查也使用了面板上传的独立副本。其 `green1` 版本通过 20 项测试；此后又有改动， 该结果不能代表当前工作区已经通过完整验收。当前 E2 尚未训练。
