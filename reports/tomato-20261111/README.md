**番茄 RF-DETR 实验记录**

- [E0 100 轮分析报告](baseline-analysis/baseline_analysis.md)
- [训练曲线](baseline-analysis/baseline_curves.png)
- [果梗预测对照图](baseline-analysis/stem_examples.png)
- [E1 启动及运行时验证](runs/e1-seg-small-cldice-c0p1-v1-seed42/runtime_validation.md)
- [E1 最近状态记录](runs/e1-seg-small-cldice-c0p1-v1-seed42/launch_status.json)
- [E1 固定配置](runs/e1-seg-small-cldice-c0p1-v1-seed42/training.yaml)
- [E2 Query MLP 采摘点模块计划](e2_query_mlp_plan.md)

E0 报告保留 2026-11-11 分析时的内容；E1 于 2026-11-12 正式启动，最新记录见 E1 文件。
目录包含原始指标、逐实例结果、分析脚本、源码快照和校验清单，不含训练权重或访问凭据。
状态文件是观察时点的快照，不会自动刷新。正式训练结果保留在服务器 output/output1 下。
