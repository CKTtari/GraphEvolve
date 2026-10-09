# GraphEvolve v18 轻量级本地测评 — 2026-10-08

两种 RSI 系统均使用 gpt-6-luna；各指标只在同一任务内比较，不作跨任务原始分数平均。

| 任务 | 指标方向 | MLEvolve RSI baseline | 旧版 GraphEvolve | v18 最佳（独立复核） | 最后一轮 | 有效轮数 / 状态 | 对 baseline |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| spooky | multi_class_log_loss ↓ | 0.345936049 | — | 0.323023329 | 0.323023329 | 20 / completed | 领先 |
| insults | roc_auc ↑ | 0.916839842 | 0.912351324 | 0.923632435 | 0.923632435 | 20 / completed | 领先 |
| NOMAD CSV-only | mean_columnwise_rmsle ↓ | 0.063968260 | 0.062516905 | 0.059997761 | 0.059997761 | 20 / completed | 领先 |

只有状态 completed 的行是已完成运行；active 行是运行中检查点。
MLEvolve baseline 来源：benchmarks/LOCAL_EVALUATION_20261002.md。insults baseline 是中断于 3/8 步运行保存的最好提交；spooky 和 NOMAD baseline 为已完成运行。
NOMAD 两边均为 CSV-only，不含晶体结构文件。本报告不是官方 MLE-bench 总榜成绩。
v18 使用各自历史 GraphEvolve 实验卡片作为只读参考，代码由新项目 agent 生成；因此这是带历史经验的续跑，不能把单次成绩差异全部归因于系统改动。
电脑重启后的续跑沿用原 project-id、session-id 和已完成轮数；技术修复及候选重拟不计入有效研究轮数。

更新时间（北京时间）：2026-10-08T11:34:33+08:00
