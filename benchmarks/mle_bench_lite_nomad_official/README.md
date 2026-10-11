# MLE-bench Lite — NOMAD 2018 tabular adapter

`prepare_from_public.py` reproduces the official MLE-bench 90/10 CSV split
(`random_state=0`) from the original 2,400-row public training CSV. The four
prepared CSVs match the MD5 checksums in MLE-bench's NOMAD task definition.
It omits the crystal geometry files, so this is a lightweight CSV-only local
evaluation of the task. The independent evaluator computes mean-column-wise
RMSLE.

GraphEvolve receives only the task contract and public CSV files and writes its
own solution. No fixed predictor is included.

Run from this directory after installing the repository:

```powershell
python prepare_from_public.py
python run_graph_evolve.py --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1 --budget-seconds 18000 --max-iterations 20 --project-id nomad-trial --state-dir runs/graphevolve_state --dashboard-port 8769
```

Prepared data and private labels are excluded from Git. The runner prints the
dashboard URL and accepts `--session-id` to resume. `mlevolve_description.md`
describes the corresponding baseline interface, with the same CSV-only scope.
See [current results](../RESULTS.md) for measured scores and comparison conditions.
