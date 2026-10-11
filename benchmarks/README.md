# Benchmark adapters

These adapters provide task interfaces and independent evaluators. None of
the public workspaces is seeded with a fixed predictor. Dataset files, private
labels, generated solutions and experiment logs are not committed.

| Adapter | Metric | Preparation |
| --- | --- | --- |
| [Spooky](mle_bench_lite_spooky/README.md) | multiclass log loss | public train mirror and MLE-bench-style 90/10 split |
| [Insults](mle_bench_lite_insults/README.md) | ROC-AUC | official prepared MLE-bench train/test CSV files |
| [NOMAD CSV-only](mle_bench_lite_nomad_official/README.md) | mean-column-wise RMSLE | public CSV mirror and prepared 90/10 split |
| [OpenML credit-g](openml_credit_g/README.md) | accuracy | optional small execution smoke task |

Install the repository with `python -m pip install -e ".[dev,benchmark]"` before
running an adapter. The MLE-bench adapters all accept:

```powershell
python run_graph_evolve.py --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1 --budget-seconds 18000 --max-iterations 20 --max-repair-steps 100 --project-id unique-project --state-dir runs/graphevolve_state
```

Run from the adapter directory. Use a distinct project ID for each independent
trial; to resume, specify its existing project and session IDs. Each runner
prints the localhost dashboard URL. Set different `--dashboard-port` values
when several adapters are running.

The current [results and comparison conditions](RESULTS.md) describe three
selected tasks. They are not a result for the full MLE-bench Lite task set.
Keep task metrics separate and record any reduced data scope, interrupted
reference run or reused experimental evidence.
