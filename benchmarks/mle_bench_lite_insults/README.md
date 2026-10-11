# MLE-bench Lite — detecting-insults-in-social-commentary

This adapter uses the prepared MLE-bench train/test split. The metric is binary
ROC-AUC, higher is better. GraphEvolve writes its own `solution.py` from the task
contract; no fixed predictor is included.

Prepare `detecting-insults-in-social-commentary` with the MLE-bench preparation
tools. Copy the prepared public `train.csv`, `test.csv` and
`sample_submission.csv` into `task/`; copy the prepared private answer
`test.csv` into `private_evaluator/`. Keep its `evaluate.py` in place. These data
files are excluded from Git.

Run from this directory after installing the repository:

```powershell
python run_graph_evolve.py --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1 --budget-seconds 18000 --max-iterations 20 --project-id insults-trial --state-dir runs/graphevolve_state --dashboard-port 8768
```

The runner prints the dashboard URL and accepts `--session-id` to resume.
`mlevolve_description.md` describes the corresponding baseline task interface.
See [current results](../RESULTS.md) for measured scores and comparison scope.
