# MLE-bench Lite - spooky-author-identification

This adapter exposes the public training text and test features, while keeping
held-out labels in `private_evaluator/`. The metric is multiclass log loss,
lower is better. GraphEvolve writes its own predictor from the task contract.

`prepare_from_hf.py` downloads a public training mirror and applies the
MLE-bench-style 90/10 split with `random_state=0`. It prepares the public CSVs
and private one-hot answer file. Dataset files are excluded from Git.

Run from this directory after installing the repository:

```powershell
python prepare_from_hf.py
python run_graph_evolve.py --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1 --budget-seconds 18000 --max-iterations 20 --max-repair-steps 100 --project-id spooky-trial --state-dir runs/graphevolve_state
```

The runner prints the localhost dashboard URL. It accepts `--session-id` to
resume the same run and `--prior-project-id` to read earlier experiment cards.
Neither option substitutes a fixed predictor. Technical repairs do not consume
research rounds, and the best valid checkpoint is retained.

`mlevolve_description.md` describes the corresponding baseline task interface.
See [current results](../RESULTS.md) for scores and comparison conditions.
