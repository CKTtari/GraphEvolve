# MLE-bench Lite — spooky-author-identification

This directory contains a single-task MLE-bench Lite run for GraphEvolve. The
task is the official `spooky-author-identification` competition format: train a
three-class text classifier and output class probabilities. The public data is
the same Kaggle competition data, prepared with the official MLE-bench 90/10
split (`random_state=0`). A public Hugging Face mirror is used when Kaggle API
authentication is unavailable.

## Prepare

```powershell
conda run -n nlphw python prepare_from_hf.py --output .
```

The generated `task/` directory contains only public inputs. The labels and
evaluator stay under `private_evaluator/` and are copied to a temporary
evaluation directory by GraphEvolve.

## Baseline

```powershell
conda run -n nlphw python baseline_solution.py
```

The baseline is a direct TF-IDF plus logistic-regression solution. It is kept
outside the agent workspace and is not supplied to GraphEvolve.

For a fair harness comparison, the accepted representation can also be run as
a fixed no-agent program:

```powershell
conda run -n nlphw python baseline_combined_solution.py
```

## GraphEvolve run

Set the API key in the current shell only:

```powershell
$env:LLM_API_KEY = "<your-key>"
conda run -n nlphw python run_graph_evolve.py `
  --model gpt-6-luna `
  --base-url https://yuzapi.fun/v1 `
  --budget-seconds 2400 `
  --max-iterations 12 `
  --max-repair-steps 100
```

The run records each code change, private score, runtime, failure and graph
decision under `runs/graph_evolve_state/`.

## First measured run

The first local run used `gpt-6-luna` through an OpenAI-compatible endpoint and
completed five serial iterations on the RTX 4060 Laptop environment.

| version | multi-class log loss | status |
| --- | ---: | --- |
| direct TF-IDF + logistic regression | 0.4395 | external baseline |
| fixed word + character TF-IDF, no agent | 0.3720 | representation control |
| GraphEvolve word + character TF-IDF | 0.3720 | adopted |
| GraphEvolve word-only comparator | 0.5661 | deferred |

The fixed representation control shows that the current 0.3720 score is not
evidence that the harness itself beats a fixed word+character predictor; it is
evidence that the run discovered and validated that representation. A stronger
harness comparison needs matched no-agent and multi-seed runs.

The project keeps technical repair steps separate from research rounds. A
repair step can continue, switch implementation, or abandon the candidate;
`100` is only a safety cap and is not a research-round budget. These numbers are
a single-task local measurement, not an official MLE-bench Lite aggregate score.
