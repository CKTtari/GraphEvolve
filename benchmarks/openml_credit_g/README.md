# OpenML credit-g smoke benchmark

This optional smoke task downloads public OpenML credit-g data and creates a
workspace with `train.csv`, `test.csv`, `test_labels.csv` and a protected
`evaluator.py`. It contains no `solution.py`; GraphEvolve writes the predictor.
Raw and prepared data are excluded from Git.

```powershell
conda run -n nlphw python prepare.py --output task
```

Configure your API key in `LLM_API_KEY` and run:

```powershell
conda run -n nlphw python run_graph_evolve.py --task task --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1
```
