# OpenML credit-g smoke benchmark

This directory downloads the public OpenML credit-g data, creates a predictor-free task workspace, and supplies a protected evaluator. The task workspace contains `train.csv`, `test.csv`, `test_labels.csv`, and `evaluator.py`; it intentionally contains no `solution.py`.

```powershell
conda run -n nlphw python prepare.py --output task
```

`direct_logreg_baseline.py` is outside the task workspace. It is only an external baseline used to verify that MethodTrail's runner and evaluator read the data correctly. It is never supplied to the Coding Agent.

Run it through the same verifier and executor used by MethodTrail:

```powershell
conda run -n nlphw python run_direct_baseline.py --task task
```

For a real Agent run, configure `DASHSCOPE_API_KEY` in the environment and run:

```powershell
conda run -n nlphw python run_methodtrail.py --task task --model qwen-plus
```
