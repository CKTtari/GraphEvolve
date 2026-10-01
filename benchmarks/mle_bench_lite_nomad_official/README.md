# MLE-bench Lite — NOMAD 2018 tabular adapter

`prepare_from_public.py` reproduces the official MLE-bench 90/10 CSV split
(`random_state=0`) from the original 2,400-row public training CSV. The four
prepared CSVs match the MD5 checksums in MLE-bench's NOMAD task definition.
It omits the crystal geometry files, so this is a lightweight CSV-only local
evaluation of the task. The independent evaluator computes mean-column-wise
RMSLE.

The fixed no-agent comparator is `baseline_solution.py`: a log-target
random-forest multi-output model. GraphEvolve receives only the task contract
and the public CSV files. Earlier experiments under `mle_bench_lite_nomad/`
used the original Kaggle 2,400/600 split and are not mixed with this project.
