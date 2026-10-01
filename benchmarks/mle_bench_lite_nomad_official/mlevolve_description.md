# MLE-bench Lite: nomad2018-predict-transparent-conductors (CSV-only)

Build a complete multi-output regression solution from `input/train.csv`. The
public workspace also contains `input/test.csv` and
`input/sample_submission.csv`. This lightweight adapter provides the official
MLE-bench 90/10 CSV split but omits crystal geometry files. Use only the public
training labels and features; do not use private labels or external data.

Predict `formation_energy_ev_natom` and `bandgap_energy_ev`. The primary metric
is the mean of the two column-wise root mean squared logarithmic errors
(RMSLE), so lower is better. Use a reproducible local validation split and
print its mean column-wise RMSLE as the final line in this format:

`Final Validation Score: <number>`

Write `submission_<node_id>.csv` with columns
`id,formation_energy_ev_natom,bandgap_energy_ev` in that order and one row per
test row. Preserve test `id` order and supply finite, non-negative predictions.
Keep each candidate runnable within the configured execution timeout.
