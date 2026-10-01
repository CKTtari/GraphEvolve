# MLE-bench Lite: detecting-insults-in-social-commentary

Build a complete machine-learning solution from `input/train.csv`. The public
workspace also contains `input/test.csv` and `input/sample_submission.csv`.
Predict the binary `Insult` label for each row of the test file. Use only the
public training labels and features; do not use private labels or external data.

The primary metric is ROC-AUC, so higher is better. Use a reproducible local
validation split and print its ROC-AUC as the final line in this format:

`Final Validation Score: <number>`

Write `submission_<node_id>.csv` with columns `Insult,Date,Comment` in that
order and one row per test row. Preserve each test row's `Date` and `Comment`
and supply a finite probability in `[0, 1]` for `Insult`. Keep each candidate
runnable within the configured execution timeout.
