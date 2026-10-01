# MLE-bench Lite: spooky-author-identification

Build a complete machine-learning solution for the provided text-classification
competition. The working directory contains `input/train.csv`,
`input/test.csv`, and `input/sample_submission.csv`.

The target is the three-class author label encoded by the training columns
`EAP`, `HPL`, and `MWS`. Train only from `input/train.csv`; do not use private
labels or any external data. Create a reproducible validation split from the
training data and report the validation **multiclass log loss**. Lower is
better.

The final program must write `submission_<node_id>.csv` (the runtime may rename
the generic submission path) with columns `id,EAP,HPL,MWS`, one probability row
per row of `input/test.csv`, and rows summing to one. It must also print the
validation log loss in the final line as:

`Final Validation Score: <number>`

Use real model inference for every test row. Keep each candidate runnable on a
single local GPU or CPU within the configured execution timeout.
