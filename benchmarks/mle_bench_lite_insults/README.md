# MLE-bench Lite — detecting-insults-in-social-commentary

This local adapter uses the prepared MLE-bench public train/test files and
keeps the private test labels in `private_evaluator/`. The metric is binary
ROC-AUC. `baseline_solution.py` is a fixed word+character TF-IDF logistic
regression comparator; GraphEvolve receives only the task contract and public
files.
