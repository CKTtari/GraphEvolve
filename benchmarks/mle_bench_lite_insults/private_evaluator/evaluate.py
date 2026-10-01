"""Independent evaluator for the prepared MLE-bench insults task."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    root = Path.cwd()
    predictions = pd.read_csv(root / "predictions.csv")
    labels = pd.read_csv(root / "test.csv")
    required = ["Insult", "Date", "Comment"]
    if list(predictions.columns) != required:
        raise ValueError(f"prediction columns must be {required}")
    if len(predictions) != len(labels):
        raise ValueError(f"expected {len(labels)} rows, got {len(predictions)}")
    if not predictions["Comment"].equals(labels["Comment"]):
        raise ValueError("prediction comments do not match private test order")
    probabilities = predictions["Insult"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Insult predictions must be finite probabilities in [0, 1]")
    score = float(roc_auc_score(labels["Insult"].astype(int), probabilities))
    output = Path(os.environ.get("METHODTRAIL_METRIC_FILE", "metrics.json"))
    output.write_text(json.dumps({"roc_auc": score}, indent=2), encoding="utf-8")
    print(json.dumps({"roc_auc": score}))


if __name__ == "__main__":
    main()
