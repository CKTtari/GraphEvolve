"""Independent log-loss evaluator for the prepared MLE-bench task."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]


def main() -> None:
    root = Path.cwd()
    predictions = pd.read_csv(root / "predictions.csv")
    labels = pd.read_csv(root / "test_labels.csv")
    if list(predictions.columns) != ["id", *CLASSES]:
        raise ValueError(f"prediction columns must be id,{','.join(CLASSES)}")
    if not predictions["id"].equals(labels["id"]):
        raise ValueError("prediction ids do not match the private test order")
    values = predictions[CLASSES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("predictions contain invalid probabilities")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("each prediction row must sum to 1")
    true = labels[CLASSES].to_numpy(dtype=int).argmax(axis=1)
    score = float(log_loss(true, np.clip(values, 1e-15, 1.0), labels=list(range(3))))
    output = Path(os.environ.get("METHODTRAIL_METRIC_FILE", "metrics.json"))
    output.write_text(
        json.dumps({"multi_class_log_loss": score}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"multi_class_log_loss": score}))


if __name__ == "__main__":
    main()
