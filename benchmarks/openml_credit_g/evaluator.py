"""Protected evaluator for the OpenML credit-g smoke benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


def main() -> None:
    root = Path.cwd()
    predictions = pd.read_csv(root / "predictions.csv")
    labels = pd.read_csv(root / "test_labels.csv")
    if list(predictions.columns) != ["prediction"]:
        raise ValueError(
            "predictions.csv must contain exactly one column named prediction"
        )
    if len(predictions) != len(labels):
        raise ValueError(
            f"expected {len(labels)} predictions, received {len(predictions)}"
        )
    values = predictions["prediction"]
    if not values.isin([0, 1]).all():
        raise ValueError("prediction values must be binary 0 or 1")
    score = accuracy_score(labels["target"], values)
    (root / "metrics.json").write_text(
        json.dumps({"accuracy": score}), encoding="utf-8"
    )
    print(json.dumps({"accuracy": score}))


if __name__ == "__main__":
    main()
