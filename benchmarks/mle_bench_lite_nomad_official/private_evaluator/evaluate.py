"""Independent evaluator for the prepared NOMAD 2018 task."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def main() -> None:
    root = Path.cwd()
    predictions = pd.read_csv(root / "predictions.csv")
    labels = pd.read_csv(root / "test.csv")
    expected = ["id", *TARGETS]
    if list(predictions.columns) != expected:
        raise ValueError(f"prediction columns must be {expected}")
    if len(predictions) != len(labels) or not predictions["id"].equals(labels["id"]):
        raise ValueError("prediction ids or row count do not match private test order")
    values = predictions[TARGETS].to_numpy(dtype=float)
    truth = labels[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("predictions must be finite and non-negative")
    score = float(
        np.mean(
            [root_mean_squared_log_error(truth[:, index], values[:, index]) for index in range(2)]
        )
    )
    output = Path(os.environ.get("METHODTRAIL_METRIC_FILE", "metrics.json"))
    output.write_text(
        json.dumps({"mean_columnwise_rmsle": score}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"mean_columnwise_rmsle": score}))


if __name__ == "__main__":
    main()
