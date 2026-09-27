"""Download OpenML credit-g and create a predictor-free MethodTrail task workspace."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

OPENML_URL = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="task")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "credit-g.csv"
    urllib.request.urlretrieve(OPENML_URL, raw_path)
    frame = pd.read_csv(raw_path)
    target = frame.pop("class").map({"good": 1, "bad": 0})
    train, test, train_target, test_target = train_test_split(
        frame,
        target,
        test_size=0.2,
        random_state=2026,
        stratify=target,
    )
    train.assign(target=train_target).to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    pd.DataFrame({"target": test_target.to_numpy()}).to_csv(
        output / "test_labels.csv", index=False
    )
    raw_path.unlink(missing_ok=True)
    template_dir = Path(__file__).parent
    shutil.copy2(template_dir / "evaluator.py", output / "evaluator.py")
    print(f"prepared {output}: {len(train)} train rows, {len(test)} test rows")


if __name__ == "__main__":
    main()
