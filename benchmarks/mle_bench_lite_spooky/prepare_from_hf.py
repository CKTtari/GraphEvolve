"""Prepare the MLE-bench spooky-author task from a public copy of train.csv.

The official MLE-bench preparer downloads the same Kaggle train archive.  This
adapter uses the public Hugging Face mirror when Kaggle authentication is not
available, then applies the official 90/10 split (random_state=0).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_URL = (
    "https://huggingface.co/datasets/hkadxqq/spooky-author-identification/"
    "resolve/main/train.csv"
)
CLASSES = ["EAP", "HPL", "MWS"]


def download_train(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        return destination
    with urllib.request.urlopen(TRAIN_URL, timeout=60) as response:
        destination.write_bytes(response.read())
    return destination


def prepare(root: Path) -> None:
    root = root.resolve()
    public = root / "task"
    private = root / "private_evaluator"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    raw_csv = download_train(root / "raw" / "train.csv")

    frame = pd.read_csv(raw_csv)
    required = {"id", "text", "author"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    # The mirror stores the same Kaggle labels as integer codes.  The official
    # competition uses the corresponding author initials in its CSV format.
    values = set(frame["author"].dropna().unique().tolist())
    if values == {0, 1, 2}:
        frame["author"] = frame["author"].map({0: "EAP", 1: "HPL", 2: "MWS"})
    elif values != set(CLASSES):
        raise ValueError(f"unexpected classes: {sorted(values)}")

    train, held_out = train_test_split(
        frame, test_size=0.1, random_state=0
    )
    train.to_csv(public / "train.csv", index=False, lineterminator="\n")
    held_out.drop(columns=["author"]).to_csv(
        public / "test.csv", index=False, lineterminator="\n"
    )

    # MLE-bench's private answer file is a one-hot submission with the same
    # columns as sample_submission.csv.
    answer = held_out[["id", "author"]].copy()
    for label in CLASSES:
        answer[label] = (answer["author"] == label).astype(int)
    answer.drop(columns=["author"]).to_csv(
        private / "test_labels.csv", index=False, lineterminator="\n"
    )

    sample = held_out[["id"]].copy()
    sample["EAP"] = 0.403493538995863
    sample["HPL"] = 0.287808366106543
    sample["MWS"] = 0.308698094897594
    sample.to_csv(public / "sample_submission.csv", index=False, lineterminator="\n")

    print(f"prepared {len(train)} train rows and {len(held_out)} held-out rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".")
    prepare(Path(parser.parse_args().output))
