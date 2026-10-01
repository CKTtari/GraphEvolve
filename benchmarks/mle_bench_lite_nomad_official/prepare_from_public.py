"""Reproduce MLE-bench NOMAD's CSV split from the public Kaggle train mirror.

The official preparer also copies crystal geometry files. This lightweight
adapter exposes the tabular CSV features only; its train/test/answer CSVs use
the same 90/10 split, random state and remapped IDs as MLE-bench.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_URL = (
    "https://raw.githubusercontent.com/csutton7/nomad_2018_kaggle_dataset/"
    "master/train.csv"
)
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def prepare(root: Path) -> None:
    root = root.resolve()
    raw = root / "raw" / "train.csv"
    public = root / "task"
    private = root / "private_evaluator"
    raw.parent.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(TRAIN_URL, raw)

    # MLE-bench's read_csv helper uses round_trip precision. Matching it also
    # preserves the original floating-point literals in the prepared CSVs.
    old_train = pd.read_csv(raw, float_precision="round_trip")
    if len(old_train) != 2400 or not {"id", *TARGETS}.issubset(old_train.columns):
        raise ValueError("unexpected NOMAD public training schema or row count")
    train, held_out = train_test_split(old_train, test_size=0.1, random_state=0)
    train = train.copy()
    held_out = held_out.copy()
    train["id"] = train["id"].map(
        {old_id: new_id for new_id, old_id in enumerate(train["id"], start=1)}
    )
    held_out["id"] = held_out["id"].map(
        {old_id: new_id for new_id, old_id in enumerate(held_out["id"], start=1)}
    )
    # Official checksums were generated with LF line endings. Specify them
    # explicitly so preparation remains byte-for-byte reproducible on Windows.
    train.to_csv(public / "train.csv", index=False, lineterminator="\n")
    held_out.drop(columns=TARGETS).to_csv(
        public / "test.csv", index=False, lineterminator="\n"
    )
    held_out.to_csv(private / "test.csv", index=False, lineterminator="\n")
    sample = pd.DataFrame(
        {
            "id": held_out["id"],
            "formation_energy_ev_natom": 0.1779,
            "bandgap_energy_ev": 1.8892,
        }
    )
    sample.to_csv(public / "sample_submission.csv", index=False, lineterminator="\n")
    print(f"prepared {len(train)} train rows and {len(held_out)} held-out rows")


if __name__ == "__main__":
    prepare(Path(__file__).parent)
