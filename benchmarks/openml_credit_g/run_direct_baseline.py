"""Run the external direct LogisticRegression baseline through MethodTrail's executor."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from methodtrail.execution import Executor, Verifier
from methodtrail.schemas import TaskContract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    source = Path(args.task).resolve()
    if not (source / "train.csv").exists():
        raise FileNotFoundError("task data is missing; run prepare.py first")
    target = (
        Path(args.runs_dir).resolve()
        / f"direct_logreg_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    )
    shutil.copytree(source, target)
    shutil.copy2(
        Path(__file__).parent / "direct_logreg_baseline.py", target / "solution.py"
    )
    payload = json.loads(
        (Path(__file__).parent / "task_contract.json").read_text(encoding="utf-8")
    )
    payload["workspace_template"] = str(target)
    contract = TaskContract.model_validate(payload)
    verification = Verifier().verify(target, contract)
    if not verification.passed:
        raise RuntimeError(json.dumps(verification.model_dump(), ensure_ascii=False))
    result = Executor().run(target, contract)
    print(
        json.dumps(
            {"workspace": str(target), **result.model_dump()},
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.return_code != 0:
        raise SystemExit(result.return_code)


if __name__ == "__main__":
    main()
