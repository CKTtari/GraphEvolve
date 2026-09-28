"""Run GraphEvolve on the prepared single MLE-bench Lite task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from methodtrail.llm import OpenAICompatibleLLM
from methodtrail.orchestrator import MethodTrail
from methodtrail.schemas import TaskContract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--base-url", default="https://yuzapi.fun/v1")
    parser.add_argument("--budget-seconds", type=int, default=2400)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument(
        "--search-policy", choices=["breadth", "balanced", "depth"], default=None,
        help="Deprecated compatibility option; path choice is now evidence-driven.",
    )
    parser.add_argument("--max-repair-steps", type=int, default=None)
    parser.add_argument("--max-repair-attempts", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--llm-log", default=None)
    parser.add_argument("--project-id", default="mle-lite-spooky-graph-evolve")
    parser.add_argument("--state-dir", default="runs/graph_evolve_state")
    args = parser.parse_args()
    root = Path(__file__).parent.resolve()
    payload = json.loads((root / "task_contract.json").read_text(encoding="utf-8"))
    payload["workspace_template"] = str((root / "task").resolve())
    payload["private_evaluator_dir"] = str((root / "private_evaluator").resolve())
    if args.max_repair_steps is not None:
        payload["max_repair_steps"] = args.max_repair_steps
    elif args.max_repair_attempts is not None:
        payload["max_repair_steps"] = args.max_repair_attempts
    contract = TaskContract.model_validate(payload)
    llm_log = args.llm_log or str(
        (Path(args.state_dir) / ".methodtrail" / "llm_calls.jsonl").resolve()
    )
    llm = OpenAICompatibleLLM(
        args.model, args.api_key_env, args.base_url, log_path=llm_log
    )
    trail = MethodTrail(
        args.state_dir,
        llm,
        project_id=args.project_id,
    )
    results = trail.run_research(contract, args.budget_seconds, args.max_iterations)
    print(
        json.dumps(
            [
                {
                    "session_id": trail.session_id,
                    "iteration": result.iteration,
                    "decision": result.assessment.decision
                    if result.assessment
                    else "recovery",
                    "metric": result.run.metric if result.run else None,
                    "metrics": result.run.metrics if result.run else {},
                    "workspace": str(result.workspace),
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
