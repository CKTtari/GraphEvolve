"""Run GraphEvolve itself on credit-g after an API key has been configured in the environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphevolve import __version__
from graphevolve.llm import OpenAICompatibleLLM
from graphevolve.orchestrator import GraphEvolve
from graphevolve.schemas import TaskContract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"GraphEvolve v{__version__}")
    parser.add_argument("--task", default="task")
    parser.add_argument("--state-dir", default="graphevolve_state")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--project-id", default=None)
    args = parser.parse_args()
    root = Path(__file__).parent
    payload = json.loads((root / "task_contract.json").read_text(encoding="utf-8"))
    payload["workspace_template"] = str(Path(args.task).resolve())
    contract = TaskContract.model_validate(payload)
    llm = OpenAICompatibleLLM(args.model, args.api_key_env, args.base_url)
    trail = GraphEvolve(
        args.state_dir,
        llm,
        session_id=args.session_id,
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
