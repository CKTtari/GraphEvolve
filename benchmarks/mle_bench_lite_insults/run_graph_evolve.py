"""Run GraphEvolve on the prepared single MLE-bench Lite task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from methodtrail.llm import OpenAICompatibleLLM
from methodtrail.orchestrator import MethodTrail
from methodtrail.schemas import TaskContract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--base-url", default="https://yuzapi.fun/v1")
    parser.add_argument("--budget-seconds", type=int, default=18000)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--search-policy", choices=["breadth", "balanced", "depth"], default=None,
        help="Deprecated compatibility option; path choice is now evidence-driven.",
    )
    parser.add_argument("--max-repair-steps", type=int, default=None)
    parser.add_argument("--max-repair-attempts", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--llm-log", default=None)
    parser.add_argument("--project-id", default="mle-lite-insults-graph-evolve")
    parser.add_argument(
        "--session-id",
        default=None,
        help="resume an existing session instead of creating a new one",
    )
    parser.add_argument(
        "--prior-project-id",
        default=None,
        help="optional completed project whose cards are read as prior evidence",
    )
    parser.add_argument("--state-dir", default="runs/graph_evolve_state")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8767)
    parser.add_argument(
        "--no-dashboard-server",
        action="store_true",
        help="Do not start the local dashboard HTTP server.",
    )
    parser.add_argument(
        "--stop-dashboard-server",
        action="store_true",
        help="Stop the local dashboard server when the research run exits.",
    )
    args = parser.parse_args()
    root = Path(__file__).parent.resolve()
    dashboard_root = Path(args.state_dir).resolve()
    dashboard_root.mkdir(parents=True, exist_ok=True)
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
        session_id=args.session_id,
        project_id=args.project_id,
        prior_project_id=args.prior_project_id,
    )
    server = None
    if not args.no_dashboard_server:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(args.dashboard_port),
                "--bind",
                args.dashboard_host,
            ],
            cwd=str(dashboard_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        # Create the session before the long loop so the monitor URL is known
        # immediately and the first dashboard export has a stable location.
        trail._ensure_project_session(contract)
        dashboard_rel = (
            Path(".methodtrail")
            / "projects"
            / args.project_id
            / "sessions"
            / str(trail.session_id)
            / "dashboard.html"
        )
        print(
            f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}/{dashboard_rel.as_posix()}",
            flush=True,
        )
        results = trail.run_research(contract, args.budget_seconds, args.max_iterations)
    finally:
        if server is not None and args.stop_dashboard_server:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
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
                    "completed_research": result.completed_research,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
