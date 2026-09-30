"""Command-line entry points for inspection and one serial MethodTrail iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import OpenAICompatibleLLM
from .orchestrator import MethodTrail
from .project import ProjectManager
from .repository import RepoMap
from .schemas import TaskContract


def _load_contract(path: str) -> TaskContract:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    template = Path(payload["workspace_template"])
    if not template.is_absolute():
        payload["workspace_template"] = str((contract_path.parent / template).resolve())
    private_evaluator = payload.get("private_evaluator_dir")
    if private_evaluator:
        private_path = Path(private_evaluator)
        if not private_path.is_absolute():
            payload["private_evaluator_dir"] = str(
                (contract_path.parent / private_path).resolve()
            )
    return TaskContract.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="methodtrail")
    commands = parser.add_subparsers(dest="command", required=True)

    show_map = commands.add_parser(
        "repo-map", help="print a compact Python repository map"
    )
    show_map.add_argument("workspace")

    run = commands.add_parser("run", help="run one autonomous experiment iteration")
    run.add_argument("--contract", required=True, help="task contract JSON")
    run.add_argument(
        "--project-root", required=True, help="directory for MethodTrail state"
    )
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    run.add_argument("--base-url", default=None)
    run.add_argument(
        "--llm-log",
        default=None,
        help="optional JSONL file containing prompts, typed responses and timings",
    )
    run.add_argument("--remaining-seconds", type=int, required=True)
    run.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="maximum serial research turns; use 1 for a single turn",
    )
    run.add_argument("--parent-variant-id", default=None)
    run.add_argument("--trigger", default=None)
    run.add_argument(
        "--session-id",
        default=None,
        help="resume a session; omit to start a new isolated session",
    )
    run.add_argument(
        "--project-id",
        default=None,
        help="project name; omit to use the task_id as the project name",
    )

    pause = commands.add_parser("pause", help="pause a session and keep its handoff")
    pause.add_argument("--project-root", required=True)
    pause.add_argument("--project-id", required=True)
    pause.add_argument("--session-id", required=True)

    rollback = commands.add_parser(
        "rollback", help="point the project incumbent at an adopted variant"
    )
    rollback.add_argument("--project-root", required=True)
    rollback.add_argument("--project-id", required=True)
    rollback.add_argument("--variant-id", required=True)

    finalize = commands.add_parser(
        "finalize", help="persist the best valid measured candidate for a completed session"
    )
    finalize.add_argument("--contract", required=True)
    finalize.add_argument("--project-root", required=True)
    finalize.add_argument("--project-id", required=True)
    finalize.add_argument("--session-id", required=True)

    export = commands.add_parser(
        "export-trajectory", help="export artifacts and graph to JSON"
    )
    export.add_argument("--project-root", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--model", required=True, help="used only to construct the client"
    )
    export.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    export.add_argument("--base-url", default=None)

    dashboard = commands.add_parser(
        "dashboard", help="write an offline HTML view of method and memory graphs"
    )
    dashboard.add_argument("--project-root", required=True)
    dashboard.add_argument("--project-id", required=True)
    dashboard.add_argument("--session-id", required=True)
    dashboard.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "repo-map":
        print(RepoMap(args.workspace).summary())
        return

    if args.command == "pause":
        manager = ProjectManager(Path(args.project_root).resolve() / ".methodtrail")
        project = manager.load_project(args.project_id)
        session = manager.load_session(project, args.session_id)
        manager.update_session(project, session, status="paused")
        print(json.dumps({"session_id": session.session_id, "status": "paused"}))
        return

    if args.command == "rollback":
        manager = ProjectManager(Path(args.project_root).resolve() / ".methodtrail")
        project = manager.load_project(args.project_id)
        project = manager.set_incumbent(project, args.variant_id)
        print(
            json.dumps(
                {
                    "task_id": project.task_id,
                    "incumbent_variant_id": project.incumbent_variant_id,
                    "incumbent_commit": project.incumbent_commit,
                }
            )
        )
        return

    if args.command == "finalize":
        contract = _load_contract(args.contract)
        trail = MethodTrail(
            args.project_root,
            llm=None,  # type: ignore[arg-type]  # This command performs no LLM call.
            session_id=args.session_id,
            project_id=args.project_id,
        )
        trail._ensure_project_session(contract)
        trail._finalize_best_valid_candidate(contract)
        assert trail.project is not None
        print(
            json.dumps(
                {
                    "incumbent_variant_id": trail.project.incumbent_variant_id,
                    "incumbent_commit": trail.project.incumbent_commit,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "dashboard":
        trail = MethodTrail(
            args.project_root,
            llm=None,  # type: ignore[arg-type]  # dashboard export performs no LLM call.
            project_id=args.project_id,
            session_id=args.session_id,
        )
        print(
            trail.export_dashboard(
                args.output,
                project_id=args.project_id,
                session_id=args.session_id,
            )
        )
        return

    llm = OpenAICompatibleLLM(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        log_path=getattr(args, "llm_log", None),
    )
    trail = MethodTrail(
        args.project_root,
        llm,
        session_id=getattr(args, "session_id", None),
        project_id=getattr(args, "project_id", None),
    )
    if args.command == "export-trajectory":
        print(trail.export_trajectory(args.output))
        return

    contract = _load_contract(args.contract)
    if args.max_iterations == 1:
        results = [
            trail.run_iteration(
                contract,
                remaining_seconds=args.remaining_seconds,
                parent_variant_id=args.parent_variant_id,
                trigger=args.trigger,
            )
        ]
    else:
        results = trail.run_research(
            contract,
            total_seconds=args.remaining_seconds,
            max_iterations=args.max_iterations,
            parent_variant_id=args.parent_variant_id,
        )
    result = results[-1]
    print(
        json.dumps(
            {
                "iteration": result.iteration,
                "completed_iterations": sum(
                    result.completed_research for result in results
                ),
                "technical_attempts": sum(
                    not result.completed_research for result in results
                ),
                "workspace": str(result.workspace),
                "decision": result.assessment.decision
                if result.assessment
                else "recovery",
                "next_question": result.next_hypothesis,
                "artifacts": result.artifact_ids,
                "session_id": trail.session_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
