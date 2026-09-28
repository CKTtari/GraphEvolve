# GraphEvolve

GraphEvolve is an autonomous experiment system for tasks that need more than one coding attempt. An LLM chooses an experiment, edits the implementation, runs it, reads measured feedback, and decides what to try next.

It is aimed at problems with a runnable evaluator but no fixed solution: model selection, feature work, training changes, ranking logic, and code repair can all become serial experiments with preserved code and results.

## Research loop

Each research round follows one main path:

    Task -> Question -> Method graph -> Code -> Execute -> Record -> Assess

The controller keeps research decisions separate from technical execution.

- Question mode `new_question` turns the current observation into a testable question.
- Question mode `conflict_refinement` turns valid but unclear results into an ablation, comparison, or segmented check.
- Method Graph expands the project-wide method pool, connects new nodes to related history, and selects one path.
- Assessment-Memory records valid evidence and updates node status and path priority.
- Repair handles technical failures inside the same research round and never changes the research question by itself.

The LLM proposes experiments and edits. GraphEvolve ranks candidate paths, applies permitted edits, runs commands, reads evaluator output, and manages versions. The evaluator supplies the metric used to keep or reject a candidate.

### Graph-organized experiment memory

Each experiment declares a compact method description: method family, changed components, changed factors, and target scope. Before one candidate is executed, every proposed candidate is attached as a branch from the active version. The system then calculates path priority and asks the LLM to select from those visible branches.

After execution, the branch is linked to its outcome. Measured facts, reusable conclusions, applicable conditions, and failures are retained separately. Later iterations retrieve a directed evidence pack rather than a flat recent log:

- **path history** explains how the current method was reached;
- **matched alternatives** expose sibling branches that isolate a related change;
- **candidate frontier** preserves unselected or deferred branches that may become useful after new evidence arrives.

The graph is stored as readable JSON and the concise experiment memory as JSONL. SQLite indexes typed artifacts and run facts.

Candidate estimates begin with the LLM's expected gain, information value, runtime, and failure risk. Once related paths have measured outcomes, GraphEvolve blends those observations back into the estimates before ranking the next branch. Evaluators may also declare auxiliary metrics and minimum metric constraints; a candidate that misses a declared constraint cannot be adopted on the primary score alone.

### Dynamic method pool

The method pool starts empty for a new project. The task contract provides data,
interfaces, metric, permitted dependencies, and resource limits; it does not
provide a predictor or a fixed candidate list. The Method Graph Agent may add
any number of distinct nodes and may stop discovery when the current question
has enough relevant alternatives.

Each node stores its implementation description, changed factors, parent
relations, measured results, conditions, cost, and failure history. Duplicate
method descriptions are attached to the existing node instead of creating a
new branch. The path priority uses the declared experiment value and measured
runtime/risk; there is no hidden breadth/depth phase switch. One selected
experiment still runs at a time, and unselected nodes remain in the project
method pool.

## Editing and safety

MethodTrail uses structured edits instead of giving the LLM unrestricted shell access.

- A new file can be created from complete content.
- Existing files are changed through an exact old-text to new-text replacement.
- The old fragment must occur exactly once. Missing or ambiguous matches are rejected before files change.
- Large existing files cannot be rewritten in one replacement.
- Editable paths, protected paths, and declared data paths restrict where edits may go.
- Every candidate uses its own Git worktree. Failed candidates never overwrite the accepted version.

Private evaluation files are kept outside the generated-code workspace. Candidate code finishes first; then required outputs are copied into a temporary evaluator directory. For hostile code, run MethodTrail inside a container or VM with restricted filesystem and network access.

## Install

Python 3.11 or newer is required.

    git clone <your-repository-url>
    cd methodtrail
    conda create -n methodtrail python=3.11 -y
    conda run -n methodtrail python -m pip install -e ".[dev]"

Install task-specific packages in the same environment when needed.

## Configure an LLM API

MethodTrail uses the OpenAI Python SDK with an OpenAI-compatible Chat Completions endpoint. Keep the key in an environment variable. MethodTrail does not write it into project metadata, candidate worktrees, or trajectories.

    $env:LLM_API_KEY = "your-key"

Pass the variable name, model, and endpoint when starting a run:

    conda run -n methodtrail python -m methodtrail run --contract .\task_contract.json --project-root .\methodtrail_state --project-id demo-project --model your-model-name --api-key-env LLM_API_KEY --base-url https://your-openai-compatible-endpoint/v1 --remaining-seconds 3600 --max-iterations 12

The CLI default is `DASHSCOPE_API_KEY` for backward compatibility; the
benchmark and examples can use `LLM_API_KEY` explicitly. The endpoint and
model are launch-time settings.

## Create a task

A task starts with a JSON contract. It declares public input files, editable source paths, expected outputs, execution commands, and the metric. The task adapter does not need to supply a predictor or a hand-written model search space.

    {
      "task_id": "toy-classification",
      "description": "Train from train.csv and write predictions.csv for test.csv.",
      "workspace_template": "./workspace",
      "allowed_data_paths": ["train.csv", "test.csv"],
      "editable_paths": ["solution.py", "src/*.py"],
      "solution_entrypoint": "solution.py",
      "run_command": ["python", "solution.py"],
      "private_evaluator_dir": "./private_evaluator",
      "private_evaluation_command": ["python", "evaluate.py"],
      "metric_file": "metrics.json",
      "required_outputs": ["predictions.csv"],
      "metric_name": "accuracy",
      "maximize_metric": true,
      "timeout_seconds": 900,
      "max_repair_steps": 100,
      "minimum_iterations": 3,
      "allowed_dependencies": ["numpy", "pandas", "scikit-learn"],
      "protected_paths": ["task_contract.json", "metrics.json"]
    }

The workspace template contains public inputs only. A private evaluator can contain hidden labels and evaluate.py; it is excluded from LLM context and candidate worktrees. It writes a metric file such as:

    {"accuracy": 0.84}

For a public task, use evaluation_command instead of the private-evaluator fields. Self-reported metrics are intended only for quick smoke tasks.

## Run and manage projects

Use a project ID to isolate code history, memory, candidates, and the accepted-version pointer. Use a session ID to pause and later continue one research thread.

    # Start a project
    conda run -n methodtrail python -m methodtrail run --contract .\task_contract.json --project-root .\methodtrail_state --project-id feature-study --model your-model-name --api-key-env LLM_API_KEY --base-url https://your-openai-compatible-endpoint/v1 --remaining-seconds 3600 --max-iterations 12

    # Pause a session
    conda run -n methodtrail python -m methodtrail pause --project-root .\methodtrail_state --project-id feature-study --session-id <session-id>

    # Resume it
    conda run -n methodtrail python -m methodtrail run --contract .\task_contract.json --project-root .\methodtrail_state --project-id feature-study --session-id <session-id> --model your-model-name --api-key-env LLM_API_KEY --base-url https://your-openai-compatible-endpoint/v1 --remaining-seconds 1800 --max-iterations 6

    # Roll back to an adopted candidate
    conda run -n methodtrail python -m methodtrail rollback --project-root .\methodtrail_state --project-id feature-study --variant-id <adopted-variant-id>

When a fixed iteration budget ends without an explicit adoption, select the best valid measured candidate from that session:

    conda run -n methodtrail python -m methodtrail finalize --contract .\task_contract.json --project-root .\methodtrail_state --project-id feature-study --session-id <session-id>

## Project layout

    .methodtrail/projects/<project-id>/
    ├── repository/                  accepted source revisions
    ├── sessions/<session-id>/        session state, handoff, JSONL trajectory
    ├── candidates/<variant-id>.json  candidate status, parent, metric, branch
    ├── wt/<short-id>/                isolated candidate worktrees
    └── memory/                       experiment cards, path graph, portfolio

Git revisions, candidate metadata, session trajectories, and experiment-memory files are directly inspectable. The SQLite artifact store indexes decisions and run facts.

## Useful commands

    conda run -n methodtrail python -m methodtrail repo-map .\workspace
    conda run -n methodtrail pytest

## Status

MethodTrail is experimental. The core loop is intentionally compact while task adapters, executor isolation, and research roles continue to evolve.
