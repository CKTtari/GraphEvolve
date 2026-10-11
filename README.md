# GraphEvolve

**v0.18 · Graph-guided autonomous experimentation**

GraphEvolve lets an LLM propose a research question, choose an experiment,
write or patch its code, run an independent evaluator, and decide what to try
next. It preserves measured results and the best valid code version.

A task supplies data interfaces, permitted edits and evaluation rules. It does
not supply a predictor or a fixed list of algorithms. The same research loop
can work on text classification, tabular regression and other tasks with a
runnable evaluator.

## Research loop

    Task → Question → Method graph → Code → Execute → Record → Assess

- **Question Agent** turns the task and measured history into a testable question.
- **Method Graph Agent** proposes or revisits candidates and chooses one experiment.
- **Coding Agent** creates the first solution, then applies local edits to the selected parent.
- **Implementation Review** checks the actual code changes against the selected experiment.
- **Assessment-Memory Agent** records results and decides whether to adopt, verify or defer.
- **Repair Agent** fixes execution and implementation failures inside the same research round.

One selected experiment runs at a time. Repair and candidate replanning do not
consume research rounds. The run ends at its total time budget, round limit,
an explicit stop decision or the absence of an executable candidate. There is
no automatic stop after a fixed number of non-improving rounds.

## Two directed graphs

The **method graph** contains pending candidates and measured outcomes. Edges
record code lineage, candidate execution, explicit evidence sources and sparse
related-method links. Stable method signatures help avoid duplicate trials.

The **experience graph** connects experiment cards through parent versions,
explicit evidence references and relevant earlier conclusions. Retrieval follows
directed ancestors and nearby branches and reports the traversed edges to the
agents. Context is bounded instead of sending the entire experiment database.

A candidate can choose an earlier measured version as its code parent and
branch from it. Failed paths remain available as evidence. Graph priorities
combine LLM estimates with measured gain, cost and risk; the LLM makes the
final choice from a compact set of alternatives.

## Code and evaluation

Existing code is changed through exact text or symbol edits. Syntax, imports,
task interfaces and implementation review are checked before full execution.
Review feedback can request a code repair or a revision of an infeasible
candidate declaration. Technical failures are kept separately from measured
research conclusions.

Candidates use isolated Git worktrees. Private evaluation inputs stay outside
the generated-code workspace; outputs are copied to a temporary evaluation
directory. The local process runner is not a security sandbox: use a restricted
container or VM when running untrusted generated code.

Running programs are checked periodically rather than killed after an implicit
five-minute timeout. An execution monitor can request termination; the shared
research deadline and an explicitly configured hard timeout remain enforceable.

## Install

Python 3.11+ and Git are required.

```powershell
git clone https://github.com/CKTtari/GraphEvolve.git
cd GraphEvolve
python -m pip install -e ".[dev,benchmark]"
python -m graphevolve --version
```

An existing Conda environment can be used:

```powershell
conda run -n nlphw python -m pip install -e ".[dev,benchmark]"
conda run -n nlphw pytest
```

## Run a task

Set your API key in an environment variable. Model and endpoint are launch-time
settings for an OpenAI-compatible API; credentials are not saved in project
metadata or trajectories.

```powershell
$env:LLM_API_KEY = "your-key"
python -m graphevolve run --contract .\task_contract.json --project-root .\graphevolve_state --project-id example --model your-model --api-key-env LLM_API_KEY --base-url https://your-endpoint/v1 --remaining-seconds 18000 --max-iterations 20
```

The task contract declares `workspace_template`, allowed data and editable paths,
execution commands, required outputs, metric name and direction. For an
independent evaluator, also declare `private_evaluator_dir` and
`private_evaluation_command`. See the [benchmark adapters](benchmarks/README.md)
for complete contracts and preparation instructions.

Resume with the same `--project-id` and `--session-id`. To expose a completed
project's cards as read-only reference, add `--prior-project-id`; its code and
graph nodes are not copied into the new project.

```powershell
python -m graphevolve dashboard --project-root .\graphevolve_state --project-id example --session-id SESSION_ID --output .\dashboard.html
python -m graphevolve pause --project-root .\graphevolve_state --project-id example --session-id SESSION_ID
python -m graphevolve rollback --project-root .\graphevolve_state --project-id example --variant-id VARIANT_ID
```

## Dashboard

The self-contained HTML dashboard shows both directed graphs, node and edge
details, candidate comparisons and the score for each research round. Nodes
can be dragged; the canvas supports pan and zoom. Colors are derived from
method-metadata similarity and are recomputed for the visible graph.

Each session exports its dashboard immediately and updates it during research.
The three MLE-bench runners start a localhost HTTP server and print its URL.
Pass `--no-dashboard-server` to disable it or `--stop-dashboard-server` to stop
it when research ends.

## Layout and versions

```text
src/graphevolve/       installable Python package
tests/                core regression tests
benchmarks/           task contracts, preparation, runners and independent evaluators
  RESULTS.md          current lightweight local evaluation
.graphevolve/         generated state beneath each chosen project root (not committed)
no-use/               local archive of superseded files (not committed)
```

`src/` separates the installable package from development files. One package
under it is intentional; modules group the agents, graphs, execution and storage.

The release number is defined once in `src/graphevolve/_version.py`. Package
metadata uses `0.18`; the CLI, dashboard and documentation display `v0.18`.

Older local `.methodtrail` stores are reopened in place when their project ID
is requested. New projects use `.graphevolve`. Saved experiment IDs and paths
are retained so previous results remain inspectable. Use `graphevolve` for
Python imports and CLI commands; the earlier package name has been retired.

## Results and design

The [current results](benchmarks/RESULTS.md) cover three selected MLE-bench Lite
tasks with the same LLM backbone. They are a lightweight local comparison,
with documented differences in run length and prior evidence, rather than
an official leaderboard result.

See [DESIGN.md](DESIGN.md) for the implemented architecture and
[REFERENCES.md](REFERENCES.md) for source references.
