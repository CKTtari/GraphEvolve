# GraphEvolve design · v0.18

GraphEvolve performs artifact-level recursive self-improvement: the LLM
iterates on research questions, implementations and experiment evidence. The
orchestration kernel and task permissions remain controlled by the project owner.

## Research and implementation

The serial controller runs Question → Method Graph → Coding → Execution →
Assessment-Memory. A round selects one executable candidate. Other candidates
stay in the project-wide method pool and can be revisited after new evidence.
The pool is not expanded on every round when existing alternatives suffice.

Candidates describe the question, method components, changed factors, code
parent, evidence parents and conditions that implementation must preserve.
The Method Graph Agent owns the relation: deepen, combine, ablate, explore or
recover. Consistency checks report disagreements instead of silently deciding
that a component removal must be an ablation.

Coding uses the selected parent's current source. Existing files receive exact
text or symbol replacements; ordinary child and recovery edits cannot rewrite
the whole entrypoint. Each patch passes deterministic workspace checks and an
independent review of the cumulative change from its code parent.

A code defect returns concrete review or execution feedback to repair. An
infeasible or contradictory candidate can return to method selection for
replanning. Repair and replanning stay within the same research round and
shared budget. An identical rejected candidate or unchanged patch cannot start
an identical review cycle without a changed implementation or declaration.

## Graphs and retrieval

The method graph is stored in `memory/experiment_graph.json`. Proposed methods
are linked before selection; executed candidates connect to measured outcomes.
The graph retains unselected branches and failed paths. Method signatures use
structured method information rather than the natural-language title alone.

The experience graph is stored in `memory/memory_graph.json`, derived from
append-only `memory/cards.jsonl`. A card records the research question, actual
changes, measured facts, conclusion, applicable conditions and next question.
Technical errors go to `memory/bugs.jsonl` instead of becoming predictive evidence.

Both graphs distinguish:

- **Lineage:** the version whose code was edited points to its child.
- **Evidence:** earlier measured versions explicitly cited by the research
  decision point to the new candidate or conclusion.
- **Related methods:** sparse links expose comparable families and shared
  factors without connecting every pair in a family.

Method retrieval includes the active path, related alternatives and pending
frontier. Memory retrieval starts from the selected code parent's card when
available, traces incoming ancestors and outgoing branches, and records the
edge trace. Lexical matching supplements this retrieval; it is not a substitute
for directed provenance.

Agents receive a bounded working context, including measured metrics, graph
relationships, short conclusions and current editable source. They can select
an earlier measured version as the parent of a new branch. Read-only prior
project cards can inform a fresh run, while its code lineage and graph IDs stay
project-local.

## Selection and stopping

The LLM proposes expected gain, information value, runtime and failure risk.
The graph calibrates these estimates with related measured outcomes. Ranking
offers gain, information, cost, risk and plateau signals; a compact candidate
view keeps promising improvements and useful alternatives visible to the LLM.
The selector makes the final choice and records its reasoning.

Plateau evidence can increase exploration preference without forcing a named
algorithm or a semantically orthogonal route. New evidence can also justify
returning to an earlier method. The best valid checkpoint is retained throughout.

Research ends at the configured round limit, total time budget, explicit Agent
stop decision or lack of an executable candidate. Non-improving or strictly
worsening rounds are not a separate fixed-count stop condition.

## Execution, persistence and monitoring

A task contract specifies data, edits, expected outputs, metric and execution
commands. Public task templates contain interfaces and data, not a predictor.
Candidates run in independent Git worktrees. The evaluator receives output
copies after candidate execution; its private inputs do not enter agent context.

Periodic progress checks inspect logs and outputs. The monitor may stop a
failed execution, while healthy long-running experiments can continue until
the shared deadline. A hard per-command cap is optional and explicit.

New state is stored under `.graphevolve/projects/<project-id>/`, with repository,
candidate, session and memory records. Resuming retains completed rounds,
technical/replanning state and the accepted version. Older `.methodtrail`
projects are read in place to preserve their worktree paths and evidence.

SQLite stores typed artifacts; JSON and JSONL keep graph and run evidence
inspectable. The dashboard exports those records rather than creating a second
set of graph relationships. It shows directed edges, candidate decisions and
round scores, with draggable nodes, pan and zoom.

## Source layout

`src/graphevolve/` is one installable package. `agents.py` and `orchestrator.py`
coordinate research; `path_graph.py` and `memory.py` organize evidence;
`workspace.py`, `repository.py` and `execution.py` handle code and programs;
`project.py`, `artifacts.py` and `portfolio.py` preserve state; `llm.py`,
`schemas.py`, `cli.py` and `dashboard.py` provide the typed external interfaces.

The release number in `_version.py` supplies packaging, CLI and dashboard
labels. Historical experiment IDs are identifiers, not release numbers, and
are not rewritten during naming migrations.
