# GraphEvolve

## Graph Engineering for Autonomous Experimentation

**GraphEvolve** is a general-purpose automated experiment system for machine-learning research with limited feedback and runtime. It turns research questions, code, configurations, execution results, and conclusions into connected artifacts. Those artifacts form an experiment-path graph that helps the system choose a useful next experiment, repair failures, and preserve evidence for later work.

The project name is **GraphEvolve: a graph-guided autonomous experiment system**.

- **Method** covers model families, feature construction, training settings, fusion rules, validation designs, and code implementations.
- **Trail** is the evidence trail showing how one experiment led to the next.

The system follows a serial research process, a project-wide method graph, a
separate memory graph, a technical repair loop, and value-aware path choice. It is not tied to
one model family or benchmark.

## 1. Goal

Long-running experiments need more than a loop that launches training jobs and compares one metric. The system needs to answer:

- What should the next execution establish?
- Which historical results are relevant to that question?
- Is a candidate worth the remaining runtime and feedback opportunity?
- When evidence conflicts, what controlled experiment can identify the cause?
- When execution fails, should the system repair code, use a lighter implementation, or test a nearby method?

MethodTrail implements **artifact-level Recursive Self-Improvement (RSI)**. Across iterations it can improve the research artifacts that generate later experiments:

- research question;
- implementation and code patch;
- model, feature, and training configuration;
- candidate generation and fusion rule;
- experiment memory and path priority;
- prompt instruction used by each research agent.

The self-improvement target is the predictor and experiment process. The orchestration kernel remains stable and configured by the project owner.

MethodTrail does not start from a supplied predictive model. A task adapter supplies only the task contract: permitted data access, input and output schemas, target metric, resource limits, submission format, and evaluation functions. The Coding Agent creates the first runnable predictor from that contract, then writes later feature, model, training, fusion, and validation changes as new code artifacts.

## 2. Design principles

1. **Each experiment has a purpose.** Before execution, the system records a hypothesis, required evidence, comparison plan, adoption condition, and fallback condition.
2. **History keeps context.** A conclusion includes where it applied, what changed, actual cost, and what remains uncertain; a score alone is insufficient.
3. **Selection considers information as well as gain.** A run is valuable when it improves a result or rules out several plausible but unhelpful directions.
4. **Failures provide later constraints.** Resource limits, invalid parameters, and failed patches update future candidate risk and repair choices.
5. **LLM output becomes executable structure.** The LLM returns typed JSON, registered task operations, or a restricted code diff rather than free-form suggestions.
6. **Important decisions are replayable.** The system preserves parent artifacts, selected and deferred candidates, code changes, runtime facts, and decision rationale.
7. **Configuration and implementation changes remain distinct.** A parameter trial edits a configuration artifact only. A change to data processing, model behavior, loss, training logic, fusion, or validation becomes an implementation artifact with an explicit code diff.
8. **Code is evaluated in layers.** A new code artifact must pass syntax and interface checks, a small smoke run, and the task's local evaluation before it can consume a full experiment budget.

## 3. Main research process

`Task → Question → Method graph → Code → Execute → Record → Assess`

| Stage | Responsibility | Output |
|---|---|---|
| Task | Assemble the current question, budget, relevant graph subgraph, global method pool, existing best result, and unresolved evidence | `ResearchState` |
| Question | Form or refine one measurable question from the task and valid evidence | `HypothesisArtifact` |
| Method graph | Add new method nodes, relate them to existing nodes, score paths, and select one executable experiment | `ExperimentSpecArtifact` |
| Code | Read the selected change request, write or patch the predictor workspace, review the applied diff against the hypothesis, then run interface and smoke checks | `ImplementationArtifact` + `ImplementationReviewArtifact` |
| Execute | Materialize source and configuration in an isolated workspace, then train and evaluate | `RunArtifact` |
| Record | Save actual source revision, config, metrics, runtime, resources, logs, and prediction files | persistent run record |
| Assess | Judge evidence, then adopt, investigate, recover, send for allowed external evaluation, or stop | `DecisionArtifact` |

`Task` assembles the present working state. It does not simply reread the task statement after every experiment.

### Specialized agents and workers

| Agent | Trigger | Responsibility | Return point |
|---|---|---|---|
| Question Agent (`new_question`) | a new task or a broad unresolved direction | form a measurable hypothesis and evidence plan | Method graph |
| Question Agent (`conflict_refinement`) | a valid run is incomplete, inconsistent, or needs an ablation | narrow the question without changing it because of a code error | Method graph |
| Method Graph Agent | a research question is ready | grow the project-wide method pool, create graph relations, score paths, and choose one node | Coding |
| Assessment-Memory Agent | a valid run has returned | compare evidence, set node status, store conditions, and choose the next research route | Method graph or Question |
| Repair Agent | code, dependency, resource, or output execution fails | repair the same candidate, switch to a nearby implementation, or abandon it; it never rewrites the research question | the same candidate or Method graph |

The controller is the only owner of runtime state. Question formation, method selection, valid-run assessment, memory writing, and technical repair are separate transitions. A repair step does not increase the research-round count and does not call the Question Agent. The system runs one experiment at a time; the method graph and append-only memory retain every useful or failed branch.

### Coding Agent

The Coding Agent is the implementation worker inside the main research process. It receives a selected experiment specification, the current code lineage, the task contract, relevant graph evidence, and any Recovery notes. It writes the first baseline implementation when no parent predictor exists; in later rounds it makes a narrowly scoped change that matches the selected experiment path.

The edit protocol is enforced by the workspace, not only requested in the
prompt. A root candidate may create the missing entrypoint. A child candidate
or any recovery step must use a unique `replace` or one-symbol
`replace_symbol` edit; whole-entrypoint `rewrite` operations are rejected for
those steps. This keeps local modification an executable invariant rather than
an LLM intention.

Its internal loop is:

`inspect relevant artifacts → plan the code change → apply a constrained patch → independent implementation review → syntax/interface checks → smoke execute → return a runnable implementation or a recovery request`.

The review is a separate typed agent transition, even when it uses the same
provider. It sees the realized diff rather than the proposed plan and checks
the selected research question, declared invariants, comparison target, and
output interface. Its typed outcome is pass, repair, or replan. Repair means
the candidate contract is coherent but the applied code must change; the
Recovery and Coding Agents receive specific issues and inspect the latest
workspace before another local edit. Replan means the candidate's own
declarations cannot be satisfied by source edits, such as a requirement to
report a private evaluator score from inside solution code or mutually
contradictory invariants. The review supplies concrete candidate adjustments;
Method Graph proposes another executable candidate under the same research
question and records a directed revision edge from the rejected proposal.
An exact repeat of the rejected contract is ineligible. Neither transition
becomes measured research evidence or consumes a research round. This gate
complements the deterministic verifier: the verifier can prove syntax,
dependencies, and required output paths, while the review checks whether the
implementation still means what the hypothesis says.

The Coding Agent has no authority to choose the research objective. Reflection and Choose define what should be tested; the Coding Agent decides how to express that test in executable code.

## 4. Artifact model

Artifacts are immutable records. A changed configuration or code revision becomes a new artifact with references to its parents, so later work can trace exactly what changed.

| Artifact | Main content | Use |
|---|---|---|
| `TaskArtifact` | data profile, target metric, task constraints, budget | defines the current research environment |
| `HypothesisArtifact` | observation, hypothesis, evidence needed, adoption/fallback conditions | states what a candidate should establish |
| ChangeRequestArtifact | mutation class, allowed files, required invariants, comparison target | separates a parameter trial from a code change before implementation begins |
| CodePlanArtifact | intended source changes, affected interfaces, expected test result | lets the Coding Agent explain an implementation plan before editing |
| `ImplementationReviewArtifact` | applied diff, invariant checks, scope result, concrete issues | blocks a semantically unrelated or hypothesis-breaking patch before execution |
| `ImplementationArtifact` | source snapshot, patch, entrypoint, dependency profile | identifies runnable model and feature code |
| `ConfigArtifact` | model family, feature switches, hyperparameters, seed, validation plan | gives exact settings |
| `ExperimentSpecArtifact` | parent graph nodes, planned changes, implementation/config refs, evidence plan | represents a selectable candidate path |
| `RunArtifact` | command, timestamps, metrics, runtime, resources, logs, outputs | records execution facts |
| `EvidenceArtifact` | comparisons, subgroup results, consistency assessment, unanswered questions | interprets the run |
| `RecoveryArtifact` | failure class, diagnosis, attempted repair, result | drives the current technical repair step; durable repair records live in `memory/bugs.jsonl` |
| `DecisionArtifact` | candidate priorities, selected path, LLM rationale, adoption outcome | explains choice |
| `PathNodeArtifact` | reusable conclusion, conditions, lineage, linked runs | becomes a path-graph node |
| `ExternalFeedbackArtifact` | evaluator reference and returned permitted metrics | captures external feedback |

All artifacts include `artifact_id`, `task_id`, parent artifact IDs, creation time, producer, schema version, a human-readable summary, a JSON payload, and file references.

Artifact lineage: `Hypothesis → Implementation + Config → ExperimentSpec → Run → Evidence → Recovery / Decision / PathNode`.

For a code-writing system, the more precise lineage is Hypothesis → ChangeRequest → CodePlan → Implementation + Config → ExperimentSpec → Run → Evidence → Decision → PathNode.

Two forms of replay are supported: execution replay recreates source, configuration, and command for a prior run; decision replay reconstructs the graph state and candidate ranking used for a prior choice.

## 5. Project, session, and version lifecycle

MethodTrail separates a research project from its individual sessions and code variants.

- A **project** owns one task contract, a Git baseline repository, and the pointer to the current accepted revision.
- A **session** records one continuous research attempt. It has a JSONL trajectory and a short handoff file for pause and resume.
- A **candidate** is a Git branch and worktree created from the current accepted revision or an adopted parent variant.
- A **run** is the measured execution performed inside that candidate worktree.
- A **memory card** stores a reusable conclusion in append-only JSONL. A separate memory graph links cards by parent variant, method family, changed factors, and related tags; the experiment-path graph stores executable method relations.

The Git commits, candidate metadata, session handoff, trajectory JSONL, memory cards, and memory graph are the replayable sources of truth. SQLite artifacts remain useful as a local audit/index layer, but the system can still inspect a project without querying a database.

Pause writes the latest accepted version, metric, next question, workspace, and artifact references into `handoff.md`. Resume reopens the session and starts the next candidate from the project’s accepted revision. Rollback moves the project’s accepted-version pointer to any earlier adopted candidate; it does not erase later exploratory evidence.

## 6. Experiment-path graph

The graph is a memory structure and a candidate-generation structure.

### Nodes

A node represents a method state that produced a reusable conclusion. It stores the research question, method family, feature view, implementation/configuration references, metrics, runtime, resource observations, supported conclusion, evidence strength, conditions where the conclusion held or failed, and unresolved questions.

Raw runs stay in the artifact store. The Memory Agent creates a path node when Assess determines that a run has produced a reusable finding, including a useful negative result.

### Edges

| Relation | Meaning |
|---|---|
| `deepen` | extend a supported idea, such as larger propagation range or longer training |
| `ablate` | isolate a component or change to measure its contribution |
| `combine` | retain useful components and test a new composition |
| `explore` | introduce a method from a new family or a new information source |
| `repair` | preserve the research question but change the implementation after failure |
| `revisit` | retest an unstable idea after new evidence changes the context |

### Why graph organization matters

A chronological list records what happened. The path graph also records how methods relate:

- several candidate methods can share a parent and differ by one deliberate change;
- a failed branch remains attached to the assumption it tested, separating a weak method idea from an implementation problem;
- a useful component can feed several later combinations;
- retrieval starts at the current question and follows relevant parents, siblings, and neighbors rather than consuming unrelated old logs.

NetworkX provides in-memory subgraph retrieval, path expansion, ranking, and visualization. The method graph JSON, memory graph JSON, and append-only memory cards are readable outside the runtime; SQLite holds the generic artifact audit trail.

### Implemented path mechanics

The executable system uses four concrete steps rather than treating the graph as a visual record:

1. **Describe a method.** Each candidate carries its method family, component map, changed factors, and target scope. This gives the graph a machine-readable basis for comparing two experiments.
2. **Attach branches before selection.** Candidate proposals are written as proposed children of the active version before ranking. The Method Graph Agent owns the semantic relation (`deepen`, `ablate`, `combine`, `explore`, or `recover`). The runtime validates the component map and stores a visible warning when it conflicts with that declaration; it never silently changes the research question. A proposal can also list multiple `evidence_parent_ids`, which become directed `informed_by` edges distinct from the single code-parent lineage edge.
3. **Retrieve a directed evidence pack.** The next iteration receives path history, matched alternatives from the same parent, and a candidate frontier of unselected or deferred branches. It does not flatten the graph into an undirected recent-history list.
4. **Write back outcomes.** Adopted, deferred, evidence-seeking, and failed executions all retain their measured facts and relation to the proposed branch. Raw measurements remain distinct from the Memory Agent's reusable conclusion and applicable conditions.

The memory graph then links the new conclusion to related cards. Its parent and
semantic edges are directed and sparse: retrieval walks incoming provenance
edges to older conclusions before considering outgoing later branches. The
returned memory record includes the traversed relation, reason, and target
change, so the Agent sees why a card was reached. This profile is supplied to
the next proposal step, so memory changes candidate generation rather than
merely increasing the size of the prompt. A proposal can name an existing
ancestor outcome as `parent_variant_id`; the orchestrator validates that ID and
creates the new workspace from that ancestor, making directed backtracking an
explicit experiment operation.

Path priority starts from the LLM's estimates, then applies a bounded historical correction. Comparable outcomes contribute observed gain, wall time, and failure rate; the correction is blended with, rather than substituted for, the LLM prior. A method family with no measured outcome receives a small information preference. A family with repeated non-improving outcomes is down-weighted without being deleted, so a genuinely new factor can still be revisited. After repeated non-improvement, candidates that only repeat the recent changed factors receive an additional saturation penalty; an explicit composition or orthogonal factor remains eligible and receives information value even before it has a positive result. Candidates that cannot fit the remaining runtime plus the finalization reserve are excluded before the LLM chooses. Task contracts can expose auxiliary metrics and minimum metric constraints, so a higher primary score cannot bypass a failed correctness or reliability requirement.

## 7. Candidate generation and selection

The method graph starts with a task root, not a prewritten predictor. The first
method nodes are proposed from the task contract, data profile, available
libraries, and the current question. A new task therefore begins from zero
method knowledge; a resumed project restores its own method pool and evidence.

The Method Graph Agent may add any number of distinct nodes in one discovery
step, including none when the current pool already contains enough relevant
alternatives. There is no fixed `proposal_count`. Each new node is normalized
against the project method pool before it is attached to the graph. Existing
nodes are revisited rather than duplicated.

Candidates come from four sources:

1. Graph-neighbor expansion derives `deepen`, `ablate`, `combine`, `repair`, or
   `revisit` paths from relevant nodes.
2. The task contract exposes data schemas, metric functions, permitted
   dependencies, resource limits, output requirements, and code interfaces. It
   does not provide a ready-made predictor.
3. The LLM proposes a new implementation or composition through the task's
   typed capability and file-access schema.
4. A repair result may add a nearby implementation node that tests the same
   research question.

Free-form suggestions are never executed directly. Every candidate becomes a ChangeRequest Artifact and then an implementation plan or controlled patch that passes workspace checks.

At the initial discovery boundary, if the proposal batch contains multiple
distinct method families or components but no composition candidate, the
controller performs one bounded proposal revision with an explicit coverage
warning. This is a generic search-space check, not a benchmark-specific
predictor. Later rounds keep an orthogonal or composition challenger in the
frontier when recent candidates share factors and fail to improve, so a local
calibration or inference tweak cannot silently become the whole search space.

### Candidate specification

```json
{
  "candidate_id": "rec_three_source_history_group_fusion",
  "parent_nodes": ["rec_two_source_ranker_v3"],
  "relation": "combine",
  "question": "Does the third candidate source help only users with complete history?",
  "changes": [
    {"kind": "candidate_source", "operation": "add", "value": "source_3"},
    {"kind": "strategy", "operation": "group_apply", "value": "complete_history_only"}
  ],
  "implementation_ref": "impl:rec_ranker_017",
  "config_ref": "cfg:rec_052",
  "evidence_plan": {
    "comparisons": ["two_source", "three_source"],
    "metrics": ["Recall@20", "NDCG@10"],
    "segments": ["short", "partial", "complete"]
  }
}
```

### Change boundary: tuning, composition, and code evolution

MethodTrail does not label every change as tuning. Before Code runs, Choose must assign exactly one primary mutation class.

| Mutation class | May change | Must not change | Artifact produced |
|---|---|---|---|
| Configuration tuning | declared numeric values: learning rate, depth, batch size, regularization, seed, training rounds, thresholds, weights | source files, data transformation logic, model topology, loss, candidate generation | ConfigArtifact version |
| Composition change | how existing agent-written components are wired, selected, or weighted | component internals unless a separate code change is requested | CompositionSpec plus generated assembly source |
| Implementation evolution | feature extraction, model architecture, training loop, loss, sampling, retrieval, ranking, fusion, validation procedure, and helper functions | task contract, metric definition, data-access policy, protected system files | ChangeRequest, CodePlan, ImplementationArtifact |
| Recovery patch | only the smallest code/config change needed to restore a blocked experiment | research question unless Recovery explicitly returns it to Reflection | RecoveryArtifact plus patch/config version |

A configuration experiment preserves the exact implementation fingerprint. Any semantic behavior change creates a new implementation artifact and a code diff. Composition is recorded separately because it can reuse existing code artifacts while still introducing a new executable predictor.

The Coding Agent receives the mutation class as a hard instruction. It may decline a configuration-only request when the requested effect requires changing code; in that case it returns an implementation-evolution request to Choose rather than silently editing source.

### Predictor creation policy

The first predictor is generated by the Coding Agent from the task contract and a small, empty task scaffold containing only input/output interfaces. The scaffold may include data loading, metric calls, and submission validation. It contains no hand-written model, feature pipeline, ranker, graph propagation routine, or fusion rule.

Later predictors come from one of three origins:

1. a code mutation of an earlier implementation artifact;
2. a new implementation written from a Reflection question and task contract;
3. a composition written from compatible earlier implementation artifacts.

Third-party libraries such as PyTorch, scikit-learn, CatBoost, or NetworkX remain permitted dependencies. Calling a library API in code written by the Coding Agent is different from providing it a finished predictor implementation.

### Path value model

Each candidate receives an experiment value and a path priority:

\[
V_s = \alpha_s G_s + \beta_s I_s
\]

\[
P_s = \frac{V_s}{1 + \gamma_s T_s + \delta_s R_s}
\]

| Term | Meaning | Produced by |
|---|---|---|
| \(G_s\) | expected improvement over the selected version | LLM estimate grounded in historical evidence |
| \(I_s\) | information value: distinguish candidates, narrow a question, or explain a conflict | LLM estimate and evidence plan |
| \(T_s\) | expected runtime divided by remaining runtime | program using historical runtime and adapter estimates |
| \(R_s\) | execution risk | program using failure history, resource demand, and complexity |
| \(P_s\) | final candidate priority | program calculation |

The default calculation uses transparent unit weights. The LLM supplies the
candidate's expected improvement and information value, while the program
derives runtime and failure risk from the method graph and execution history.
Task-specific stage coefficients are not hidden in the contract. Any change to
the scoring rule is itself a versioned harness change and is recorded in the
project history.

The LLM sees program-sorted candidates. If it selects a lower-ranked path, it must record a concrete reason. Selected and deferred candidates both remain in the Decision Artifact.

### Code-variant archive

Every implementation written by the Coding Agent is stored as a code variant linked to its parent variant and experiment-path node. The archive keeps more than a single current-best program:

- a score-leading variant for exploitation;
- a low-cost variant for fast evidence collection;
- a reliable variant with few execution failures;
- distinct variants that solve the task through materially different assumptions.

The Method Graph Agent retrieves candidates from this archive and from graph
neighbors. It can deepen a strong implementation, compare a sibling variant
that changes one factor, combine two compatible variants, or revive a deferred
branch when new evidence changes the estimate.

The archive uses measured metric, runtime, reliability, and relationship to the active question. It does not discard a slower or lower-scoring variant when that variant provides a capability or evidence path the current best implementation lacks.

### Layered code evaluation

Code changes are evaluated in increasing-cost order:

1. source syntax and imports;
2. task-contract and output-schema checks;
3. optional task-declared smoke command;
4. optional task-declared test command;
5. full task execution;
6. task-declared independent evaluation command.

At each layer, the executor writes observations back to the active Run Artifact. A failure before full training is a technical signal for Recovery and path-risk estimates; it is not a measured research conclusion.

## 8. Question, Assessment, Recovery, and Memory

### Question Agent: `new_question`

Reflection converts an initial observation into a precise research target. It receives the current task, relevant path nodes, recent evidence, and remaining budget. It returns a hypothesis, evidence request, comparison plan, success condition, fallback condition, and candidate-generation hints.

Example:

```text
Observation: a third retrieval source increases candidate coverage.
Hypothesis: it improves ranking for users with complete history.
Evidence: compare two-source and three-source pipelines overall and by history group.
Adoption: positive evidence for the target group without material overall regression.
Fallback: retain the two-source ranker and test the third source as a low-weight component.
```

The Question Agent sets the task target. The Method Graph Agent selects the
concrete model, feature, parameter, and implementation changes.

### Question Agent: `conflict_refinement`

Evidence handles results that remain uncertain. It treats conflicts as a localization problem.

Trigger examples:

- local validation improves while external evaluation declines;
- selection split and independent confirmation disagree;
- an overall metric is flat while a subgroup changes substantially;
- several changes occur in one run;
- different seeds produce inconsistent results.

The same Question Agent compares a candidate with its parent, identifies
changed components, specifies missing evidence, and requests ablations, matched
controls, group comparisons, or repeated confirmation. It returns a narrower
question to the Method Graph Agent; a technical error never triggers this mode.

### Recovery Agent

Recovery starts after Execute finds a technical problem. It is an internal
repair loop, not a research iteration. The current hypothesis, method target,
and parent node stay fixed until the candidate runs successfully or the Repair
Agent explicitly abandons it.

| Repair direction | Action |
|---|---|
| Parameter repair | correct invalid or unstable settings while preserving the intended comparison |
| Lower-cost implementation | use a cheaper implementation that tests the same question |
| Neighbor-path replacement | choose a graph-adjacent method that can test a similar question |
| Code patch | write a constrained local patch in the isolated run workspace |
| Abandon candidate | record the technical cause and return to the method graph without changing the question |

The LLM receives relevant source files, traceback, task interface, current
hypothesis, similar past repairs, and (for a semantic review failure) the
typed `ImplementationReviewArtifact` with failed checks and concrete issues.
The Repair Agent must answer that checklist with local edits before the patch
is reviewed again. It must inspect the latest workspace after every applied patch;
an earlier failed plan is not a source of truth for the next plan. It returns
`continue_repair`, `switch_implementation`, or `abandon_candidate` for
technical failures. The implementation reviewer can instead return
`replan_candidate` when the selected candidate contract itself is impossible;
the controller returns to method selection with review feedback rather than
asking for another code patch. The
controller allows a large technical-step safety cap (`max_repair_steps`,
default 100). Repeated review failures alone never abandon a candidate;
`abandon_candidate` must be an explicit Recovery Agent decision supported by a
technical impossibility diagnosis. These steps do not count
as research rounds. The executor runs syntax checks and an adapter smoke test
before launching full training. Each repair step becomes a Recovery Artifact
and a separate bug record; only a successful model run with a measured metric
creates an experiment-memory conclusion and increments the research-round
counter.

### Memory Agent

The Assessment-Memory Agent runs after a valid measurement. It:

1. summarizes the question, actual change, result, and supporting evidence;
2. states the conditions where the conclusion applies or does not apply;
3. creates or updates a method node, including useful negative and promising
   results;
4. connects parent, sibling, and future candidate paths;
5. updates node status and path priority for the next selection.

The final output is always chosen from the best measured valid node. A lower
scoring but informative or promising node remains in the graph and can be
expanded later; it never silently replaces the project incumbent.

Example memory note:

```text
Conclusion: directed multi-hop features improve classification after one-hop evidence is positive.
Support: independent validation and external classification feedback both improved.
Conditions: sparse attributes, directed graph representation, CatBoost classifier family.
Open question: does the gain persist with a graph-neural backbone?
Suggested paths: deepen multi-hop, ablate directionality, combine with probability features.
```

## 9. Execution and evaluation

The current implementation creates an isolated Git worktree for every candidate:

```text
.methodtrail/projects/<project_id>/
├── repository/                 # Git baseline
├── sessions/<session_id>/      # trajectory and handoff
├── wt/<short-variant-id>/      # candidate Git worktree
├── candidates/<variant-id>.json
└── memory/
    ├── cards.jsonl
    └── bugs.jsonl
```

Each candidate worktree contains:

```text
<candidate-worktree>/
├── task data hydrated from the template and ignored by Git
├── protected evaluator inherited from the baseline
├── generated or modified implementation files
├── .methodtrail.diff
├── predictions.csv or task-specific outputs
└── metrics.json written by the evaluator
```

The executor runs only commands declared by the active task contract. It records wall-clock time, exit status, truncated stdout/stderr, evaluator output, metric files, and required-output checks. CPU/GPU telemetry is a later extension.

The LLM can modify declared experiment code only inside this workspace. It cannot directly modify credentials, original datasets, the orchestration kernel, or arbitrary files outside its run folder.

### Evaluation layers

| Layer | Purpose |
|---|---|
| Syntax and dependency check | reject invalid source or imports outside the declared dependency list |
| Smoke command | run a short task-declared check before full execution |
| Test command | run a task-declared test suite before full execution |
| Evaluation command | read generated outputs and write the metric independently of model code |

An external feedback connector can later feed its result into Evidence. The current implementation supports local evaluation commands only.

## 9. Provider and task interfaces

### LLM providers

MethodTrail currently ships with one OpenAI-compatible client. The endpoint and
model are launch-time settings; the benchmark configuration uses `gpt-6-luna`
through its configured compatible endpoint.

The provider layer supports typed JSON responses, model identifiers, and failure propagation into Recovery. Credentials are read from the API-key environment variable selected at launch; the default is `DASHSCOPE_API_KEY`.

### External evaluator

The current evaluator interface is a task-declared local command that reads outputs and writes a metric file. Automatic remote submission is outside the current implementation.

### Task adapter

Current adapters create task contracts for prepared MLE-bench, SWE-bench, and Terminal-Bench workspaces. A task contract declares data paths, commands, metric behavior, dependencies, protected files, and resource limits; it never supplies a predictor.

Task-specific adapters remain separate from the core package and can be added without changing the research loop.

## 10. Storage, state machine, and repository layout

### Current persistent state

SQLite currently stores typed artifacts in one append-only `artifacts` table with parent IDs. The experiment graph is stored as JSON, and the Pareto version set is stored as JSON. Splitting those records into specialized relational tables is a later storage optimization.

### Orchestrator states

The sequence is `INITIALIZE → ASSEMBLE_TASK_STATE → QUESTION_IF_NEEDED → EXPAND_METHOD_GRAPH → SCORE_AND_CHOOSE → MATERIALIZE → EXECUTE → RECORD → ASSESS`.

From Execute, technical failures enter a separate `REPAIR` state. Repair steps
do not increment the research-round counter. A successful repair returns to the
same candidate; an abandoned repair returns to method selection. From Assess,
the system can adopt and remember a result, keep a promising or incomplete
branch for more evidence, expand the method graph, or stop.

The multi-round runner allows up to 20 completed research rounds by default. It
also stops after four consecutive completed rounds fail to improve the current
incumbent, in addition to an Agent `stop` decision or exhausted time budget.
Technical repair attempts can repeat inside one round and do not consume the
research-round count. The `dashboard` command writes an offline HTML view of
the method graph, experiment-memory graph, round filter, directed edge labels,
node/edge change details, and trajectory events.

### Current repository

The project contains `agents`, `artifacts`, `execution`, `path_graph`, `portfolio`, `repository`, `workspace`, `orchestrator`, `llm`, and `adapters` modules; benchmark helpers and tests live alongside them. Remote-submission integrations are intentionally outside the core package.

## 11. Implementation sequence

1. Artifact schemas, SQLite repository, run-state machine, and run file layout.
2. OpenAI-compatible providers with typed JSON validation.
3. Task-contract protocol plus MLE-bench, SWE-bench, and Terminal-Bench contract adapters with no built-in predictors.
4. Isolated execution workspace, independent metric collection, optional smoke/test commands, and output validation.
5. Experiment graph, subgraph retrieval, candidate expansion, and path-value calculation.
6. Question, Method Graph, Assessment-Memory, Repair, and Coding Agent collaboration.
7. Code-variant archive, controlled patch flow, and repair-history integration.
8. External evaluator connector and feedback ingestion.
9. Replay commands, graph visualization, reports, and benchmark-scale integration tests.

Each stage must run through the actual orchestrator and actual training code. The project will not use simulated scores as a substitute for execution.

## 12. Implementation assumptions to confirm

1. The runtime is provider-agnostic. The current benchmark uses an
   OpenAI-compatible endpoint with `gpt-6-luna`; no provider-specific model is
   assumed by the agent roles.
2. The design includes an external-evaluation connector interface. Automatic submission remains disabled until an evaluator API and authorization are provided.
3. The LLM may patch task adapters and experiment implementations inside a run workspace; GraphEvolve core, credentials, and configured data roots remain read-only.
4. One experiment runs at a time by default. The selected adapter may use a configured GPU.
5. Future task adapters share the GraphEvolve artifact and graph system without becoming part of its core policy.

## 13. Implementation references

The design draws practical implementation patterns from OpenEvolve and the Awesome RSI research map. OpenEvolve contributes the code-variant archive, artifact feedback, and layered evaluation ideas. Awesome RSI informs the artifact-level RSI scope: MethodTrail improves the research artifacts used by later experiments while keeping its orchestration kernel under explicit project control. See REFERENCES.md for source links and the exact mapping.

## 14. Coding-agent architecture

MethodTrail has one deterministic research controller, four decision roles, and
one coding worker. The four decision roles correspond to question formation,
method-graph expansion, valid-run assessment/memory, and technical repair.
Coding writes the selected implementation but never changes the research
objective or adoption rule.

| Role | Decides | Does not decide |
|---|---|---|
| Question Agent | what question and evidence plan should guide the next experiment, in `new_question` or `conflict_refinement` mode | exact source-code implementation |
| Method Graph Agent | which new or existing method node should be evaluated next | whether code is technically valid |
| Assessment-Memory Agent | how a valid result changes node status, conditions, and path priority | whether to invent a metric or overwrite the global best |
| Repair Agent | whether to continue repair, switch implementation, or abandon a technical candidate | whether the research hypothesis is disproved by a code failure |
| Coding Agent | how a selected change becomes source code and tests | the research objective or adoption decision |

Three functions are deliberately programmatic services rather than additional LLM agents:

| Service | Function |
|---|---|
| Context Assembler | builds a compact context pack from task contract, repository map, selected graph neighborhood, relevant code symbols, tests, errors, and run facts |
| Verifier | runs interface checks, tests, metrics, resource checks, and prediction-format validation; it never asks the LLM to grade its own code |
| Portfolio Manager | maintains Pareto-relevant code variants using quality, cost, reliability, information value, and diversity; the path graph computes candidate priority |

This division keeps LLM reasoning focused on questions and code while the executor remains the source of operational facts.

### Repository context for Coding Agent

Before asking the Coding Agent to edit source, Context Assembler builds a repository map containing:

```text
file tree and task-owned directories
symbols and import relations for relevant source files
active entrypoint and test commands
public task-adapter interfaces
parent implementation diff and configuration
recent traceback or failed assertion when applicable
related graph nodes and their evidence summaries
```

The agent receives this compact context pack instead of the entire repository. It can request further files through a read-only retrieval tool. This supports multi-file changes without filling the context window with unrelated run logs.

The runtime enforces this boundary in `methodtrail.agents._context`: the
default serialized pack is capped at 80,000 characters. It keeps the latest
directed graph neighborhood and method pool as compact rows (node IDs,
parents, edge traces, relation, family, factors, metric, and short change
logic), preserves every candidate index passed to selection, and truncates
source excerpts and prose independently. The artifact store remains the
complete source of truth. A byte-identical patch is also detected before the
next implementation review; the controller emits a typed no-progress failure
and forwards it to Recovery/Coding instead of paying for an identical review.

### Workspace and code lifecycle

Each implementation variant lives in a separate workspace created from a parent source snapshot. If the project is a Git repository, the executor uses a worktree or temporary branch; otherwise it copies the parent implementation into the run workspace. The lifecycle is:

```text
ChangeRequest
→ CodePlan
→ source edit or new implementation
→ static/interface verification
→ smoke execution
→ local evaluation
→ independent confirmation when warranted
→ code-variant archive and path-graph update
```

The system retains the source diff, generated tests, execution observations, and decision that followed. A source revision that fails early checks remains a recovery artifact; it does not enter the adopted code-variant archive.

## 15. Two-level RSI

MethodTrail separates predictor evolution from harness evolution.

### Predictor RSI

Predictor RSI changes artifacts that solve the active ML task:

```text
feature code
model architecture
training loop and loss
candidate generation
ranking or fusion logic
validation procedure
configuration and composition
```

These changes are evaluated against task metrics, runtime, resource use, reliability, and evidence quality.

### Harness RSI

Harness RSI changes artifacts that decide how the system conducts research, without modifying the orchestration kernel:

```text
Reflection, Choose, Evidence, Recovery, and Coding prompt templates
context-pack retrieval and compression policy
candidate-mutation templates
path-value coefficient policy
code-plan and test-generation templates
```

A HarnessArtifact is evaluated on a fixed set of held-out research episodes: task success, improvement per unit compute, valid-code rate, repair rate, and evidence quality. A harness variant is adopted only when it improves these episode-level measures without degrading basic safety or reproducibility checks.

This gives MethodTrail a concrete RSI path beyond predictor tuning. It stays narrower than unrestricted self-rewriting: task contracts, execution permissions, protected paths, verifier rules, and the orchestration state machine remain fixed system policy.

## 16. Trajectory data and evaluation flywheel

Every completed research episode is normalized into a structured trajectory:

```text
task state
retrieved context
hypothesis and evidence plan
candidate set and path priorities
chosen change request
code plan, patch, and tests
execution observations and errors
local/external evaluation
adoption or recovery decision
path-graph update
```

The trajectory store supports three later uses:

1. **In-context retrieval**: retrieve similar successful plans, code fixes, and evidence designs for a new experiment.
2. **Preference data**: construct preference pairs from selected versus deferred candidates, valid versus invalid patches, and successful versus failed recovery plans.
3. **Offline policy improvement**: export task-conditioned trajectories and reward signals for later SFT, preference optimization, or Agent RL research.

The project initially consumes these trajectories for retrieval and analysis. It does not claim to train a foundation model or run production-scale Agent RL. The interfaces make those later experiments possible.

### Reward and evaluation signals

The evaluator records a vector rather than one opaque reward:

```text
task metric improvement
information value realized
wall-clock and accelerator cost
code validity and test pass rate
recovery success and repair rounds
prediction or submission validity
reproduction consistency
```

Portfolio Manager uses this vector for Pareto retention. Offline learning exports can combine it into a task-specific reward, while keeping the original dimensions available for analysis and reward redesign.

## 17. Candidate search policy

The graph provides the candidate neighborhood and the project-wide method pool.
The current implementation uses transparent priority ranking plus Pareto
retention. The Method Graph Agent decides how many new nodes are useful for the
current question; the program removes duplicate nodes, attaches relations,
calculates priorities, and keeps unselected nodes available for later visits.

There is no task-level breadth/depth switch. A later search policy can be added
without changing the artifact format:

| Policy | Suitable situation | What it adds |
|---|---|---|
| priority ranking | few candidates and reliable local measurement | direct, explainable choice from expected gain, information value, cost, and risk |
| contextual bandit | repeated task families with comparable outcomes | learns which path types tend to pay off under similar task states |
| tree search | several dependent code changes where early choices affect later options | compares short experiment sequences instead of only the immediate next run |
| evolutionary population | many diverse code variants with cheap evaluation | retains useful variants instead of repeatedly restarting from one incumbent |

The initial system should implement priority ranking and Pareto retention first. Bandit, tree-search, and population policies need accumulated trajectories and a dependable evaluator; adding them before those foundations would create more machinery than useful decisions.

## 18. Benchmark and comparison plan

MethodTrail will be evaluated on two primary public benchmarks and one optional engineering benchmark. Each benchmark exercises a different part of the system, so results will be reported separately rather than averaged into one opaque score.

| Benchmark | What it tests | MethodTrail task adapter | Main measure |
|---|---|---|---|
| MLE-bench | autonomous ML experimentation: data understanding, code writing, training, validation, and prediction generation | `MLEBenchAdapter` creates a task contract from the competition files and metric | normalized competition score, valid-submission rate, best score under a fixed runtime budget |
| SWE-bench Verified | repository-level issue fixing with real tests | `SWEBenchAdapter` converts an issue and repository checkout into a code-change task contract | resolved-issue rate, valid patch rate, repair rounds, wall-clock time |
| Terminal-Bench (optional) | terminal operation, environment handling, and multi-step execution | `TerminalBenchAdapter` exposes its task specification and command environment | task success, timeout rate, recovery success |

The first implementation should start with a small, reproducible MLE-bench subset and a small SWE-bench Verified subset. Terminal-Bench is added after the workspace runner and recovery logic are stable.

### Implemented smoke benchmark

`benchmarks/openml_credit_g` provides a public OpenML credit-g smoke task while the larger benchmark adapters are being integrated. The task workspace contains training data, test features, and an evaluator; it contains no predictor. On September 26, 2026, the external `direct_logreg_baseline.py` ran through MethodTrail's verifier and executor on the fixed 800/200 split and obtained **0.730 accuracy**. This verifies data download, isolated execution, prediction-file checking, and independently written metrics. It is an execution check, not a MethodTrail Agent result.

Three comparison systems are sufficient for the first report:

| Comparison | Shared conditions | Purpose |
|---|---|---|
| Direct Coding Agent | same LLM, task contract, workspace, tools, and execution budget; one plan writes and runs code without graph memory or specialized return paths | measures the value of structured research control over one-shot coding |
| Sequential Experiment Agent | same LLM, task contract, workspace, tools, and execution budget; reads chronological run summaries and selects the next trial without graph relations or value-aware path ranking | measures whether graph-linked memory and path choice reduce aimless iteration |
| Public reference system | SWE-agent or OpenHands on SWE-bench; AIDE where its MLE-bench implementation is available | gives a recognizable external reference under each project’s documented setup |

The comparison report will use final task performance, time to first valid result, best result within the same budget, valid-code rate, technical-recovery rate, and number of completed experiments. It will not present a cross-benchmark average because the benchmarks measure different abilities.

## 19. Industry-aligned technical stack

| Layer | MethodTrail implementation | Why it matters |
|---|---|---|
| Language and packaging | Python 3.11+, typed dataclasses/Pydantic schemas, `uv` or pip environment files | reproducible typed artifacts and service boundaries |
| LLM integration | OpenAI-compatible provider, structured outputs, retry/error handling | model-agnostic Coding and research roles |
| Repository understanding | `ripgrep`, AST or tree-sitter symbol index, import graph, test map | repo-level code retrieval and targeted editing |
| Code quality | ruff, pyright, pytest, adapter-provided smoke tests | separate technical correctness from LLM self-assessment |
| Workspace isolation | per-run worktree/copy workspace; optional Docker runtime | safe code mutation and parallel-safe provenance |
| Experiment execution | subprocess runner, timeout/resource observer, GPU telemetry where available | measurable cost and reliable Recovery triggers |
| Memory and graph | SQLite, JSON artifacts, NetworkX | structured trajectory store and graph-based retrieval |
| Evaluation | local metric suite, independent confirmation, external evaluator interface, reproduction runner | measurable research and coding outcomes |
| Analysis and future learning | trajectory export, preference-pair builder, reward-vector export | connects the system to post-training and Agent RL workflows |

Detailed industry and open-source alignment is recorded in TECH_STACK_ALIGNMENT.md.
