# Lightweight local evaluation — 2026-10-01

These are independent local task scores, not an official MLE-bench aggregate.
All RSI runs used the same gpt-6-luna backbone. GraphEvolve and MLEvolve have
different search controllers and run lengths. Each task uses its own metric;
raw scores from different tasks must not be averaged.

| Task | Metric | GraphEvolve current run (best / final) | MLEvolve local reproduction | Fixed no-agent comparator |
| --- | --- | --- | --- | --- |
| spooky-author-identification | multiclass log loss, lower is better | v8 partial: 0.411168 / 0.411168, 3 measured rounds; v7 complete: 0.409342 / 0.409613, 12 rounds | 0.345936, earlier completed spooky reproduction | word + character TF-IDF: 0.371989 |
| detecting-insults-in-social-commentary | ROC-AUC, higher is better | 0.912351 / 0.912351, 12 rounds | 0.916840 from the best saved submission of an interrupted 3-step run | 0.910535 |
| nomad2018-predict-transparent-conductors | mean-column-wise RMSLE, lower is better | 0.062517 / 0.062517, 6 rounds | 0.063968, completed 8-step run | 0.063791 |

The historical GraphEvolve spooky v6 project reached **0.31482771348040794**
in round 15. It remains the project's best known spooky result. It is archived
under no-use/benchmarks/mle_bench_lite_spooky/runs/graph_evolve_5h_20260930_v6.
The newer v7 and v8 scores are worse; the later controller changes have not
yet demonstrated an improvement on this task. The v6 path tested a word/character
out-of-fold blend, classwise stacking, character ComplementNB, and BM25 word
features. The newer searches stayed much closer to their initial branches and
did not reach that sequence. A good historical code version was preserved, but
the new project IDs intentionally started with separate method and memory
graphs, so v6 evidence was not in their Agent context. This explains part of
the direction gap without proving that any one prompt change caused it.

The v8 run was stopped after round 3 when the configured API returned
INSUFFICIENT_BALANCE. The private evaluator had already measured round 3; its
valid 0.41116768094208284 result was recovered into the project, method graph,
memory graph, and dashboard without inventing an assessment-agent conclusion.
This run started before the review-to-replan change, so it does not test that
change in a live benchmark. Further paid LLM runs require API capacity.

The insults MLEvolve run saved a valid private-evaluator submission after
three steps but did not complete the planned eight steps; its score is a
partial-run reference. A later restart also stopped on the same API balance
error and did not supersede that saved result. The NOMAD comparison uses the
official prepared CSV split on both systems, but excludes the crystal geometry
files. It is therefore a CSV-only local evaluation of that task.

Source and evidence locations:

- Historical MLEvolve spooky workspace:
  E:/projects/afac/q3/external/mlevolve_runs/workspace/20260928_203604_mlevolve-luna-spooky
- Its privately evaluated best code and score:
  E:/projects/afac/q3/external/mlevolve_runs/mlevolve_best_eval/solution.py
  and E:/projects/afac/q3/external/mlevolve_runs/mlevolve_best_eval/metrics.json
- GraphEvolve spooky v8:
  benchmarks/mle_bench_lite_spooky/runs/graph_evolve_5h_20261001_v8
- GraphEvolve insults:
  benchmarks/mle_bench_lite_insults/runs/graph_evolve_5h_20261001
- GraphEvolve NOMAD:
  benchmarks/mle_bench_lite_nomad_official/runs/graph_evolve_5h_20261001_official
- MLEvolve insults and NOMAD reproductions:
  cache/mlevolve_runs/20261001_113000_insults-luna-repro and
  cache/mlevolve_runs/20261001_111508_nomad-official-csv-luna-repro-v2

The earlier MLEvolve spooky files belong to the separate AFAC project and were
neither copied into nor moved by this GraphEvolve cleanup.
