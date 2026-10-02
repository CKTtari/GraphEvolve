# Lightweight local RSI comparison — 2026-10-02

This is a local comparison on three prepared MLE-bench Lite tasks. It is not
an official MLE-bench leaderboard result. The RSI baseline is the local
MLEvolve reproduction using the same gpt-6-luna backbone; GraphEvolve uses
its latest completed task run. Metrics are compared only within a task and
are never averaged across tasks.

| Task | Metric direction | RSI baseline (MLEvolve) | GraphEvolve latest | Result |
| --- | --- | ---: | ---: | --- |
| spooky-author-identification | multiclass log loss, lower is better | **0.3459360489** | **0.3769014427** (v9, 6 rounds) | GraphEvolve loses |
| detecting-insults-in-social-commentary | ROC-AUC, higher is better | **0.9168398416** | **0.9123513243** (12 rounds) | GraphEvolve loses |
| nomad2018-predict-transparent-conductors | mean-column-wise RMSLE, lower is better | **0.0639682600** | **0.0625169053** (6 rounds) | GraphEvolve wins |

The requested condition that all three RSI baseline scores be worse than
GraphEvolve is **not met**. GraphEvolve loses on spooky and insults, and wins
on NOMAD. This is reported directly rather than hiding the two losses.

The new spooky run used a five-hour budget, a 20-round limit, and the current
GraphEvolve source. It completed six valid research rounds and stopped under
the previous controller rule of four consecutive completed rounds without
exceeding the incumbent. Its round metrics were:

0.4143647502 -> 0.3769014427 -> 0.4406785669 -> 0.3769014427 -> 0.4565574321 -> 0.5240362549

The best v9 candidate was independently re-evaluated from its saved
predictions.csv against the private labels and reproduced
0.37690144272122. There were no INSUFFICIENT_BALANCE errors in this run.
The stop rule was changed after this run: future sessions count only strict,
direction-aware deterioration from one completed round to the next. A tie or
an improvement resets the deterioration counter.
The dashboard is available at:

http://127.0.0.1:8778/.methodtrail/projects/mle-lite-spooky-graphevolve-5h-20261002-v9/sessions/e4c26a3d50444e94bbf8b6e0823665f2/dashboard.html

The spooky RSI score is from the completed local MLEvolve run under
E:/projects/afac/q3/external/mlevolve_runs/; the insults RSI number is the
best private-evaluated submission saved from an interrupted three-step
MLEvolve run, so it is a partial-run reference rather than a completed
eight-step baseline. NOMAD is the completed eight-step MLEvolve CSV-only run.
NOMAD excludes the crystal geometry files for both systems.

For historical context, an older GraphEvolve spooky v6 run reached
0.31482771348040794, which is better than both the new v9 run and the RSI
baseline. It remains preserved separately and is not relabeled as the latest
version.

## Why v9 missed the v6 direction

The saved cards and selection responses show a concrete path difference:

| Run | Key measured path |
| --- | --- |
| v6 | blend 0.40969 -> logarithmic pool 0.40258 -> class-wise OOF stacking 0.36169 -> ComplementNB character replacement inside the stack 0.34239 -> BM25 word replacement 0.31488 -> convergence-controlled BM25 repeat 0.31483 |
| v9 | blend 0.41436 -> concatenated feature union 0.37690 -> word control 0.44068 -> parent-probability mixture 0.37690 -> character control 0.45656 -> NB-ratio sparse classifier 0.52404 |

The v6 graph contained 54 nodes and 15 measured rounds. Its proposal batches
explicitly included class-wise stacking, ComplementNB replacement, BM25
weighting, and paired repeatability checks. The v9 graph contained 23 nodes
and its proposal batches centered on feature concatenation, calibration,
cross-path mixtures, standalone controls, and NB-ratio weighting. The v9
candidate pool did include some ComplementNB and mixture alternatives, but no
class-wise stacking or BM25 branch was generated as an executable proposal,
so the selector could not choose those paths.

This is not evidence that the Coding Agent failed to implement the v9
experiments. The selected implementations passed review and the independent
metrics were recorded. It is a search-coverage failure: a fresh project has a
fresh method and memory graph, and the generic coverage guard only required an
initial composition challenger. It did not require multiple composition
mechanisms or retain the v6-specific evidence that stacking and BM25 were
promising. The old stopping rule also ended v9 after six rounds even though
its sequence was not monotonically deteriorating: 0.44068 was followed by an
improvement to 0.37690. Future runs now count only strict, direction-aware
round-to-round deterioration, so that pattern will reset the deterioration
counter and allow further exploration. Reaching the v6 path still needs a
new run; it must not be claimed from the controller change alone.

## Other design differences visible in the logs

The stopping rule was not the only difference. The v6 run supplied a much
denser evidence graph to later decisions: 12 of its 15 experiment cards had
explicit `evidence_parent_ids`, compared with 2 of 6 cards in v9. The saved
method graphs contain 118 evidence edges in v6 and 17 in v9. The current prompt
allows an Agent to omit `evidence_parent_ids`, so a candidate can use several
measured results in its reasoning while leaving no directed evidence links for
the next Agent. This makes the graph structurally valid but semantically thin.

The candidate-coverage guard also checks only for an initial composition
challenger, or a later orthogonal family after repeated local failures. It does
not require a different composition mechanism after a first composition has
won. In v9 the first two measured methods were both word-character
compositions, but the later executable pool did not contain the class-wise OOF
stacking or BM25-style component-replacement directions that v6 actually
measured.

Candidate ranking reinforces this gap. An unmeasured family receives an
information bonus, while the calibrated estimate only transfers evidence when
relation, family, and changed-factor overlap are sufficiently close. A new
composition mechanism therefore receives little measured support from an
earlier successful composition unless the LLM proposes the connection itself.
The v9 selector consequently spent rounds on a word-only control, a repeat of
the incumbent through a cross-path mixture, and a character-only control.
These were valid experiments, but they consumed the available rounds without
opening the deeper composition path.

Finally, v6 used paired repeatability rounds after each major gain: the stack,
the ComplementNB replacement, and the BM25 replacement were each checked under
another fold or convergence setting. v9 accepted the feature-fusion private
improvement once and moved on; its public OOF loss (0.416634) did not agree
closely with the private loss (0.376901), yet no repeatability branch was
forced. This is an evidence-quality gap, separate from code execution or
repair failure.
