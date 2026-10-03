# Local evaluation and controller audits — 2026-10-03/04

These are lightweight local evaluations, not official MLE-bench aggregate
scores. Metrics with different units are reported separately.

| Task | Same-backbone MLEvolve RSI baseline | GraphEvolve latest measured result | Comparison |
| --- | --- | --- | --- |
| spooky (log loss, lower better) | 0.345936 | v12: 0.329863; 20 measured rounds, completed | GraphEvolve wins this local score comparison |
| insults (ROC-AUC, higher better) | 0.916840; interrupted after 3/8 steps | 0.912351; 12 measured rounds | GraphEvolve loses; baseline run incomplete |
| NOMAD (mean-column-wise RMSLE, lower better) | 0.063968; 8 steps | 0.062517; 6 measured rounds | GraphEvolve wins on the CSV-only task |

The three-win target has not been achieved. Only spooky was rerun under the
latest evaluated controller (`b01f18f`); insults and NOMAD are earlier runs.
Historical spooky v6 remains stronger at 0.314828; v9 measured 0.376901.
v10's saved incumbent prediction
file was independently rechecked at 0.35683099178266003, with matching IDs,
finite nonnegative probabilities and normalized rows.

## Verified v10 failure

Source version: `7fd8712`. Project: `mle-lite-spooky-graphevolve-5h-20261002-v10`.
Session: `41c1e1f345704bb0bf37586626104088`.

The resumed process exited at approximately 13:59 China time on October 3,
before its remaining time budget expired. There was no budget or research
stop event. Three consecutive malformed `CodePlanArtifact` responses escaped
the repair controller as a RuntimeError. The persisted session remained
`active`, incorrectly suggesting that the dead process was still running.

Round 7 performed 43 repairs over about 81 minutes. Its 43 failures comprised
24 implementation-review failures, 8 patch-application failures and 11
execution failures. Round 8 accumulated 40 recorded failures before the
final schema error: 23 patch failures, 14 review failures and 3 execution
failures. These were internal repair attempts, not additional research rounds.

Two code defects explain repeated problems visible in those logs:

1. `RepoMap.retrieve` returned only a 5,000-character prefix when an editable
   source exceeded 20,000 characters. Prompt assembly separately shortened
   source content to 9,000 characters, or less under context pressure. The
   last failed model responses explicitly said the entry guards were outside
   the supplied excerpt and declined to invent an exact patch.
2. The review received only the latest patch, although earlier repairs stayed
   in the workspace. Several reviews explicitly complained that the diff did
   not implement the selected probability-handling change, while admitting
   that the current source already contained it. The controller now supplies
   the cumulative delta from an immutable parent snapshot and the original
   selected candidate contract.

Round 7 changed the measured score from 0.35683099178266087 to
0.35683099178266003. The strict comparison promoted this floating-point tie
as an improvement. Promotion and deterioration now use a numerical tolerance.
The research prompt also directs already verified output hygiene to technical
maintenance unless it resolves a recorded uncertainty relevant to the research
conclusion. No benchmark-specific predictive algorithm was added to prompts.

These findings do not establish that any particular algorithm sampled by v6
was caused by a design difference. Some earlier report passages inferred too
much from candidate coverage and graph density in individual trajectories;
those observations alone are not a controlled comparison of system designs.

## Follow-up

v11 completed 20 measured rounds at **16:49 China time on October 3**,
under source revision `5a446c8`. It stopped at the research-round limit,
with time remaining. The host reboot interrupted its first attempt; the user
confirmed another application caused the OOM. Continuation preserved the
same project and session and used the remaining original budget.

Project: `mle-lite-spooky-graphevolve-5h-20261003-v11`.
Session: `0cbc63f4435743e29f78ef9de7c42adb`.

The best and final measured log loss were both **0.3542728330169538**.
Independent reevaluation of the saved incumbent file reproduced that value
exactly. The best method used concatenated character TF-IDF (2–6 grams),
word unigrams and half-weighted word bigrams with logistic regression;
the public validation sweep selected C=16. It improved on v10, but remained
worse than MLEvolve's 0.345936 and historical v6's 0.314828.

The resumed run took about 91 minutes. Across both pre-reboot and resumed
segments there were 245 logged LLM calls, zero logged LLM errors and about
64.5 minutes of LLM request time. Repair resolved seven candidates. Four
candidate-contract rejections returned to replanning within a research
round; they did not count as measured experiments.

The measured path was:

| Round | Main change | Log loss |
| --- | --- | --- |
| 1 | Character-only reference | 0.418143 |
| 3 | Character plus word-unigram features | 0.372251 |
| 5 | Downweight word bigrams | 0.366735 |
| 6 | Word-feature ComplementNB control | 0.645893 |
| 8–13 | Separate-expert blends and normalization | 0.413378 to 0.366735 |
| 14 | Regularization-strength sweep | 0.354273 |
| 15–20 | Probability checks, further regularization, validation and serialization diagnostics | 0.354273 throughout |

The late search still spent research rounds investigating an evaluator
normalization warning. In the final incumbent CSV, the largest row-sum error
was **1.2014e-7**, within the independent evaluator's accepted tolerance.
Normalizing the rows before reevaluation gave 0.35427284002426795, a change
of approximately **7e-9**. This recorded diagnostic has negligible score
impact; it does not establish a new predictive direction. Technical warnings
still need a clearer boundary from research uncertainty, and repeated
diagnostics should not become the entire experiment trajectory.

The auxiliary monitor initially asserted a stricter row-sum tolerance than
the evaluator and assumed a categorical label column, although the private
labels use three one-hot columns. Those monitor checks were corrected to
reproduce the actual evaluator. The final completion snapshot and dashboard
now agree with the persisted completed session. No research-source change
was made during this audit.

The insults and NOMAD GraphEvolve scores above are earlier runs, not reruns
under `b01f18f`. A six-score comparison of fully completed baseline runs and
the latest controller on all three tasks therefore remains unfinished.

## v12 follow-up — October 4

The audited design fixes were committed as `b01f18f` and evaluated in a fresh,
isolated spooky project with `gpt-6-luna`, a five-hour budget and a 20-round cap.
v12 completed 20 measured rounds at 00:29:43 China time on October 4, after
4 hours 40 minutes. It stopped at the round cap with about 20 minutes remaining.
Repairs and four candidate-contract replans did not count as research rounds.

The best retained source and predictions independently reproduce
**0.32986289055145557**. This beats v11 and the historical same-backbone
MLEvolve score, but **does not restore v6's 0.314828 or reach 0.31**. The final
experiment submitted a control at 0.37399610678172257; it did not replace the
best version. v12 made 337 LLM requests (about 110 minutes), while 23 single
execution timeouts consumed about 115 minutes. Valid executions took about
55 minutes. Benchmark data and the contract remained unchanged.

Memory metadata reached the actual prompts after the fixes, but the run also
revealed unresolved failures: truncated diagnostic summaries, an incorrectly
described parent predictor, missing parent-source references in repair, and
review acceptance of a different submission target. Consequently the scores
are reproducible, while some method-change attributions are not reliable.
See [the complete v12 report](SPOOKY_V12_EVALUATION_20261003.md) for the code,
trajectory, evidence and limits. This single rerun is not a controlled estimate
of each design fix's independent effect or of performance across seeds.
