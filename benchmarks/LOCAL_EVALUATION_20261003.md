# Local evaluation and v10 interruption audit — 2026-10-03

These are lightweight local evaluations, not official MLE-bench aggregate
scores. Metrics with different units are reported separately.

| Task | Same-backbone MLEvolve RSI baseline | GraphEvolve latest measured result | Comparison |
| --- | --- | --- | --- |
| spooky (log loss, lower better) | 0.345936 | v10: 0.356831; 7 measured rounds, interrupted during round 8 | GraphEvolve loses |
| insults (ROC-AUC, higher better) | 0.916840; interrupted after 3/8 steps | 0.912351; 12 measured rounds | GraphEvolve loses; baseline run incomplete |
| NOMAD (mean-column-wise RMSLE, lower better) | 0.063968; 8 steps | 0.062517; 6 measured rounds | GraphEvolve wins on the CSV-only task |

The three-win target has not been achieved. Historical spooky v6 remains
stronger at 0.314828; v9 measured 0.376901. v10's saved incumbent prediction
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

The repaired controller is to be evaluated in a new isolated project, with a
fresh five-hour budget and 20 research rounds. v10's artifacts, failures and
scores remain intact. A follow-up run is pending until its first independently
measured result is available; it must not be reported as completed or as a
performance improvement based only on regression tests.
