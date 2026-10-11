# GraphEvolve v0.18 · lightweight local evaluation

Completed on 2026-10-08; saved predictions for both systems independently
re-evaluated on 2026-10-11. All six re-evaluated scores match their saved records.

Both systems used `gpt-6-luna`. Each comparison uses the same task split and
independent evaluator. This is a local evaluation of three selected MLE-bench
Lite tasks, not an official MLE-bench leaderboard score.

| Task | Metric | MLEvolve reference | GraphEvolve best / final retained | Completed research rounds |
| --- | --- | ---: | ---: | ---: |
| spooky-author-identification | multiclass log loss ↓ | 0.345936049 | 0.323023329 | 20 |
| detecting-insults-in-social-commentary | ROC-AUC ↑ | 0.916839842 | 0.923632435 | 20 |
| nomad2018-predict-transparent-conductors, CSV-only | mean-column-wise RMSLE ↓ | 0.063968260 | 0.059997761 | 20 |

GraphEvolve's retained result is better than the saved MLEvolve reference on
each task. The relative changes are 6.6234%, 0.7409% and 6.2070%, respectively.
Insults also has an absolute AUC increase of 0.6793 percentage points.

The equal-weight mean relative change is **4.5238%**. It is computed as
`(reference - result) / reference` for losses and
`(result - reference) / reference` for AUC, then averaged across the three tasks.
This is a descriptive, project-defined summary; it is not an official metric.
Raw scores from different tasks are not averaged.

## Comparison conditions

- MLEvolve was reproduced locally with the same LLM backbone. Spooky and
  NOMAD reference runs completed; the Insults reference saved its best valid
  submission after 3 of 8 planned steps and was interrupted. It is a partial-run
  reference, not a completed baseline evaluation.
- GraphEvolve used read-only cards from its earlier experiments. These runs
  are not cold starts; historical evidence and different search lengths prevent
  attributing the score differences solely to the system design.
- NOMAD uses the prepared 90/10 CSV split for both systems and omits crystal
  geometry files. It is a CSV-only evaluation, not the full competition setup.
- These are individual completed runs, not multi-seed estimates of expected
  performance. No statistical-significance or equal-compute claim is made.
- Research rounds include completed, independently scored experiments. Code
  repair and candidate replanning are recorded separately.
- The retained checkpoint is selected using repeated evaluator feedback;
  these results do not constitute a separate untouched final test estimate.

## Evidence and reproduction

The public repository contains task contracts, preparation instructions and
independent evaluators. Generated predictions, datasets, private labels,
worktrees and LLM logs remain local and are excluded from Git.

Local evidence consists of candidate metadata, session trajectories and saved
`predictions.csv` for both systems. Re-evaluation runs the same task evaluator
on those saved predictions. Historical project IDs are preserved verbatim for
traceability; the public release is named v0.18.

See [README.md](README.md) for the three adapters. MLEvolve's submission-format
task descriptions are retained alongside them for baseline reproduction.
