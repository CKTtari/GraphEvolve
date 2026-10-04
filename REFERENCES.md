# MethodTrail implementation references

These references inform the implementation choices in DESIGN.md. They are not copied into MethodTrail as dependencies.

## OpenEvolve

OpenEvolve is an open-source evolutionary coding system. MethodTrail adopts three implementation ideas from its public design:

- retain multiple useful code variants rather than only one current-best program;
- return execution artifacts, including errors, to later code-generation decisions;
- use a cascade of lower-cost checks before expensive evaluation.

MethodTrail applies those ideas to graph-linked ML experiments. Its code variants are tied to hypotheses, evidence, runtime, conditions, and experiment-path relations rather than treated as an unstructured population.

Source: https://github.com/codelion/openevolve

## Awesome RSI

Awesome RSI describes recursive self-improvement across memory, harness and prompt optimization, self-modifying coding agents, automated research, and evolutionary search. MethodTrail uses the narrower artifact-level interpretation: it improves the research artifacts used to generate later experiments while keeping the orchestration kernel and execution policy under project control.

Source: https://github.com/lobehub/awesome-rsi

## Open-ended evolutionary search

Recent RSI-style coding systems converge on archive-based search rather than a
single hill-climbing chain. Darwin Gödel Machine keeps a growing tree of agent
variants and can branch from lower-scoring stepping stones; parent selection
balances measured quality with exploration. AlphaEvolve pairs an automated
evaluator with an evolutionary program database, while ShinkaEvolve adds
novelty rejection and exploration/exploitation parent sampling. These systems
do not use a fixed “stop after N bad rounds” rule as their main diversity
mechanism; they keep the best checkpoint and allocate further evaluations to
promising or underexplored archive regions until a budget or explicit stopping
condition is reached.

Sources:

- https://arxiv.org/abs/2505.22954
- https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- https://arxiv.org/abs/2509.19349

## Engineering choice

MethodTrail combines the two directions as follows:

- an outer research process selects hypotheses and experiment paths;
- an inner Coding Agent produces implementation artifacts and repair patches;
- an experiment-path graph stores code lineage and evidence relationships;
- the value model orders paths using expected gain, information value, runtime,
  execution risk, and a bounded plateau-triggered novelty/uncertainty bonus;
- the best checkpoint is retained while a persistent plateau counter increases
  exploration pressure; this counter is not a stopping counter;
- layered evaluation filters broken or incompatible code before full training.
