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

## Engineering choice

MethodTrail combines the two directions as follows:

- an outer research process selects hypotheses and experiment paths;
- an inner Coding Agent produces implementation artifacts and repair patches;
- an experiment-path graph stores code lineage and evidence relationships;
- the value model orders paths using expected gain, information value, runtime, and execution risk;
- layered evaluation filters broken or incompatible code before full training.
