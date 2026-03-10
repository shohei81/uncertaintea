# AGENTS.md

## Project Goal

UncertainTea is a GPU-native probabilistic programming project in Julia.
The intended user experience is Gen-like syntax with a static backend that compiles to GPU-friendly execution.

## Current Phase

This repository is in the design and documentation phase.
Before adding runtime code, read these files:

- [docs/research.md](docs/research.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/dsl.md](docs/dsl.md)

## Working Rules

- Preserve the core direction: `Gen-like syntax, static semantics`
- Treat GPU support as a first-class requirement, not an afterthought
- Prefer a CPU reference backend first, then add GPU backends
- Keep dynamic-trace features isolated from the static GPU path
- Do not introduce Turing compatibility constraints into the core runtime unless explicitly requested

## Documentation Rules

- If architecture assumptions change, update `docs/architecture.md`
- If ecosystem understanding changes, update `docs/research.md` with a date and source links
- If surface syntax or semantics change, update `docs/dsl.md`
- If a change affects contributor workflow, update this `AGENTS.md`
- Keep repository documentation in English
- When discussing Gen-like syntax, check the official Gen docs and use Gen terminology accurately

## Code Direction

- Prefer explicit IR types over ad hoc dictionaries
- Prefer dense parameter layouts over heterogeneous trace containers
- Prefer backend-agnostic interfaces with backend-specific implementations
- Default to ASCII unless an existing file clearly requires otherwise
- Keep comments sparse and factual

## Inference Priorities

Initial implementation priority:

1. Static model lowering
2. CPU reference `logjoint`
3. Batched VI / SVI
4. Importance sampling / SMC
5. Batched HMC

Defer by default:

- NUTS
- trans-dimensional inference
- arbitrary dynamic control flow on GPU

## Validation Expectations

- New model-lowering behavior should come with CPU-side tests
- GPU code should be validated against CPU reference behavior
- Performance claims should be backed by a reproducible benchmark or a documented measurement plan
