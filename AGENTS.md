# AGENTS.md

## Project Goal

UncertainTea is a GPU-native probabilistic programming project in Julia.
The intended user experience is Gen-like syntax with a static backend that compiles to GPU-friendly execution.

## Current Phase

The CPU reference runtime is implemented: the static DSL, compiled `logjoint`
and gradients, HMC/NUTS/SMC/ADVI inference, batched scoring with analytic
gradients, and a range of distributions and transforms. GPU work has moved from
source emission to a real device backend — KernelAbstractions kernels (with a
Metal extension) for the batched logjoint, gradient, and HMC/ADVI inner loops.

Before making changes, read these files:

- [docs/research.md](docs/research.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/dsl.md](docs/dsl.md)
- [docs/device-backend.md](docs/device-backend.md) — the KernelAbstractions device path

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
- Keep large subsystems split across focused source files; do not regrow monolithic files like `src/inference.jl`
- Aim to keep individual source and test files under roughly 1000 lines
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
- Keep large test suites split across multiple files instead of regrowing `test/runtests.jl`
- Name test files under `test/uncertaintea/core/` after what they cover
  (e.g. `dist_truncated.jl`, `device_masked_nuts.jl`), never after the PR that
  introduced them; a new feature's tests go into the topically matching file or
  a new descriptively named one
- CI (.github/workflows/ci.yml) shards the suite via `UNCERTAINTEA_TEST_GROUP`
  (groups defined in `test/uncertaintea/core.jl`); a plain local `Pkg.test()`
  still runs everything. Register a new test file in the `core_test_files`
  list with its topical group (the suite errors on unregistered files)
