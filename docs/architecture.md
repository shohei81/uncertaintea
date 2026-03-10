# Architecture Direction

Date: 2026-03-10

## Core Thesis

UncertainTea should look much closer to Gen on the surface than the previous draft did.
The key idea is:

- surface syntax close to official Gen
- implementation strategy centered on a static GPU-friendly IR

In other words, the user-facing model should feel like a generative-function system,
while the compiled runtime behaves more like a static log-density compiler.

## Surface Syntax Direction

The frontend should mirror official Gen patterns wherever practical:

- `@tea` or `@tea (static)` for model definitions
- tilde syntax for random choices
- explicit addresses with `{...}`
- hierarchical addresses with `=>`
- observations supplied externally through `choicemap`-style constraints

This is a better fit for the stated goal than an `@observe`-centric DSL.

## Architectural Layers

### 1. Frontend

Responsibilities:

- parse `@tea` and `@tea (static)` definitions
- resolve tilde expressions and choice addresses
- distinguish the static GPU subset from future dynamic features
- lower surface syntax into an internal execution plan

Design rules:

- stay close to Gen terminology: generative function, choice, trace, choicemap
- reject or reroute unsupported dynamic behavior early
- keep a clear split between "Gen-like syntax" and "GPU-compatible semantics"

### 2. Static IR

Minimum internal components:

- `ModelSpec`: static specification of a compiled model
- `ChoiceSpec`: metadata for each random choice
- `AddressSpec`: normalized address representation
- `ParameterLayout`: dense unconstrained parameter layout
- `ExecutionPlan`: compiled `logjoint`, replay, and initialization plan

Core invariants:

- the set and order of choices are fixed for a compiled execution plan
- each choice has a fixed shape
- each address is statically enumerable
- the hot path can be evaluated with dense arrays and backend-friendly control flow

### 3. Backends

CPU backend:

- reference implementation
- correctness oracle
- fallback path for unsupported dynamic behavior

GPU backend:

- start with `CUDA.jl` and `Metal.jl`
- use high-level array operations first
- use `KernelAbstractions.jl` or backend-specific kernels only for hotspots
- make chain, particle, and minibatch dimensions first-class
- lower the static execution plan into a symbolic primitive-only backend plan

Current backend-lowering subset:

- distribution families: `normal`, `lognormal`, `bernoulli`
- primitive calls: `:`, `=>`, `+`, `-`, `*`, `/`, `^`, `%`, `exp`, `log`,
  `log1p`, `sqrt`, `abs`, `min`, `max`
- batched backend execution assumes synchronized loop iterables across the batch
- backend environments separate numeric slots, index slots, and generic slots
- backend score evaluation uses direct family kernels for supported distributions
- address and loop-index evaluation use an integer-specific backend path
- distribution arguments and numeric deterministic assignments use a dedicated
  numeric backend evaluator instead of the generic expression path
- supported distribution choices lower to family-specific backend plan nodes
  rather than a symbol-dispatched generic choice node
- compiled and backend evaluators now build concrete addresses through tuple
  recursion and use normalized direct `ChoiceMap` lookup on the hot path
- `ChoiceMap` now maintains an address index so normalized lookup no longer
  scans linearly through stored entries
- synchronized backend loops with a single observed choice whose address depends
  only on the loop iterator now use a loop-local fast path instead of the
  generic body scorer
- batched backend workspaces now reuse their environment and temporary buffers
  across repeated evaluations with the same batch shape
- batched gradient caches now keep monomorphic column caches and write gradients
  directly into the shared output matrix
- batched HMC now keeps a sampler workspace for momentum, proposals,
  diagnostics, and constrained-position scratch so repeated sampling no longer
  rebuilds those buffers every iteration
- supported backend numeric expressions now evaluate over the whole batch using
  reusable scratch vectors, reducing per-column recursive interpretation on the
  batched path
- backend index expressions and dynamic address parts now use reusable integer
  scratch buffers, so batched address evaluation no longer reinterprets each
  index expression independently for every column
- unsupported expressions fall back to the compiled CPU evaluator on the
  batched path

### 4. Inference

Recommended order:

1. `ADVI/SVI`
2. importance sampling
3. `SIR/SMC`
4. batched `HMC`
5. `NUTS` only after the static path is mature

Why:

- the first three are naturally batch-oriented
- batched `HMC` has regular state updates
- `NUTS` brings dynamic tree-building and adaptation complexity

## Recommended Execution Modes

### `@tea (static)`

This should be the primary mode and the main GPU target.

Requirements:

- the choice set is fixed for a compiled execution plan
- loop extents are static or shape-specialized
- addresses are statically enumerable
- distribution parameters are backend-compatible

### `@tea`

This can exist as a future, more dynamic mode.

Possible future use cases:

- data-dependent trace growth
- richer programmable inference features
- experimental dynamic generative functions

This mode should start as CPU-only fallback, not as a GPU target.

## Conditioning Model

To stay close to Gen, conditioning should be external:

- models define random choices
- observed values are supplied through a `choicemap`-like structure
- the runtime provides `generate`, `assess`, replay, and later convenience wrappers

This avoids baking a separate observed-site syntax into the core language.

## Memory and Parameter Layout

Preferred:

- a single unconstrained vector or dense `param x chain` tensor
- normalized address metadata compiled once per execution plan
- dense observed-data layouts for batch-heavy likelihoods

Avoid:

- dynamic dictionaries in the GPU hot path
- heterogeneous trace containers in performance-critical kernels
- scalar-loop-heavy `logpdf` implementations

## Proposed Package Structure

Initial direction:

- `src/UncertainTea.jl`
- `src/frontend/`
- `src/ir/`
- `src/constraints/`
- `src/choicemaps/`
- `src/backends/cpu/`
- `src/backends/cuda/`
- `src/backends/metal/`
- `src/inference/`
- `src/distributions/`
- `test/`
- `docs/`

## Near-Term Deliverables

1. define `ModelSpec`, `ChoiceSpec`, and `AddressSpec`
2. define the `choicemap` and `generate` API surface
3. implement CPU-side lowering and `logjoint`
4. implement the minimal Gen-like static DSL
5. add a batched `logjoint` reference path with a stable public API
6. lower that batched path into a compiled evaluator before GPU backend work

## Non-Goals for v0

- full Gen compatibility
- Turing-compatible runtime semantics
- unrestricted dynamic trace manipulation on the GPU
- trans-dimensional inference
