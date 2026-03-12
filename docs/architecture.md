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
  `log1p`, `sqrt`, `abs`, `min`, `max`, `clamp`
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
- batched gradient caches now first try a plan-aware backend gradient evaluator
  for the differentiable backend subset and only fall back to the flat
  `ForwardDiff` objective or the older column-wise cache when that subset does
  not apply
- the differentiable backend primitive subset now includes `abs`, `min`,
  `max`, `clamp`, `%` with a literal divisor, and `^` with a literal exponent
  in addition to the earlier arithmetic and log/exp primitives
- batched HMC now keeps a sampler workspace for momentum, proposals,
  diagnostics, and constrained-position scratch so repeated sampling no longer
  rebuilds those buffers every iteration
- batched HMC now also carries the current gradient state across accepted
  proposals and uses a combined value+gradient path at the final leapfrog
  position when the backend-plan-aware gradient evaluator is available
- batched HMC warmup now mirrors the single-chain structure more closely:
  dual-averaging step-size adaptation, optional reasonable-step-size search,
  divergence-aware acceptance aggregation, and windowed shared diagonal mass
  adaptation from pooled non-divergent chain positions using winsorized online
  variance updates whose clip threshold anneals from a loose early-window value
  to a tighter steady-state value, with acceptance-aware weights so rejected
  states do not count like fully refreshed samples; both schedules are now
  explicitly tied to the current slow-window length, and each slow-window
  update is surfaced as a diagnostic summary after sampling and carried into
  the high-level `summarize` output, including a richer plain-text display for
  interactive inspection
- batched NUTS now exists as a CPU reference state machine with `param x chain`
  storage and pooled warmup adaptation; the first trajectory doubling step now
  uses a batched leapfrog-and-selection step, and its result now initializes
  each chain's reusable continuation state directly; single-chain `nuts`
  initializes its continuation state through the same first-step helper, and
  both paths now seed that transition from the same reusable subtree scratch
  states; on the batched path, continuation frontier/proposal vectors and
  continuation control metadata now live in batched workspace buffers, and the
  per-chain subtree scratch itself is now view-backed by those batch-owned
  matrices; per-chain gradient caches on that path also now target
  workspace-backed gradient columns, and they reuse a shared
  `ForwardDiff` objective/config when batch inputs are homogeneous; continuation
  subtrees now also run through the batched leapfrog and batched value+gradient
  path in same-depth cohorts before falling back, with subtree energy,
  acceptance, and log-weight bookkeeping now stored in batch-owned scratch, and
  the continuation merge path now keeps proposal-selection/log-weight scratch
  there as well; turning checks on that path now also run through batched
  helpers over the workspace matrices, with frontier/proposal column copies now
  routed through masked matrix-copy helpers, and subtree start-state
  initialization now uses that same masked-copy path instead of per-chain state
  cloning; logjoint values for those batched subtree/continuation states now
  also have workspace-owned vector storage with explicit sync helpers; the
  first-step batched NUTS initialization now likewise loads `current/next`
  tree states through batched helpers and restores rejected/divergent proposal
  columns through masked batch copies before the remaining chain-local
  selection logic continues; final batched proposal summary statistics now also
  run through batch helpers over workspace vectors rather than a per-chain
  epilogue loop, and the single-chain NUTS path now uses the same acceptance /
  energy / moved-summary helper logic, with continuation proposal energy and
  energy-error values now carried as state on both paths; the remaining
  chain-local batched fallback now also records subtree summary and selected
  subtree proposal energy/error in workspace vectors before continuation merge,
  and its frontier/proposal copies plus turning checks now also go through
  one-chain masked batch helpers; the single-chain subtree builder now also
  carries its metadata through a reusable scratch summary object attached to
  `NUTSSubtreeWorkspace`, and scalar continuation merge now consumes that
  scratch through a dedicated subtree-merge helper while scalar frontier /
  proposal copies and turning updates also run through dedicated helpers, and
  direction sampling / active checks / subtree-start selection are now shared
  helper concepts across scalar and batched continuation loops; the batched
  depth-cohort scheduler now also runs through explicit reset / select /
  activate / initialize / advance / merge helpers, with continuation gating
  masks and selected cohort depth/count now stored in a dedicated scheduler
  state object on the batched workspace, and the scheduler itself now advances through explicit
  `idle/expand/merge/done` phases with a remaining-step counter; accepted /
  divergent flags, step directions, tree depths, and integration counters now
  also live in a dedicated batched NUTS control object instead of as top-level
  workspace fields, and that control object now lowers to a small
  `IdleIR`/`ExpandIR`/`MergeIR`/`DoneIR` snapshot that the scheduler step can
  dispatch through directly, with `ExpandIR`/`MergeIR` reloading the active
  control masks they need from the IR payload itself through an intermediate
  executable control-block layer, and then through a 1-step descriptor that
  names the phase-local scratch masks touched by expand/merge; a final
  step-state object then bundles that descriptor with the numeric subtree
  energy/log-weight scratch for the current batched scheduler step, and a
  kernel-frame object then makes the concrete matrix/vector buffers for that
  step explicit as well; the current CPU path now also flattens that frame
  into a phase-local kernel-access object whose fields directly expose the
  buffers touched by each step, then lowers each primitive step into a typed
  dataflow descriptor with explicit logical read/write buffer sets, alias
  classes, and a fixed intra-program dependency table, and derives a
  phase-local schedule plus buffer lifecycle metadata, resource groups, and
  barrier placements from those steps, and then lowers that metadata into a
  backend execution block with concrete buffer bindings and barrier hints, and
  then into a device-plan skeleton with segment-local slots and device-stage
  barrier hints, and finally into a target-plan layer that can choose
  target-specific allocation and barrier policy for `:gpu`, `:metal`, and
  `:cuda`, and then into a launch-plan layer with concrete argument/shared
  bindings and per-stage executor skeletons, and then into a backend
  executor-plan layer with target-specific argument classes and kernel
  symbols, and then into a codegen-plan layer with backend module symbols,
  generated entry symbols, and per-stage generated-argument descriptors, and
  then into an artifact-plan layer with backend artifact symbols and generated
  kernel layouts, and then into a source-plan layer with generated stub source
  entrypoints and backend-specific argument declarations, and then into a
  module-plan layer with materialized source blobs and backend-specific
  filenames, and then into a bundle-plan layer that groups stage modules
  behind a backend bundle manifest, and then into a package-plan layer with a
  writeable root layout and concrete file entries, and then into a reusable
  GPU-backend bundle/package-layout substrate plus an emitter stub that can
  write the generated files to disk, and
  then
  wraps that access layer in a
  small kernel program with a fixed per-phase op sequence, so the control
  skeleton is increasingly declarative, and those op sequences now feed
  phase-specialized program executors rather than a monolithic kernel-op
  interpreter, with each program also exposing a typed primitive-step table
  under the phase-specific executor; the
  remaining chain-local subtree builder also reuses a per-chain
  current/next/left/right/proposal scratch workspace, but deeper tree growth
  is still performed chain-by-chain rather than through a backend-lowered
  batched tree kernel, and subtree expansion now returns metadata while leaving
  mutable frontier state in that scratch workspace; the continuation loop
  itself now mutates an explicit reusable continuation-state object
- supported backend numeric expressions now evaluate over the whole batch using
  reusable scratch vectors, reducing per-column recursive interpretation on the
  batched path
- backend index expressions and dynamic address parts now use reusable integer
  scratch buffers, so batched address evaluation no longer reinterprets each
  index expression independently for every column
- synchronized backend loops now derive their reference iterable from
  batch-wide evaluated range endpoints instead of building one iterable per
  batch element before comparison
- observed-loop fast paths now batch constraint lookup per synchronized address
  and reuse a shared observed-value buffer instead of re-querying each
  `ChoiceMap` inside the innermost scoring loop
- supported backend choice steps now preload their batched choice values into a
  shared numeric buffer, so `normal`, `lognormal`, and `bernoulli` score
  evaluation no longer performs value lookup inside the per-column scoring loop
- unsupported expressions fall back to the compiled CPU evaluator on the
  batched path

### 4. Inference

Recommended order:

1. `ADVI/SVI`
2. importance sampling
3. `SIR/SMC`
4. batched `HMC`
5. CPU reference iterative `NUTS`
6. batched / GPU-oriented `NUTS` only after the static path is mature

Why:

- the first three are naturally batch-oriented
- batched `HMC` has regular state updates
- `NUTS` brings dynamic tree-building and adaptation complexity, so the first
  version should be an iterative CPU reference path instead of a direct GPU
  target

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
