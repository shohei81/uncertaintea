# Batched Inference Design

Date: 2026-03-10

## Motivation

UncertainTea now has a good CPU reference path for single-chain `logjoint`,
gradients, HMC, multi-chain HMC, and summaries.

The next architectural step is not `NUTS`.
It is a batched execution path that makes `chain`, `particle`, or `minibatch`
dimensions explicit in the evaluator.

This is the bridge from the current CPU implementation to a future GPU-native
backend.

## Design Goal

Add a batched evaluator in phases:

1. CPU reference batching with the final public API
2. compiled batched execution on CPU
3. backend-specific lowering for `CUDA.jl` and `Metal.jl`

The public API should stabilize in phase 1 even if the implementation is still
loop-based internally.

## Batch Axis Convention

The primary convention is:

- parameter arrays use shape `num_params x batch`
- outputs use shape `batch`
- gradient outputs use shape `num_params x batch`

This matches the long-term target layout for:

- `theta[param, chain]`
- `grad[param, chain]`
- `weight[particle]`

## Phase 1 API

The initial API should be:

- `batched_logjoint(model, params, args, constraints)`
- `batched_logjoint_unconstrained(model, params, args, constraints)`
- `batched_logjoint_gradient_unconstrained(model, params, args, constraints)`
- `batched_hmc(model, args, constraints; ...)`

For repeated gradient evaluations on a fixed batch shape, phase 1 also supports
an explicit cache:

- `BatchedLogjointGradientCache(model, params, args, constraints)`
- `batched_logjoint_gradient_unconstrained!(cache, params)`

Accepted batching modes:

- shared `args::Tuple` for every batch element
- `Vector{<:Tuple}` for per-batch arguments
- shared `ChoiceMap` for every batch element
- `Vector{ChoiceMap}` for per-batch constraints

The first `batched_hmc` implementation is intentionally narrow:

- fixed-step HMC only
- shared diagonal mass matrix across the whole batch
- warmup as burn-in only
- CPU reference path built on the compiled batched evaluator

Phase 1 intentionally uses the existing single-item evaluator internally.
That gives us:

- a stable API
- clear shape conventions
- a correctness oracle for later optimized implementations

The current implementation already moves part of this work into a compiled
batched evaluator and a reusable gradient cache, but it is still a CPU
reference path rather than a GPU kernel design.

The next lowering layer is now explicit:

- `backend_execution_plan(model)` produces a symbolic primitive-only plan for
  the GPU-friendly subset
- `backend_report(model)` reports whether a model stays inside that subset
- unsupported models keep working through the compiled CPU fallback path
- batched backend execution currently requires synchronized loop iterables across
  the batch; divergent loop shapes fall back to the compiled CPU path
- backend-plan environments now split numeric slots, index slots, and true
  generic slots so arithmetic-heavy bindings can live in dense `Float64`
  storage and loop/address state can live in dense integer storage
- backend-plan distribution scoring now uses family-specific direct score kernels
  instead of constructing distribution objects on the hot path
- address construction and `1:n` loop evaluation now use an integer-specific
  evaluator instead of the generic expression path
- distribution arguments and numeric deterministic assignments now use a
  numeric-specific evaluator instead of the generic expression path
- supported choices now lower to family-specific backend plan nodes, so hot
  scoring no longer branches on a distribution family symbol
- normalized address lookup now uses a direct `ChoiceMap` helper, and concrete
  addresses are assembled with tuple recursion instead of `Any[]` accumulation
- `ChoiceMap` itself now keeps an address index, so normalized lookup is no
  longer linear in the number of stored constraints
- synchronized repeated-observation loops now have a narrow fast path when the
  loop body is a single observed choice and the address depends only on the
  iterator
- batched backend evaluation now reuses its environment, totals buffer, and
  unconstrained temporary buffers when the batch shape stays fixed
- batched gradient caches now reuse typed column caches and let `ForwardDiff`
  write directly into the shared gradient matrix
- batched HMC now reuses sampler-local momentum, proposal, diagnostics, and
  constrained-position buffers instead of reallocating them on each iteration
- backend numeric expressions for supported models now evaluate batch-wide into
  reusable scratch buffers instead of recursing separately for each column
- backend index expressions and dynamic address parts now use reusable integer
  scratch buffers on the batched path, reducing per-column address evaluation
- synchronized loop iterables now compare batch-wide evaluated range endpoints
  instead of materializing one iterable object per column before checking
- supported observed-loop scoring now batches `ChoiceMap` value lookup for each
  synchronized address and reuses a shared observed-value buffer across items
- batched supported choice scoring now fills a shared numeric choice-value
  buffer before scoring, removing per-column choice lookup calls from the hot
  scoring loop for `normal`, `lognormal`, and `bernoulli`

## Phase 2 Execution Strategy

After the public API lands, the next step is a compiled batched evaluator.

Requirements:

- no dynamic dictionaries on the hot path
- one compiled execution plan reused across the whole batch
- batched environments represented with dense arrays
- deterministic assignments and distribution arguments lowered once
- a symbolic backend plan that avoids arbitrary Julia callables on the hot path

The target abstraction is not "run the single evaluator in a loop on GPU".
The target abstraction is "evaluate one static execution plan over batched
storage".

## Phase 3 GPU Strategy

Once the batched evaluator exists on CPU, GPU support can follow by lowering the
same batched representation to:

- `CUDA.jl`
- `Metal.jl`
- possibly `KernelAbstractions.jl` for selected kernels

The initial GPU scope should stay narrow:

- continuous latent choices only
- fixed parameter dimension
- static execution plans only
- vectorized likelihoods first

## Non-Goals for the Batched Path

- dynamic trace growth
- trans-dimensional models
- arbitrary heterogeneous batch elements
- direct GPU support for the current single-trace runtime path

## Immediate Implementation Plan

1. add phase 1 batched evaluator functions in `src/`
2. add regression tests that compare batched outputs with per-column scalar calls
3. add batch-specific `args` and `ChoiceMap` tests
4. keep phase 1 implementation simple and correct
5. optimize only after the API and tests are stable
