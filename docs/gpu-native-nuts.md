# GPU-Native NUTS Design Note

Date: 2026-03-11

## Thesis

UncertainTea should not target a recursive, dynamically allocated NUTS tree on
GPU.

The realistic target is:

- static `logjoint` and gradient evaluation
- iterative tree doubling with fixed-shape loop state
- many-chain batching over `theta[param, chain]`
- backend-specific lowering only after the batched CPU reference is stable

## Why This Direction

The ecosystem points in the same direction.

- JAX lowers `lax.while_loop` to a single loop primitive, but loop-carried
  state must keep a fixed shape and dtype.
- NumPyro and BlackJAX use iterative control flow and accelerator-friendly
  chain parallelism instead of a pointer-heavy runtime tree representation.
- Stan's GPU story accelerates likelihood evaluation, not the full dynamic NUTS
  control flow.

Primary references:

- [JAX `lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html)
- [NumPyro HMC utilities](https://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc_util.html)
- [NumPyro MCMC implementation](https://num.pyro.ai/en/0.15.3/_modules/numpyro/infer/mcmc.html)
- [BlackJAX NUTS](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
- [TensorFlow Probability `NoUTurnSampler`](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler)
- [Stan OpenCL guide](https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html)

## Practical Implications For UncertainTea

The implementation path should be:

1. CPU reference iterative NUTS
2. sampler state shaped for batched execution
3. backend-lowered batched NUTS for the GPU-friendly subset
4. `Metal.jl` / `CUDA.jl` lowering only after the batched state machine is stable

This means:

- no recursive tree node allocation on the hot path
- no host-side trace mutation during tree growth
- no requirement that arbitrary Julia control flow runs inside the sampler loop

## Current CPU Reference Scope

The current CPU NUTS path is intentionally narrow:

- single-chain and multi-chain wrappers
- a batched reference wrapper with `param x chain` state and pooled warmup
- a batched first doubling step, so the first narrowing step away from a
  chain-local implementation is already in place even for deeper trees
- direct initialization of reusable continuation-state objects from that first
  doubling step, so the frontier handoff no longer needs a separate staged
  frontier representation
- the single-chain `nuts` reference path now uses the same first-step
  continuation initialization logic before it enters the remaining
  chain-local tree-expansion loop
- that shared first-step path now also seeds the mutable subtree scratch state
  directly, so the remaining CPU reference control flow does not need fresh
  `NUTSState` allocation just to enter continuation
- the batched reference path now keeps continuation frontier/proposal vectors
  and continuation control metadata in batch-owned buffers, leaving mainly the
  `ForwardDiff` config/objective as chain-local state while subtree scratch and
  gradient output storage are batch-owned; homogeneous batch inputs now let
  even that config/objective layer be shared across chains, and continuation
  subtrees now stay on the batched leapfrog and batched value+gradient path
  cohort by cohort whenever multiple active chains share a tree depth; subtree
  energy and acceptance/log-weight bookkeeping for that path also now live in
  batch-owned scratch, the continuation merge step now keeps its
  proposal-selection/log-weight scratch there too, turning checks now run
  through batched helpers over the workspace matrices, and frontier/proposal
  buffer copies plus subtree start-state initialization now flow through
  masked matrix-copy helpers; the same path now also keeps subtree and
  continuation logjoint values in workspace-owned vectors with explicit sync
  points for the reference scalar state objects, and the first batched NUTS
  step now uses the same batch-owned load/restore helpers before the remaining
  chain-local selection logic runs; the final proposal summary path now also
  uses batch helpers for energy, acceptance, and moved-state aggregation, while
  the scalar reference path now shares the same summary formulas, and both
  paths now keep proposal energy/error as continuation-state data rather than
  recomputing them at the very end; even the remaining chain-local batched
  fallback now stages subtree summary and proposal-energy metadata in
  workspace-owned vectors before merge, and its frontier/proposal copies plus
  turning check now also run through one-chain masked batch helpers; the
  single-chain subtree builder now mirrors that shape by keeping subtree
  metadata in reusable scratch attached to `NUTSSubtreeWorkspace`, and its
  continuation merge now consumes that scratch through a dedicated helper while
  frontier/proposal copies and merged-turning updates also run through their
  own scalar helpers; direction sampling, active checks, and subtree-start
  selection are now helperized across scalar and batched continuation loops,
  and the batched depth-cohort scheduler now also stages reset/depth-select/
  cohort-activate/initialize/advance/merge through explicit helpers, with
  continuation-active and subtree-started masks plus selected depth/count kept
  in a dedicated scheduler state on the batched workspace, and with an explicit
  `idle/expand/merge/done` scheduler phase plus remaining-step counter; the
  per-chain accepted/divergent flags, sampled directions, and tree-depth /
  integration counters now also live in a dedicated control state object, which
  can in turn be snapshotted as a small control IR
  (`IdleIR`/`ExpandIR`/`MergeIR`/`DoneIR`) for scheduler-step dispatch, with
  the expand/merge steps reloading their active masks from that IR payload
  through an explicit executable control-block layer and then a 1-step
  descriptor for the phase-local scratch masks they touch; the current CPU
  path now also makes the phase-local numeric energy/log-weight scratch
  explicit as a step-state object layered on top, and then exposes the
  concrete numeric matrix/vector buffers for one subtree step as a kernel
  frame built from that state, then flattens that frame into a phase-local
  kernel-access object whose fields directly name the buffers touched by the
  step, then lowers each primitive kernel step into a typed dataflow
  descriptor with explicit logical read/write buffer sets, alias classes, and
  a fixed intra-program dependency table, derives a phase-local schedule and
  buffer lifecycle metadata from those descriptors, plus a small
  kernel-program wrapper with a fixed per-phase op sequence whose execution
  now runs through
  phase-specialized program handlers instead of a single generic op loop, with
  a typed primitive-step table underneath each phase program
- per-chain current/next subtree scratch for the remaining CPU reference tree
  expansion, reducing integration-step allocations while the control flow is
  still chain-local
- scratch-backed subtree frontier/proposal states, so subtree growth no longer
  clones those state vectors at every intermediate integration step
- metadata-only subtree returns, so the remaining CPU reference control flow is
  a little closer to an explicit mutable state machine
- reusable continuation-state objects for the chain-local loop, so the control
  flow no longer threads a large frontier tuple through every continuation step
- iterative subtree doubling
- multinomial proposal selection
- dual-averaging step-size adaptation
- the same diagonal mass-matrix warmup machinery used by HMC
- existing summary, R-hat, ESS, and diagnostics reuse

It is a reference implementation for correctness and API shape.
It is not yet the GPU implementation.

## Planned GPU Subset

The first GPU-native NUTS target should assume:

- continuous latent choices only
- fixed parameter dimension
- static backend-lowered execution plans
- shared diagonal mass matrix across a batch of chains
- fixed `max_tree_depth`

This is compatible with the existing batched evaluator direction and avoids
reintroducing dynamic trace semantics into the hot path.
