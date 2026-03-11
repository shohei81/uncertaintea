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
  even that config/objective layer be shared across chains
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
