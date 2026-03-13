Date: 2026-03-12

# Vector Latent Backend Lowering

## Goal

UncertainTea now supports vector-valued latent slots in the CPU reference path.
The immediate examples are:

- simplex-valued `dirichlet`
- restricted diagonal `mvnormal`

Those families already work with:

- static lowering into `ModelSpec` / `ExecutionPlan`
- constrained and unconstrained parameter conversion
- CPU reference `logjoint`
- batched fallback evaluation
- HMC and NUTS in the CPU path
- `batched_advi`, importance sampling, and SIR in the CPU reference path
- adaptive tempered `batched_smc`, including rejuvenation moves, in the CPU
  reference path, with random-walk, tempered HMC, and tempered NUTS kernels
  available
- the tempered `:nuts` rejuvenation path now batch-shares the initial
  value-and-gradient evaluation and the first continuation subtree, and only
  uses chain-local continuation for deeper tree growth

At this point:

- restricted diagonal `mvnormal` does lower into the backend-native subset
- simplex-valued `dirichlet` now also lowers into that subset for static
  concentration vectors

This note records the staged design that produced the first native vector
families and the remaining limits beyond them.

## Current State

The current parameter layout supports a distinction between:

- unconstrained parameter span
- constrained flattened value span
- per-choice transform

That is enough for vector-valued latents in the CPU path because the evaluator
can materialize whole values from a `ParameterSlotSpec` before calling
distribution `logpdf`.

The backend-native path now covers two vector families:

- restricted diagonal `mvnormal`
- static `dirichlet`

That support is still intentionally family-specific:

- vector score nodes are explicit family kernels, not generic multivariate
  distribution objects
- backend workspaces still separate numeric/index/generic slots, but vector
  choices bind through explicit value spans
- manual backend gradients remain specialized rather than deriving from a
  generic tensor AD layer

## Why `dirichlet` and `mvnormal` Differ

`mvnormal` is the simpler first target because the current built-in family is
restricted to a diagonal covariance parameterization:

- latent dimension is statically known
- constrained and unconstrained dimensions match
- score evaluation is a reduction over independent scalar normal terms

`dirichlet` is harder:

- constrained value dimension is `K`
- unconstrained dimension is `K - 1`
- the transform itself is vector-valued and contributes a nontrivial Jacobian
- backend gradients need transform-aware vector chain rules

For that reason, the first backend-native vector family was diagonal
`mvnormal`, followed by `dirichlet` once simplex-aware chain rules were added.

## Required Backend Changes

### 1. Vector-Aware Choice Nodes

The backend plan needs choice node families that carry value-span information
explicitly instead of pretending every latent slot is scalar.

Minimum requirement:

- a vector latent plan step knows its parameter slot ordinal
- it knows the constrained value span length
- its score kernel knows whether the value comes from parameters or constraints

### 2. Vector Buffer Binding

The backend workspace and lowering metadata need segment-aware bindings for:

- unconstrained parameter rows
- constrained value rows
- temporary vector scratch
- reduction outputs used by score kernels

This should stay dense and batch-major, not trace-object based.

### 3. Family-Specific Vector Kernels

The backend-native subset should not start from a generic
`Distribution <: Any -> logpdf` abstraction. It should lower supported vector
families to explicit kernels.

Initial target:

- `mvnormal(mean_vec, scale_vec)` with diagonal covariance

Later target:

- `dirichlet(alpha_vec)`

### 4. Manual Gradient Support

The current backend manual gradient path is scalar-oriented. Vector families
need:

- vector score accumulation
- vector-to-vector transform chain rules
- reduction-aware gradient writes back into dense parameter buffers

Again, `mvnormal` is the easier first step because its latent transform is an
identity map and its score is a sum of scalar normal terms.

## Proposed Phases

### Phase 1: Keep the Boundary Explicit

This phase is complete for the currently supported families:

- restricted diagonal `mvnormal`
- static `dirichlet`

### Phase 2: Backend-Native Observed Vector Likelihoods

Add observed-only vector score kernels before latent vector kernels.

This allows the backend to validate:

- vector address/value transport
- vector score accumulation
- vector scratch reuse

without immediately coupling to latent transforms.

### Phase 3: Latent Diagonal `mvnormal`

Add a backend-native latent vector slot for restricted diagonal `mvnormal`.

This is now implemented because:

- the parameter transform is a vector-valued identity
- constrained and unconstrained spans match
- score and gradient logic are comparatively simple

Current limitation after Phase 3:

- only the restricted diagonal `mvnormal` family is backend-native
- richer transformed vector families still remain CPU/fallback only

### Phase 4: Latent `dirichlet`

After vector-valued identity slots work in the backend-native path, add
simplex-valued vector slots for `dirichlet`.

This phase is now implemented for static concentration vectors through:

- backend-native simplex transforms
- Jacobian accumulation
- transform-aware manual gradients

## Non-Goals

This plan does not imply:

- arbitrary dense covariance `mvnormal`
- generic multivariate `Distributions.jl` compatibility in the core IR
- dynamic vector shapes on the GPU path

The intended rule remains:

`small optimized built-ins first, wider compatibility later`
