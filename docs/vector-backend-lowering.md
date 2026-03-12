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

At this point:

- restricted diagonal `mvnormal` does lower into the backend-native subset
- `dirichlet` still remains outside that subset

This note records the intended staged design for closing that gap.

## Current State

The current parameter layout supports a distinction between:

- unconstrained parameter span
- constrained flattened value span
- per-choice transform

That is enough for vector-valued latents in the CPU path because the evaluator
can materialize whole values from a `ParameterSlotSpec` before calling
distribution `logpdf`.

The backend-native path is not there yet because its choice nodes are still
fundamentally scalar-oriented:

- backend score nodes assume scalar distribution arguments and scalar choice
  values
- backend workspaces distinguish numeric/index/generic slots, but not
  value-span-aware vector slots
- backend manual gradients assume scalar latent choices plus observed discrete
  likelihoods

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

For that reason, the first backend-native vector family should be diagonal
`mvnormal`, not `dirichlet`.

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

Continue reporting vector latent families as outside the backend-native subset.

This is the current behavior and should remain explicit in `backend_report`.

### Phase 2: Backend-Native Observed Vector Likelihoods

Add observed-only vector score kernels before latent vector kernels.

This allows the backend to validate:

- vector address/value transport
- vector score accumulation
- vector scratch reuse

without immediately coupling to latent transforms.

### Phase 3: Latent Diagonal `mvnormal`

Add a backend-native latent vector slot for restricted diagonal `mvnormal`.

This is now the first implemented end-to-end vector target because:

- the parameter transform is a vector-valued identity
- constrained and unconstrained spans match
- score and gradient logic are comparatively simple

Current limitation inside Phase 3:

- backend-native batched scoring supports diagonal `mvnormal`
- batched gradients for that family now use the manual backend gradient path
- richer vector families such as `dirichlet` still remain CPU/fallback only

### Phase 4: Latent `dirichlet`

After vector-valued identity slots work in the backend-native path, add
simplex-valued vector slots for `dirichlet`.

That phase needs:

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
