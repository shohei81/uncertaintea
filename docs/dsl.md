# Minimal DSL Proposal

Date: 2026-03-10

## Design Goal

The DSL should feel recognizably close to official Gen syntax, while still compiling
to a static GPU-friendly execution plan.

The key principle is:

`Gen-like syntax, static GPU semantics`

This document intentionally corrects the earlier draft after checking the official
Gen documentation.

## What "Closer to Gen" Means

The minimal UncertainTea DSL should borrow these ideas directly:

- `@tea` and `@tea (static)` in the role that `@gen` and `@gen (static)` play in Gen
- tilde syntax for random choices
- explicit choice addresses with `{...}`
- hierarchical addresses with `=>`
- external conditioning through `choicemap` and `generate`

It should not center the language around `@observe`.

## Minimal Surface Syntax

### Scalar Model

```julia
@tea (static) function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))
(trace, logw) = generate(gaussian_mean, (), constraints)
```

Semantics:

- `mu ~ normal(...)` creates a random choice, binds it to `mu`, and uses `:mu` as the implicit address
- `{:y} ~ normal(...)` creates an explicitly addressed choice without introducing a named local binding
- `generate` conditions the model with externally supplied constraints

### Explicit Address on the Left-Hand Side

```julia
@tea (static) function latent_scale_model()
    mu ~ normal(0.0f0, 1.0f0)
    log_sigma ~ normal(0.0f0, 1.0f0)
    sigma = exp(log_sigma)
    y = ({:y} ~ normal(mu, sigma))
    return (; mu, sigma, y)
end
```

This style stays close to Gen's explicit addressed choice syntax.

### Repeated Choices

```julia
@tea (static) function iid_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i in 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

constraints = choicemap((:y => i, ys[i]) for i in eachindex(ys))
(trace, logw) = generate(iid_model, (length(ys),), constraints)
```

This is intentionally closer to Gen than the earlier `@plate`-based draft.

In the static GPU path, the loop extent `n` must be compile-time constant or part of
the shape-specialized execution cache.

### Broadcast (Vectorized) Observations

A dot-call on the right-hand side of `~` scores N observations as ONE dense vector
choice instead of N loop-addressed choices — this is the flagship GPU-lowering form:

```julia
@tea (static) function regression(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

constraints = choicemap(:y => ys)          # ONE vector value at address (:y,)
(trace, logw) = generate(regression, (xs,), constraints)
```

Semantics:

- The choice has a single address; its value is a `Vector`.
- Each element scores against the broadcast-elementwise arguments. Arguments may be
  real scalars or vectors of the observation's length (scalar-or-N broadcast only);
  a length mismatch throws at scoring time.
- Only `normal.(...)` is currently supported. Dot-calling any other distribution
  family throws an `ArgumentError` at macro-expansion time.
- **Generate requires a length source**: sampling the choice (i.e. running the model
  with the observation unconstrained) needs at least one vector argument to infer
  the sample length. With all-scalar arguments the observation must be constrained,
  otherwise `generate` throws an informative `ArgumentError`.
- Backend lowering emits a native `BackendBroadcastNormalChoicePlanStep`, so
  `backend_report(model).supported == true` and the batched logjoint / manual
  gradient paths score the vector observation densely per batch column.

### `iid` Latent Vectors

`iid(dist_call, n)` declares a latent vector of `n` independent draws from a scalar
distribution under a single address (and a single `n`-wide parameter slot):

```julia
@tea (static) function factors()
    eps ~ iid(normal(0.0f0, 1.0f0), 12)      # 12-wide slot, VectorIdentityTransform
    scales ~ iid(lognormal(0.0f0, 1.0f0), 3) # 3-wide slot, VectorLogTransform
    return eps
end
```

Rules:

- **`n` must be a literal `Int`** — a non-literal count throws an `ArgumentError`
  at macro-expansion time. This holds for latents and observations alike (kept
  literal-only for simplicity).
- The per-element transform follows the base family: `normal`/`laplace`/`studentt`
  use `VectorIdentityTransform(n)`; `lognormal`/`exponential`/`gamma`/
  `inversegamma`/`weibull` use `VectorLogTransform(n)`; `beta` uses
  `VectorLogitTransform(n)`.
- `iid` may also appear as an observation (constrain a length-`n` vector at the
  address; no parameter slot is created for observations).
- `iid` latents currently run through the compiled/AD fallback rather than the
  backend-native batched path; `backend_report` reports them honestly as
  unsupported.

### Nested Calls

```julia
@tea (static) function step(prev)
    z ~ normal(prev, 1.0f0)
    return z
end

@tea (static) function chain_model(T)
    z = ({:z => 1} ~ step(0.0f0))
    for t in 2:T
        z = ({:z => t} ~ step(z))
    end
    return z
end
```

This preserves the "generative function call at an address" flavor from Gen.

## Conditioning Model

The minimal conditioning model should follow Gen's style:

- models declare random choices
- data is supplied through constraints
- the runtime provides `choicemap`, `generate`, `assess`, and replay

Possible future sugar:

- `condition(model, constraints)`
- helpers for turning named tuples or arrays into structured choicemaps

But the core semantics should stay external, not inline.

## Static Semantics for `@tea (static)`

The initial GPU-targeted subset should require:

- the set of choices is fixed for a compiled execution plan
- each choice has a fixed shape
- each address is statically enumerable
- loop bounds are static or shape-specialized
- no recursion
- no trans-dimensional structure
- no data-dependent creation of new addresses

Allowed:

- arithmetic and deterministic local computation
- simple control flow that does not change the set of choices
- repeated structure with statically enumerable addresses

Rejected or sent to CPU fallback:

- data-dependent growth of the trace
- runtime-generated address structure
- variable-length latent structure
- dynamic control flow that changes which choices exist

## Address Rules

The initial static mode should accept:

- implicit symbol addresses from `x ~ dist(...)`
- explicit addresses like `{:x}`
- hierarchical addresses like `{:layer => 1 => :weight}`
- repeated addresses generated from static loops, such as `{:y => i}`
- tuple addresses such as `{(:y, i)}`, which flatten to the same normalized
  parts as `{:y => i}`

The address language should lower into a normalized `AddressSpec` rather than a
runtime dictionary key scheme.

## Lowering Model

The examples above should lower into something conceptually like:

```julia
ModelSpec(
    choices = [
        ChoiceSpec(:mu, Normal(), shape=(), constraint=Identity()),
        ChoiceSpec(:y, Normal(), shape=(), constraint=Identity())
    ],
    layout = ParameterLayout(offsets=(mu=1,)),
    return_plan = ...
)
```

The backend should then provide:

- `initial_trace(spec, rng, batch_shape)`
- `generate(spec_or_model, args, constraints)`
- `assess(spec_or_model, args, constraints)`
- `logjoint(spec, unconstrained_params, constraints, args)`
- `replay(spec, unconstrained_params, args)`

`assess` requires a complete choicemap covering every choice in the model and
returns the full log density; it throws if any choice is missing.

## Distribution Interface

The initial GPU-targeted distribution set should stay small:

- `normal`
- `lognormal`
- `laplace`
- `exponential`
- `gamma`
- `inversegamma`
- `weibull`
- `beta`
- `dirichlet`
- `bernoulli`
- `binomial`
- `geometric`
- `negativebinomial`
- `poisson`
- `studentt`
- `categorical`
- `truncatednormal`
- `truncatedstudentt`
- `mixture`
- a restricted diagonal `mvnormal`
- `mvnormaldense` (dense covariance via a Cholesky factor)
- `lkjcholesky` (LKJ prior over correlation Cholesky factors)
- simple transformed distributions

Requirements:

- backend-compatible `logpdf`
- clear batch semantics
- a well-defined constrained/unconstrained transform when needed
- `binomial` is treated as a built-in distribution inside `@tea`, but direct
  constructor calls outside the DSL should use `UncertainTea.binomial(...)`
  because `Base` already defines a different `binomial`
- `dirichlet` now supports static simplex sizes in both the CPU reference path
  and the current backend-native static subset, with unconstrained HMC/NUTS
  flowing through a simplex transform
- `mvnormal` currently supports static diagonal vector sizes in the CPU
  reference path and unconstrained HMC/NUTS through a vector-valued identity
  transform, and restricted diagonal forms now lower to the backend-native
  subset as the first vector-valued built-in family
- `mvnormaldense(mu, scale_tril)` is the dense-covariance multivariate normal in
  Cholesky parameterization: `scale_tril` is a d×d lower-triangular factor `L`
  with strictly positive diagonal and covariance `L * L'`. Only the lower
  triangle of `scale_tril` is read — any upper-triangular content is ignored —
  so a full runtime matrix (a model argument or deterministic binding) works
  without wrapping. It is CPU-reference only (honestly reported unsupported by
  `backend_report`; the batched path uses the ForwardDiff fallback). As a
  **latent** (parameter slot sampled by HMC/NUTS) the mean must have a
  statically known length (vector literal/tuple), mirroring the diagonal
  `mvnormal` rule; with a non-static mean it is observation-only (no slot).
- `lkjcholesky(d, eta)` is the LKJ prior over the Cholesky factor of a `d`×`d`
  correlation matrix, scored on the column-major **packed** lower triangle
  (length `d*(d+1)/2`, diagonal included) so the value stays a flat vector. The
  dimension `d` must be a literal integer `>= 2` — for latents and observations
  alike; a non-literal `d` raises an `ArgumentError` at macro-expansion time.
  `eta` may be any expression. Latents flow through `CholeskyCorrTransform`
  (Stan's canonical partial correlation parameterization, `d*(d-1)/2`
  unconstrained coordinates), so HMC/NUTS explores exactly the below-diagonal
  free coordinates. The family is CPU-reference only (honestly reported
  unsupported by `backend_report`; batched calls use the ForwardDiff fallback).
  The `scale_cholesky(scales, packed_corr_chol)` helper un-packs the factor and
  scales row `i` by `scales[i]`, producing the dense lower-triangular
  `scale_tril` for `mvnormaldense`; it is a plain function usable as a
  deterministic binding inside `@tea`. A hierarchical covariance prior then
  reads:

  ```julia
  @tea static function hierarchical_cov_model(zero_mean, n)
      Omega ~ lkjcholesky(3, 2.0)              # packed correlation Cholesky
      tau ~ iid(lognormal(0.0f0, 0.3f0), 3)    # per-dimension scales
      Ltril = scale_cholesky(tau, Omega)       # dense scale_tril = diag(tau) * L
      for i in 1:n
          {:y => i} ~ mvnormaldense(zero_mean, Ltril)
      end
      return Omega
  end
  ```
- `truncatednormal(mu, sigma, lower, upper)` and
  `truncatedstudentt(nu, mu, sigma, lower, upper)` renormalize the base density
  over `[lower, upper]` (infinite bounds are allowed on either side). As
  **observations** the bounds may be any expression (model arguments,
  deterministic bindings, etc.), and both families lower to the backend-native
  batched path with analytic gradients (`backend_report(model).supported ==
  true`) — with one restriction for `truncatedstudentt`: the normalizer uses the
  regularized incomplete beta, whose `nu`-derivative has no closed form, so the
  backend-native (and ForwardDiff) gradient is only available when `nu` is a
  **constant** (a literal degrees-of-freedom). A latent- or argument-flowing `nu`
  is honestly reported unsupported and runs through the compiled CPU logjoint
  fallback (value only; its gradient is intractable). As **latents** (parameter
  slots sampled by HMC/NUTS) both truncated families draw through a bounded
  parameter transform not implemented in the batched backend, so they fall back
  to the ForwardDiff column path; both bounds must be literal statics — a `Number`
  or `Inf`/`-Inf` — so the unconstraining transform is fixed at model build time:
  both finite uses a scaled-logit `BoundedTransform`, a single finite bound uses
  `LowerBoundedTransform`/`UpperBoundedTransform`, and two infinite bounds degrade
  to `IdentityTransform`. Declaring a truncated latent with a dynamic (non-literal)
  bound raises an `ArgumentError` at macro-expansion time.
- `mixture(weights, components...)` marginalizes a finite mixture with
  `logpdf(mix, x) = logsumexp_k(log(w_k) + logpdf(component_k, x))`. The `weights`
  argument may be a literal tuple/vector or any runtime expression — including a
  latent simplex supplied by a `dirichlet` slot — and is validated (nonnegative,
  summing to 1 within `1e-8`, one per component). Components are inline
  distribution constructor calls with fixed families. Mixtures are CPU-reference
  only (honestly reported unsupported by `backend_report`, but they still run
  through the compiled CPU logjoint and the batched ForwardDiff fallback). As
  **observations** the components may be any families. As **latents** (parameter
  slots sampled by HMC/NUTS) every component must be a real-line location-scale
  family (`normal`, `laplace`, `studentt`) so an `IdentityTransform` is exact;
  declaring a latent mixture with any other component family raises an
  `ArgumentError` at macro-expansion time.

## User-Defined Distributions

A distribution defined outside the package participates in `@tea` models on
the CPU reference path once registered with `register_distribution`. The
family stays honestly unsupported in `backend_report`/`device_report` (same
tier as the built-in CPU-only families), so scoring and gradients use the
compiled CPU logjoint and the ForwardDiff column fallback.

The builder must return a subtype of `AbstractTeaDistribution` implementing
`UncertainTea.logpdf(dist, x)` (return `-Inf` outside the support, and keep it
ForwardDiff-Dual-friendly if the family will be a latent) and
`Random.rand(rng::AbstractRNG, dist)`:

```julia
using UncertainTea, Random

struct SkewNormalDist{T<:Real} <: AbstractTeaDistribution
    location::T
    scale::T
    shape::T
    function SkewNormalDist(location::T, scale::T, shape::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("skewnormal requires scale > 0"))
        return new{T}(location, scale, shape)
    end
end

skewnormal(location, scale, shape) = SkewNormalDist(promote(location, scale, shape)...)

function UncertainTea.logpdf(dist::SkewNormalDist, x)
    z = (x - dist.location) / dist.scale
    return log(2) - z^2 / 2 - log(2 * pi) / 2 - log(dist.scale) +
           log(UncertainTea._std_normal_cdf(dist.shape * z))
end

function Random.rand(rng::AbstractRNG, dist::SkewNormalDist)
    delta = dist.shape / sqrt(1 + dist.shape^2)
    u0, v = randn(rng), randn(rng)
    return dist.location + dist.scale * (delta * abs(u0) + sqrt(1 - delta^2) * v)
end

register_distribution(:skewnormal; builder=skewnormal, transform=IdentityTransform())

@tea static function model()
    x ~ skewnormal(0.0, 1.0, 3.0)
    {:y} ~ normal(x, 0.5)
    return x
end
```

`transform` declares the unconstrained parameterization for latent use --
`IdentityTransform()` (real line), `LogTransform()` (positive),
`LogitTransform()` ((0,1)), or `BoundedTransform(lower, upper)`. Omit it for
observation-only families; a latent then gets no parameter slot.

Rules and caveats:

- Register **before** defining models that use the family (registration is
  consulted at model definition).
- Re-registration overwrites, but already-defined models keep the builder they
  were compiled with.
- Built-in family names and expression primitives (`exp`, `log`, ...) cannot
  be registered.
- Broadcast observations (`family.(...)`) and `iid(family(...), n)` are not
  supported for registered families.

## Inference-Oriented Consequences

### VI / SVI

This syntax lowers cleanly to a fixed latent representation and batch-heavy likelihoods.
The current CPU reference runtime now exposes `batched_advi` on top of the
same static unconstrained parameter layout, and the current backend-native
vector subset (`mvnormal` diagonal, `dirichlet`) flows through that path.

### SMC

Particle axes map naturally onto the same compiled model structure.
The current reference implementation exposes `batched_importance_sampling` and
`batched_sir`, and `batched_smc` now denotes an adaptive tempered multi-stage
SMC bridge from a Gaussian proposal to the target density, with optional
batched rejuvenation moves after resampling stages. The current move kernels
are `:random_walk`, fixed-step tempered `:hmc`, and CPU-reference tempered
`:nuts`.

### HMC

Feasible after the static path exists, but batched `HMC` is a better first target than `NUTS`.

## Recommended First Milestone

The first real subset should include:

1. `@tea (static)`
2. tilde syntax
3. implicit and explicit addresses
4. hierarchical addresses with `=>`
5. `choicemap`
6. `generate`
7. a small distribution set starting with `normal`, `lognormal`,
   `laplace`, `exponential`, `gamma`, `inversegamma`, `weibull`, `beta`,
   `dirichlet`, `bernoulli`, `binomial`, `geometric`, `negativebinomial`,
   `poisson`, `studentt`, and `categorical`

That is enough to build a CPU reference backend and a GPU-oriented static lowering path
without committing to full Gen compatibility.
