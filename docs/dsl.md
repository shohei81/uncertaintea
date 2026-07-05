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
- `truncatednormal(mu, sigma, lower, upper)` and
  `truncatedstudentt(nu, mu, sigma, lower, upper)` renormalize the base density
  over `[lower, upper]` (infinite bounds are allowed on either side). They are
  CPU-reference only: they are honestly reported as unsupported by
  `backend_report`, but models still run through the compiled CPU logjoint and
  the batched ForwardDiff fallback. As **observations** the bounds may be any
  expression (model arguments, deterministic bindings, etc.). As **latents**
  (parameter slots sampled by HMC/NUTS) both bounds must be literal statics —
  a `Number` or `Inf`/`-Inf` — so the unconstraining transform is fixed at model
  build time: both finite uses a scaled-logit `BoundedTransform`, a single finite
  bound uses `LowerBoundedTransform`/`UpperBoundedTransform`, and two infinite
  bounds degrade to `IdentityTransform`. Declaring a truncated latent with a
  dynamic (non-literal) bound raises an `ArgumentError` at macro-expansion time.
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
