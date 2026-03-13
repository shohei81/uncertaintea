# UncertainTea

UncertainTea is an experimental Julia probabilistic programming package with a
Gen-like frontend and a static execution model designed for GPU-friendly
backends.

The project is built around one constraint: keep model structure static enough
that `logjoint`, batched chains or particles, and parameter transforms can run
over dense layouts and backend-friendly control flow. The CPU reference runtime
is available today; GPU lowering and code emission are under active development.

## Status

UncertainTea `0.1.0` is an experimental release.

- The static DSL, CPU evaluation path, and several inference algorithms are
  implemented.
- GPU work currently focuses on backend lowering, support checks, and package
  emission for a supported static subset.
- APIs and model restrictions may change as the static IR and backend contract
  continue to converge.

## What It Provides

- Gen-like modeling with `@tea` and `@tea (static)`, tilde syntax, explicit
  addresses, hierarchical addresses, and external conditioning via `choicemap`
- Static model introspection through `modelspec`, `parameterlayout`,
  `executionplan`, and backend reports
- CPU reference evaluation with `generate`, `assess`, `logjoint`,
  unconstrained transforms, and batched logjoint/gradient APIs
- Inference methods including `hmc`, `nuts`, `hmc_chains`, `nuts_chains`,
  `batched_hmc`, `batched_nuts`, `batched_advi`,
  `batched_importance_sampling`, `batched_sir`, and `batched_smc`
- Experimental GPU-oriented lowering and code emission helpers such as
  `backend_report`, `backend_execution_plan`, and `emit_backend_package`

## Installation

UncertainTea currently targets Julia 1.10+.

```julia
using Pkg
Pkg.add(url="https://github.com/shohei81/uncertaintea.git")
```

For local development:

```julia
using Pkg
Pkg.develop(path="/path/to/uncertaintea")
```

## Quick Start

```julia
using Random
using UncertainTea

@tea (static) function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))

trace, logw = generate(gaussian_mean, (), constraints; rng=MersenneTwister(1))
params = parameter_vector(trace)
joint = logjoint(gaussian_mean, params, (), constraints)

chains = hmc_chains(
    gaussian_mean,
    (),
    constraints;
    num_chains=4,
    num_samples=100,
    num_warmup=100,
    step_size=0.2,
    num_leapfrog_steps=8,
    rng=MersenneTwister(2),
)

summary = summarize(chains)
println(trace[:mu])
println(logw)
println(joint)
println(summary.parameters[1].mean)
```

## Current Direction

UncertainTea is intentionally not centered on Turing compatibility or
unrestricted dynamic traces. The main path is:

- Gen-like surface syntax
- static semantics
- dense parameter layouts
- CPU reference first, GPU backends second

The current built-in distribution set includes `normal`, `lognormal`,
`laplace`, `exponential`, `gamma`, `inversegamma`, `weibull`, `beta`,
`dirichlet`, diagonal `mvnormal`, `bernoulli`, `binomial`, `geometric`,
`negativebinomial`, `poisson`, `studentt`, and `categorical`.

## Documentation

- [Documentation index](docs/README.md)
- [Research notes](docs/research.md)
- [Architecture direction](docs/architecture.md)
- [Minimal DSL proposal](docs/dsl.md)
- [Batched inference design](docs/batched-inference.md)
- [GPU-native NUTS notes](docs/gpu-native-nuts.md)
- [Vector backend lowering notes](docs/vector-backend-lowering.md)
- [Repository agent guide](AGENTS.md)

## License

Apache 2.0
