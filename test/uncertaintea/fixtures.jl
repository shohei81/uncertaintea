# Shared fixtures: models and values used by more than one test file
# (dsl_static_models_and_backend.jl and sampling.jl). test/runtests.jl includes
# this file once, before the core suite. To run any core test file standalone:
#
#   using Test, Random, UncertainTea
#   include("test/uncertaintea/fixtures.jl")
#   include("test/uncertaintea/core/<file>.jl")
#
# Everything here is deterministic: every RNG-consuming call seeds its own
# MersenneTwister, so fixture values do not depend on evaluation order.

using Test
using Random
using UncertainTea

@tea static function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))

@tea static function iid_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i in 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

ys = Float32[0.1f0, -0.2f0, 0.4f0]
repeated = choicemap((:y => i, ys[i]) for i in eachindex(ys))
trace2, logw2 = generate(iid_model, (length(ys),), repeated; rng=MersenneTwister(2))
params2 = [Float64(trace2[:mu])]

@tea static function unsupported_backend_model()
    mu ~ normal(0.0f0, 1.0f0)
    sigma = sin(mu)
    {:y} ~ normal(mu, 1.0f0)
    return sigma
end

@tea static function positive_step()
    sigma ~ lognormal(0.0f0, 0.5f0)
    return sigma
end

@tea static function observed_positive_step()
    sigma = ({:state} ~ positive_step())
    {:y} ~ normal(sigma, 1.0f0)
    return sigma
end

positive_step_trace, _ = generate(observed_positive_step, (), choicemap((:y, 1.2f0)); rng=MersenneTwister(14))
positive_step_unconstrained = transform_to_unconstrained(positive_step_trace)

gaussian_batch_params = reshape(Float64[-0.4, 0.0, 0.6], 1, 3)
gaussian_batch_constraints = [
    choicemap((:y, -0.1f0)),
    choicemap((:y, 0.3f0)),
    choicemap((:y, 0.7f0)),
]
gaussian_batch_gradient_cache = BatchedLogjointGradientCache(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)

iid_batch_params = reshape(Float64[-0.2, 0.4], 1, 2)
iid_batch_args = [(2,), (3,)]
iid_batch_constraints = [
    choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0)),
    choicemap((:y => 1, 0.5f0), (:y => 2, 0.2f0), (:y => 3, -0.1f0)),
]

positive_batch_unconstrained = reshape(
    [
        positive_step_unconstrained[1] - 0.2,
        positive_step_unconstrained[1],
        positive_step_unconstrained[1] + 0.2,
    ],
    1,
    3,
)
positive_batch_constraints = [
    choicemap((:y, 0.8f0)),
    choicemap((:y, 1.2f0)),
    choicemap((:y, 1.5f0)),
]
