# Observed-loop shared-address fast path for the batched gradient walker
# (issue #140): a loop whose body is a single observed scalar choice with an
# iterator-only address resolves the constraint address ONCE per iteration and
# fills all chains with one lookup (`_batched_observed_choice_values!`),
# mirroring the scoring walker's `_score_backend_observed_loop_choice!` fast
# path. The generic per-chain address build + Dict lookup in
# `_batched_choice_numeric_values!` is O(observations x chains); this path is
# O(observations). Values, gradients, and logjoint are bitwise identical to
# the generic walk (validated in
# test/uncertaintea/core/batched_observed_loop_gradient.jl).

# Test seam: lets the bitwise-identity test drive the generic walk on the same
# model. Always true in production.
const _BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH = Ref(true)

# The family set mirrors the scoring walker's `_score_backend_observed_loop_choice!`
# methods; anything else (or a non-single-observed-choice body) takes the
# generic per-iteration body walk.
_backend_observed_loop_gradient_supported(choice) = false
_backend_observed_loop_gradient_supported(
    ::Union{
        BackendNormalChoicePlanStep,
        BackendLognormalChoicePlanStep,
        BackendExponentialChoicePlanStep,
        BackendGammaChoicePlanStep,
        BackendInverseGammaChoicePlanStep,
        BackendWeibullChoicePlanStep,
        BackendBetaChoicePlanStep,
        BackendStudentTChoicePlanStep,
        BackendLaplaceChoicePlanStep,
        BackendBernoulliChoicePlanStep,
        BackendPoissonChoicePlanStep,
        BackendGeometricChoicePlanStep,
        BackendBinomialChoicePlanStep,
        BackendNegativeBinomialChoicePlanStep,
        BackendCategoricalChoicePlanStep,
    },
) = true

function _score_backend_observed_loop_and_gradient!(
    choice,
    step::BackendLoopPlanStep,
    reference_iterable,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    # the observed choice has no parameter slot, so its value-gradient seed is
    # identically zero on every iteration: zero the shared scratch once for
    # the whole loop (the continuous per-iteration methods read it; the
    # discrete families take no value seed)
    fill!(_batched_backend_gradient_scratch!(cache, 1), zero(T))
    for item in reference_iterable
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        address = _concrete_address(env, choice.address, 1)
        _batched_observed_choice_values!(env.observed_values, constraints, address)
        _score_backend_observed_loop_choice_gradient!(choice, totals, gradients, cache, env)
    end
    return totals, gradients
end

# --- per-family per-iteration accumulate ---------------------------------------
#
# Each method mirrors its `_score_backend_step_and_gradient!` counterpart with
# the same scratch depths, minus the per-chain value gather (the driver has
# already filled `env.observed_values`) and the binding-slot assignment (the
# loop-observed contract guarantees `binding_slot === nothing`). The
# value-gradient seed at gradient scratch depth 1 is pre-zeroed by the driver.

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    return _accumulate_normal_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendLognormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    return _accumulate_lognormal_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendExponentialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    rate_values = _batched_numeric_scratch!(env, 1)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, choice.rate, 3)
    return _accumulate_exponential_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        rate_values,
        rate_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    rate_values = _batched_numeric_scratch!(env, 2)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, choice.rate, 5)
    return _accumulate_gamma_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        rate_values,
        rate_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_inversegamma_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendWeibullChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_weibull_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBetaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    alpha_gradients = _batched_backend_gradient_scratch!(cache, 2)
    beta_values = _batched_numeric_scratch!(env, 2)
    beta_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(alpha_values, alpha_gradients, cache, env, choice.alpha, 4)
    _eval_backend_numeric_expr_and_gradient!(beta_values, beta_gradients, cache, env, choice.beta, 5)
    return _accumulate_beta_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        alpha_values,
        alpha_gradients,
        beta_values,
        beta_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendStudentTChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    nu_values = _batched_numeric_scratch!(env, 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    mu_values = _batched_numeric_scratch!(env, 2)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 3)
    sigma_values = _batched_numeric_scratch!(env, 3)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 4)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, choice.nu, 5)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 7)
    return _accumulate_studentt_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        nu_values,
        nu_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendLaplaceChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_laplace_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendPoissonChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, choice.lambda, 2)
    return _accumulate_poisson_gradient!(totals, gradients, lambda_values, lambda_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendGeometricChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_geometric_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_index_value_expr!(trials_values, env, choice.trials, 2)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_binomial_gradient!(
        totals,
        gradients,
        trials_values,
        probability_values,
        probability_gradients,
        env.observed_values,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    successes_values = _batched_numeric_scratch!(env, 1)
    successes_gradients = _batched_backend_gradient_scratch!(cache, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(successes_values, successes_gradients, cache, env, choice.successes, 3)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 4)
    return _accumulate_negativebinomial_gradient!(
        totals,
        gradients,
        successes_values,
        successes_gradients,
        probability_values,
        probability_gradients,
        env.observed_values,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendCategoricalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(choice.probabilities))
    probability_gradients = ntuple(index -> _batched_backend_gradient_scratch!(cache, index), length(choice.probabilities))
    for (index, probability) in enumerate(choice.probabilities)
        _eval_backend_numeric_expr_and_gradient!(
            probability_values[index],
            probability_gradients[index],
            cache,
            env,
            probability,
            length(choice.probabilities) + index,
        )
    end
    return _accumulate_categorical_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end
