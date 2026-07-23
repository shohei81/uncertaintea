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

# On top of the fast path, issue #141 stages the loop's observations once per
# gradient cache (the constraints are fixed for a cache's lifetime) and, for a
# normal family whose parameter expressions are loop-INVARIANT (they never
# read the iterator slot), hoists the parameter evaluation out of the loop and
# reduces over observations per chain with a single P x B chain-rule
# application at the end: O(observations x parameters x chains) becomes
# O(observations x chains + parameters x chains). The reduction reassociates
# the observation sum, so it is tolerance-equal (~1e-12), not bitwise; when
# the parameter expressions are loop-varying the walk keeps the per-iteration
# fast path (still fed from the staged observation vector, which IS bitwise).

# Test seams: let the validation tests drive the generic walk / the
# non-hoisted loop on the same model. Always true in production.
const _BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH = Ref(true)
const _BATCHED_GRADIENT_OBSERVED_LOOP_HOIST = Ref(true)

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
    # shared constraints: stage the observation vector once per (step,
    # iterable) on the gradient cache (issue #141). Per-chain constraint
    # vectors keep the per-iteration shared-address lookup.
    observed =
        constraints isa ChoiceMap ?
        _gathered_observed_loop_values!(cache, env, step, choice, reference_iterable, constraints) : nothing

    if !isnothing(observed) &&
       _BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[] &&
       _backend_observed_loop_gradient_hoistable(choice, step.iterator_slot)
        _score_backend_observed_loop_gradient_hoisted!(choice, observed, totals, gradients, cache, env)
        return totals, gradients
    end

    # the observed choice has no parameter slot, so its value-gradient seed is
    # identically zero on every iteration: zero the shared scratch once for
    # the whole loop (the continuous per-iteration methods read it; the
    # discrete families take no value seed)
    fill!(_batched_backend_gradient_scratch!(cache, 1), zero(T))
    for (position, item) in enumerate(reference_iterable)
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        if isnothing(observed)
            address = _concrete_address(env, choice.address, 1)
            _batched_observed_choice_values!(env.observed_values, constraints, address)
        else
            fill!(env.observed_values, observed[position])
        end
        _score_backend_observed_loop_choice_gradient!(choice, totals, gradients, cache, env)
    end
    return totals, gradients
end

# Gather the loop's observed values into a vector staged on the gradient
# cache. The cache's constraints are immutable for its lifetime and the
# address depends only on the iterator, so the entry is keyed by the loop step
# and revalidated against the current iterable (a marginalize branch or
# argument-driven bound could change it between calls). The staged values are
# produced by the same normalize + convert pipeline as
# `_batched_observed_choice_values!`, so replaying them is bitwise identical
# to the per-iteration lookup.
function _gathered_observed_loop_values!(
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    step::BackendLoopPlanStep,
    choice,
    reference_iterable,
    constraints::ChoiceMap,
) where {T<:AbstractFloat}
    entry = get(cache.observed_loop_values, step, nothing)
    if entry isa Tuple{Any,Vector{T}} && entry[1] == reference_iterable
        return entry[2]
    end
    values = Vector{T}(undef, length(reference_iterable))
    for (position, item) in enumerate(reference_iterable)
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        address = _concrete_address(env, choice.address, 1)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values[position] = _batched_backend_observed_value(constrained_value, T)
    end
    cache.observed_loop_values[step] = (reference_iterable, values)
    return values
end

# --- loop-invariant hoisting + per-chain observation reduction (issue #141) ----

# Does a backend expression read the given slot? Conservative on anything the
# walker does not recognize.
_backend_expr_reads_slot(::BackendLiteralExpr, slot::Int) = false
_backend_expr_reads_slot(expr::BackendSlotExpr, slot::Int) = expr.slot == slot
_backend_expr_reads_slot(expr::BackendPrimitiveExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr::BackendTupleExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr::BackendBlockExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr, slot::Int) = true

# The loop body is a single observed choice with no binding slot, so the only
# binding that changes across iterations is the iterator: a parameter
# expression that never reads it is loop-invariant.
_backend_observed_loop_gradient_hoistable(choice, iterator_slot::Int) = false
function _backend_observed_loop_gradient_hoistable(choice::BackendNormalChoicePlanStep, iterator_slot::Int)
    return !_backend_expr_reads_slot(choice.mu, iterator_slot) &&
           !_backend_expr_reads_slot(choice.sigma, iterator_slot)
end

# Loop-invariant normal observations: evaluate mu/sigma (values and gradients)
# ONCE, reduce (sum logpdf, sum dmu, sum dsigma) over the staged observations
# per chain, then apply the parameter x chain rule once. The observed value
# gradient seed is zero, so no dvalue term appears. The per-observation terms
# are the exact `_backend_normal_logpdf` / `_accumulate_normal_gradient!`
# formulas (including NaN totals for a non-positive sigma, issue #98); only
# the summation order differs.
function _score_backend_observed_loop_gradient_hoisted!(
    choice::BackendNormalChoicePlanStep,
    observed::Vector{T},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    observation_count = length(observed)
    half_log_2pi = T(log(2 * pi) / 2)
    @inbounds for batch_index in eachindex(totals)
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        inv_sigma = 1 / sigma
        log_sigma = sigma > zero(T) ? log(sigma) : T(NaN)
        total_accumulator = zero(T)
        dmu_accumulator = zero(T)
        dsigma_accumulator = zero(T)
        @simd for position = 1:observation_count
            z = (observed[position] - mu) * inv_sigma
            total_accumulator += -log_sigma - half_log_2pi - z * z / 2
            dmu_accumulator += z * inv_sigma
            dsigma_accumulator += (z * z - 1) * inv_sigma
        end
        totals[batch_index] += total_accumulator
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dmu_accumulator * mu_gradients[parameter_index, batch_index] +
                dsigma_accumulator * sigma_gradients[parameter_index, batch_index]
        end
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
