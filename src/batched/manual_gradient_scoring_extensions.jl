function _accumulate_laplace_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    scale_values::AbstractVector{Float64},
    scale_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_laplace_logpdf(mu, scale, value)
        delta = value - mu
        sign_delta = delta > 0 ? 1.0 : (delta < 0 ? -1.0 : 0.0)
        dvalue = -sign_delta / scale
        dmu = sign_delta / scale
        dscale = -1 / scale + abs(delta) / (scale * scale)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_geometric_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    probability_values::AbstractVector{Float64},
    probability_gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
)
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_geometric_logpdf(probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        derivative = 1 / probability - count / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_negativebinomial_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    successes_values::AbstractVector{Float64},
    successes_gradients::AbstractMatrix{Float64},
    probability_values::AbstractVector{Float64},
    probability_gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
)
    for batch_index in eachindex(totals)
        successes = successes_values[batch_index]
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_negativebinomial_logpdf(successes, probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        dsuccesses = digamma(count + successes) - digamma(successes) + log(probability)
        dprobability = successes / probability - count / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dsuccesses * successes_gradients[parameter_index, batch_index] +
                dprobability * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_laplace_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, scale_values, scale_gradients)
    isnothing(step.binding_slot) || _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    isnothing(step.parameter_slot) || throw(BatchedBackendFallback("batched backend gradient does not support Geometric latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_geometric_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support NegativeBinomial latent parameters"))
    value_values = env.observed_values
    successes_values = _batched_numeric_scratch!(env, 1)
    successes_gradients = _batched_backend_gradient_scratch!(cache, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(successes_values, successes_gradients, cache, env, step.successes, 3)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 4)
    _accumulate_negativebinomial_gradient!(
        totals,
        gradients,
        successes_values,
        successes_gradients,
        probability_values,
        probability_gradients,
        value_values,
    )

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 5)),
        )
    end
    return totals, gradients
end
