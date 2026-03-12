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

function _accumulate_mvnormal_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{<:AbstractVector},
    value_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
)
    for batch_index in eachindex(totals)
        for component_index in eachindex(value_values)
            value = value_values[component_index][batch_index]
            mu = mu_values[component_index][batch_index]
            sigma = sigma_values[component_index][batch_index]
            totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dvalue = -z * inv_sigma
            dmu = z * inv_sigma
            dsigma = (z * z - 1) * inv_sigma
            component_value_gradients = value_gradients[component_index]
            component_mu_gradients = mu_gradients[component_index]
            component_sigma_gradients = sigma_gradients[component_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dvalue * component_value_gradients[parameter_index, batch_index] +
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dsigma * component_sigma_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _accumulate_dirichlet_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    parameter_index::Union{Nothing,Int},
    value_values::AbstractVector{<:AbstractVector},
    alpha_values::AbstractVector{<:AbstractVector},
    alpha_gradients::AbstractVector{<:AbstractMatrix},
)
    value_length = length(value_values)
    choice_gradients = Vector{Float64}(undef, value_length)
    alpha_derivatives = Vector{Float64}(undef, value_length)
    for batch_index in eachindex(totals)
        total_alpha = 0.0
        accumulator = 0.0
        for component_index in 1:value_length
            alpha = alpha_values[component_index][batch_index]
            alpha > 0 || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
            total_alpha += alpha
            accumulator -= loggamma(alpha)
        end
        accumulator += loggamma(total_alpha)

        total = 0.0
        valid = true
        weighted_choice_gradient = 0.0
        for component_index in 1:value_length
            value = value_values[component_index][batch_index]
            value > 0 || begin
                valid = false
                break
            end
            total += value
            alpha = alpha_values[component_index][batch_index]
            accumulator += (alpha - 1) * log(value)
            choice_gradient = (alpha - 1) / value
            choice_gradients[component_index] = choice_gradient
            weighted_choice_gradient += choice_gradient * value
            alpha_derivatives[component_index] = digamma(total_alpha) - digamma(alpha) + log(value)
        end

        if !valid || abs(total - 1) > sqrt(eps(total)) * value_length * 16
            totals[batch_index] += -Inf
            continue
        end

        totals[batch_index] += accumulator
        for component_index in 1:value_length
            alpha_derivative = alpha_derivatives[component_index]
            component_alpha_gradients = alpha_gradients[component_index]
            for parameter_row in axes(gradients, 1)
                gradients[parameter_row, batch_index] +=
                    alpha_derivative * component_alpha_gradients[parameter_row, batch_index]
            end
        end
        isnothing(parameter_index) && continue
        for component_index in 1:(value_length - 1)
            unconstrained_row = parameter_index + component_index - 1
            constrained_value = value_values[component_index][batch_index]
            gradients[unconstrained_row, batch_index] +=
                constrained_value * (choice_gradients[component_index] - weighted_choice_gradient)
        end
    end
    return totals, gradients
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment{Float64},
    slot_gradients::Array{Float64,3},
    slot::Int,
    values::AbstractVector{<:AbstractVector},
    gradients::AbstractVector{<:AbstractMatrix},
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
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
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index in 1:step.value_length]
    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, step.value_length + index) for index in 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index in 1:step.value_length]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * step.value_length + index) for index in 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index in 1:step.value_length
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index],
            mu_gradients[component_index],
            cache,
            env,
            step.mu[component_index],
            3 * step.value_length + 1,
        )
        _eval_backend_numeric_expr_and_gradient!(
            sigma_values[component_index],
            sigma_gradients[component_index],
            cache,
            env,
            step.sigma[component_index],
            3 * step.value_length + 2,
        )
    end

    _accumulate_mvnormal_gradient!(
        totals,
        gradients,
        choice_values,
        choice_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendDirichletChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    alpha_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    alpha_gradients = [
        _batched_backend_gradient_scratch!(cache, step.value_length + index) for index in 1:step.value_length
    ]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index in 1:step.value_length
        _eval_backend_numeric_expr_and_gradient!(
            alpha_values[component_index],
            alpha_gradients[component_index],
            cache,
            env,
            step.alpha[component_index],
            2 * step.value_length + 1,
        )
    end

    _accumulate_dirichlet_gradient!(
        totals,
        gradients,
        step.parameter_index,
        choice_values,
        alpha_values,
        alpha_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, alpha_gradients)
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
