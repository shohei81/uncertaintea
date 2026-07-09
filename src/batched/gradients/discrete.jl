# Hand-derived analytic batched logjoint gradients: discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

function _accumulate_bernoulli_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        derivative = value != 0 ? 1 / probability : -1 / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_binomial_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    trials_values::AbstractVector{Int},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
        count = _poisson_count(value)
        if isnothing(count) || count > trials
            continue
        elseif count == 0
            derivative = -trials / (1 - probability)
        elseif count == trials
            derivative = count / probability
        else
            derivative = count / probability - (trials - count) / (1 - probability)
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_poisson_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    lambda_values::AbstractVector{T},
    lambda_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        lambda = lambda_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_poisson_logpdf(lambda, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        derivative = count / lambda - 1
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * lambda_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_categorical_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::Tuple,
    probability_gradients::Tuple,
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probabilities = map(values -> values[batch_index], probability_values)
        value = value_values[batch_index]
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
        index = _categorical_index(value, length(probabilities))
        isnothing(index) && continue
        derivative = 1 / probabilities[index]
        selected_gradients = probability_gradients[index]
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * selected_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Bernoulli latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

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
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Binomial latent parameters"))
    value_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_binomial_gradient!(totals, gradients, trials_values, probability_values, probability_gradients, value_values)

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
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support categorical latent parameters"))
    value_values = env.observed_values
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    probability_gradients = ntuple(index -> _batched_backend_gradient_scratch!(cache, index), length(step.probabilities))
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr_and_gradient!(
            probability_values[index],
            probability_gradients[index],
            cache,
            env,
            probability,
            length(step.probabilities) + index,
        )
    end
    _accumulate_categorical_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, length(step.probabilities) + 1)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendPoissonChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Poisson latent parameters"))
    value_values = env.observed_values
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, step.lambda, 2)
    _accumulate_poisson_gradient!(totals, gradients, lambda_values, lambda_gradients, value_values)

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

function _accumulate_geometric_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
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
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    successes_values::AbstractVector{T},
    successes_gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
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
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Geometric latent parameters"))
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
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
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
