function _backend_laplace_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || throw(ArgumentError("laplace requires scale > 0"))
    return -log(2 * scale_) - abs(xx - mu_) / scale_
end

function _backend_inversegamma_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || throw(ArgumentError("inversegamma requires shape > 0"))
    scale_ > zero(scale_) || throw(ArgumentError("inversegamma requires scale > 0"))
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape_ * log(scale_) - loggamma(shape_) -
           (shape_ + one(shape_)) * log(xx) -
           scale_ / xx
end

function _backend_weibull_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || throw(ArgumentError("weibull requires shape > 0"))
    scale_ > zero(scale_) || throw(ArgumentError("weibull requires scale > 0"))
    xx < zero(xx) && return oftype(xx, -Inf)
    if xx == zero(xx)
        if shape_ < one(shape_)
            return oftype(xx, Inf)
        elseif shape_ == one(shape_)
            return -log(scale_)
        end
        return oftype(xx, -Inf)
    end
    log_ratio = log(xx) - log(scale_)
    return log(shape_) + (shape_ - one(shape_)) * log(xx) -
           shape_ * log(scale_) - exp(shape_ * log_ratio)
end

function _backend_geometric_logpdf(probability, x)
    probability_ = float(probability)
    zero(probability_) < probability_ <= one(probability_) ||
        throw(ArgumentError("geometric requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0
        return log(probability_)
    elseif probability_ == one(probability_)
        return oftype(probability_, -Inf)
    end
    return log(probability_) + count * log1p(-probability_)
end

function _backend_negativebinomial_logpdf(successes, probability, x)
    successes_, probability_ = promote(successes, probability)
    successes_ > zero(successes_) || throw(ArgumentError("negativebinomial requires successes > 0"))
    zero(probability_) < probability_ <= one(probability_) ||
        throw(ArgumentError("negativebinomial requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0 && probability_ == one(probability_)
        return zero(probability_)
    elseif probability_ == one(probability_)
        return oftype(probability_, -Inf)
    end
    return loggamma(count + successes_) - loggamma(successes_) - _logfactorial_like(probability_, count) +
           successes_ * log(probability_) + count * log1p(-probability_)
end

function _backend_beta_logpdf(alpha, beta_parameter, x)
    xx, alpha_, beta_ = promote(x, alpha, beta_parameter)
    alpha_ > zero(alpha_) || throw(ArgumentError("beta requires alpha > 0"))
    beta_ > zero(beta_) || throw(ArgumentError("beta requires beta > 0"))
    zero(xx) < xx < one(xx) || return oftype(xx, -Inf)
    return loggamma(alpha_ + beta_) - loggamma(alpha_) - loggamma(beta_) +
           (alpha_ - one(alpha_)) * log(xx) +
           (beta_ - one(beta_)) * log1p(-xx)
end

function _backend_binomial_logpdf(trials, probability, x)
    probability_ = float(probability)
    zero(probability_) <= probability_ <= one(probability_) ||
        throw(ArgumentError("binomial requires 0 <= p <= 1"))
    trial_count = _binomial_trials(trials)
    isnothing(trial_count) && throw(ArgumentError("binomial requires integer trials >= 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    count <= trial_count || return oftype(probability_, -Inf)
    log_combination = _logbinomial_like(probability_, trial_count, count)
    if count == 0 && count == trial_count
        return log_combination
    elseif count == 0
        return log_combination + trial_count * log1p(-probability_)
    elseif count == trial_count
        return log_combination + count * log(probability_)
    end
    return log_combination +
           count * log(probability_) +
           (trial_count - count) * log1p(-probability_)
end

function _backend_categorical_logpdf(probabilities::Tuple, x)
    length(probabilities) > 0 || throw(ArgumentError("categorical requires at least one probability"))
    total = zero(float(first(probabilities)))
    for probability in probabilities
        probability_ = float(probability)
        zero(probability_) <= probability_ <= one(probability_) ||
            throw(ArgumentError("categorical requires 0 <= p <= 1"))
        total += probability_
    end
    tolerance = sqrt(eps(total)) * max(length(probabilities), 1) * 8
    abs(total - one(total)) <= tolerance || throw(ArgumentError("categorical probabilities must sum to 1"))
    index = _categorical_index(x, length(probabilities))
    isnothing(index) && return oftype(total, -Inf)
    return log(float(probabilities[index]))
end

function _backend_mvnormal_observed_value(value, expected_length::Int, ::Type{T}=Float64) where {T<:Real}
    values = value isa Tuple ? collect(value) : value
    values isa AbstractVector || throw(ArgumentError("mvnormal expects a vector or tuple value"))
    length(values) == expected_length ||
        throw(ArgumentError("mvnormal expects a value of length $expected_length, got $(length(values))"))
    return T[convert(T, float(item)) for item in values]
end

function _backend_choice_vector_value(
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractVector,
    constraints::ChoiceMap,
    address,
)
    if !isnothing(value_index)
        return collect(view(params, value_index:(value_index + value_length - 1)))
    end
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
    return _backend_mvnormal_observed_value(constrained_value, value_length, Float64)
end

function _batched_choice_vector_values!(
    destination::AbstractMatrix,
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractMatrix,
    constraints::ChoiceMap,
    address_parts::Tuple,
)
    size(destination) == (value_length, size(params, 2)) ||
        throw(DimensionMismatch("expected mvnormal destination of size ($(value_length), $(size(params, 2))), got $(size(destination))"))
    if !isnothing(value_index)
        copyto!(destination, view(params, value_index:(value_index + value_length - 1), :))
        return destination
    end
    for batch_index in 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(destination))
        for component_index in 1:value_length
            destination[component_index, batch_index] = values[component_index]
        end
    end
    return destination
end

function _batched_choice_vector_values!(
    destinations::AbstractVector{<:AbstractVector},
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractMatrix,
    constraints::ChoiceMap,
    address_parts::Tuple,
)
    length(destinations) == value_length ||
        throw(DimensionMismatch("expected $value_length mvnormal component buffers, got $(length(destinations))"))
    if !isnothing(value_index)
        for component_index in 1:value_length
            copyto!(destinations[component_index], view(params, value_index + component_index - 1, :))
        end
        return destinations
    end
    for batch_index in 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(first(destinations)))
        for component_index in 1:value_length
            destinations[component_index][batch_index] = values[component_index]
        end
    end
    return destinations
end

function _batched_choice_vector_values!(
    destinations::AbstractVector{<:AbstractVector},
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractMatrix,
    constraints::AbstractVector,
    address_parts::Tuple,
)
    length(destinations) == value_length ||
        throw(DimensionMismatch("expected $value_length mvnormal component buffers, got $(length(destinations))"))
    length(constraints) == size(params, 2) ||
        throw(DimensionMismatch("expected $(size(params, 2)) batched constraints, got $(length(constraints))"))
    if !isnothing(value_index)
        for component_index in 1:value_length
            copyto!(destinations[component_index], view(params, value_index + component_index - 1, :))
        end
        return destinations
    end
    for batch_index in 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(first(destinations)))
        for component_index in 1:value_length
            destinations[component_index][batch_index] = values[component_index]
        end
    end
    return destinations
end

function _batched_choice_vector_values!(
    destination::AbstractMatrix,
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractMatrix,
    constraints::AbstractVector,
    address_parts::Tuple,
)
    size(destination) == (value_length, size(params, 2)) ||
        throw(DimensionMismatch("expected mvnormal destination of size ($(value_length), $(size(params, 2))), got $(size(destination))"))
    length(constraints) == size(params, 2) ||
        throw(DimensionMismatch("expected $(size(params, 2)) batched constraints, got $(length(constraints))"))
    if !isnothing(value_index)
        copyto!(destination, view(params, value_index:(value_index + value_length - 1), :))
        return destination
    end
    for batch_index in 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(destination))
        for component_index in 1:value_length
            destination[component_index, batch_index] = values[component_index]
        end
    end
    return destination
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractMatrix,
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = collect(view(values, :, batch_index))
    end
    env.assigned[slot] = true
    return env
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractVector{<:AbstractVector},
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
end

function _backend_mvnormal_logpdf(mu::Tuple, sigma::Tuple, x)
    (length(mu) == length(sigma) && length(mu) == length(x)) ||
        throw(ArgumentError("mvnormal requires matching mean, scale, and value lengths"))
    total = _backend_normal_logpdf(mu[1], sigma[1], x[1])
    for index in 2:length(mu)
        total += _backend_normal_logpdf(mu[index], sigma[index], x[index])
    end
    return total
end

function _score_backend_step!(
    step::BackendLaplaceChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_laplace_logpdf(mu, scale, value)
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    mu = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mu)...)
    sigma = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigma)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvnormal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendInverseGammaChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_inversegamma_logpdf(shape, scale, value)
end

function _score_backend_step!(
    step::BackendWeibullChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_weibull_logpdf(shape, scale, value)
end

function _score_backend_step!(
    step::BackendGeometricChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _poisson_count(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_geometric_logpdf(probability, value)
end

function _score_backend_step!(
    step::BackendNegativeBinomialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    successes = _eval_backend_numeric_expr(env, step.successes)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _poisson_count(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_negativebinomial_logpdf(successes, probability, value)
end

function _score_backend_step!(
    step::BackendBetaChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    alpha = _eval_backend_numeric_expr(env, step.alpha)
    beta_parameter = _eval_backend_numeric_expr(env, step.beta)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_beta_logpdf(alpha, beta_parameter, value)
end

function _score_backend_step!(
    step::BackendBinomialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    trials = _eval_backend_index_value_expr(env, step.trials)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _binomial_trials(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_binomial_logpdf(trials, probability, value)
end

function _score_backend_step!(
    step::BackendCategoricalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probabilities = map(expr -> _eval_backend_numeric_expr(env, expr), step.probabilities)
    index = _categorical_index(value, length(probabilities))
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(index) ? value : index)
    return _backend_categorical_logpdf(probabilities, value)
end

function _score_backend_step!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_laplace_logpdf(mu, scale, value)
        if !isnothing(step.binding_slot)
            env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)

    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index in 1:step.value_length]
    for component_index in 1:step.value_length
        _eval_backend_numeric_expr!(mu_values[component_index], env, step.mu[component_index], 3 * step.value_length + 1)
        _eval_backend_numeric_expr!(sigma_values[component_index], env, step.sigma[component_index], 3 * step.value_length + 2)
        for batch_index in 1:env.batch_size
            totals[batch_index] += _backend_normal_logpdf(
                mu_values[component_index][batch_index],
                sigma_values[component_index][batch_index],
                choice_values[component_index][batch_index],
            )
        end
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

function _score_backend_step!(
    step::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_inversegamma_logpdf(shape, scale, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                value isa Integer || throw(
                    BatchedBackendFallback("index backend slot $(step.binding_slot) received non-integer choice value"),
                )
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendWeibullChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                value isa Integer || throw(
                    BatchedBackendFallback("index backend slot $(step.binding_slot) received non-integer choice value"),
                )
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_geometric_logpdf(probability, value)
        if !isnothing(step.binding_slot)
            count = _poisson_count(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-geometric choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    successes_values = _batched_numeric_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(successes_values, env, step.successes, 3)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        successes = successes_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_negativebinomial_logpdf(successes, probability, value)
        if !isnothing(step.binding_slot)
            count = _poisson_count(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-negativebinomial choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendBetaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        alpha = alpha_values[batch_index]
        beta_parameter = beta_values[batch_index]
        totals[batch_index] += _backend_beta_logpdf(alpha, beta_parameter, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                value isa Integer || throw(
                    BatchedBackendFallback("index backend slot $(step.binding_slot) received non-integer choice value"),
                )
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
        if !isnothing(step.binding_slot)
            count = _binomial_trials(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-binomial choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr!(probability_values[index], env, probability, length(step.probabilities) + index)
    end
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        probabilities = map(values -> values[batch_index], probability_values)
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
        if !isnothing(step.binding_slot)
            index = _categorical_index(value, length(probabilities))
            isnothing(index) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-categorical choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), index)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = index
            else
                env.generic_values[step.binding_slot][batch_index] = index
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        totals[batch_index] += _backend_laplace_logpdf(mu_values[batch_index], scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_inversegamma_logpdf(shape, scale, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendWeibullChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    probability_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        totals[batch_index] += _backend_geometric_logpdf(probability_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    successes_values = _batched_numeric_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(successes_values, env, step.successes, 3)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        totals[batch_index] += _backend_negativebinomial_logpdf(
            successes_values[batch_index],
            probability_values[batch_index],
            observed_values[batch_index],
        )
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBetaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        alpha = alpha_values[batch_index]
        beta_parameter = beta_values[batch_index]
        totals[batch_index] += _backend_beta_logpdf(alpha, beta_parameter, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    observed_values = env.observed_values
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr!(probability_values[index], env, probability, length(step.probabilities) + index)
    end
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        probabilities = map(values -> values[batch_index], probability_values)
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
    end
    return totals
end
