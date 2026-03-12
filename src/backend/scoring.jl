function _backend_normal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || throw(ArgumentError("normal requires sigma > 0"))
    z = (xx - mu_) / sigma_
    return -log(sigma_) - log(2 * pi) / 2 - z * z / 2
end

function _backend_lognormal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || throw(ArgumentError("lognormal requires sigma > 0"))
    xx > zero(xx) || return oftype(xx, -Inf)
    return _backend_normal_logpdf(mu_, sigma_, log(xx)) - log(xx)
end

function _backend_bernoulli_logpdf(p, x)
    probability = p
    zero(probability) <= probability <= one(probability) ||
        throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
    value = x isa Bool ? x : x != 0
    return value ? log(probability) : log1p(-probability)
end

function _backend_choice_value(parameter_slot::Union{Nothing,Int}, params::AbstractVector, constraints::ChoiceMap, address)
    if !isnothing(parameter_slot)
        return params[parameter_slot]
    end
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _backend_choice_value(
    parameter_slot::Union{Nothing,Int},
    params::AbstractMatrix,
    constraint_map::ChoiceMap,
    address,
    batch_index::Int,
)
    if !isnothing(parameter_slot)
        return params[parameter_slot, batch_index]
    end
    found, constrained_value = _choice_tryget_normalized(constraint_map, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _backend_observed_choice_value(constraint_map::ChoiceMap, address)
    found, constrained_value = _choice_tryget_normalized(constraint_map, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _batched_backend_observed_value(value, ::Type{T}=Float64) where {T<:Real}
    if value isa Bool
        return value ? one(T) : zero(T)
    elseif value isa Real
        return convert(T, value)
    end
    throw(BatchedBackendFallback("batched backend observed choice values must be real or Bool, got $(typeof(value))"))
end

function _batched_observed_choice_values!(
    destination::AbstractVector,
    constraints::ChoiceMap,
    address,
)
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
    fill!(destination, _batched_backend_observed_value(constrained_value, eltype(destination)))
    return destination
end

function _batched_observed_choice_values!(
    destination::AbstractVector,
    constraints::AbstractVector,
    address,
)
    length(destination) == length(constraints) ||
        throw(DimensionMismatch("expected $(length(destination)) batched constraints, got $(length(constraints))"))
    for batch_index in eachindex(destination, constraints)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

function _batched_choice_numeric_values!(
    destination::AbstractVector,
    parameter_slot::Int,
    params::AbstractMatrix,
    constraints,
    address_parts::Tuple,
)
    size(params, 2) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched params columns, got $(size(params, 2))"))
    copyto!(destination, view(params, parameter_slot, :))
    return destination
end

function _batched_choice_numeric_values!(
    destination::AbstractVector,
    ::Nothing,
    params::AbstractMatrix,
    constraints::ChoiceMap,
    address_parts::Tuple,
)
    size(params, 2) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched params columns, got $(size(params, 2))"))
    for batch_index in eachindex(destination)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

function _batched_choice_numeric_values!(
    destination::AbstractVector,
    ::Nothing,
    params::AbstractMatrix,
    constraints::AbstractVector,
    address_parts::Tuple,
)
    size(params, 2) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched params columns, got $(size(params, 2))"))
    length(constraints) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched constraints, got $(length(constraints))"))
    for batch_index in eachindex(destination, constraints)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

_score_backend_steps(::Tuple{}, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap) = 0.0
_score_backend_steps!(totals::AbstractVector, ::Tuple{}, env::BatchedPlanEnvironment, params::AbstractMatrix, constraints) = totals

function _score_backend_steps(steps::Tuple, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap)
    return _score_backend_step!(first(steps), env, params, constraints) +
           _score_backend_steps(Base.tail(steps), env, params, constraints)
end

function _score_backend_steps!(
    totals::AbstractVector,
    steps::Tuple,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    _score_backend_step!(first(steps), totals, env, params, constraints)
    return _score_backend_steps!(totals, Base.tail(steps), env, params, constraints)
end

function _score_backend_step!(
    step::BackendNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_normal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendLognormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_lognormal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendBernoulliChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probability = _eval_backend_numeric_expr(env, step.probability)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_bernoulli_logpdf(probability, value)
end

function _score_backend_step!(
    step::BackendDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    _environment_set!(env, step.binding_slot, _eval_backend_expr(env, step.expr))
    return 0.0
end

function _batched_constraint(constraints::ChoiceMap, batch_index::Int)
    return constraints
end

function _batched_constraint(constraints::AbstractVector, batch_index::Int)
    return constraints[batch_index]
end

function _batched_environment_set_shared!(env::BatchedPlanEnvironment, slot::Int, value)
    if env.numeric_slots[slot]
        value isa Real && !(value isa Bool) || throw(
            BatchedBackendFallback("numeric backend slot $slot received non-real shared value"),
        )
        env.numeric_values[slot, :] .= convert(eltype(env.numeric_values), value)
    elseif env.index_slots[slot]
        value isa Integer || throw(
            BatchedBackendFallback("index backend slot $slot received non-integer shared value"),
        )
        env.index_values[slot, :] .= Int(value)
    else
        values = env.generic_values[slot]
        for batch_index in 1:env.batch_size
            values[batch_index] = value
        end
    end
    env.assigned[slot] = true
    return nothing
end

function _batched_environment_set!(env::BatchedPlanEnvironment, slot::Int, values::AbstractVector)
    length(values) == env.batch_size ||
        throw(DimensionMismatch("expected $(env.batch_size) batched values, got $(length(values))"))
    if env.numeric_slots[slot]
        for batch_index in 1:env.batch_size
            value = values[batch_index]
            value isa Real && !(value isa Bool) || throw(
                BatchedBackendFallback("numeric backend slot $slot received non-real batched value"),
            )
            env.numeric_values[slot, batch_index] = convert(eltype(env.numeric_values), value)
        end
    elseif env.index_slots[slot]
        for batch_index in 1:env.batch_size
            value = values[batch_index]
            value isa Integer || throw(
                BatchedBackendFallback("index backend slot $slot received non-integer batched value"),
            )
            env.index_values[slot, batch_index] = Int(value)
        end
    else
        storage = env.generic_values[slot]
        for batch_index in 1:env.batch_size
            storage[batch_index] = values[batch_index]
        end
    end
    env.assigned[slot] = true
    return nothing
end

function _batched_environment_restore!(
    env::BatchedPlanEnvironment,
    slot::Int,
    previous_value,
    was_assigned::Bool,
)
    if was_assigned
        if env.index_slots[slot]
            env.index_values[slot, :] .= previous_value
        else
            env.generic_values[slot] .= previous_value
        end
        env.assigned[slot] = true
    else
        env.assigned[slot] = false
    end
    return nothing
end

function _score_backend_step!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
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
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
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
    step::BackendBernoulliChoicePlanStep,
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
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
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

function _score_backend_observed_loop_choice!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index in 1:env.batch_size
        value = observed_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBernoulliChoicePlanStep,
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
        value = observed_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
    end
    return totals
end

function _score_backend_step!(
    step::BackendDeterministicPlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    if env.numeric_slots[step.binding_slot]
        _eval_backend_numeric_expr!(view(env.numeric_values, step.binding_slot, :), env, step.expr)
        env.assigned[step.binding_slot] = true
        return totals
    elseif env.index_slots[step.binding_slot]
        _eval_backend_index_value_expr!(view(env.index_values, step.binding_slot, :), env, step.expr)
        env.assigned[step.binding_slot] = true
        return totals
    end

    values = env.generic_values[step.binding_slot]
    for batch_index in 1:env.batch_size
        values[batch_index] = _eval_backend_expr(env, step.expr, batch_index)
    end
    env.assigned[step.binding_slot] = true
    return totals
end

function _score_backend_step!(
    step::BackendLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_backend_index_iterable_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = 0.0

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _score_backend_steps(step.body, env, params, constraints)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

function _score_backend_step!(
    step::BackendLoopPlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    reference_iterable = _batched_index_iterable_reference(env, step.iterable)

    had_previous = env.assigned[step.iterator_slot]
    previous_value = if had_previous
        copy(env.index_values[step.iterator_slot, :])
    else
        Int[]
    end
    loop_choice = _backend_loop_observed_choice(step)
    if !isnothing(loop_choice)
        for item in reference_iterable
            _batched_environment_set_shared!(env, step.iterator_slot, item)
            address = _concrete_address(env, loop_choice.address, 1)
            _score_backend_observed_loop_choice!(loop_choice, totals, env, params, constraints, address)
        end
    else
        for item in reference_iterable
            _batched_environment_set_shared!(env, step.iterator_slot, item)
            _score_backend_steps!(totals, step.body, env, params, constraints)
        end
    end

    _batched_environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return totals
end
