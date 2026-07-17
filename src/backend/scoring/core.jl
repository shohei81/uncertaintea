# Backend-native scalar and batched scoring: shared machinery.

function _batched_backend_observed_value(value, ::Type{T}=Float64) where {T<:Real}
    if value isa Bool
        return value ? one(T) : zero(T)
    elseif value isa Real
        return convert(T, value)
    end
    throw(BatchedBackendFallback("batched backend observed choice values must be real or Bool, got $(typeof(value))"))
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
        for batch_index = 1:env.batch_size
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
        for batch_index = 1:env.batch_size
            value = values[batch_index]
            value isa Real && !(value isa Bool) || throw(
                BatchedBackendFallback("numeric backend slot $slot received non-real batched value"),
            )
            env.numeric_values[slot, batch_index] = convert(eltype(env.numeric_values), value)
        end
    elseif env.index_slots[slot]
        for batch_index = 1:env.batch_size
            value = values[batch_index]
            value isa Integer || throw(
                BatchedBackendFallback("index backend slot $slot received non-integer batched value"),
            )
            env.index_values[slot, batch_index] = Int(value)
        end
    else
        storage = env.generic_values[slot]
        for batch_index = 1:env.batch_size
            storage[batch_index] = values[batch_index]
        end
    end
    env.assigned[slot] = true
    return nothing
end

# Full-environment snapshot/restore for suffix re-evaluation under
# enumeration (the batched mirror of `_environment_snapshot`): suffix steps
# may rebind slots from their own prior values, so restoring only the
# enumerated binding would leak one branch's mutations into the next.
function _batched_environment_snapshot(env::BatchedPlanEnvironment)
    return (
        copy(env.numeric_values),
        copy(env.index_values),
        [copy(values) for values in env.generic_values],
        copy(env.assigned),
    )
end

function _batched_environment_restore_snapshot!(env::BatchedPlanEnvironment, snapshot::Tuple)
    copyto!(env.numeric_values, snapshot[1])
    copyto!(env.index_values, snapshot[2])
    for (values, saved) in zip(env.generic_values, snapshot[3])
        copyto!(values, saved)
    end
    copyto!(env.assigned, snapshot[4])
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
    for batch_index = 1:env.batch_size
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

function _backend_choice_vector_value(
    value_index::Union{Nothing,Int},
    value_length::Int,
    params::AbstractVector,
    constraints::ChoiceMap,
    address,
)
    if !isnothing(value_index)
        return collect(view(params, value_index:(value_index+value_length-1)))
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
        throw(
            DimensionMismatch(
                "expected mvnormal destination of size ($(value_length), $(size(params, 2))), got $(size(destination))",
            ),
        )
    if !isnothing(value_index)
        copyto!(destination, view(params, value_index:(value_index+value_length-1), :))
        return destination
    end
    for batch_index = 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(destination))
        for component_index = 1:value_length
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
        for component_index = 1:value_length
            copyto!(destinations[component_index], view(params, value_index + component_index - 1, :))
        end
        return destinations
    end
    for batch_index = 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(first(destinations)))
        for component_index = 1:value_length
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
        for component_index = 1:value_length
            copyto!(destinations[component_index], view(params, value_index + component_index - 1, :))
        end
        return destinations
    end
    for batch_index = 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(first(destinations)))
        for component_index = 1:value_length
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
        throw(
            DimensionMismatch(
                "expected mvnormal destination of size ($(value_length), $(size(params, 2))), got $(size(destination))",
            ),
        )
    length(constraints) == size(params, 2) ||
        throw(DimensionMismatch("expected $(size(params, 2)) batched constraints, got $(length(constraints))"))
    if !isnothing(value_index)
        copyto!(destination, view(params, value_index:(value_index+value_length-1), :))
        return destination
    end
    for batch_index = 1:size(params, 2)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_mvnormal_observed_value(constrained_value, value_length, eltype(destination))
        for component_index = 1:value_length
            destination[component_index, batch_index] = values[component_index]
        end
    end
    return destination
end
