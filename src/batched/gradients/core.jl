# Hand-derived analytic batched logjoint gradients: shared machinery.

function _score_backend_steps_and_gradient!(
    ::Tuple{},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    return totals, gradients
end

function _score_backend_steps_and_gradient!(
    steps::Tuple,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    _score_backend_step_and_gradient!(first(steps), totals, gradients, cache, env, params, constraints)
    return _score_backend_steps_and_gradient!(Base.tail(steps), totals, gradients, cache, env, params, constraints)
end

function _assign_backend_choice_value!(
    env::BatchedPlanEnvironment{T},
    slot_gradients::Array{T,3},
    slot::Int,
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    if env.numeric_slots[slot]
        copyto!(view(env.numeric_values, slot, :), values)
        _store_slot_gradient!(slot_gradients, slot, gradients)
    elseif env.index_slots[slot]
        for batch_index = 1:env.batch_size
            value = values[batch_index]
            index = _integer_like_choice_value(value)
            isnothing(index) && throw(
                BatchedBackendFallback("index backend slot $slot received non-integer choice value"),
            )
            env.index_values[slot, batch_index] = index
        end
    else
        storage = env.generic_values[slot]
        for batch_index = 1:env.batch_size
            storage[batch_index] = values[batch_index]
        end
    end
    env.assigned[slot] = true
    return env
end

function _integer_like_choice_value(value)
    if value isa Integer
        return Int(value)
    elseif value isa Real && isfinite(value)
        truncated = trunc(value)
        value == truncated || return nothing
        return Int(truncated)
    end
    return nothing
end

function _score_backend_step_and_gradient!(
    step::BackendDeterministicPlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    if env.numeric_slots[step.binding_slot]
        values = view(env.numeric_values, step.binding_slot, :)
        slot_gradients = view(cache.slot_gradients, :, step.binding_slot, :)
        _eval_backend_numeric_expr_and_gradient!(values, slot_gradients, cache, env, step.expr)
        env.assigned[step.binding_slot] = true
        return totals, gradients
    end

    _score_backend_step!(step, totals, env, params, constraints)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLoopPlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    reference_iterable = _batched_index_iterable_reference(env, step.iterable)
    had_previous = env.assigned[step.iterator_slot]
    previous_value = had_previous ? copy(env.index_values[step.iterator_slot, :]) : Int[]

    for item in reference_iterable
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        _score_backend_steps_and_gradient!(step.body, totals, gradients, cache, env, params, constraints)
    end

    _batched_environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return totals, gradients
end

function _batched_backend_logjoint_and_gradient_unconstrained!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    model::TeaModel,
    cache::BatchedBackendGradientCache,
    params::AbstractMatrix,
) where {T<:AbstractFloat}
    size(params, 1) == size(gradients, 1) ||
        throw(DimensionMismatch("expected $(size(gradients, 1)) parameters, got $(size(params, 1))"))
    size(params, 2) == size(gradients, 2) ||
        throw(DimensionMismatch("expected $(size(gradients, 2)) batch elements, got $(size(params, 2))"))

    workspace = cache.workspace
    env = _prepare_batched_environment!(workspace, cache.args, size(params, 2), T)
    fill!(totals, zero(T))
    fill!(gradients, zero(T))
    fill!(cache.slot_gradients, zero(T))

    layout = workspace.layout
    constrained = _batched_constrained_buffer!(workspace, workspace.constrained_parameter_count, size(params, 2), T)
    logabsdet = _batched_logabsdet_buffer!(workspace, size(params, 2), T)
    for slot in layout.slots
        # scalar slots: params is indexed by the UNCONSTRAINED row, the
        # constrained matrix by the VALUE row; the two drift apart after a
        # dimension-changing (simplex/cholesky) predecessor (issue #36)
        parameter_row = slot.index
        value_row = slot.value_index
        if slot.transform isa IdentityTransform || slot.transform isa NoncenteredTransform
            # noncentered slots pass z through: the z-space plan step scores
            # N(z; 0, 1) and carries theta itself, so no Jacobian or
            # chain-rule correction applies
            for batch_index = 1:size(params, 2)
                constrained[value_row, batch_index] = T(params[parameter_row, batch_index])
            end
        elseif slot.transform isa VectorIdentityTransform
            source_indices = parameterindices(slot)
            destination_indices = parametervalueindices(slot)
            copyto!(view(constrained, destination_indices, :), view(params, source_indices, :))
        elseif slot.transform isa LogTransform
            for batch_index = 1:size(params, 2)
                unconstrained_value = T(params[parameter_row, batch_index])
                constrained_value = exp(unconstrained_value)
                constrained[value_row, batch_index] = constrained_value
                logabsdet[batch_index] += unconstrained_value
            end
        elseif slot.transform isa LogitTransform
            for batch_index = 1:size(params, 2)
                unconstrained_value = T(params[parameter_row, batch_index])
                constrained_value = to_constrained(slot.transform, unconstrained_value)
                constrained[value_row, batch_index] = constrained_value
                logabsdet[batch_index] += logabsdetjac(slot.transform, unconstrained_value)
            end
        elseif slot.transform isa SimplexTransform
            source_indices = parameterindices(slot)
            destination_indices = parametervalueindices(slot)
            for batch_index = 1:size(params, 2)
                constrained_view = view(constrained, destination_indices, batch_index)
                unconstrained_view = view(params, source_indices, batch_index)
                _to_constrained_simplex!(constrained_view, slot.transform, unconstrained_view)
                logabsdet[batch_index] += _simplex_logabsdet(constrained_view)
            end
        elseif slot.transform isa CholeskyCorrTransform
            source_indices = parameterindices(slot)
            destination_indices = parametervalueindices(slot)
            for batch_index = 1:size(params, 2)
                constrained_view = view(constrained, destination_indices, batch_index)
                unconstrained_view = view(params, source_indices, batch_index)
                logabsdet[batch_index] +=
                    _to_constrained_cholesky_corr!(constrained_view, slot.transform, unconstrained_view)
            end
        else
            throw(BatchedBackendFallback("batched backend gradient does not support transform $(typeof(slot.transform))"))
        end
    end

    _score_backend_steps_and_gradient!(workspace.backend_plan.steps, totals, gradients, cache, env, constrained, cache.constraints)

    for slot in layout.slots
        if slot.transform isa IdentityTransform
            continue
        elseif slot.transform isa NoncenteredTransform
            continue
        elseif slot.transform isa VectorIdentityTransform
            continue
        elseif slot.transform isa LogTransform
            parameter_row = slot.index
            value_row = slot.value_index
            for batch_index = 1:size(params, 2)
                gradients[parameter_row, batch_index] =
                    gradients[parameter_row, batch_index] * constrained[value_row, batch_index] + one(T)
            end
        elseif slot.transform isa LogitTransform
            parameter_row = slot.index
            value_row = slot.value_index
            for batch_index = 1:size(params, 2)
                constrained_value = constrained[value_row, batch_index]
                gradients[parameter_row, batch_index] =
                    gradients[parameter_row, batch_index] * constrained_value * (1 - constrained_value) +
                    (1 - 2 * constrained_value)
            end
        elseif slot.transform isa SimplexTransform
            source_indices = parameterindices(slot)
            destination_indices = parametervalueindices(slot)
            for batch_index = 1:size(params, 2)
                constrained_view = view(constrained, destination_indices, batch_index)
                for (local_index, parameter_index) in enumerate(source_indices)
                    gradients[parameter_index, batch_index] +=
                        1 - slot.value_length * constrained_view[local_index]
                end
            end
        elseif slot.transform isa CholeskyCorrTransform
            # log-abs-det = sum_ij [log(1 - w_ij^2) + 0.5 sum_{k<j} log(1 - w_ik^2)]
            # in w = tanh(z) terms, so each below-diagonal coordinate carries the
            # weight 1 + (i - 1 - j)/2 and d/dz log(1 - tanh(z)^2) = -2 tanh(z).
            source_indices = parameterindices(slot)
            d = slot.transform.size
            for batch_index = 1:size(params, 2)
                position = 0
                for row = 2:d, col = 1:(row-1)
                    position += 1
                    parameter_index = source_indices[position]
                    gradients[parameter_index, batch_index] -=
                        (row - col + 1) * tanh(T(params[parameter_index, batch_index]))
                end
            end
        end
    end

    for batch_index in eachindex(totals)
        totals[batch_index] += logabsdet[batch_index]
    end
    return totals, gradients
end
