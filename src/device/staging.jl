# One-time host staging. Given (args, constraints), resolve every observed choice
# value into a dense `observed::Matrix{T}` (row per choice instance in kernel
# pre-order, column per batch element), and resolve each loop's trip count / start.
#
# The traversal walks the second-stage BackendExecutionPlan in the SAME pre-order
# the device kernel uses, so the observation cursor in the kernel lines up with the
# rows produced here. It reuses the batched-backend helpers (`_batched_backend_address_parts`,
# `_batched_choice_numeric_values!`, `_batched_index_iterable_reference`) so the
# resolution logic is identical to the CPU batched path.

struct DeviceStagingBundle{T}
    observed::Matrix{T}
    trip_counts::Vector{Int32}
    loop_starts::Vector{Int32}
end

function _device_staging_environment(
    model::TeaModel,
    backend::BackendExecutionPlan,
    args,
    batch_size::Int,
    ::Type{T},
) where {T}
    layout = executionplan(model).environment_layout
    env = BatchedPlanEnvironment(
        layout,
        backend.numeric_slots,
        backend.index_slots,
        backend.generic_slots,
        batch_size,
        T,
    )
    fill!(env.assigned, false)

    argument_slots = layout.argument_slots
    argument_count = length(argument_slots)
    if args isa Tuple
        length(args) == argument_count ||
            throw(DimensionMismatch("expected $argument_count model arguments, got $(length(args))"))
        for (slot, value) in zip(argument_slots, args)
            _batched_environment_set_shared!(env, slot, value)
        end
    else
        length(args) == batch_size ||
            throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
        values = Vector{Any}(undef, batch_size)
        for argument_index = 1:argument_count
            slot = argument_slots[argument_index]
            for batch_index = 1:batch_size
                batch_args = args[batch_index]
                length(batch_args) == argument_count ||
                    throw(DimensionMismatch("expected $argument_count model arguments, got $(length(batch_args))"))
                values[batch_index] = batch_args[argument_index]
            end
            _batched_environment_set!(env, slot, values)
        end
    end
    return env
end

function _stage_observed_row!(rows, step, env, constraints, dummy_params, ::Type{T}) where {T}
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    row = Vector{T}(undef, size(dummy_params, 2))
    _batched_choice_numeric_values!(row, step.parameter_slot, dummy_params, constraints, address_parts)
    push!(rows, row)
    return nothing
end

function _stage_deterministic!(step, env)
    slot = step.binding_slot
    try
        if env.index_slots[slot]
            _eval_backend_index_value_expr!(view(env.index_values, slot, :), env, step.expr)
            env.assigned[slot] = true
        elseif env.numeric_slots[slot]
            _eval_backend_numeric_expr!(view(env.numeric_values, slot, :), env, step.expr)
            env.assigned[slot] = true
        end
    catch err
        # A deterministic that depends on latent parameters (unavailable during
        # staging) or on a generic value simply cannot be staged; it is only needed
        # here if it feeds a loop iterable / address, in which case that evaluation
        # raises its own precise error below.
        (err isa BatchedBackendFallback || err isa ArgumentError) || rethrow()
    end
    return nothing
end

function _stage_step!(
    rows,
    step::BackendChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    isnothing(step.parameter_slot) || return nothing # latent: no observed row
    _stage_observed_row!(rows, step, env, constraints, dummy_params, T)
    return nothing
end

# A vector observation stages as `value_length` consecutive rows in component
# order; the kernel-side cursor advances by the step's compile-time dimension,
# preserving the pre-order alignment invariant with a stride.
function _stage_step!(
    rows,
    step::Union{BackendMvNormalChoicePlanStep,BackendDirichletChoicePlanStep},
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    isnothing(step.parameter_slot) || return nothing # latent: no observed rows
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    block = Matrix{T}(undef, step.value_length, size(dummy_params, 2))
    _batched_choice_vector_values!(block, nothing, step.value_length, dummy_params, constraints, address_parts)
    for component_index = 1:step.value_length
        push!(rows, block[component_index, :])
    end
    return nothing
end

function _stage_step!(
    rows,
    step::BackendDeterministicPlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    _stage_deterministic!(step, env)
    return nothing
end

function _stage_step!(
    rows,
    step::BackendLoopPlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    loop_counter[] += 1
    lid = loop_counter[]
    reference = _batched_index_iterable_reference(env, step.iterable)
    Base.step(reference) == 1 ||
        throw(ArgumentError("device staging only supports unit-step loop ranges; see device_lowering_report(model)"))
    trip_counts[lid] = Int32(length(reference))
    loop_starts[lid] = isempty(reference) ? Int32(0) : Int32(first(reference))

    had_previous = env.assigned[step.iterator_slot]
    previous = had_previous ? copy(env.index_values[step.iterator_slot, :]) : Int[]
    for item in reference
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        for inner in step.body
            _stage_step!(rows, inner, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, T)
        end
    end
    _batched_environment_restore!(env, step.iterator_slot, previous, had_previous)
    return nothing
end

function _stage_device_observations(
    model::TeaModel,
    plan::DeviceExecutionPlan{T},
    args,
    constraints,
    batch_size::Int,
) where {T}
    backend = _backend_execution_plan(model)
    env = _device_staging_environment(model, backend, args, batch_size, T)
    dummy_params = Matrix{T}(undef, 0, batch_size)
    rows = Vector{Vector{T}}()
    trip_counts = zeros(Int32, Int(plan.loop_count))
    loop_starts = zeros(Int32, Int(plan.loop_count))
    loop_counter = Ref(0)
    for step in backend.steps
        _stage_step!(rows, step, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, T)
    end

    observed = Matrix{T}(undef, length(rows), batch_size)
    for (row_index, row) in enumerate(rows)
        @inbounds observed[row_index, :] .= row
    end
    return DeviceStagingBundle{T}(observed, trip_counts, loop_starts)
end
