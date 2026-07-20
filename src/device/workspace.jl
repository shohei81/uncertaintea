# Device-resident workspace: holds the lowered plan, the staged observation buffers,
# and reusable device buffers for params / slots / totals. Buffers are allocated via
# `KernelAbstractions.allocate(backend, T, dims...)`, so the same code path serves the
# CPU reference backend and any GPU backend (e.g. Metal) unchanged.

# Precision traits. `default_device_precision` names the natural precision of a
# backend; `_device_supports_float64` gates the Float64 request. The Metal extension
# overrides both for MetalBackend (Float32-only).
default_device_precision(::KernelAbstractions.Backend) = Float64
_device_supports_float64(::KernelAbstractions.Backend) = true

function _check_device_precision(backend::KernelAbstractions.Backend, precision::Type)
    if precision === Float64 && !_device_supports_float64(backend)
        throw(
            ArgumentError(
                "backend $(typeof(backend)) does not support Float64 arithmetic; " *
                "request precision=$(default_device_precision(backend)) instead",
            ),
        )
    end
    return nothing
end

function _device_unsupported_message(model::TeaModel, issues::Vector{String})
    io = IOBuffer()
    print(io, "model $(model.name) cannot be lowered to the device logjoint path.")
    if !isempty(issues)
        print(io, " Issues:")
        for issue in issues
            print(io, "\n  - ", issue)
        end
    end
    print(io, "\nSee device_lowering_report(model) for details.")
    return String(take!(io))
end

mutable struct DeviceBatchedWorkspace{T,B<:KernelAbstractions.Backend,P<:DeviceExecutionPlan{T}}
    model::TeaModel
    backend::B
    plan::P
    batch_size::Int
    parameter_count::Int
    slot_count::Int
    params_device::Any
    slots_device::Any
    observed_device::Any
    totals_device::Any
    trip_counts_device::Any
    loop_starts_device::Any
    # model arguments, kept for staging into the lazily allocated gradient
    # slot scratch (issue #38)
    args::Any
    # Gradient buffers (allocated lazily on the first gradient call). `grad_slots_device`
    # is a `DeviceDual{T}` scratch laid out (slot_count x parameter_count x batch).
    gradients_device::Any
    grad_slots_device::Any
end

function DeviceBatchedWorkspace(
    model::TeaModel,
    batch_size::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
)
    _check_device_precision(backend, precision)
    # fill missing trailing model arguments from the model's defaults, so the
    # device path agrees with generate and the CPU scoring entry points
    args =
        args isa Tuple ? _complete_model_args(model, args) :
        Tuple[_complete_model_args(model, batch_args) for batch_args in args]
    T = precision
    issues, plan = _lower_device_plan(model, T)
    isnothing(plan) && throw(ArgumentError(_device_unsupported_message(model, issues)))

    batch_size = Int(batch_size)
    bundle = _stage_device_observations(model, plan, args, constraints, batch_size)

    parameter_count = parametercount(parameterlayout(model))
    slot_count = Int(plan.slot_count)

    params_device = KernelAbstractions.allocate(backend, T, parameter_count, batch_size)
    slots_device = KernelAbstractions.allocate(backend, T, slot_count, batch_size)
    _device_stage_arguments!(slots_device, model, args, batch_size, T)
    observed_device = KernelAbstractions.allocate(backend, T, size(bundle.observed, 1), batch_size)
    copyto!(observed_device, bundle.observed)
    totals_device = KernelAbstractions.allocate(backend, T, batch_size)
    trip_counts_device = KernelAbstractions.allocate(backend, Int32, length(bundle.trip_counts))
    loop_starts_device = KernelAbstractions.allocate(backend, Int32, length(bundle.loop_starts))
    copyto!(trip_counts_device, bundle.trip_counts)
    copyto!(loop_starts_device, bundle.loop_starts)

    return DeviceBatchedWorkspace{T,typeof(backend),typeof(plan)}(
        model,
        backend,
        plan,
        batch_size,
        parameter_count,
        slot_count,
        params_device,
        slots_device,
        observed_device,
        totals_device,
        trip_counts_device,
        loop_starts_device,
        args,
        nothing,
        nothing,
    )
end

# Allocates (once) the device buffers the gradient kernel needs: a plain-`T`
# `parameter_count x batch` gradient matrix and a `DeviceDual{T}` slot scratch laid
# out `(slot_count, parameter_count, batch)` so each `(parameter, batch)` thread owns
# its own slot column.
function _device_ensure_gradient_buffers!(workspace::DeviceBatchedWorkspace{T}) where {T}
    if isnothing(workspace.gradients_device)
        workspace.gradients_device =
            KernelAbstractions.allocate(workspace.backend, T, workspace.parameter_count, workspace.batch_size)
    end
    if isnothing(workspace.grad_slots_device)
        workspace.grad_slots_device = KernelAbstractions.allocate(
            workspace.backend,
            DeviceDual{T},
            workspace.slot_count,
            workspace.parameter_count,
            workspace.batch_size,
        )
        _device_stage_gradient_arguments!(
            workspace.grad_slots_device,
            workspace.model,
            workspace.args,
            workspace.parameter_count,
            workspace.batch_size,
            T,
        )
    end
    return nothing
end

# Kernels never write argument slots, so staging their values once covers
# every launch (issue #38). Non-Real arguments are skipped: expressions
# referencing them are not device-lowerable in the first place, and integer
# loop bounds are consumed host-side through the trip-count staging.
function _device_stage_arguments!(slots_device, model::TeaModel, args, batch_size::Int, ::Type{T}) where {T}
    argument_slots = executionplan(model).environment_layout.argument_slots
    isempty(argument_slots) && return nothing
    staged = Matrix{T}(undef, 1, batch_size)
    for (argument_index, slot) in enumerate(argument_slots)
        if args isa Tuple
            value = args[argument_index]
            value isa Real || continue
            fill!(staged, convert(T, value))
        else
            all(args[batch_index][argument_index] isa Real for batch_index = 1:batch_size) || continue
            for batch_index = 1:batch_size
                staged[1, batch_index] = convert(T, args[batch_index][argument_index])
            end
        end
        copyto!(view(slots_device, slot:slot, :), staged)
    end
    return nothing
end

function _device_stage_gradient_arguments!(
    grad_slots_device,
    model::TeaModel,
    args,
    parameter_count::Int,
    batch_size::Int,
    ::Type{T},
) where {T}
    argument_slots = executionplan(model).environment_layout.argument_slots
    isempty(argument_slots) && return nothing
    staged = Array{DeviceDual{T}}(undef, 1, parameter_count, batch_size)
    for (argument_index, slot) in enumerate(argument_slots)
        if args isa Tuple
            value = args[argument_index]
            value isa Real || continue
            fill!(staged, DeviceDual{T}(convert(T, value), zero(T)))
        else
            all(args[batch_index][argument_index] isa Real for batch_index = 1:batch_size) || continue
            for batch_index = 1:batch_size
                dual = DeviceDual{T}(convert(T, args[batch_index][argument_index]), zero(T))
                for parameter_index = 1:parameter_count
                    staged[1, parameter_index, batch_index] = dual
                end
            end
        end
        copyto!(view(grad_slots_device, slot:slot, :, :), staged)
    end
    return nothing
end

function _device_upload_params!(workspace::DeviceBatchedWorkspace{T}, params::AbstractMatrix) where {T}
    if eltype(params) === T
        copyto!(workspace.params_device, params)
    else
        copyto!(workspace.params_device, convert(Array{T}, Array(params)))
    end
    return nothing
end

"""
    device_batched_logjoint!(workspace, params) -> Vector

Runs the fused device kernel for `params` (unconstrained, `parameter_count × batch`)
reusing the workspace's staged observations and device buffers, and returns the
per-column unconstrained logjoint (including the transform log-abs-det). Successive
calls with different `params` are independent (staging is not mutated).
"""
function device_batched_logjoint!(workspace::DeviceBatchedWorkspace{T}, params::AbstractMatrix) where {T}
    size(params, 1) == workspace.parameter_count ||
        throw(DimensionMismatch("expected $(workspace.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == workspace.batch_size ||
        throw(DimensionMismatch("expected $(workspace.batch_size) batch elements, got $(size(params, 2))"))

    _device_upload_params!(workspace, params)
    kernel = _device_logjoint_kernel!(workspace.backend)
    kernel(
        workspace.totals_device,
        workspace.plan,
        workspace.params_device,
        workspace.observed_device,
        workspace.slots_device,
        workspace.trip_counts_device,
        workspace.loop_starts_device;
        ndrange=workspace.batch_size,
    )
    KernelAbstractions.synchronize(workspace.backend)

    result = Vector{T}(undef, workspace.batch_size)
    copyto!(result, workspace.totals_device)
    return result
end
