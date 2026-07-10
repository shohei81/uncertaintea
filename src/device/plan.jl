# Device execution plan: an `isbits` mirror of the second-stage BackendExecutionPlan.
#
# Design notes
# ------------
# * Addresses are fully ERASED. Observed choice values are resolved on the host
#   (staging.jl) into a dense `observed[row, col]` matrix; the kernel walks the
#   plan maintaining an observation cursor that advances identically to staging.
# * Primitive operators are lifted to a type parameter (`DevicePrimitiveExpr{Op}`,
#   `Op::Symbol`) so the kernel dispatches at compile time with no runtime branch.
# * `Union{Nothing,Int}` is avoided everywhere (it is not an isbits layout). Slots
#   use `Int32` with `0` meaning "no binding". A choice step's `value_source::Int32`
#   encodes the value origin: `> 0` -> unconstrained parameter row (latent);
#   `< 0` -> observed (value comes from the staged observation cursor).
# * Loops are non-nested and carry a static `loop_id::Int32`; the trip count and
#   start index are staged per loop into device Int32 buffers.

# ---- device expressions --------------------------------------------------------

abstract type AbstractDeviceExpr end

struct DeviceLiteralExpr{T} <: AbstractDeviceExpr
    value::T
end

struct DeviceSlotExpr <: AbstractDeviceExpr
    slot::Int32
end

struct DevicePrimitiveExpr{Op,A<:Tuple} <: AbstractDeviceExpr
    args::A
end

DevicePrimitiveExpr(op::Symbol, args::Tuple) = DevicePrimitiveExpr{op,typeof(args)}(args)

# ---- device plan steps ---------------------------------------------------------

abstract type AbstractDevicePlanStep end
abstract type AbstractDeviceChoiceStep <: AbstractDevicePlanStep end

struct DeviceNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# reparam=:noncentered normal: z read raw from the unconstrained params,
# scored against N(0, 1); the binding materializes theta = mu + sigma * z.
struct DeviceNoncenteredNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

struct DeviceLognormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceExponentialChoiceStep{R} <: AbstractDeviceChoiceStep
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGammaChoiceStep{SH,R} <: AbstractDeviceChoiceStep
    shape::SH
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceLaplaceChoiceStep{M,S} <: AbstractDeviceChoiceStep
    loc::M
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBetaChoiceStep{A,B} <: AbstractDeviceChoiceStep
    alpha::A
    beta::B
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceStudentTChoiceStep{N,M,S} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceInverseGammaChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceWeibullChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBernoulliChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBinomialChoiceStep{N,P} <: AbstractDeviceChoiceStep
    trials::N
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGeometricChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceNegativeBinomialChoiceStep{R,P} <: AbstractDeviceChoiceStep
    successes::R
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceCategoricalChoiceStep{PS<:Tuple} <: AbstractDeviceChoiceStep
    probabilities::PS
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DevicePoissonChoiceStep{L} <: AbstractDeviceChoiceStep
    lambda::L
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceDeterministicStep{E} <: AbstractDevicePlanStep
    expr::E
    binding_slot::Int32
end

struct DeviceLoopStep{B<:Tuple} <: AbstractDevicePlanStep
    loop_id::Int32
    iterator_slot::Int32
    body::B
end

struct DeviceExecutionPlan{T,S<:Tuple}
    steps::S
    slot_count::Int32
    loop_count::Int32
end

DeviceExecutionPlan{T}(steps::S, slot_count, loop_count) where {T,S<:Tuple} =
    DeviceExecutionPlan{T,S}(steps, Int32(slot_count), Int32(loop_count))

function Base.show(io::IO, plan::DeviceExecutionPlan{T}) where {T}
    print(
        io,
        "DeviceExecutionPlan{",
        T,
        "}(steps=",
        length(plan.steps),
        ", slots=",
        plan.slot_count,
        ", loops=",
        plan.loop_count,
        ")",
    )
end

# ---- lowering ------------------------------------------------------------------

const DEVICE_SUPPORTED_PRIMITIVES = (:+, :-, :*, :/, :^, :exp, :log, :log1p, :sqrt, :abs, :min, :max, :clamp)

function _device_issue!(issues::Vector{String}, message::String)
    push!(issues, message)
    return nothing
end

_device_slot32(slot::Nothing) = Int32(0)
_device_slot32(slot::Integer) = Int32(slot)

function _device_transform_code(transform::AbstractParameterTransform, issues::Vector{String}, context::String)
    if transform isa IdentityTransform
        return DEVICE_TRANSFORM_IDENTITY
    elseif transform isa LogTransform
        return DEVICE_TRANSFORM_LOG
    elseif transform isa LogitTransform
        return DEVICE_TRANSFORM_LOGIT
    end
    _device_issue!(
        issues,
        "device lowering does not support the $(nameof(typeof(transform))) parameter transform (vector transforms such as Simplex/VectorIdentity are unsupported) in $context",
    )
    return nothing
end

function _lower_device_expr(expr::BackendLiteralExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    value = expr.value
    if value isa Real && !(value isa Bool)
        return DeviceLiteralExpr(convert(T, value))
    end
    _device_issue!(issues, "device lowering only supports real numeric literals, got $(typeof(value)) in $context")
    return nothing
end

function _lower_device_expr(expr::BackendSlotExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if generic_slots[expr.slot]
        _device_issue!(
            issues,
            "device lowering cannot feed generic (non-numeric) slot $(expr.slot) into a numeric expression in $context",
        )
        return nothing
    end
    return DeviceSlotExpr(Int32(expr.slot))
end

function _lower_device_expr(expr::BackendPrimitiveExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if !(expr.op in DEVICE_SUPPORTED_PRIMITIVES)
        _device_issue!(issues, "device lowering does not support primitive `$(expr.op)` in $context")
        return nothing
    end
    lowered = map(arg -> _lower_device_expr(arg, generic_slots, T, issues, context), expr.arguments)
    any(isnothing, lowered) && return nothing
    return DevicePrimitiveExpr(expr.op, tuple(lowered...))
end

function _lower_device_expr(expr::BackendBlockExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if length(expr.arguments) == 1
        return _lower_device_expr(first(expr.arguments), generic_slots, T, issues, context)
    end
    _device_issue!(issues, "device lowering does not support multi-statement block expressions in $context")
    return nothing
end

function _lower_device_expr(expr::AbstractBackendExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    _device_issue!(issues, "device lowering does not support $(nameof(typeof(expr))) in $context")
    return nothing
end

# Latent value-source / transform, or observed marker. Returns (value_source, transform_code)
# or nothing (issue pushed).
function _device_choice_value_source(
    step::BackendChoicePlanStep,
    layout::ParameterLayout,
    in_loop::Bool,
    issues::Vector{String},
    family::String,
)
    if isnothing(step.parameter_slot)
        return (Int32(-1), DEVICE_TRANSFORM_IDENTITY)
    end
    if in_loop
        _device_issue!(issues, "device lowering does not support latent $family choices inside a loop")
        return nothing
    end
    # backend steps carry the slot's VALUE row (issue #36), so recover the
    # slot spec by that row rather than indexing by ordinal
    slot_position = findfirst(s -> s.value_index == step.parameter_slot, layout.slots)
    if isnothing(slot_position)
        _device_issue!(issues, "device lowering could not resolve the parameter slot for a $family latent")
        return nothing
    end
    slot = layout.slots[slot_position]
    if slot.value_length != 1 || slot.dimension != 1
        _device_issue!(issues, "device lowering only supports scalar latent parameters, got a vector $family latent")
        return nothing
    end
    tcode = _device_transform_code(slot.transform, issues, "$family latent")
    isnothing(tcode) && return nothing
    return (Int32(slot.index), tcode)
end

# ---- per-family choice lowering ----

function _lower_device_step!(
    out,
    step::BackendNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.mu, step.sigma, DeviceNormalChoiceStep, backend, layout, T, issues, in_loop, "normal")
end
function _lower_device_step!(
    out,
    step::BackendLognormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.mu,
        step.sigma,
        DeviceLognormalChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "lognormal",
    )
end
function _lower_device_step!(
    out,
    step::BackendGammaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.shape, step.rate, DeviceGammaChoiceStep, backend, layout, T, issues, in_loop, "gamma")
end
function _lower_device_step!(
    out,
    step::BackendLaplaceChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.mu, step.scale, DeviceLaplaceChoiceStep, backend, layout, T, issues, in_loop, "laplace")
end
function _lower_device_step!(
    out,
    step::BackendBetaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.alpha, step.beta, DeviceBetaChoiceStep, backend, layout, T, issues, in_loop, "beta")
end

function _lower_device_step!(
    out,
    step::BackendInverseGammaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.shape,
        step.scale,
        DeviceInverseGammaChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "inversegamma",
    )
end
function _lower_device_step!(
    out,
    step::BackendWeibullChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.shape,
        step.scale,
        DeviceWeibullChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "weibull",
    )
end
function _lower_device_step!(
    out,
    step::BackendBinomialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.trials,
        step.probability,
        DeviceBinomialChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "binomial",
    )
end
function _lower_device_step!(
    out,
    step::BackendNegativeBinomialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.successes,
        step.probability,
        DeviceNegativeBinomialChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "negativebinomial",
    )
end
function _lower_device_step!(
    out,
    step::BackendStudentTChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "studentt")
    nu = _lower_device_expr(step.nu, backend.generic_slots, T, issues, "studentt argument")
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "studentt argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "studentt argument")
    (isnothing(src) || isnothing(nu) || isnothing(mu) || isnothing(sigma)) && return nothing
    value_source, tcode = src
    push!(out, DeviceStudentTChoiceStep(nu, mu, sigma, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end
function _lower_device_step!(
    out,
    step::BackendGeometricChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "geometric")
    p = _lower_device_expr(step.probability, backend.generic_slots, T, issues, "geometric argument")
    (isnothing(src) || isnothing(p)) && return nothing
    value_source, tcode = src
    push!(out, DeviceGeometricChoiceStep(p, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end
function _lower_device_step!(
    out,
    step::BackendCategoricalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "categorical")
    probabilities = map(
        probability -> _lower_device_expr(probability, backend.generic_slots, T, issues, "categorical argument"),
        step.probabilities,
    )
    (isnothing(src) || any(isnothing, probabilities)) && return nothing
    value_source, tcode = src
    push!(out, DeviceCategoricalChoiceStep(probabilities, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_two_arg!(
    out,
    step,
    arg1_expr,
    arg2_expr,
    Ctor,
    backend,
    layout,
    ::Type{T},
    issues,
    in_loop,
    family,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, family)
    a1 = _lower_device_expr(arg1_expr, backend.generic_slots, T, issues, "$family argument")
    a2 = _lower_device_expr(arg2_expr, backend.generic_slots, T, issues, "$family argument")
    (isnothing(src) || isnothing(a1) || isnothing(a2)) && return nothing
    value_source, tcode = src
    push!(out, Ctor(a1, a2, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendExponentialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "exponential")
    rate = _lower_device_expr(step.rate, backend.generic_slots, T, issues, "exponential argument")
    (isnothing(src) || isnothing(rate)) && return nothing
    value_source, tcode = src
    push!(out, DeviceExponentialChoiceStep(rate, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendBernoulliChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "bernoulli")
    p = _lower_device_expr(step.probability, backend.generic_slots, T, issues, "bernoulli argument")
    (isnothing(src) || isnothing(p)) && return nothing
    value_source, tcode = src
    push!(out, DeviceBernoulliChoiceStep(p, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendPoissonChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "poisson")
    lambda = _lower_device_expr(step.lambda, backend.generic_slots, T, issues, "poisson argument")
    (isnothing(src) || isnothing(lambda)) && return nothing
    value_source, tcode = src
    push!(out, DevicePoissonChoiceStep(lambda, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

# Any other supported-by-CPU-backend choice family that we deliberately do not
# yet lower to the device.
function _lower_device_step!(
    out,
    step::BackendNoncenteredNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    if in_loop
        _device_issue!(issues, "device lowering does not support latent noncentered normal choices inside a loop")
        return nothing
    end
    isnothing(step.parameter_slot) && begin
        _device_issue!(issues, "noncentered normal requires a parameter slot")
        return nothing
    end
    slot_position = findfirst(s -> s.value_index == step.parameter_slot, layout.slots)
    if isnothing(slot_position) || layout.slots[slot_position].dimension != 1
        _device_issue!(issues, "device lowering could not resolve the noncentered normal parameter slot")
        return nothing
    end
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "noncentered normal argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "noncentered normal argument")
    (isnothing(mu) || isnothing(sigma)) && return nothing
    push!(
        out,
        DeviceNoncenteredNormalChoiceStep(
            mu,
            sigma,
            Int32(layout.slots[slot_position].index),
            _device_slot32(step.binding_slot),
        ),
    )
    return nothing
end

function _lower_device_step!(out, step::BackendChoicePlanStep, backend, layout, ::Type{T}, issues, loop_counter, in_loop) where {T}
    _device_issue!(issues, "device lowering does not support the $(nameof(typeof(step))) distribution family yet")
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendDeterministicPlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    slot = step.binding_slot
    if backend.numeric_slots[slot]
        expr = _lower_device_expr(step.expr, backend.generic_slots, T, issues, "deterministic assignment")
        isnothing(expr) && return nothing
        push!(out, DeviceDeterministicStep(expr, Int32(slot)))
    end
    # Index/generic deterministic slots only feed loop iterables / addresses, which
    # are resolved on the host during staging; the kernel does not need them.
    return nothing
end

function _lower_device_step!(out, step::BackendLoopPlanStep, backend, layout, ::Type{T}, issues, loop_counter, in_loop) where {T}
    if in_loop
        _device_issue!(issues, "device lowering does not support nested loops")
        return nothing
    end
    loop_counter[] += Int32(1)
    loop_id = loop_counter[]
    body_vec = Any[]
    for inner in step.body
        _lower_device_step!(body_vec, inner, backend, layout, T, issues, loop_counter, true)
    end
    any(isnothing, body_vec) && return nothing
    push!(out, DeviceLoopStep(loop_id, Int32(step.iterator_slot), tuple(body_vec...)))
    return nothing
end

function _lower_device_plan(model::TeaModel, ::Type{T}) where {T}
    issues = String[]
    backend = _backend_execution_plan(model)
    if isnothing(backend)
        report = backend_report(model)
        if isempty(report.issues)
            _device_issue!(issues, "model $(model.name) is not representable in the second-stage backend")
        else
            append!(issues, report.issues)
        end
        return issues, nothing
    end

    layout = parameterlayout(model)
    loop_counter = Ref(Int32(0))
    out = Any[]
    for step in backend.steps
        _lower_device_step!(out, step, backend, layout, T, issues, loop_counter, false)
    end

    if !isempty(issues)
        return issues, nothing
    end

    # argument slots are staged once per workspace (issue #38); a model that
    # rebinds an argument symbol would have kernels overwrite that slot and
    # break workspace reuse, so reject the rebinding shape outright
    argument_slots = BitSet(executionplan(model).environment_layout.argument_slots)
    if !isempty(argument_slots) && _device_steps_rebind_argument(out, argument_slots)
        _device_issue!(
            issues,
            "device lowering does not support rebinding a model argument symbol; rename the binding",
        )
        return issues, nothing
    end

    slot_count = length(executionplan(model).environment_layout.symbols)
    steps = tuple(out...)
    plan = DeviceExecutionPlan{T}(steps, slot_count, loop_counter[])
    return issues, plan
end

"""
    device_lowering_report(model; precision=Float64) -> (supported::Bool, issues::Vector{String})

Reports whether `model` can be lowered to the device (KernelAbstractions) logjoint
path. `issues` is empty iff `supported` is `true`; otherwise each entry is a precise
explanation of what is not representable.
"""
function _device_steps_rebind_argument(steps, argument_slots::BitSet)
    for step in steps
        binding = hasproperty(step, :binding_slot) ? step.binding_slot : Int32(-1)
        binding isa Integer && Int(binding) in argument_slots && return true
        hasproperty(step, :body) && _device_steps_rebind_argument(step.body, argument_slots) && return true
    end
    return false
end

function device_lowering_report(model::TeaModel; precision::Type=Float64)
    issues, plan = _lower_device_plan(model, precision)
    return (isempty(issues) && !isnothing(plan), issues)
end
