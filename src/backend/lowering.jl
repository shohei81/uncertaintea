const GPU_BACKEND_SUPPORTED_PRIMITIVES = Symbol[
    Symbol(":"),
    Symbol("=>"),
    :+,
    :-,
    :*,
    :/,
    :^,
    :%,
    :exp,
    :log,
    :log1p,
    :sqrt,
    :abs,
    :min,
    :max,
    :clamp,
]

const GPU_BACKEND_SUPPORTED_DISTRIBUTIONS = Symbol[:normal, :lognormal, :exponential, :bernoulli, :poisson]

abstract type AbstractBackendExpr end
abstract type AbstractBackendAddressPart end
abstract type AbstractBackendPlanStep end
abstract type BackendChoicePlanStep <: AbstractBackendPlanStep end

struct BackendLiteralExpr{T} <: AbstractBackendExpr
    value::T
end

struct BackendSlotExpr <: AbstractBackendExpr
    slot::Int
end

struct BackendPrimitiveExpr{A<:Tuple} <: AbstractBackendExpr
    op::Symbol
    arguments::A
end

struct BackendTupleExpr{A<:Tuple} <: AbstractBackendExpr
    arguments::A
end

struct BackendBlockExpr{A<:Tuple} <: AbstractBackendExpr
    arguments::A
end

struct BackendAddressLiteralPart{T} <: AbstractBackendAddressPart
    value::T
end

struct BackendAddressExprPart{E<:AbstractBackendExpr} <: AbstractBackendAddressPart
    expr::E
end

struct BackendAddressSpec{P<:Tuple}
    parts::P
end

struct BackendNormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendLognormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendExponentialChoicePlanStep{R<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    rate::R
    parameter_slot::Union{Nothing,Int}
end

struct BackendBernoulliChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendPoissonChoicePlanStep{L<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    lambda::L
    parameter_slot::Union{Nothing,Int}
end

struct BackendDeterministicPlanStep{E<:AbstractBackendExpr} <: AbstractBackendPlanStep
    binding_slot::Int
    expr::E
end

struct BackendLoopPlanStep{I<:AbstractBackendExpr,B<:Tuple} <: AbstractBackendPlanStep
    iterator_slot::Int
    iterable::I
    body::B
end

struct BackendExecutionPlan{S<:Tuple}
    target::Symbol
    steps::S
    numeric_slots::BitVector
    index_slots::BitVector
    generic_slots::BitVector
end

struct BackendLoweringReport
    target::Symbol
    supported::Bool
    issues::Vector{String}
    supported_families::Vector{Symbol}
    supported_primitives::Vector{Symbol}
end

struct BackendLoweringResult
    report::BackendLoweringReport
    plan::Union{Nothing,BackendExecutionPlan}
end

struct BatchedBackendFallback <: Exception
    message::String
end

mutable struct BatchedPlanEnvironment{T<:Real}
    layout::EnvironmentLayout
    numeric_slots::BitVector
    index_slots::BitVector
    generic_slots::BitVector
    numeric_values::Matrix{T}
    index_values::Matrix{Int}
    generic_values::Vector{Vector{Any}}
    numeric_scratch::Vector{Vector{T}}
    index_scratch::Vector{Vector{Int}}
    observed_values::Vector{T}
    assigned::BitVector
    batch_size::Int
end

function BatchedPlanEnvironment(
    layout::EnvironmentLayout,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    batch_size::Int,
    ::Type{T}=Float64,
) where {T<:Real}
    numeric_values = Matrix{T}(undef, length(layout.symbols), batch_size)
    index_values = zeros(Int, length(layout.symbols), batch_size)
    generic_values = [Vector{Any}(undef, batch_size) for _ in layout.symbols]
    return BatchedPlanEnvironment(
        layout,
        copy(numeric_slots),
        copy(index_slots),
        copy(generic_slots),
        numeric_values,
        index_values,
        generic_values,
        Vector{Vector{T}}(),
        Vector{Vector{Int}}(),
        Vector{T}(undef, batch_size),
        falses(length(layout.symbols)),
        batch_size,
    )
end

function Base.show(io::IO, report::BackendLoweringReport)
    print(
        io,
        "BackendLoweringReport(target=",
        report.target,
        ", supported=",
        report.supported,
        ", issues=",
        length(report.issues),
        ")",
    )
end

function Base.show(io::IO, plan::BackendExecutionPlan)
    print(
        io,
        "BackendExecutionPlan(target=",
        plan.target,
        ", steps=",
        length(plan.steps),
        ", numeric_slots=",
        count(identity, plan.numeric_slots),
        ", index_slots=",
        count(identity, plan.index_slots),
        ")",
    )
end

function Base.showerror(io::IO, err::BatchedBackendFallback)
    print(io, err.message)
end

function _backend_issue!(issues::Vector{String}, message::String)
    push!(issues, message)
    return nothing
end

function _supported_backend_primitive(op::Symbol)
    return op in GPU_BACKEND_SUPPORTED_PRIMITIVES
end

function _supported_backend_distribution(family::Symbol)
    return family in GPU_BACKEND_SUPPORTED_DISTRIBUTIONS
end

function _backend_primitive_name(model::TeaModel, callee)
    if callee isa Symbol
        return _supported_backend_primitive(callee) ? callee : nothing
    elseif callee isa GlobalRef
        if (callee.mod === Base || callee.mod === _evaluation_module(model)) &&
           _supported_backend_primitive(callee.name)
            return callee.name
        end
    end
    return nothing
end

function _backend_lower_expr(model::TeaModel, layout::EnvironmentLayout, expr, issues::Vector{String}, context::String)
    if expr isa QuoteNode
        return BackendLiteralExpr(expr.value)
    elseif expr isa Symbol
        slot = _environment_slot(layout, expr)
        if !isnothing(slot)
            return BackendSlotExpr(slot)
        end
        _backend_issue!(issues, "unsupported free symbol `$expr` in $context")
        return nothing
    elseif expr isa GlobalRef
        value = getfield(expr.mod, expr.name)
        if value isa Number || value isa Bool || value isa Char || value isa String
            return BackendLiteralExpr(value)
        end
        _backend_issue!(issues, "unsupported global reference `$(expr.mod).$(expr.name)` in $context")
        return nothing
    elseif expr isa LineNumberNode
        return nothing
    elseif expr isa Expr
        if expr.head == :call
            op = _backend_primitive_name(model, expr.args[1])
            if isnothing(op)
                _backend_issue!(issues, "unsupported call `$(expr.args[1])` in $context")
                return nothing
            end
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args[2:end])
            any(isnothing, arguments) && return nothing
            return BackendPrimitiveExpr(op, tuple(arguments...))
        elseif expr.head == :tuple
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
            any(isnothing, arguments) && return nothing
            return BackendTupleExpr(tuple(arguments...))
        elseif expr.head == :block
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
            filtered = Tuple(arg for arg in arguments if !isnothing(arg))
            return BackendBlockExpr(filtered)
        end

        _backend_issue!(issues, "unsupported expression head `$(expr.head)` in $context")
        return nothing
    elseif expr isa Number || expr isa Bool || expr isa Char || expr isa String
        return BackendLiteralExpr(expr)
    end

    _backend_issue!(issues, "unsupported literal `$(repr(expr))` in $context")
    return nothing
end

function _backend_lower_address(model::TeaModel, layout::EnvironmentLayout, address::AddressSpec, issues::Vector{String})
    parts = map(address.parts) do part
        if part isa AddressLiteralPart
            BackendAddressLiteralPart(part.value)
        else
            lowered = _backend_lower_expr(model, layout, part.value, issues, "address")
            isnothing(lowered) && return nothing
            BackendAddressExprPart(lowered)
        end
    end
    any(isnothing, parts) && return nothing
    return BackendAddressSpec(tuple(parts...))
end

function _backend_lower_step(model::TeaModel, layout::EnvironmentLayout, step::ChoicePlanStep, issues::Vector{String})
    step.rhs isa DistributionSpec || begin
        _backend_issue!(issues, "unsupported choice RHS $(typeof(step.rhs)) in backend lowering")
        return nothing
    end
    _supported_backend_distribution(step.rhs.family) || begin
        _backend_issue!(issues, "unsupported distribution family `$(step.rhs.family)` in backend lowering")
        return nothing
    end

    address = _backend_lower_address(model, layout, step.address, issues)
    arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, "distribution argument"), step.rhs.arguments)
    (isnothing(address) || any(isnothing, arguments)) && return nothing
    if step.rhs.family === :normal
        length(arguments) == 2 || begin
            _backend_issue!(issues, "normal expects exactly 2 backend arguments")
            return nothing
        end
        return BackendNormalChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], step.parameter_slot)
    elseif step.rhs.family === :lognormal
        length(arguments) == 2 || begin
            _backend_issue!(issues, "lognormal expects exactly 2 backend arguments")
            return nothing
        end
        return BackendLognormalChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], step.parameter_slot)
    elseif step.rhs.family === :exponential
        length(arguments) == 1 || begin
            _backend_issue!(issues, "exponential expects exactly 1 backend argument")
            return nothing
        end
        return BackendExponentialChoicePlanStep(step.binding_slot, address, arguments[1], step.parameter_slot)
    elseif step.rhs.family === :bernoulli
        length(arguments) == 1 || begin
            _backend_issue!(issues, "bernoulli expects exactly 1 backend argument")
            return nothing
        end
        return BackendBernoulliChoicePlanStep(step.binding_slot, address, arguments[1], step.parameter_slot)
    elseif step.rhs.family === :poisson
        length(arguments) == 1 || begin
            _backend_issue!(issues, "poisson expects exactly 1 backend argument")
            return nothing
        end
        return BackendPoissonChoicePlanStep(step.binding_slot, address, arguments[1], step.parameter_slot)
    end

    _backend_issue!(issues, "unsupported distribution family `$(step.rhs.family)` in backend lowering")
    return nothing
end

function _backend_lower_step(model::TeaModel, layout::EnvironmentLayout, step::DeterministicPlanStep, issues::Vector{String})
    expr = _backend_lower_expr(model, layout, step.expr, issues, "deterministic assignment")
    isnothing(expr) && return nothing
    return BackendDeterministicPlanStep(step.binding_slot, expr)
end

function _backend_lower_step(model::TeaModel, layout::EnvironmentLayout, step::LoopPlanStep, issues::Vector{String})
    iterable = _backend_lower_expr(model, layout, step.iterable, issues, "loop iterable")
    body = map(inner -> _backend_lower_step(model, layout, inner, issues), step.body)
    (isnothing(iterable) || any(isnothing, body)) && return nothing
    return BackendLoopPlanStep(step.iterator_slot, iterable, tuple(body...))
end

function _mark_backend_numeric_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    (index_slots[slot] || generic_slots[slot]) && return false
    numeric_slots[slot] && return false
    numeric_slots[slot] = true
    return true
end

function _mark_backend_index_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    changed = false
    if !index_slots[slot]
        index_slots[slot] = true
        changed = true
    end
    if numeric_slots[slot]
        numeric_slots[slot] = false
        changed = true
    end
    if generic_slots[slot]
        generic_slots[slot] = false
        changed = true
    end
    return changed
end

function _mark_backend_generic_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    changed = false
    index_slots[slot] && return false
    if !generic_slots[slot]
        generic_slots[slot] = true
        changed = true
    end
    if numeric_slots[slot]
        numeric_slots[slot] = false
        changed = true
    end
    return changed
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _mark_backend_index_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
        return nothing
    end
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _backend_expr_is_numeric(expr::BackendLiteralExpr)
    return expr.value isa Real && !(expr.value isa Bool)
end

_backend_expr_is_numeric(expr::BackendSlotExpr) = true

function _backend_expr_is_numeric(expr::BackendPrimitiveExpr)
    (expr.op === Symbol(":") || expr.op === Symbol("=>")) && return false
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _backend_expr_is_numeric(expr::BackendTupleExpr)
    return false
end

function _backend_expr_is_numeric(expr::BackendBlockExpr)
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _mark_backend_choice_address_slots!(
    address::BackendAddressSpec,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for part in address.parts
        if part isa BackendAddressExprPart
            _mark_backend_index_expr_slots!(part.expr, numeric_slots, index_slots, generic_slots)
        end
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLognormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendExponentialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.rate, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendBernoulliChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendPoissonChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.lambda, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendDeterministicPlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    if _backend_expr_is_numeric(step.expr)
        _mark_backend_numeric_expr_slots!(step.expr, numeric_slots, index_slots, generic_slots)
        _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    else
        _mark_backend_generic_expr_slots!(step.expr, numeric_slots, index_slots, generic_slots)
        _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLoopPlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.iterator_slot)
    _mark_backend_index_expr_slots!(step.iterable, numeric_slots, index_slots, generic_slots)
    for inner in step.body
        _collect_backend_slot_kinds!(inner, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

_backend_iterator_only_index_expr(::BackendLiteralExpr, iterator_slot::Int) = true
_backend_iterator_only_index_expr(expr::BackendSlotExpr, iterator_slot::Int) = expr.slot == iterator_slot

function _backend_iterator_only_index_expr(expr::BackendPrimitiveExpr, iterator_slot::Int)
    expr.op === Symbol(":") && return false
    expr.op === Symbol("=>") && return false
    return all(arg -> _backend_iterator_only_index_expr(arg, iterator_slot), expr.arguments)
end

_backend_iterator_only_index_expr(::BackendTupleExpr, iterator_slot::Int) = false

function _backend_iterator_only_index_expr(expr::BackendBlockExpr, iterator_slot::Int)
    return all(arg -> _backend_iterator_only_index_expr(arg, iterator_slot), expr.arguments)
end

function _backend_iterator_only_address(address::BackendAddressSpec, iterator_slot::Int)
    for part in address.parts
        if part isa BackendAddressExprPart &&
           !_backend_iterator_only_index_expr(part.expr, iterator_slot)
            return false
        end
    end
    return true
end

function _backend_loop_observed_choice(step::BackendLoopPlanStep)
    length(step.body) == 1 || return nothing
    choice = first(step.body)
    choice isa BackendChoicePlanStep || return nothing
    isnothing(choice.parameter_slot) || return nothing
    isnothing(choice.binding_slot) || return nothing
    _backend_iterator_only_address(choice.address, step.iterator_slot) || return nothing
    return choice
end

function _derive_backend_slot_kinds(layout::EnvironmentLayout, steps::Tuple)
    numeric_slots = falses(length(layout.symbols))
    index_slots = falses(length(layout.symbols))
    generic_slots = falses(length(layout.symbols))
    for step in steps
        _collect_backend_slot_kinds!(step, numeric_slots, index_slots, generic_slots)
    end
    return numeric_slots, index_slots, generic_slots
end

function _lower_backend_execution_plan(model::TeaModel; target::Symbol=:gpu)
    target === :gpu || throw(ArgumentError("only :gpu backend lowering is currently supported"))
    plan = executionplan(model)
    issues = String[]
    steps = map(step -> _backend_lower_step(model, plan.environment_layout, step, issues), plan.steps)
    report = BackendLoweringReport(
        target,
        isempty(issues),
        issues,
        copy(GPU_BACKEND_SUPPORTED_DISTRIBUTIONS),
        copy(GPU_BACKEND_SUPPORTED_PRIMITIVES),
    )
    if any(isnothing, steps)
        return BackendLoweringResult(report, nothing)
    end
    numeric_slots, index_slots, generic_slots = _derive_backend_slot_kinds(plan.environment_layout, tuple(steps...))
    return BackendLoweringResult(
        report,
        BackendExecutionPlan(target, tuple(steps...), numeric_slots, index_slots, generic_slots),
    )
end

function _backend_lowering(model::TeaModel; target::Symbol=:gpu)
    target === :gpu || throw(ArgumentError("only :gpu backend lowering is currently supported"))
    cached = model.backend_cache[]
    if isnothing(cached)
        cached = _lower_backend_execution_plan(model; target=target)
        model.backend_cache[] = cached
    end
    return cached::BackendLoweringResult
end

function _backend_execution_plan(model::TeaModel; target::Symbol=:gpu)
    return _backend_lowering(model; target=target).plan
end

function backend_report(model::TeaModel; target::Symbol=:gpu)
    return _backend_lowering(model; target=target).report
end

function backend_execution_plan(model::TeaModel; target::Symbol=:gpu)
    result = _backend_lowering(model; target=target)
    isnothing(result.plan) && throw(
        ArgumentError("model $(model.name) is not supported for $(target) backend lowering; see backend_report(model)"),
    )
    return result.plan
end
