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
]

const GPU_BACKEND_SUPPORTED_DISTRIBUTIONS = Symbol[:normal, :lognormal, :bernoulli]

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

struct BackendBernoulliChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
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

mutable struct BatchedPlanEnvironment
    layout::EnvironmentLayout
    numeric_slots::BitVector
    index_slots::BitVector
    generic_slots::BitVector
    numeric_values::Matrix{Float64}
    index_values::Matrix{Int}
    generic_values::Vector{Vector{Any}}
    numeric_scratch::Vector{Vector{Float64}}
    index_scratch::Vector{Vector{Int}}
    assigned::BitVector
    batch_size::Int
end

function BatchedPlanEnvironment(
    layout::EnvironmentLayout,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    batch_size::Int,
)
    numeric_values = zeros(Float64, length(layout.symbols), batch_size)
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
        Vector{Vector{Float64}}(),
        Vector{Vector{Int}}(),
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
    elseif step.rhs.family === :bernoulli
        length(arguments) == 1 || begin
            _backend_issue!(issues, "bernoulli expects exactly 1 backend argument")
            return nothing
        end
        return BackendBernoulliChoicePlanStep(step.binding_slot, address, arguments[1], step.parameter_slot)
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

function _eval_backend_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return expr.value
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return expr.value
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _environment_value(env, expr.slot)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    env.assigned[expr.slot] || throw(ArgumentError("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        return env.numeric_values[expr.slot, batch_index]
    elseif env.index_slots[expr.slot]
        return env.index_values[expr.slot, batch_index]
    end
    return env.generic_values[expr.slot][batch_index]
end

function _backend_primitive(op::Symbol, args...)
    if op === Symbol(":")
        return getfield(Base, Symbol(":"))(args...)
    elseif op === Symbol("=>")
        length(args) == 2 || throw(ArgumentError("`=>` expects exactly 2 arguments"))
        return args[1] => args[2]
    elseif op === :+
        return +(args...)
    elseif op === :-
        return -(args...)
    elseif op === :*
        return *(args...)
    elseif op === :/
        return /(args...)
    elseif op === :^
        return ^(args...)
    elseif op === :%
        return %(args...)
    elseif op === :exp
        length(args) == 1 || throw(ArgumentError("`exp` expects exactly 1 argument"))
        return exp(args[1])
    elseif op === :log
        length(args) == 1 || throw(ArgumentError("`log` expects exactly 1 argument"))
        return log(args[1])
    elseif op === :log1p
        length(args) == 1 || throw(ArgumentError("`log1p` expects exactly 1 argument"))
        return log1p(args[1])
    elseif op === :sqrt
        length(args) == 1 || throw(ArgumentError("`sqrt` expects exactly 1 argument"))
        return sqrt(args[1])
    elseif op === :abs
        length(args) == 1 || throw(ArgumentError("`abs` expects exactly 1 argument"))
        return abs(args[1])
    elseif op === :min
        return min(args...)
    elseif op === :max
        return max(args...)
    end

    throw(ArgumentError("unsupported backend primitive `$op`"))
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    arguments = tuple((_eval_backend_expr(env, arg) for arg in expr.arguments)...)
    return _backend_primitive(expr.op, arguments...)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    arguments = tuple((_eval_backend_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return _backend_primitive(expr.op, arguments...)
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    return tuple((_eval_backend_expr(env, arg) for arg in expr.arguments)...)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    return tuple((_eval_backend_expr(env, arg, batch_index) for arg in expr.arguments)...)
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_expr(env, arg)
    end
    return value
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_expr(env, arg, batch_index)
    end
    return value
end

function _backend_numeric_error(env::PlanEnvironment, message::String)
    throw(ArgumentError(message))
end

function _backend_numeric_error(env::BatchedPlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _require_numeric_value(env, value, context::String)
    value isa Real && !(value isa Bool) && return float(value)
    _backend_numeric_error(env, "$context requires real values, got $(typeof(value))")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return _require_numeric_value(env, expr.value, "backend numeric expression")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return _require_numeric_value(env, expr.value, "batched backend numeric expression")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _require_numeric_value(env, _environment_value(env, expr.slot), "backend numeric slot")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    return _require_numeric_value(env, _eval_backend_expr(env, expr, batch_index), "batched backend numeric slot")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "backend numeric expression cannot use `$(expr.op)`")
    end
    arguments = tuple((_eval_backend_numeric_expr(env, arg) for arg in expr.arguments)...)
    return _require_numeric_value(env, _backend_primitive(expr.op, arguments...), "backend numeric primitive")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "batched backend numeric expression cannot use `$(expr.op)`")
    end
    arguments = tuple((_eval_backend_numeric_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return _require_numeric_value(
        env,
        _backend_primitive(expr.op, arguments...),
        "batched backend numeric primitive",
    )
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    _backend_numeric_error(env, "backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    _backend_numeric_error(env, "batched backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_numeric_expr(env, arg)
    end
    return value
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_numeric_expr(env, arg, batch_index)
    end
    return value
end

function _batched_numeric_scratch!(env::BatchedPlanEnvironment, depth::Int)
    depth > 0 || throw(ArgumentError("batched numeric scratch depth must be positive"))
    while length(env.numeric_scratch) < depth
        push!(env.numeric_scratch, Vector{Float64}(undef, env.batch_size))
    end
    buffer = env.numeric_scratch[depth]
    length(buffer) == env.batch_size || resize!(buffer, env.batch_size)
    return buffer
end

function _batched_index_scratch!(env::BatchedPlanEnvironment, depth::Int)
    depth > 0 || throw(ArgumentError("batched index scratch depth must be positive"))
    while length(env.index_scratch) < depth
        push!(env.index_scratch, Vector{Int}(undef, env.batch_size))
    end
    buffer = env.index_scratch[depth]
    length(buffer) == env.batch_size || resize!(buffer, env.batch_size)
    return buffer
end

function _apply_backend_numeric_unary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
)
    for batch_index in eachindex(destination)
        destination[batch_index] = _require_numeric_value(
            env,
            _backend_primitive(op, destination[batch_index]),
            "batched backend numeric primitive",
        )
    end
    return destination
end

function _apply_backend_numeric_binary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
    rhs::AbstractVector,
)
    length(destination) == length(rhs) ||
        throw(DimensionMismatch("expected backend numeric vectors of matching length, got $(length(destination)) and $(length(rhs))"))
    for batch_index in eachindex(destination, rhs)
        destination[batch_index] = _require_numeric_value(
            env,
            _backend_primitive(op, destination[batch_index], rhs[batch_index]),
            "batched backend numeric primitive",
        )
    end
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(destination, _require_numeric_value(env, expr.value, "batched backend numeric expression"))
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendSlotExpr,
    depth::Int=1,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(destination, view(env.numeric_values, expr.slot, :))
        return destination
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(destination)
            destination[batch_index] = Float64(env.index_values[expr.slot, batch_index])
        end
        return destination
    end
    _backend_numeric_error(env, "batched backend numeric slot $(expr.slot) is not numeric")
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "batched backend numeric expression cannot use `$(expr.op)`")
    end
    isempty(expr.arguments) && _backend_numeric_error(env, "batched backend numeric primitive requires arguments")

    _eval_backend_numeric_expr!(destination, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_unary!(destination, env, expr.op)
    end

    temp = _batched_numeric_scratch!(env, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_numeric_expr!(temp, env, argument, depth + 1)
        _apply_backend_numeric_binary!(destination, env, expr.op, temp)
    end
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_numeric_error(env, "batched backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for arg in expr.arguments
        _eval_backend_numeric_expr!(destination, env, arg, depth)
    end
    return destination
end

function _backend_index_error(env::PlanEnvironment, message::String)
    throw(ArgumentError(message))
end

function _backend_index_error(env::BatchedPlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _require_index_value(env, value, context::String)
    value isa Integer && return Int(value)
    _backend_index_error(env, "$context requires integer values, got $(typeof(value))")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return _require_index_value(env, expr.value, "backend index expression")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return _require_index_value(env, expr.value, "batched backend index expression")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _require_index_value(env, _environment_value(env, expr.slot), "backend index slot")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    return _require_index_value(env, _eval_backend_expr(env, expr, batch_index), "batched backend index slot")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    expr.op === Symbol(":") && _backend_index_error(env, "backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "backend index value expression cannot be a pair")
    arguments = tuple((_eval_backend_index_value_expr(env, arg) for arg in expr.arguments)...)
    value = _backend_primitive(expr.op, arguments...)
    return _require_index_value(env, value, "backend index primitive")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    expr.op === Symbol(":") && _backend_index_error(env, "batched backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "batched backend index value expression cannot be a pair")
    arguments = tuple((_eval_backend_index_value_expr(env, arg, batch_index) for arg in expr.arguments)...)
    value = _backend_primitive(expr.op, arguments...)
    return _require_index_value(env, value, "batched backend index primitive")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    _backend_index_error(env, "backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    _backend_index_error(env, "batched backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_index_value_expr(env, arg)
    end
    return value
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_index_value_expr(env, arg, batch_index)
    end
    return value
end

function _apply_backend_index_unary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
)
    for batch_index in eachindex(destination)
        destination[batch_index] = _require_index_value(
            env,
            _backend_primitive(op, destination[batch_index]),
            "batched backend index primitive",
        )
    end
    return destination
end

function _apply_backend_index_binary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
    rhs::AbstractVector,
)
    length(destination) == length(rhs) ||
        throw(DimensionMismatch("expected backend index vectors of matching length, got $(length(destination)) and $(length(rhs))"))
    for batch_index in eachindex(destination, rhs)
        destination[batch_index] = _require_index_value(
            env,
            _backend_primitive(op, destination[batch_index], rhs[batch_index]),
            "batched backend index primitive",
        )
    end
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(destination, _require_index_value(env, expr.value, "batched backend index expression"))
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendSlotExpr,
    depth::Int=1,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.index_slots[expr.slot]
        copyto!(destination, view(env.index_values, expr.slot, :))
        return destination
    elseif env.numeric_slots[expr.slot]
        for batch_index in eachindex(destination)
            destination[batch_index] = _require_index_value(
                env,
                env.numeric_values[expr.slot, batch_index],
                "batched backend index slot",
            )
        end
        return destination
    end
    _backend_index_error(env, "batched backend index slot $(expr.slot) is not index-compatible")
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op === Symbol(":") && _backend_index_error(env, "batched backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "batched backend index value expression cannot be a pair")
    isempty(expr.arguments) && _backend_index_error(env, "batched backend index primitive requires arguments")

    _eval_backend_index_value_expr!(destination, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_index_unary!(destination, env, expr.op)
    end

    temp = _batched_index_scratch!(env, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_index_value_expr!(temp, env, argument, depth + 1)
        _apply_backend_index_binary!(destination, env, expr.op, temp)
    end
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_index_error(env, "batched backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for arg in expr.arguments
        _eval_backend_index_value_expr!(destination, env, arg, depth)
    end
    return destination
end

function _eval_backend_index_iterable_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    expr.op === Symbol(":") || _backend_index_error(env, "backend loop iterable must lower to `:`")
    arguments = tuple((_eval_backend_index_value_expr(env, arg) for arg in expr.arguments)...)
    return getfield(Base, Symbol(":"))(arguments...)
end

function _eval_backend_index_iterable_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    expr.op === Symbol(":") || _backend_index_error(env, "batched backend loop iterable must lower to `:`")
    arguments = tuple((_eval_backend_index_value_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return getfield(Base, Symbol(":"))(arguments...)
end

function _eval_backend_index_iterable_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _eval_backend_index_iterable_expr(env, arg)
        else
            _backend_index_error(env, "backend loop iterable block must end in `:`")
        end
    end
    return value
end

function _eval_backend_index_iterable_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _eval_backend_index_iterable_expr(env, arg, batch_index)
        else
            _backend_index_error(env, "batched backend loop iterable block must end in `:`")
        end
    end
    return value
end

function _batched_index_iterable_reference(
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op === Symbol(":") || _backend_index_error(env, "batched backend loop iterable must lower to `:`")
    reserved_depth = depth + length(expr.arguments)
    argument_buffers = ntuple(length(expr.arguments)) do argument_index
        buffer = _batched_index_scratch!(env, depth + argument_index - 1)
        _eval_backend_index_value_expr!(buffer, env, expr.arguments[argument_index], reserved_depth)
        buffer
    end

    reference_arguments = ntuple(length(argument_buffers)) do argument_index
        values = argument_buffers[argument_index]
        reference_value = values[1]
        for batch_index in 2:env.batch_size
            values[batch_index] == reference_value || throw(
                BatchedBackendFallback(
                    "batched backend evaluation requires synchronized loop iterables across the batch",
                ),
            )
        end
        reference_value
    end
    return getfield(Base, Symbol(":"))(reference_arguments...)
end

function _batched_index_iterable_reference(
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _batched_index_iterable_reference(env, arg, depth)
        else
            _backend_index_error(env, "batched backend loop iterable block must end in `:`")
        end
    end
    return value
end

_concrete_backend_address_parts(env::PlanEnvironment, ::Tuple{}) = ()

function _concrete_backend_address_parts(env::PlanEnvironment, parts::Tuple)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        _eval_backend_index_value_expr(env, part.expr)
    end
    return (head, _concrete_backend_address_parts(env, Base.tail(parts))...)
end

function _concrete_address(env::PlanEnvironment, address::BackendAddressSpec)
    return _concrete_backend_address_parts(env, address.parts)
end

_concrete_backend_address_parts(env::BatchedPlanEnvironment, ::Tuple{}, batch_index::Int) = ()

function _concrete_backend_address_parts(env::BatchedPlanEnvironment, parts::Tuple, batch_index::Int)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        _eval_backend_index_value_expr(env, part.expr, batch_index)
    end
    return (head, _concrete_backend_address_parts(env, Base.tail(parts), batch_index)...)
end

function _concrete_address(env::BatchedPlanEnvironment, address::BackendAddressSpec, batch_index::Int)
    return _concrete_backend_address_parts(env, address.parts, batch_index)
end

_batched_backend_address_parts(env::BatchedPlanEnvironment, ::Tuple{}, depth::Int=1) = ()

function _batched_backend_address_parts(env::BatchedPlanEnvironment, parts::Tuple, depth::Int=1)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        values = _batched_index_scratch!(env, depth)
        _eval_backend_index_value_expr!(values, env, part.expr, depth + 1)
        values
    end
    next_depth = part isa BackendAddressLiteralPart ? depth : depth + 1
    return (head, _batched_backend_address_parts(env, Base.tail(parts), next_depth)...)
end

_concrete_batched_address(::Tuple{}, batch_index::Int) = ()

function _concrete_batched_address(parts::Tuple, batch_index::Int)
    source = first(parts)
    head = source isa AbstractVector ? source[batch_index] : source
    return (head, _concrete_batched_address(Base.tail(parts), batch_index)...)
end

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
        env.numeric_values[slot, :] .= Float64(value)
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
            env.numeric_values[slot, batch_index] = Float64(value)
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
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    for batch_index in 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_choice_value(step.parameter_slot, params, constraint_map, address, batch_index)
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = Float64(value)
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
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    for batch_index in 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_choice_value(step.parameter_slot, params, constraint_map, address, batch_index)
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = Float64(value)
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
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    for batch_index in 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_choice_value(step.parameter_slot, params, constraint_map, address, batch_index)
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = Float64(value)
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
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    for batch_index in 1:env.batch_size
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_observed_choice_value(constraint_map, address)
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
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    for batch_index in 1:env.batch_size
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_observed_choice_value(constraint_map, address)
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
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    for batch_index in 1:env.batch_size
        constraint_map = _batched_constraint(constraints, batch_index)
        value = _backend_observed_choice_value(constraint_map, address)
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
