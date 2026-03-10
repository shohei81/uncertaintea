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

struct BackendChoicePlanStep{A<:Tuple,AD<:BackendAddressSpec} <: AbstractBackendPlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    family::Symbol
    arguments::A
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
    generic_slots::BitVector
    numeric_values::Matrix{Float64}
    generic_values::Vector{Vector{Any}}
    assigned::BitVector
    batch_size::Int
end

function BatchedPlanEnvironment(
    layout::EnvironmentLayout,
    numeric_slots::BitVector,
    generic_slots::BitVector,
    batch_size::Int,
)
    numeric_values = zeros(Float64, length(layout.symbols), batch_size)
    generic_values = [Vector{Any}(undef, batch_size) for _ in layout.symbols]
    return BatchedPlanEnvironment(
        layout,
        copy(numeric_slots),
        copy(generic_slots),
        numeric_values,
        generic_values,
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

    return BackendChoicePlanStep(step.binding_slot, address, step.rhs.family, tuple(arguments...), step.parameter_slot)
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

function _mark_backend_numeric_slot!(numeric_slots::BitVector, generic_slots::BitVector, slot::Int)
    generic_slots[slot] && return false
    numeric_slots[slot] && return false
    numeric_slots[slot] = true
    return true
end

function _mark_backend_generic_slot!(numeric_slots::BitVector, generic_slots::BitVector, slot::Int)
    changed = false
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
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_numeric_slot!(numeric_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _mark_backend_generic_expr_slots!(expr, numeric_slots, generic_slots)
        return nothing
    end
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_generic_slot!(numeric_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, generic_slots)
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
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _backend_expr_is_numeric(expr::BackendBlockExpr)
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _collect_backend_slot_kinds!(
    step::BackendChoicePlanStep,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    for part in step.address.parts
        if part isa BackendAddressExprPart
            _mark_backend_generic_expr_slots!(part.expr, numeric_slots, generic_slots)
        end
    end
    for arg in step.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, generic_slots)
    end
    if !isnothing(step.binding_slot)
        if step.family === :bernoulli
            _mark_backend_generic_slot!(numeric_slots, generic_slots, step.binding_slot)
        else
            _mark_backend_numeric_slot!(numeric_slots, generic_slots, step.binding_slot)
        end
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendDeterministicPlanStep,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    if _backend_expr_is_numeric(step.expr)
        _mark_backend_numeric_expr_slots!(step.expr, numeric_slots, generic_slots)
        _mark_backend_numeric_slot!(numeric_slots, generic_slots, step.binding_slot)
    else
        _mark_backend_generic_expr_slots!(step.expr, numeric_slots, generic_slots)
        _mark_backend_generic_slot!(numeric_slots, generic_slots, step.binding_slot)
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLoopPlanStep,
    numeric_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_generic_slot!(numeric_slots, generic_slots, step.iterator_slot)
    _mark_backend_generic_expr_slots!(step.iterable, numeric_slots, generic_slots)
    for inner in step.body
        _collect_backend_slot_kinds!(inner, numeric_slots, generic_slots)
    end
    return nothing
end

function _derive_backend_slot_kinds(layout::EnvironmentLayout, steps::Tuple)
    numeric_slots = falses(length(layout.symbols))
    generic_slots = falses(length(layout.symbols))
    for step in steps
        _collect_backend_slot_kinds!(step, numeric_slots, generic_slots)
    end
    return numeric_slots, generic_slots
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
    numeric_slots, generic_slots = _derive_backend_slot_kinds(plan.environment_layout, tuple(steps...))
    return BackendLoweringResult(report, BackendExecutionPlan(target, tuple(steps...), numeric_slots, generic_slots))
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

function _concrete_address(env::PlanEnvironment, address::BackendAddressSpec)
    parts = Any[]
    for part in address.parts
        if part isa BackendAddressLiteralPart
            push!(parts, part.value)
        else
            push!(parts, _eval_backend_expr(env, part.expr))
        end
    end
    return Tuple(parts)
end

function _concrete_address(env::BatchedPlanEnvironment, address::BackendAddressSpec, batch_index::Int)
    parts = Any[]
    for part in address.parts
        if part isa BackendAddressLiteralPart
            push!(parts, part.value)
        else
            push!(parts, _eval_backend_expr(env, part.expr, batch_index))
        end
    end
    return Tuple(parts)
end

function _backend_distribution(family::Symbol, arguments::Tuple)
    family === :normal && return normal(arguments...)
    family === :lognormal && return lognormal(arguments...)
    family === :bernoulli && return bernoulli(arguments...)
    throw(ArgumentError("unsupported backend distribution family `$family`"))
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
    step::BackendChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = if !isnothing(step.parameter_slot)
        params[step.parameter_slot]
    elseif haskey(constraints, address)
        constraints[address]
    else
        throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
    end

    arguments = tuple((_eval_backend_expr(env, arg) for arg in step.arguments)...)
    dist = _backend_distribution(step.family, arguments)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logpdf(dist, value)
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
    previous_value::Vector{Any},
    was_assigned::Bool,
)
    if was_assigned
        env.generic_values[slot] .= previous_value
        env.assigned[slot] = true
    else
        env.assigned[slot] = false
    end
    return nothing
end

function _score_backend_step!(
    step::BackendChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    for batch_index in 1:env.batch_size
        address = _concrete_address(env, step.address, batch_index)
        constraint_map = _batched_constraint(constraints, batch_index)
        value = if !isnothing(step.parameter_slot)
            params[step.parameter_slot, batch_index]
        elseif haskey(constraint_map, address)
            constraint_map[address]
        else
            throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        end

        arguments = tuple((_eval_backend_expr(env, arg, batch_index) for arg in step.arguments)...)
        dist = _backend_distribution(step.family, arguments)
        totals[batch_index] += logpdf(dist, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = Float64(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
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
        for batch_index in 1:env.batch_size
            value = _eval_backend_expr(env, step.expr, batch_index)
            value isa Real && !(value isa Bool) || throw(
                BatchedBackendFallback("numeric backend slot $(step.binding_slot) produced a non-real deterministic value"),
            )
            env.numeric_values[step.binding_slot, batch_index] = Float64(value)
        end
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
    iterable = _eval_backend_expr(env, step.iterable)
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
    iterables = Vector{Any}(undef, env.batch_size)
    for batch_index in 1:env.batch_size
        iterables[batch_index] = _eval_backend_expr(env, step.iterable, batch_index)
    end

    reference_iterable = first(iterables)
    for batch_index in 2:env.batch_size
        iterables[batch_index] == reference_iterable || throw(
            BatchedBackendFallback(
                "batched backend evaluation requires synchronized loop iterables across the batch",
            ),
        )
    end

    had_previous = env.assigned[step.iterator_slot]
    previous_value = had_previous ? copy(env.generic_values[step.iterator_slot]) : Vector{Any}()
    for item in reference_iterable
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        _score_backend_steps!(totals, step.body, env, params, constraints)
    end

    _batched_environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return totals
end
