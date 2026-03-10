function _evaluation_module(model::TeaModel)
    return parentmodule(model.impl)
end

mutable struct PlanEnvironment
    layout::EnvironmentLayout
    values::Vector{Any}
    assigned::BitVector
end

function PlanEnvironment(layout::EnvironmentLayout)
    return PlanEnvironment(layout, Vector{Any}(undef, length(layout.symbols)), falses(length(layout.symbols)))
end

function _environment_slot(layout::EnvironmentLayout, symbol::Symbol)
    return get(layout.slot_by_symbol, symbol, nothing)
end

function _environment_hasvalue(env::PlanEnvironment, slot::Int)
    return env.assigned[slot]
end

function _environment_hasvalue(env::PlanEnvironment, symbol::Symbol)
    slot = _environment_slot(env.layout, symbol)
    return !isnothing(slot) && _environment_hasvalue(env, slot)
end

function _environment_value(env::PlanEnvironment, slot::Int)
    env.assigned[slot] || throw(ArgumentError("environment slot $slot is not assigned"))
    return env.values[slot]
end

function _environment_value(env::PlanEnvironment, symbol::Symbol)
    slot = _environment_slot(env.layout, symbol)
    isnothing(slot) && throw(ArgumentError("environment does not track symbol `$symbol`"))
    return _environment_value(env, slot)
end

function _environment_set!(env::PlanEnvironment, slot::Int, value)
    env.values[slot] = value
    env.assigned[slot] = true
    return value
end

function _environment_restore!(env::PlanEnvironment, slot::Int, previous_value, was_assigned::Bool)
    if was_assigned
        env.values[slot] = previous_value
        env.assigned[slot] = true
    else
        env.assigned[slot] = false
    end
    return nothing
end

function _resolve_eval_symbol(model::TeaModel, env::PlanEnvironment, sym::Symbol)
    _environment_hasvalue(env, sym) && return _environment_value(env, sym)

    module_ = _evaluation_module(model)
    if isdefined(module_, sym)
        return getfield(module_, sym)
    elseif isdefined(@__MODULE__, sym)
        return getfield(@__MODULE__, sym)
    elseif isdefined(Base, sym)
        return getfield(Base, sym)
    end

    throw(ArgumentError("lower-level logjoint could not resolve symbol `$sym` in model $(model.name)"))
end

function _resolve_eval_callable(model::TeaModel, env::PlanEnvironment, callee)
    if callee isa Symbol
        if callee === Symbol(":")
            return getfield(Base, Symbol(":"))
        elseif callee === Symbol("=>")
            return getfield(Base, Symbol("=>"))
        end
    end

    return _eval_plan_expr(model, env, callee)
end

function _eval_plan_expr(model::TeaModel, env::PlanEnvironment, expr)
    if expr isa QuoteNode
        return expr.value
    elseif expr isa Symbol
        return _resolve_eval_symbol(model, env, expr)
    elseif expr isa GlobalRef
        return getfield(expr.mod, expr.name)
    elseif expr isa Expr
        if expr.head == :call
            f = _resolve_eval_callable(model, env, expr.args[1])
            args = map(arg -> _eval_plan_expr(model, env, arg), expr.args[2:end])
            return f(args...)
        elseif expr.head == :block
            value = nothing
            for arg in expr.args
                arg isa LineNumberNode && continue
                value = _eval_plan_expr(model, env, arg)
            end
            return value
        elseif expr.head == :tuple
            return tuple((_eval_plan_expr(model, env, arg) for arg in expr.args)...)
        end

        throw(ArgumentError("unsupported expression in lower-level logjoint: $expr"))
    end

    return expr
end

function _concrete_address(model::TeaModel, env::PlanEnvironment, address::AddressSpec)
    parts = Any[]
    for part in address.parts
        if part isa AddressLiteralPart
            push!(parts, part.value)
        else
            push!(parts, _eval_plan_expr(model, env, part.value))
        end
    end
    return Tuple(parts)
end

function _distribution_from_spec(model::TeaModel, env::PlanEnvironment, rhs::DistributionSpec)
    constructor = getfield(@__MODULE__, rhs.family)
    args = map(arg -> _eval_plan_expr(model, env, arg), rhs.arguments)
    return constructor(args...)
end

function _score_distribution_instance!(
    model::TeaModel,
    step::ChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(model, env, step.address)
    value = if !isnothing(step.parameter_slot)
        params[step.parameter_slot]
    elseif haskey(constraints, address)
        constraints[address]
    else
        throw(ArgumentError("lower-level logjoint requires a provided value for choice $(address)"))
    end

    dist = _distribution_from_spec(model, env, step.rhs)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logpdf(dist, value)
end

function _score_plan_step!(
    model::TeaModel,
    step::ChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    isempty(step.scopes) ||
        throw(ArgumentError("ChoicePlanStep scopes must be lowered into LoopPlanStep before evaluation"))

    if step.rhs isa DistributionSpec
        return _score_distribution_instance!(model, step, env, params, constraints)
    end

    throw(ArgumentError("lower-level logjoint does not support RHS type $(typeof(step.rhs))"))
end

function _score_plan_step!(
    model::TeaModel,
    step::DeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    _environment_set!(env, step.binding_slot, _eval_plan_expr(model, env, step.expr))
    return 0.0
end

function _score_plan_step!(
    model::TeaModel,
    step::LoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_plan_expr(model, env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = 0.0

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        for body_step in step.body
            total += _score_plan_step!(model, body_step, env, params, constraints)
        end
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

function logjoint(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    rng::AbstractRNG=Random.default_rng(),
)
    plan = executionplan(model)
    expected = parametercount(plan.parameter_layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))
    length(args) == length(modelspec(model).arguments) ||
        throw(DimensionMismatch("expected $(length(modelspec(model).arguments)) model arguments, got $(length(args))"))

    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end

    total = 0.0
    for step in plan.steps
        total += _score_plan_step!(model, step, env, params, constraints)
    end
    return total
end

function logjoint_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    rng::AbstractRNG=Random.default_rng(),
)
    constrained, logabsdet = transform_to_constrained_with_logabsdet(model, params)
    return logjoint(model, constrained, args, constraints; rng=rng) + logabsdet
end

struct LogjointGradientCache{F,C,V}
    objective::F
    config::C
    buffer::V
end

function _logjoint_gradient_cache(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    objective = theta -> logjoint_unconstrained(model, theta, args, constraints)
    config = ForwardDiff.GradientConfig(objective, seed)
    buffer = similar(seed)
    return LogjointGradientCache(objective, config, buffer)
end

function _logjoint_gradient!(cache::LogjointGradientCache, params::AbstractVector)
    ForwardDiff.gradient!(cache.buffer, cache.objective, params, cache.config)
    return cache.buffer
end

function logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    cache = _logjoint_gradient_cache(model, seed, args, constraints)
    gradient = similar(seed)
    ForwardDiff.gradient!(gradient, cache.objective, seed, cache.config)
    return gradient
end
