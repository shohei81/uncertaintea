function _evaluation_module(model::TeaModel)
    return parentmodule(model.impl)
end

function _resolve_eval_symbol(model::TeaModel, env::Dict{Symbol,Any}, sym::Symbol)
    haskey(env, sym) && return env[sym]

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

function _resolve_eval_callable(model::TeaModel, env::Dict{Symbol,Any}, callee)
    if callee isa Symbol
        if callee === Symbol(":")
            return getfield(Base, Symbol(":"))
        elseif callee === Symbol("=>")
            return getfield(Base, Symbol("=>"))
        end
    end

    return _eval_plan_expr(model, env, callee)
end

function _eval_plan_expr(model::TeaModel, env::Dict{Symbol,Any}, expr)
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

function _concrete_address(model::TeaModel, env::Dict{Symbol,Any}, address::AddressSpec)
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

function _distribution_from_spec(model::TeaModel, env::Dict{Symbol,Any}, rhs::DistributionSpec)
    constructor = getfield(@__MODULE__, rhs.family)
    args = map(arg -> _eval_plan_expr(model, env, arg), rhs.arguments)
    return constructor(args...)
end

function _score_distribution_instance!(
    model::TeaModel,
    step::ChoicePlanStep,
    env::Dict{Symbol,Any},
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
    isnothing(step.binding) || (env[step.binding] = value)
    return Float64(logpdf(dist, value))
end

function _score_plan_step!(
    model::TeaModel,
    step::ChoicePlanStep,
    env::Dict{Symbol,Any},
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
    env::Dict{Symbol,Any},
    params::AbstractVector,
    constraints::ChoiceMap,
)
    env[step.binding] = _eval_plan_expr(model, env, step.expr)
    return 0.0
end

function _score_plan_step!(
    model::TeaModel,
    step::LoopPlanStep,
    env::Dict{Symbol,Any},
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_plan_expr(model, env, step.iterable)
    had_previous = haskey(env, step.iterator)
    previous_value = had_previous ? env[step.iterator] : nothing
    total = 0.0

    for item in iterable
        env[step.iterator] = item
        for body_step in step.body
            total += _score_plan_step!(model, body_step, env, params, constraints)
        end
    end

    if had_previous
        env[step.iterator] = previous_value
    else
        delete!(env, step.iterator)
    end
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

    env = Dict{Symbol,Any}()
    for (name, value) in zip(modelspec(model).arguments, args)
        env[name] = value
    end

    total = 0.0
    for step in plan.steps
        total += _score_plan_step!(model, step, env, params, constraints)
    end
    return total
end
