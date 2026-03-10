mutable struct RuntimeContext
    rng::AbstractRNG
    constraints::ChoiceMap
    overrides::ChoiceMap
    choices::ChoiceMap
    log_weight::Float64
    prefix::Address
    require_all_choices_scored::Bool
end

struct TeaTrace
    model::TeaModel
    args::Tuple
    choices::ChoiceMap
    retval::Any
    log_weight::Float64
end

RuntimeContext(
    rng::AbstractRNG,
    constraints::ChoiceMap;
    overrides::ChoiceMap=ChoiceMap(),
    require_all_choices_scored::Bool=false,
) = RuntimeContext(rng, constraints, overrides, ChoiceMap(), 0.0, (), require_all_choices_scored)

function _join_address(prefix::Address, address)
    return (prefix..., normalize_address(address)...)
end

function _execute(model::TeaModel, ctx::RuntimeContext, args...)
    return model.impl(ctx, args...)
end

function choice(ctx::RuntimeContext, address, dist::AbstractTeaDistribution)
    full_address = _join_address(ctx.prefix, address)
    if haskey(ctx.overrides, full_address)
        value = ctx.overrides[full_address]
        ctx.log_weight += Float64(logpdf(dist, value))
    elseif haskey(ctx.constraints, full_address)
        value = ctx.constraints[full_address]
        ctx.log_weight += Float64(logpdf(dist, value))
    elseif ctx.require_all_choices_scored
        throw(ArgumentError("logjoint requires a provided value for choice $(full_address)"))
    else
        value = rand(ctx.rng, dist)
    end
    _pushchoice!(ctx.choices, full_address, value)
    return value
end

function choice(ctx::RuntimeContext, address, call::TeaCall)
    previous_prefix = ctx.prefix
    ctx.prefix = _join_address(ctx.prefix, address)
    value = _execute(call.model, ctx, call.args...)
    ctx.prefix = previous_prefix
    return value
end

function choice(::RuntimeContext, address, rhs)
    throw(ArgumentError("unsupported right-hand side for choice at $(normalize_address(address)): $(typeof(rhs))"))
end

function _run_model(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    rng::AbstractRNG=Random.default_rng(),
    overrides::ChoiceMap=choicemap(),
    require_all_choices_scored::Bool=false,
)
    ctx = RuntimeContext(
        rng,
        constraints;
        overrides=overrides,
        require_all_choices_scored=require_all_choices_scored,
    )
    retval = _execute(model, ctx, args...)
    trace = TeaTrace(model, args, ctx.choices, retval, ctx.log_weight)
    return trace, ctx.log_weight
end

function generate(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap(); rng::AbstractRNG=Random.default_rng())
    return _run_model(model, args, constraints; rng=rng)
end

function assess(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap(); rng::AbstractRNG=Random.default_rng())
    _, log_weight = generate(model, args, constraints; rng=rng)
    return log_weight
end

Base.getindex(trace::TeaTrace, address) = trace.choices[address]
Base.length(trace::TeaTrace) = length(trace.choices)

function Base.show(io::IO, trace::TeaTrace)
    print(io, "TeaTrace(", trace.model.name, ", ", length(trace), " choices)")
end
