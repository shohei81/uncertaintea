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

abstract type AbstractCompiledExpr end
abstract type AbstractCompiledAddressPart end
abstract type AbstractCompiledPlanStep end

struct CompiledLiteralExpr{T} <: AbstractCompiledExpr
    value::T
end

struct CompiledSlotExpr <: AbstractCompiledExpr
    slot::Int
end

struct CompiledCallExpr{C<:AbstractCompiledExpr,A<:Tuple} <: AbstractCompiledExpr
    callee::C
    arguments::A
end

struct CompiledTupleExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledVectorExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledBlockExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledAddressLiteralPart{T} <: AbstractCompiledAddressPart
    value::T
end

struct CompiledAddressDynamicPart{E<:AbstractCompiledExpr} <: AbstractCompiledAddressPart
    expr::E
end

struct CompiledAddressSpec{P<:Tuple}
    parts::P
end

# reparam=:noncentered walk data: theta = location + scale * z, or
# exp(location + scale * z) for the log-space (lognormal) variant.
struct CompiledNoncentered
    location::Any
    scale::Any
    logspace::Bool
end

# marginalize=:enumerate walk data (docs/discrete-enumeration.md): the
# compile-time support values to enumerate. bernoulli: (false, true);
# categorical: (1, ..., K) from the literal probability-vector length.
struct CompiledMarginalize
    support::Tuple
end

struct CompiledChoicePlanStep{A<:Tuple,AD<:CompiledAddressSpec,C} <: AbstractCompiledPlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    constructor::C
    arguments::A
    parameter_value_indices::Union{Nothing,UnitRange{Int}}
    parameter_slot::Union{Nothing,Int}
    # compiled location/scale (plus log-space flag) for reparam=:noncentered
    # latents; `nothing` for centered choices
    noncentered::Union{Nothing,CompiledNoncentered}
    # compile-time enumeration support for marginalize=:enumerate latents;
    # `nothing` for ordinary choices
    marginalize::Union{Nothing,CompiledMarginalize}
end

struct CompiledDeterministicPlanStep{E<:AbstractCompiledExpr} <: AbstractCompiledPlanStep
    binding_slot::Int
    expr::E
end

struct CompiledLoopPlanStep{I<:AbstractCompiledExpr,B<:Tuple} <: AbstractCompiledPlanStep
    iterator_slot::Int
    iterable::I
    body::B
end

struct CompiledExecutionPlan{S<:Tuple}
    steps::S
end

# If `callee` is a dotted operator symbol (e.g. `.*`, `.+`), return the underlying
# scalar operator function; otherwise `nothing`. Used to lower broadcast argument
# expressions of broadcast (dot-call) distribution observations.
function _broadcast_operator_function(callee)
    callee isa Symbol || return nothing
    name = string(callee)
    (length(name) >= 2 && name[1] == '.') || return nothing
    base = Symbol(name[2:end])
    isdefined(Base, base) || return nothing
    value = getfield(Base, base)
    return value isa Function ? value : nothing
end

function _resolve_compile_symbol(model::TeaModel, layout::EnvironmentLayout, sym::Symbol)
    slot = _environment_slot(layout, sym)
    if !isnothing(slot)
        return CompiledSlotExpr(slot)
    elseif sym === Symbol(":")
        return CompiledLiteralExpr(getfield(Base, Symbol(":")))
    elseif sym === Symbol("=>")
        return CompiledLiteralExpr(getfield(Base, Symbol("=>")))
    end

    module_ = _evaluation_module(model)
    if isdefined(module_, sym)
        return CompiledLiteralExpr(getfield(module_, sym))
    elseif isdefined(@__MODULE__, sym)
        return CompiledLiteralExpr(getfield(@__MODULE__, sym))
    elseif isdefined(Base, sym)
        return CompiledLiteralExpr(getfield(Base, sym))
    end

    throw(ArgumentError("lower-level logjoint could not resolve symbol `$sym` in model $(model.name)"))
end

function _compile_plan_expr(model::TeaModel, layout::EnvironmentLayout, expr)
    if expr isa QuoteNode
        return CompiledLiteralExpr(expr.value)
    elseif expr isa Symbol
        return _resolve_compile_symbol(model, layout, expr)
    elseif expr isa GlobalRef
        return CompiledLiteralExpr(getfield(expr.mod, expr.name))
    elseif expr isa Expr
        if expr.head == :call
            base_op = _broadcast_operator_function(expr.args[1])
            if !isnothing(base_op)
                # A dotted operator like `.*`: evaluate as `broadcast(op, args...)` so
                # scalar/vector broadcast arguments combine with standard rules.
                arguments = tuple(
                    CompiledLiteralExpr(base_op),
                    (_compile_plan_expr(model, layout, arg) for arg in expr.args[2:end])...,
                )
                return CompiledCallExpr(CompiledLiteralExpr(broadcast), arguments)
            end
            callee = _compile_plan_expr(model, layout, expr.args[1])
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args[2:end])...)
            return CompiledCallExpr(callee, arguments)
        elseif expr.head == :block
            arguments = tuple((
                _compile_plan_expr(model, layout, arg) for arg in expr.args if !(arg isa LineNumberNode)
            )...)
            return CompiledBlockExpr(arguments)
        elseif expr.head == :tuple
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return CompiledTupleExpr(arguments)
        elseif expr.head == :vect
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return CompiledVectorExpr(arguments)
        elseif expr.head == :ref
            callee = CompiledLiteralExpr(getindex)
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return CompiledCallExpr(callee, arguments)
        end

        throw(ArgumentError("unsupported expression in lower-level logjoint compilation: $expr"))
    end

    return CompiledLiteralExpr(expr)
end

function _compile_address(layout::EnvironmentLayout, model::TeaModel, address::AddressSpec)
    parts = tuple((
        begin
            if part isa AddressLiteralPart
                CompiledAddressLiteralPart(part.value)
            else
                CompiledAddressDynamicPart(_compile_plan_expr(model, layout, part.value))
            end
        end for part in address.parts
    )...)
    return CompiledAddressSpec(parts)
end

function _compile_plan_step(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
)
    step.rhs isa DistributionSpec || step.rhs isa BroadcastDistributionSpec ||
        throw(ArgumentError("compiled lower-level logjoint only supports distribution choice steps"))
    arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in step.rhs.arguments)...)
    constructor = if step.rhs isa BroadcastDistributionSpec
        getfield(@__MODULE__, :BroadcastNormalDist)
    elseif !isnothing(step.rhs.builder)
        step.rhs.builder
    else
        getfield(@__MODULE__, step.rhs.family)
    end
    parameter_value_indices =
        isnothing(step.parameter_slot) ? nothing : parametervalueindices(parameter_layout.slots[step.parameter_slot])
    noncentered = nothing
    if step.rhs isa DistributionSpec && step.rhs.reparam === :noncentered
        if step.rhs.family === :iid
            base = step.rhs.arguments[1]
            location_index, scale_index = _noncentered_location_scale_indices(base.args[1])
            noncentered = CompiledNoncentered(
                _compile_plan_expr(model, layout, base.args[location_index+1]),
                _compile_plan_expr(model, layout, base.args[scale_index+1]),
                base.args[1] === :lognormal,
            )
        else
            location_index, scale_index = _noncentered_location_scale_indices(step.rhs.family)
            noncentered = CompiledNoncentered(
                arguments[location_index],
                arguments[scale_index],
                step.rhs.family === :lognormal,
            )
        end
    end
    marginalize = nothing
    if step.rhs isa DistributionSpec && step.rhs.marginalize === :enumerate
        marginalize = CompiledMarginalize(_marginalize_support(step.rhs))
    end
    return CompiledChoicePlanStep(
        step.binding_slot,
        _compile_address(layout, model, step.address),
        constructor,
        arguments,
        parameter_value_indices,
        step.parameter_slot,
        noncentered,
        marginalize,
    )
end

function _marginalize_support(rhs::DistributionSpec)
    rhs.family === :bernoulli && return (false, true)
    if rhs.family === :categorical
        probabilities = rhs.arguments[1]
        support_size = if probabilities isa Expr && probabilities.head == :vect
            length(probabilities.args)
        elseif probabilities isa AbstractVector
            length(probabilities)
        else
            throw(
                ArgumentError(
                    "marginalize=:enumerate requires a literal categorical probability vector, got `$probabilities`",
                ),
            )
        end
        return ntuple(identity, support_size)
    end
    throw(ArgumentError("marginalize=:enumerate is not supported for family `$(rhs.family)`"))
end

function _compile_plan_step(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::DeterministicPlanStep,
)
    return CompiledDeterministicPlanStep(step.binding_slot, _compile_plan_expr(model, layout, step.expr))
end

function _compile_plan_step(model::TeaModel, layout::EnvironmentLayout, parameter_layout::ParameterLayout, step::LoopPlanStep)
    body = tuple((_compile_plan_step(model, layout, parameter_layout, inner) for inner in step.body)...)
    return CompiledLoopPlanStep(step.iterator_slot, _compile_plan_expr(model, layout, step.iterable), body)
end

function _compile_execution_plan(model::TeaModel)
    raw_plan = executionplan(model)
    compiled_steps = tuple(
        (_compile_plan_step(model, raw_plan.environment_layout, raw_plan.parameter_layout, step) for step in raw_plan.steps)...,
    )
    return CompiledExecutionPlan(compiled_steps)
end

function _compiled_execution_plan(model::TeaModel)
    cached = model.evaluator_cache[]
    if isnothing(cached)
        cached = _compile_execution_plan(model)
        model.evaluator_cache[] = cached
    end
    return cached::CompiledExecutionPlan
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledLiteralExpr)
    return expr.value
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledSlotExpr)
    return _environment_value(env, expr.slot)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledCallExpr)
    callee = _eval_compiled_expr(env, expr.callee)
    arguments = tuple((_eval_compiled_expr(env, arg) for arg in expr.arguments)...)
    return callee(arguments...)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledTupleExpr)
    return tuple((_eval_compiled_expr(env, arg) for arg in expr.arguments)...)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledVectorExpr)
    return Any[_eval_compiled_expr(env, arg) for arg in expr.arguments]
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_compiled_expr(env, arg)
    end
    return value
end

_concrete_compiled_address_parts(env::PlanEnvironment, ::Tuple{}) = ()

function _concrete_compiled_address_parts(env::PlanEnvironment, parts::Tuple)
    part = first(parts)
    head = if part isa CompiledAddressLiteralPart
        part.value
    else
        _eval_compiled_expr(env, part.expr)
    end
    return (head, _concrete_compiled_address_parts(env, Base.tail(parts))...)
end

function _concrete_address(env::PlanEnvironment, address::CompiledAddressSpec)
    return _normalize_concrete_address(_concrete_compiled_address_parts(env, address.parts))
end

function _compiled_distribution(step::CompiledChoicePlanStep, env::PlanEnvironment)
    arguments = tuple((_eval_compiled_expr(env, arg) for arg in step.arguments)...)
    return step.constructor(arguments...)
end

function _parameter_slot_value(layout::ParameterLayout, slot_index::Int, params::AbstractVector)
    slot = layout.slots[slot_index]
    indices = parametervalueindices(slot)
    length(indices) == 1 && return params[first(indices)]
    return collect(view(params, indices))
end

function _parameter_slot_value(indices::UnitRange{Int}, params::AbstractVector)
    length(indices) == 1 && return params[first(indices)]
    return collect(view(params, indices))
end

_score_compiled_steps(::Tuple{}, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap) = 0.0

function _score_compiled_steps(steps::Tuple, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap)
    head = first(steps)
    tail = Base.tail(steps)
    if head isa CompiledChoicePlanStep && !isnothing(head.marginalize)
        return _score_marginalized_choice!(head, tail, env, params, constraints)
    end
    return _score_plan_step!(head, env, params, constraints) +
           _score_compiled_steps(tail, env, params, constraints)
end

# marginalize=:enumerate (docs/discrete-enumeration.md): the fold above makes
# a marginalized choice own its suffix -- bind each compile-time support
# value, score the remaining steps, and logsumexp-combine, so the returned
# density is the marginal over the discrete latent. A constrained value
# short-circuits to the plain joint (conditioning on the latent stays free).
# Nested marginalized latents recurse through the suffix scoring (product
# enumeration).
function _score_marginalized_choice!(
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    dist = _compiled_distribution(step, env)
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    if found
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, constrained_value)
        return logpdf(dist, constrained_value) + _score_compiled_steps(tail, env, params, constraints)
    end

    had_previous = !isnothing(step.binding_slot) && _environment_hasvalue(env, step.binding_slot)
    previous_value = had_previous ? _environment_value(env, step.binding_slot) : nothing
    terms = _marginalized_suffix_terms(step.marginalize.support, dist, step, tail, env, params, constraints)
    isnothing(step.binding_slot) || _environment_restore!(env, step.binding_slot, previous_value, had_previous)

    # max-shifted logsumexp, mirroring `logpdf(::MixtureDist, x)`
    shift = maximum(terms)
    isfinite(shift) || return oftype(shift, -Inf)
    total = zero(shift)
    for term in terms
        total += exp(term - shift)
    end
    return shift + log(total)
end

function _marginalized_suffix_terms(
    ::Tuple{},
    dist,
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    return ()
end

function _marginalized_suffix_terms(
    support::Tuple,
    dist,
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    value = first(support)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    term = logpdf(dist, value) + _score_compiled_steps(tail, env, params, constraints)
    return (term, _marginalized_suffix_terms(Base.tail(support), dist, step, tail, env, params, constraints)...)
end

function _score_plan_step!(
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = if !isnothing(step.parameter_value_indices)
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("lower-level logjoint requires a provided value for choice $(address)"))
        constrained_value
    end

    dist = _compiled_distribution(step, env)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logpdf(dist, value)
end

function _score_plan_step!(
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return 0.0
end

function _score_plan_step!(
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = 0.0

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _score_compiled_steps(step.body, env, params, constraints)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

function logjoint(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    plan = executionplan(model)
    compiled_plan = _compiled_execution_plan(model)
    expected = parametervaluecount(plan.parameter_layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))
    length(args) == length(modelspec(model).arguments) ||
        throw(DimensionMismatch("expected $(length(modelspec(model).arguments)) model arguments, got $(length(args))"))

    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end

    return _score_compiled_steps(compiled_plan.steps, env, params, constraints)
end

function logjoint_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    constrained, logabsdet = transform_to_constrained_with_logabsdet(model, params, args)
    return logjoint(model, constrained, args, constraints) + logabsdet
end

# --- dependent-transform plan walk (reparam=:noncentered) ---------------------
#
# Noncentered slots hold the standardized z while the constrained space keeps
# theta = location + scale * z, where location/scale are expressions over
# model arguments and earlier latents. The transform therefore runs as a walk
# over the compiled plan with an environment, visiting slots in execution
# order; centered slots use the ordinary per-slot transforms. Observations
# are bound as NaN poison: a location/scale that depends on one fails loudly.

# positions of (location, scale) among the family's arguments
_noncentered_location_scale_indices(family::Symbol) = family === :studentt ? (2, 3) : (1, 2)

# Poison bound to slotless choices during the walk. Unlike NaN it cannot be
# swallowed by comparisons or branching (`NaN > 0` is silently false): it is
# not a Number, so arithmetic and ordered comparisons raise MethodError, and
# equality is overloaded to throw outright.
struct _TransformUnknownValue end

const _TRANSFORM_UNKNOWN_MESSAGE =
    "reparam=:noncentered location/scale expressions may only depend on model arguments " *
    "and earlier latents with parameter slots; this model routes a choice without a slot " *
    "(an observation or a discrete latent) into one"

Base.:(==)(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))

# Evaluate an expression during the walk, converting the poison's MethodError
# into the informative rejection.
function _walk_transform_eval(env::PlanEnvironment, expr)
    try
        return _eval_compiled_expr(env, expr)
    catch err
        if err isa MethodError && any(arg isa _TransformUnknownValue for arg in err.args)
            throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
        end
        rethrow()
    end
end

function _dependent_transform_walk!(
    destination::AbstractVector,
    model::TeaModel,
    params::AbstractVector,
    args::Tuple,
    inverse::Bool,
)
    plan = executionplan(model)
    compiled_plan = _compiled_execution_plan(model)
    length(args) == length(modelspec(model).arguments) || throw(
        DimensionMismatch(
            "transforming a model with reparam=:noncentered latents requires its " *
            "$(length(modelspec(model).arguments)) model arguments, got $(length(args))",
        ),
    )
    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    return _walk_transform_steps!(destination, compiled_plan.steps, env, plan.parameter_layout, params, inverse)
end

_walk_transform_steps!(destination, ::Tuple{}, env, layout, params, inverse) = zero(eltype(destination))

function _walk_transform_steps!(destination, steps::Tuple, env, layout, params, inverse)
    return _walk_transform_step!(destination, first(steps), env, layout, params, inverse) +
           _walk_transform_steps!(destination, Base.tail(steps), env, layout, params, inverse)
end

function _walk_transform_step!(destination, step::CompiledDeterministicPlanStep, env, layout, params, inverse)
    _environment_set!(env, step.binding_slot, _walk_transform_eval(env, step.expr))
    return zero(eltype(destination))
end

function _walk_transform_step!(destination, step::CompiledLoopPlanStep, env, layout, params, inverse)
    iterable = _walk_transform_eval(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = zero(eltype(destination))
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _walk_transform_steps!(destination, step.body, env, layout, params, inverse)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

function _walk_transform_step!(destination, step::CompiledChoicePlanStep, env, layout, params, inverse)
    if isnothing(step.parameter_slot)
        # observation or slotless latent: its value is unknown during a
        # transform; poison the binding so any dependence fails loudly
        isnothing(step.binding_slot) ||
            _environment_set!(env, step.binding_slot, _TransformUnknownValue())
        return zero(eltype(destination))
    end
    slot = layout.slots[step.parameter_slot]
    if isnothing(step.noncentered)
        if inverse
            _transform_slot_to_unconstrained!(destination, slot, params)
            value = _parameter_slot_value(parametervalueindices(slot), params)
            isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
            return zero(eltype(destination))
        end
        logabsdet = _transform_slot_to_constrained!(destination, slot, params)
        value = _parameter_slot_value(parametervalueindices(slot), destination)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return logabsdet
    end

    location = _walk_transform_eval(env, step.noncentered.location)
    scale = _walk_transform_eval(env, step.noncentered.scale)
    (location isa _TransformUnknownValue || scale isa _TransformUnknownValue) &&
        throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
    (location isa Real && scale isa Real && isfinite(location) && isfinite(scale)) || throw(
        ArgumentError(
            "reparam=:noncentered location/scale must evaluate to finite reals; got " *
            "location=$location, scale=$scale for the choice bound to :$(slot.binding)",
        ),
    )
    logspace = step.noncentered.logspace
    if inverse
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            constrained_value = params[value_index]
            unconstrained_value = logspace ? log(constrained_value) : constrained_value
            destination[parameter_index] = (unconstrained_value - location) / scale
        end
        value = _parameter_slot_value(parametervalueindices(slot), params)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return zero(eltype(destination))
    end
    logabsdet = zero(eltype(destination))
    for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
        affine = location + scale * params[parameter_index]
        destination[value_index] = logspace ? exp(affine) : affine
        logabsdet += log(scale) + (logspace ? affine : zero(affine))
    end
    value = _parameter_slot_value(parametervalueindices(slot), destination)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logabsdet
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
    buffer::AbstractVector=similar(collect(params)),
)
    seed = collect(params)
    length(buffer) == length(seed) ||
        throw(DimensionMismatch("expected gradient buffer of length $(length(seed)), got $(length(buffer))"))
    objective = theta -> logjoint_unconstrained(model, theta, args, constraints)
    config = ForwardDiff.GradientConfig(objective, seed)
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
