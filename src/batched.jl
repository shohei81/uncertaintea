mutable struct BatchedLogjointWorkspace{BP,CP,E}
    backend_plan::BP
    compiled_plan::CP
    environment::E
    parameter_count::Int
    argument_count::Int
    argument_slots::Vector{Int}
    constrained_buffer::Base.RefValue{Any}
    batched_environment::Base.RefValue{Any}
    batched_totals_buffer::Base.RefValue{Any}
    batched_constrained_buffer::Base.RefValue{Any}
    batched_logabsdet_buffer::Base.RefValue{Any}
    batched_argument_buffer::Base.RefValue{Any}
end

struct BatchedGradientObjective{M,W}
    model::M
    workspace::W
    args::Any
    constraints::ChoiceMap
end

function (objective::BatchedGradientObjective)(theta)
    return _logjoint_unconstrained_with_workspace!(
        objective.model,
        objective.workspace,
        theta,
        objective.args,
        objective.constraints,
    )
end

struct BatchedGradientColumnCache{F,C,V}
    objective::F
    config::C
    buffer::V
end

struct BatchedFlatGradientObjective{M,W,A,C}
    model::M
    workspace::W
    args::A
    constraints::C
    parameter_count::Int
    batch_size::Int
end

function (objective::BatchedFlatGradientObjective)(theta)
    params = reshape(theta, objective.parameter_count, objective.batch_size)
    totals = _batched_totals_buffer!(objective.workspace, objective.batch_size, eltype(theta))
    _batched_logjoint_unconstrained_with_workspace!(
        totals,
        objective.model,
        objective.workspace,
        params,
        objective.args,
        objective.constraints,
    )

    total = zero(eltype(theta))
    for value in totals
        total += value
    end
    return total
end

struct BatchedFlatGradientCache{O,C,B}
    objective::O
    config::C
    flat_buffer::B
end

struct BatchedBackendGradientCache{W,S,G,A,C}
    workspace::W
    slot_gradients::S
    gradient_scratch::G
    args::A
    constraints::C
end

struct BatchedLogjointGradientCache{C,B,F,G<:AbstractMatrix}
    model::TeaModel
    column_caches::C
    backend_cache::B
    flat_cache::F
    gradient_buffer::G
    parameter_count::Int
    batch_size::Int
end

function BatchedLogjointWorkspace(model::TeaModel)
    plan = executionplan(model)
    return BatchedLogjointWorkspace(
        _backend_execution_plan(model),
        _compiled_execution_plan(model),
        PlanEnvironment(plan.environment_layout),
        parametercount(plan.parameter_layout),
        length(modelspec(model).arguments),
        copy(plan.environment_layout.argument_slots),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
    )
end

function _validate_batched_params(model::TeaModel, params::AbstractMatrix)
    expected = parametercount(parameterlayout(model))
    size(params, 1) == expected || throw(DimensionMismatch("expected $expected parameters, got $(size(params, 1))"))
    return size(params, 2)
end

function _validate_batched_args(args::Tuple, batch_size::Int)
    return args
end

function _validate_batched_args(args::AbstractVector, batch_size::Int)
    length(args) == batch_size || throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
    for batch_args in args
        batch_args isa Tuple || throw(ArgumentError("batched args must be a tuple or a vector of tuples"))
    end
    return args
end

function _validate_batched_constraints(constraints::ChoiceMap, batch_size::Int)
    return constraints
end

function _validate_batched_constraints(constraints::AbstractVector, batch_size::Int)
    length(constraints) == batch_size || throw(DimensionMismatch("expected $batch_size batched choicemaps, got $(length(constraints))"))
    for batch_constraints in constraints
        batch_constraints isa ChoiceMap || throw(ArgumentError("batched constraints must be a ChoiceMap or a vector of ChoiceMaps"))
    end
    return constraints
end

_batched_args(args::Tuple, index::Int) = args
_batched_args(args::AbstractVector, index::Int) = args[index]

_batched_constraints(constraints::ChoiceMap, index::Int) = constraints
_batched_constraints(constraints::AbstractVector, index::Int) = constraints[index]

function _reset_environment!(env::PlanEnvironment)
    fill!(env.assigned, false)
    return env
end

function _prepare_environment!(workspace::BatchedLogjointWorkspace, args::Tuple)
    length(args) == workspace.argument_count ||
        throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(args))"))

    env = workspace.environment
    _reset_environment!(env)
    for (slot, value) in zip(workspace.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    return env
end

function _batched_environment!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}=Float64) where {T<:Real}
    env = workspace.batched_environment[]
    if !(env isa BatchedPlanEnvironment{T}) || env.batch_size != batch_size
        env = BatchedPlanEnvironment(
            workspace.environment.layout,
            workspace.backend_plan.numeric_slots,
            workspace.backend_plan.index_slots,
            workspace.backend_plan.generic_slots,
            batch_size,
            T,
        )
        workspace.batched_environment[] = env
    end
    return env
end

function _batched_argument_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    buffer = workspace.batched_argument_buffer[]
    if !(buffer isa Vector{Any}) || length(buffer) != batch_size
        buffer = Vector{Any}(undef, batch_size)
        workspace.batched_argument_buffer[] = buffer
    end
    return buffer
end

function _prepare_batched_environment!(
    workspace::BatchedLogjointWorkspace,
    args,
    batch_size::Int,
    ::Type{T}=Float64,
) where {T<:Real}
    env = _batched_environment!(workspace, batch_size, T)
    fill!(env.assigned, false)

    if args isa Tuple
        length(args) == workspace.argument_count ||
            throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(args))"))
        for (slot, value) in zip(workspace.argument_slots, args)
            _batched_environment_set_shared!(env, slot, value)
        end
    else
        length(args) == batch_size ||
            throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
        values = _batched_argument_buffer!(workspace, batch_size)
        for argument_index in 1:workspace.argument_count
            slot = workspace.argument_slots[argument_index]
            for batch_index in 1:batch_size
                batch_args = args[batch_index]
                length(batch_args) == workspace.argument_count ||
                    throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(batch_args))"))
                values[batch_index] = batch_args[argument_index]
            end
            _batched_environment_set!(env, slot, values)
        end
    end

    return env
end

function _constrained_buffer!(workspace::BatchedLogjointWorkspace, params::AbstractVector)
    buffer = workspace.constrained_buffer[]
    if !(buffer isa AbstractVector) || length(buffer) != workspace.parameter_count || eltype(buffer) != eltype(params)
        buffer = similar(params, workspace.parameter_count)
        workspace.constrained_buffer[] = buffer
    end
    return buffer
end

function _batched_totals_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    return _batched_totals_buffer!(workspace, batch_size, Float64)
end

function _batched_totals_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}) where {T<:Real}
    buffer = workspace.batched_totals_buffer[]
    if !(buffer isa Vector{T}) || length(buffer) != batch_size
        buffer = zeros(T, batch_size)
        workspace.batched_totals_buffer[] = buffer
    else
        fill!(buffer, zero(T))
    end
    return buffer
end

function _batched_constrained_buffer!(workspace::BatchedLogjointWorkspace, parameter_count::Int, batch_size::Int)
    return _batched_constrained_buffer!(workspace, parameter_count, batch_size, Float64)
end

function _batched_constrained_buffer!(
    workspace::BatchedLogjointWorkspace,
    parameter_count::Int,
    batch_size::Int,
    ::Type{T},
) where {T<:Real}
    buffer = workspace.batched_constrained_buffer[]
    if !(buffer isa Matrix{T}) || size(buffer) != (parameter_count, batch_size)
        buffer = Matrix{T}(undef, parameter_count, batch_size)
        workspace.batched_constrained_buffer[] = buffer
    end
    return buffer
end

function _batched_logabsdet_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    return _batched_logabsdet_buffer!(workspace, batch_size, Float64)
end

function _batched_logabsdet_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}) where {T<:Real}
    buffer = workspace.batched_logabsdet_buffer[]
    if !(buffer isa Vector{T}) || length(buffer) != batch_size
        buffer = zeros(T, batch_size)
        workspace.batched_logabsdet_buffer[] = buffer
    else
        fill!(buffer, zero(T))
    end
    return buffer
end

function _logjoint_with_workspace!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    length(params) == workspace.parameter_count ||
        throw(DimensionMismatch("expected $(workspace.parameter_count) parameters, got $(length(params))"))
    env = _prepare_environment!(workspace, args)
    if !isnothing(workspace.backend_plan)
        return _score_backend_steps(workspace.backend_plan.steps, env, params, constraints)
    end
    return _score_compiled_steps(workspace.compiled_plan.steps, env, params, constraints)
end

function _logjoint_with_batched_backend!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    env = _prepare_batched_environment!(workspace, args, batch_size, eltype(params))
    totals = _batched_totals_buffer!(workspace, batch_size, eltype(params))
    _score_backend_steps!(totals, workspace.backend_plan.steps, env, params, constraints)
    return totals
end

function _fallback_batched_logjoint!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    values = Vector{Float64}(undef, batch_size)
    for batch_index in 1:batch_size
        values[batch_index] = _logjoint_with_workspace!(
            workspace,
            view(params, :, batch_index),
            _batched_args(args, batch_index),
            _batched_constraints(constraints, batch_index),
        )
    end
    return values
end

function _logjoint_unconstrained_with_workspace!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    length(params) == workspace.parameter_count ||
        throw(DimensionMismatch("expected $(workspace.parameter_count) parameters, got $(length(params))"))

    layout = parameterlayout(model)
    constrained = _constrained_buffer!(workspace, params)
    logabsdet = workspace.parameter_count == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        unconstrained_value = params[slot.index]
        constrained[slot.index] = to_constrained(slot.transform, unconstrained_value)
        logabsdet += logabsdetjac(slot.transform, unconstrained_value)
    end
    return _logjoint_with_workspace!(workspace, constrained, args, constraints) + logabsdet
end

function _logjoint_unconstrained_batched_backend!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    parameter_count, batch_size = size(params)
    length(destination) == batch_size ||
        throw(DimensionMismatch("expected unconstrained batched destination of length $batch_size, got $(length(destination))"))
    layout = parameterlayout(model)
    value_type = eltype(destination)
    constrained = _batched_constrained_buffer!(workspace, parameter_count, batch_size, value_type)
    logabsdet = _batched_logabsdet_buffer!(workspace, batch_size, value_type)
    for slot in layout.slots
        slot_index = slot.index
        for batch_index in 1:batch_size
            unconstrained_value = params[slot_index, batch_index]
            constrained[slot_index, batch_index] = to_constrained(slot.transform, unconstrained_value)
            logabsdet[batch_index] += logabsdetjac(slot.transform, unconstrained_value)
        end
    end
    totals = _logjoint_with_batched_backend!(workspace, constrained, args, constraints)
    for batch_index in 1:batch_size
        destination[batch_index] = totals[batch_index] + logabsdet[batch_index]
    end
    return destination
end

function _logjoint_unconstrained_batched_backend!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _logjoint_unconstrained_batched_backend!(values, model, workspace, params, args, constraints)
end

function _fallback_batched_logjoint_unconstrained!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    length(destination) == batch_size ||
        throw(DimensionMismatch("expected unconstrained batched destination of length $batch_size, got $(length(destination))"))
    for batch_index in 1:batch_size
        destination[batch_index] = _logjoint_unconstrained_with_workspace!(
            model,
            workspace,
            view(params, :, batch_index),
            _batched_args(args, batch_index),
            _batched_constraints(constraints, batch_index),
        )
    end
    return destination
end

function _fallback_batched_logjoint_unconstrained!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _fallback_batched_logjoint_unconstrained!(values, model, workspace, params, args, constraints)
end

function _batched_logjoint_unconstrained_with_workspace!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    if !isnothing(workspace.backend_plan)
        try
            return _logjoint_unconstrained_batched_backend!(destination, model, workspace, params, args, constraints)
        catch err
            if !(err isa BatchedBackendFallback)
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint_unconstrained!(destination, model, workspace, params, args, constraints)
end

function _batched_logjoint_unconstrained_with_workspace!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _batched_logjoint_unconstrained_with_workspace!(values, model, workspace, params, args, constraints)
end

const BACKEND_GRADIENT_SUPPORTED_PRIMITIVES = Set([:+, :-, :*, :/, :exp, :log, :log1p, :sqrt])

_backend_gradient_supported_expr(::BackendLiteralExpr) = true
_backend_gradient_supported_expr(::BackendSlotExpr) = true
_backend_gradient_supported_expr(::BackendTupleExpr) = false

function _backend_gradient_supported_expr(expr::BackendPrimitiveExpr)
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES || return false
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_expr(expr::BackendBlockExpr)
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_step(step::BackendNormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendLognormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendDeterministicPlanStep, numeric_slots::BitVector)
    return numeric_slots[step.binding_slot] ? _backend_gradient_supported_expr(step.expr) : true
end

function _backend_gradient_supported_step(step::BackendLoopPlanStep, numeric_slots::BitVector)
    return all(inner -> _backend_gradient_supported_step(inner, numeric_slots), step.body)
end

_backend_gradient_supported_step(step::BackendNormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLognormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)

function _backend_gradient_supported(plan::BackendExecutionPlan)
    return all(step -> _backend_gradient_supported_step(step, plan.numeric_slots), plan.steps)
end

function _batched_backend_gradient_scratch!(cache::BatchedBackendGradientCache, depth::Int)
    depth > 0 || throw(ArgumentError("batched backend gradient scratch depth must be positive"))
    parameter_count = size(cache.slot_gradients, 1)
    batch_size = size(cache.slot_gradients, 3)
    while length(cache.gradient_scratch) < depth
        push!(cache.gradient_scratch, Matrix{Float64}(undef, parameter_count, batch_size))
    end
    buffer = cache.gradient_scratch[depth]
    if size(buffer) != (parameter_count, batch_size)
        buffer = Matrix{Float64}(undef, parameter_count, batch_size)
        cache.gradient_scratch[depth] = buffer
    end
    return buffer
end

function _zero_gradient!(destination::AbstractMatrix{Float64})
    fill!(destination, 0.0)
    return destination
end

function _copy_slot_gradient!(
    destination::AbstractMatrix{Float64},
    slot_gradients::Array{Float64,3},
    slot::Int,
)
    for batch_index in axes(destination, 2), parameter_index in axes(destination, 1)
        destination[parameter_index, batch_index] = slot_gradients[parameter_index, slot, batch_index]
    end
    return destination
end

function _store_slot_gradient!(
    slot_gradients::Array{Float64,3},
    slot::Int,
    source::AbstractMatrix{Float64},
)
    for batch_index in axes(source, 2), parameter_index in axes(source, 1)
        slot_gradients[parameter_index, slot, batch_index] = source[parameter_index, batch_index]
    end
    return slot_gradients
end

function _fill_choice_gradient!(
    destination::AbstractMatrix{Float64},
    parameter_slot::Union{Nothing,Int},
)
    fill!(destination, 0.0)
    isnothing(parameter_slot) && return destination
    for batch_index in axes(destination, 2)
        destination[parameter_slot, batch_index] = 1.0
    end
    return destination
end

function _apply_backend_numeric_gradient_unary!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    env::BatchedPlanEnvironment{Float64},
    op::Symbol,
)
    if op === :-
        for batch_index in eachindex(values)
            values[batch_index] = -values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] = -gradients[parameter_index, batch_index]
        end
    elseif op === :exp
        for batch_index in eachindex(values)
            value = exp(values[batch_index])
            values[batch_index] = value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= value
            end
        end
    elseif op === :log
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log(value)
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= value
            end
        end
    elseif op === :log1p
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log1p(value)
            denominator = 1 + value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= denominator
            end
        end
    elseif op === :sqrt
        for batch_index in eachindex(values)
            root = sqrt(values[batch_index])
            values[batch_index] = root
            factor = 2 * root
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= factor
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support unary primitive `$(op)`")
    end
    return values, gradients
end

function _apply_backend_numeric_gradient_binary!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    env::BatchedPlanEnvironment{Float64},
    op::Symbol,
    rhs_values::AbstractVector{Float64},
    rhs_gradients::AbstractMatrix{Float64},
)
    if op === :+
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] += rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :-
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] -= rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] -= rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :*
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    gradients[parameter_index, batch_index] * rhs_value +
                    lhs_value * rhs_gradients[parameter_index, batch_index]
            end
            values[batch_index] = lhs_value * rhs_value
        end
    elseif op === :/
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            denominator = rhs_value * rhs_value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    (gradients[parameter_index, batch_index] * rhs_value -
                     lhs_value * rhs_gradients[parameter_index, batch_index]) / denominator
            end
            values[batch_index] = lhs_value / rhs_value
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support binary primitive `$(op)`")
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(values, _require_numeric_value(env, expr.value, "batched backend numeric expression"))
    fill!(gradients, 0.0)
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendSlotExpr,
    depth::Int=1,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(values, view(env.numeric_values, expr.slot, :))
        _copy_slot_gradient!(gradients, cache.slot_gradients, expr.slot)
        return values, gradients
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(values)
            values[batch_index] = Float64(env.index_values[expr.slot, batch_index])
        end
        fill!(gradients, 0.0)
        return values, gradients
    end
    _backend_numeric_error(env, "batched backend gradient slot $(expr.slot) is not numeric")
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES ||
        _backend_numeric_error(env, "batched backend gradient does not support primitive `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched backend gradient primitive requires arguments")

    _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_gradient_unary!(values, gradients, env, expr.op)
    end

    temp_values = _batched_numeric_scratch!(env, depth)
    temp_gradients = _batched_backend_gradient_scratch!(cache, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_numeric_expr_and_gradient!(temp_values, temp_gradients, cache, env, argument, depth + 1)
        _apply_backend_numeric_gradient_binary!(values, gradients, env, expr.op, temp_values, temp_gradients)
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_numeric_error(env, "batched backend gradient expression cannot be a tuple")
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for argument in expr.arguments
        _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, argument, depth)
    end
    return values, gradients
end

function _set_numeric_binding!(
    env::BatchedPlanEnvironment{Float64},
    slot_gradients::Array{Float64,3},
    slot::Int,
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
)
    copyto!(view(env.numeric_values, slot, :), values)
    _store_slot_gradient!(slot_gradients, slot, gradients)
    env.assigned[slot] = true
    return env
end

function _score_backend_steps_and_gradient!(
    ::Tuple{},
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    return totals, gradients
end

function _score_backend_steps_and_gradient!(
    steps::Tuple,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    _score_backend_step_and_gradient!(first(steps), totals, gradients, cache, env, params, constraints)
    return _score_backend_steps_and_gradient!(Base.tail(steps), totals, gradients, cache, env, params, constraints)
end

function _accumulate_normal_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    sigma_values::AbstractVector{Float64},
    sigma_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
        z = (value - mu) / sigma
        inv_sigma = 1 / sigma
        dvalue = -z * inv_sigma
        dmu = z * inv_sigma
        dsigma = (z * z - 1) * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_lognormal_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    sigma_values::AbstractVector{Float64},
    sigma_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
        if !(value > 0)
            continue
        end
        log_value = log(value)
        z = (log_value - mu) / sigma
        inv_sigma = 1 / sigma
        dvalue = (-(z * inv_sigma) - 1) / value
        dmu = z * inv_sigma
        dsigma = (z * z - 1) * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_bernoulli_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    probability_values::AbstractVector{Float64},
    probability_gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
)
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        derivative = value != 0 ? 1 / probability : -1 / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_normal_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, sigma_values, sigma_gradients)
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_lognormal_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, sigma_values, sigma_gradients)
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    isnothing(step.parameter_slot) || throw(BatchedBackendFallback("batched backend gradient does not support Bernoulli latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        if env.numeric_slots[step.binding_slot]
            copyto!(view(env.numeric_values, step.binding_slot, :), value_values)
        elseif env.index_slots[step.binding_slot]
            for batch_index in 1:env.batch_size
                env.index_values[step.binding_slot, batch_index] = Int(round(value_values[batch_index]))
            end
        else
            values = env.generic_values[step.binding_slot]
            for batch_index in 1:env.batch_size
                values[batch_index] = value_values[batch_index]
            end
        end
        env.assigned[step.binding_slot] = true
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendDeterministicPlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    if env.numeric_slots[step.binding_slot]
        values = view(env.numeric_values, step.binding_slot, :)
        slot_gradients = view(cache.slot_gradients, :, step.binding_slot, :)
        _eval_backend_numeric_expr_and_gradient!(values, slot_gradients, cache, env, step.expr)
        env.assigned[step.binding_slot] = true
        return totals, gradients
    end

    _score_backend_step!(step, totals, env, params, constraints)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLoopPlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    reference_iterable = _batched_index_iterable_reference(env, step.iterable)
    had_previous = env.assigned[step.iterator_slot]
    previous_value = had_previous ? copy(env.index_values[step.iterator_slot, :]) : Int[]

    for item in reference_iterable
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        _score_backend_steps_and_gradient!(step.body, totals, gradients, cache, env, params, constraints)
    end

    _batched_environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return totals, gradients
end

function _batched_backend_logjoint_and_gradient_unconstrained!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    model::TeaModel,
    cache::BatchedBackendGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == size(gradients, 1) ||
        throw(DimensionMismatch("expected $(size(gradients, 1)) parameters, got $(size(params, 1))"))
    size(params, 2) == size(gradients, 2) ||
        throw(DimensionMismatch("expected $(size(gradients, 2)) batch elements, got $(size(params, 2))"))

    workspace = cache.workspace
    env = _prepare_batched_environment!(workspace, cache.args, size(params, 2), Float64)
    fill!(totals, 0.0)
    fill!(gradients, 0.0)
    fill!(cache.slot_gradients, 0.0)

    layout = parameterlayout(model)
    constrained = _batched_constrained_buffer!(workspace, size(params, 1), size(params, 2), Float64)
    logabsdet = _batched_logabsdet_buffer!(workspace, size(params, 2), Float64)
    for slot in layout.slots
        slot_index = slot.index
        if slot.transform isa IdentityTransform
            for batch_index in 1:size(params, 2)
                constrained[slot_index, batch_index] = Float64(params[slot_index, batch_index])
            end
        elseif slot.transform isa LogTransform
            for batch_index in 1:size(params, 2)
                unconstrained_value = Float64(params[slot_index, batch_index])
                constrained_value = exp(unconstrained_value)
                constrained[slot_index, batch_index] = constrained_value
                logabsdet[batch_index] += unconstrained_value
            end
        else
            throw(BatchedBackendFallback("batched backend gradient does not support transform $(typeof(slot.transform))"))
        end
    end

    _score_backend_steps_and_gradient!(workspace.backend_plan.steps, totals, gradients, cache, env, constrained, cache.constraints)

    for slot in layout.slots
        if slot.transform isa IdentityTransform
            continue
        elseif slot.transform isa LogTransform
            slot_index = slot.index
            for batch_index in 1:size(params, 2)
                gradients[slot_index, batch_index] =
                    gradients[slot_index, batch_index] * constrained[slot_index, batch_index] + 1.0
            end
        end
    end

    for batch_index in eachindex(totals)
        totals[batch_index] += logabsdet[batch_index]
    end
    return totals, gradients
end

function _batched_backend_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
)
    backend_plan = _backend_execution_plan(model)
    isnothing(backend_plan) && return nothing
    _backend_gradient_supported(backend_plan) || return nothing

    workspace = BatchedLogjointWorkspace(model)
    cache = BatchedBackendGradientCache(
        workspace,
        zeros(Float64, size(params, 1), length(workspace.environment.layout.symbols), size(params, 2)),
        Matrix{Float64}[],
        batch_args,
        batch_constraints,
    )
    totals = _batched_totals_buffer!(workspace, size(params, 2), Float64)
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(totals, gradient_buffer, model, cache, params)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    return cache
end

function _batched_gradient_column!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    destination::AbstractVector,
    seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    objective = theta -> _logjoint_unconstrained_with_workspace!(model, workspace, theta, args, constraints)
    config = ForwardDiff.GradientConfig(objective, seed)
    ForwardDiff.gradient!(destination, objective, seed, config)
    return destination
end

function _batched_gradient_column_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
    batch_index::Int,
)
    workspace = BatchedLogjointWorkspace(model)
    objective = BatchedGradientObjective(
        model,
        workspace,
        _batched_args(batch_args, batch_index),
        _batched_constraints(batch_constraints, batch_index),
    )
    seed = collect(view(params, :, batch_index))
    config = ForwardDiff.GradientConfig(objective, seed)
    buffer = view(gradient_buffer, :, batch_index)
    return BatchedGradientColumnCache(objective, config, buffer)
end

function _batched_flat_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
)
    workspace = BatchedLogjointWorkspace(model)
    objective = BatchedFlatGradientObjective(
        model,
        workspace,
        batch_args,
        batch_constraints,
        size(params, 1),
        size(params, 2),
    )
    seed = collect(vec(params))
    probe = _batched_totals_buffer!(workspace, size(params, 2), eltype(params))
    try
        _batched_logjoint_unconstrained_with_workspace!(probe, model, workspace, params, batch_args, batch_constraints)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    config = ForwardDiff.GradientConfig(objective, seed)
    return BatchedFlatGradientCache(objective, config, vec(gradient_buffer))
end

function BatchedLogjointGradientCache(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_params(model, params)
    batch_args = _validate_batched_args(args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    parameter_count = size(params, 1)
    gradient_buffer = Matrix{Float64}(undef, parameter_count, batch_size)
    if batch_size == 0
        return BatchedLogjointGradientCache(model, Any[], nothing, nothing, gradient_buffer, parameter_count, batch_size)
    end

    backend_cache = _batched_backend_gradient_cache(model, gradient_buffer, params, batch_args, batch_constraints)
    if !isnothing(backend_cache)
        return BatchedLogjointGradientCache(model, Any[], backend_cache, nothing, gradient_buffer, parameter_count, batch_size)
    end

    flat_cache = isnothing(_backend_execution_plan(model)) ? nothing :
        _batched_flat_gradient_cache(model, gradient_buffer, params, batch_args, batch_constraints)
    if !isnothing(flat_cache)
        return BatchedLogjointGradientCache(model, Any[], nothing, flat_cache, gradient_buffer, parameter_count, batch_size)
    end

    first_cache = _batched_gradient_column_cache(model, gradient_buffer, params, batch_args, batch_constraints, 1)
    column_caches = Vector{typeof(first_cache)}(undef, batch_size)
    column_caches[1] = first_cache
    for batch_index in 2:batch_size
        column_caches[batch_index] = _batched_gradient_column_cache(
            model,
            gradient_buffer,
            params,
            batch_args,
            batch_constraints,
            batch_index,
        )
    end

    return BatchedLogjointGradientCache(model, column_caches, nothing, nothing, gradient_buffer, parameter_count, batch_size)
end

function batched_logjoint_gradient_unconstrained!(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    if !isnothing(cache.backend_cache)
        totals = _batched_totals_buffer!(cache.backend_cache.workspace, cache.batch_size, Float64)
        _batched_backend_logjoint_and_gradient_unconstrained!(totals, cache.gradient_buffer, cache.model, cache.backend_cache, params)
        return cache.gradient_buffer
    end

    if !isnothing(cache.flat_cache)
        ForwardDiff.gradient!(cache.flat_cache.flat_buffer, cache.flat_cache.objective, vec(params), cache.flat_cache.config)
        return cache.gradient_buffer
    end

    for batch_index in 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        ForwardDiff.gradient!(column_cache.buffer, column_cache.objective, view(params, :, batch_index), column_cache.config)
    end
    return cache.gradient_buffer
end

function batched_logjoint_gradient_unconstrained(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    return copy(batched_logjoint_gradient_unconstrained!(cache, params))
end

function batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_params(model, params)
    batch_args = _validate_batched_args(args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return Float64[]

    workspace = BatchedLogjointWorkspace(model)
    if !isnothing(workspace.backend_plan)
        try
            return _logjoint_with_batched_backend!(workspace, params, batch_args, batch_constraints)
        catch err
            if !(err isa BatchedBackendFallback)
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint!(workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_params(model, params)
    batch_args = _validate_batched_args(args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return Float64[]

    workspace = BatchedLogjointWorkspace(model)
    values = Vector{Float64}(undef, batch_size)
    return _batched_logjoint_unconstrained_with_workspace!(values, model, workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    cache = BatchedLogjointGradientCache(model, params, args, constraints)
    return batched_logjoint_gradient_unconstrained(cache, params)
end
