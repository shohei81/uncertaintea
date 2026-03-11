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

struct BatchedLogjointGradientCache{C,F,G<:AbstractMatrix}
    model::TeaModel
    column_caches::C
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
        return BatchedLogjointGradientCache(model, Any[], nothing, gradient_buffer, parameter_count, batch_size)
    end

    flat_cache = isnothing(_backend_execution_plan(model)) ? nothing :
        _batched_flat_gradient_cache(model, gradient_buffer, params, batch_args, batch_constraints)
    if !isnothing(flat_cache)
        return BatchedLogjointGradientCache(model, Any[], flat_cache, gradient_buffer, parameter_count, batch_size)
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

    return BatchedLogjointGradientCache(model, column_caches, nothing, gradient_buffer, parameter_count, batch_size)
end

function batched_logjoint_gradient_unconstrained!(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

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
