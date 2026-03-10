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

struct BatchedGradientColumnCache{F,C,V,W}
    objective::F
    config::C
    buffer::V
    workspace::W
end

struct BatchedLogjointGradientCache
    model::TeaModel
    column_caches::Vector{Any}
    gradient_buffer::Matrix{Float64}
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

function _batched_environment!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    env = workspace.batched_environment[]
    if !(env isa BatchedPlanEnvironment) || env.batch_size != batch_size
        env = BatchedPlanEnvironment(
            workspace.environment.layout,
            workspace.backend_plan.numeric_slots,
            workspace.backend_plan.index_slots,
            workspace.backend_plan.generic_slots,
            batch_size,
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

function _prepare_batched_environment!(workspace::BatchedLogjointWorkspace, args, batch_size::Int)
    env = _batched_environment!(workspace, batch_size)
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
    buffer = workspace.batched_totals_buffer[]
    if !(buffer isa Vector{Float64}) || length(buffer) != batch_size
        buffer = zeros(Float64, batch_size)
        workspace.batched_totals_buffer[] = buffer
    else
        fill!(buffer, 0.0)
    end
    return buffer
end

function _batched_constrained_buffer!(workspace::BatchedLogjointWorkspace, parameter_count::Int, batch_size::Int)
    buffer = workspace.batched_constrained_buffer[]
    if !(buffer isa Matrix{Float64}) || size(buffer) != (parameter_count, batch_size)
        buffer = Matrix{Float64}(undef, parameter_count, batch_size)
        workspace.batched_constrained_buffer[] = buffer
    end
    return buffer
end

function _batched_logabsdet_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    buffer = workspace.batched_logabsdet_buffer[]
    if !(buffer isa Vector{Float64}) || length(buffer) != batch_size
        buffer = zeros(Float64, batch_size)
        workspace.batched_logabsdet_buffer[] = buffer
    else
        fill!(buffer, 0.0)
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
    env = _prepare_batched_environment!(workspace, args, batch_size)
    totals = _batched_totals_buffer!(workspace, batch_size)
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
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    parameter_count, batch_size = size(params)
    layout = parameterlayout(model)
    constrained = _batched_constrained_buffer!(workspace, parameter_count, batch_size)
    logabsdet = _batched_logabsdet_buffer!(workspace, batch_size)
    for slot in layout.slots
        slot_index = slot.index
        for batch_index in 1:batch_size
            unconstrained_value = params[slot_index, batch_index]
            constrained[slot_index, batch_index] = to_constrained(slot.transform, unconstrained_value)
            logabsdet[batch_index] += logabsdetjac(slot.transform, unconstrained_value)
        end
    end
    return _logjoint_with_batched_backend!(workspace, constrained, args, constraints) .+ logabsdet
end

function _fallback_batched_logjoint_unconstrained!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    values = Vector{Float64}(undef, batch_size)
    for batch_index in 1:batch_size
        values[batch_index] = _logjoint_unconstrained_with_workspace!(
            model,
            workspace,
            view(params, :, batch_index),
            _batched_args(args, batch_index),
            _batched_constraints(constraints, batch_index),
        )
    end
    return values
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
    column_caches = Vector{Any}(undef, batch_size)

    for batch_index in 1:batch_size
        workspace = BatchedLogjointWorkspace(model)
        batch_args_i = _batched_args(batch_args, batch_index)
        batch_constraints_i = _batched_constraints(batch_constraints, batch_index)
        seed = collect(view(params, :, batch_index))
        objective = theta -> _logjoint_unconstrained_with_workspace!(model, workspace, theta, batch_args_i, batch_constraints_i)
        config = ForwardDiff.GradientConfig(objective, seed)
        buffer = similar(seed)
        column_caches[batch_index] = BatchedGradientColumnCache(objective, config, buffer, workspace)
    end

    return BatchedLogjointGradientCache(model, column_caches, gradient_buffer, parameter_count, batch_size)
end

function batched_logjoint_gradient_unconstrained!(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    for batch_index in 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        ForwardDiff.gradient!(column_cache.buffer, column_cache.objective, view(params, :, batch_index), column_cache.config)
        cache.gradient_buffer[:, batch_index] = column_cache.buffer
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
    if !isnothing(workspace.backend_plan)
        try
            return _logjoint_unconstrained_batched_backend!(model, workspace, params, batch_args, batch_constraints)
        catch err
            if !(err isa BatchedBackendFallback)
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint_unconstrained!(model, workspace, params, batch_args, batch_constraints)
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
