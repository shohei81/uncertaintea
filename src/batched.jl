mutable struct BatchedLogjointWorkspace{P,E,B}
    compiled_plan::P
    environment::E
    parameter_count::Int
    argument_count::Int
    argument_slots::Vector{Int}
    constrained_buffer::B
end

function BatchedLogjointWorkspace(model::TeaModel)
    plan = executionplan(model)
    return BatchedLogjointWorkspace(
        _compiled_execution_plan(model),
        PlanEnvironment(plan.environment_layout),
        parametercount(plan.parameter_layout),
        length(modelspec(model).arguments),
        copy(plan.environment_layout.argument_slots),
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

function _constrained_buffer!(workspace::BatchedLogjointWorkspace, params::AbstractVector)
    buffer = workspace.constrained_buffer[]
    if !(buffer isa AbstractVector) || length(buffer) != workspace.parameter_count || eltype(buffer) != eltype(params)
        buffer = similar(params, workspace.parameter_count)
        workspace.constrained_buffer[] = buffer
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
    return _score_compiled_steps(workspace.compiled_plan.steps, env, params, constraints)
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
    values = Vector{Float64}(undef, batch_size)
    for batch_index in 1:batch_size
        values[batch_index] = _logjoint_with_workspace!(
            workspace,
            view(params, :, batch_index),
            _batched_args(batch_args, batch_index),
            _batched_constraints(batch_constraints, batch_index),
        )
    end
    return values
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
    for batch_index in 1:batch_size
        values[batch_index] = _logjoint_unconstrained_with_workspace!(
            model,
            workspace,
            view(params, :, batch_index),
            _batched_args(batch_args, batch_index),
            _batched_constraints(batch_constraints, batch_index),
        )
    end
    return values
end

function batched_logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_params(model, params)
    batch_args = _validate_batched_args(args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    gradients = Matrix{Float64}(undef, size(params, 1), batch_size)
    batch_size == 0 && return gradients

    workspace = BatchedLogjointWorkspace(model)
    for batch_index in 1:batch_size
        seed = collect(view(params, :, batch_index))
        _batched_gradient_column!(
            model,
            workspace,
            view(gradients, :, batch_index),
            seed,
            _batched_args(batch_args, batch_index),
            _batched_constraints(batch_constraints, batch_index),
        )
    end
    return gradients
end
