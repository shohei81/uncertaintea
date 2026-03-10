function _validate_batched_params(model::TeaModel, params::AbstractMatrix)
    expected = parametercount(parameterlayout(model))
    size(params, 1) == expected || throw(DimensionMismatch("expected $expected parameters, got $(size(params, 1))"))
    return size(params, 2)
end

function _batched_args(args::Tuple, index::Int, batch_size::Int)
    return args
end

function _batched_args(args::AbstractVector, index::Int, batch_size::Int)
    length(args) == batch_size || throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
    batch_args = args[index]
    batch_args isa Tuple || throw(ArgumentError("batched args must be a tuple or a vector of tuples"))
    return batch_args
end

function _batched_constraints(constraints::ChoiceMap, index::Int, batch_size::Int)
    return constraints
end

function _batched_constraints(constraints::AbstractVector, index::Int, batch_size::Int)
    length(constraints) == batch_size || throw(DimensionMismatch("expected $batch_size batched choicemaps, got $(length(constraints))"))
    batch_constraints = constraints[index]
    batch_constraints isa ChoiceMap || throw(ArgumentError("batched constraints must be a ChoiceMap or a vector of ChoiceMaps"))
    return batch_constraints
end

function batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_params(model, params)
    batch_size == 0 && return Float64[]

    values = Vector{Float64}(undef, batch_size)
    for batch_index in 1:batch_size
        values[batch_index] = logjoint(
            model,
            view(params, :, batch_index),
            _batched_args(args, batch_index, batch_size),
            _batched_constraints(constraints, batch_index, batch_size),
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
    batch_size == 0 && return Float64[]

    values = Vector{Float64}(undef, batch_size)
    for batch_index in 1:batch_size
        values[batch_index] = logjoint_unconstrained(
            model,
            view(params, :, batch_index),
            _batched_args(args, batch_index, batch_size),
            _batched_constraints(constraints, batch_index, batch_size),
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
    gradients = Matrix{Float64}(undef, size(params, 1), batch_size)
    for batch_index in 1:batch_size
        gradients[:, batch_index] = logjoint_gradient_unconstrained(
            model,
            view(params, :, batch_index),
            _batched_args(args, batch_index, batch_size),
            _batched_constraints(constraints, batch_index, batch_size),
        )
    end
    return gradients
end
