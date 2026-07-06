function _static_address(address::AddressSpec)
    parts = Any[]
    for part in address.parts
        part isa AddressLiteralPart || throw(ArgumentError("parameter vectors require static addresses, got $address"))
        push!(parts, part.value)
    end
    return Tuple(parts)
end

to_constrained(::IdentityTransform, value) = value
to_unconstrained(::IdentityTransform, value) = value
to_constrained(::VectorIdentityTransform, value::AbstractVector) = value
to_unconstrained(::VectorIdentityTransform, value::AbstractVector) = value
to_constrained(::LogTransform, value) = exp(value)
to_unconstrained(::LogTransform, value) = log(value)
to_constrained(::LogitTransform, value) = inv(one(value) + exp(-value))
to_unconstrained(::LogitTransform, value) = log(value) - log1p(-value)
logabsdetjac(::IdentityTransform, value) = zero(value)
logabsdetjac(::VectorIdentityTransform, value::AbstractVector) = isempty(value) ? 0.0 : zero(value[firstindex(value)])
logabsdetjac(::LogTransform, value) = value
function logabsdetjac(::LogitTransform, value)
    constrained = to_constrained(LogitTransform(), value)
    return log(constrained) + log1p(-constrained)
end

to_constrained(::VectorLogTransform, value::AbstractVector) = map(exp, value)
to_unconstrained(::VectorLogTransform, value::AbstractVector) = map(log, value)
function logabsdetjac(::VectorLogTransform, value::AbstractVector)
    isempty(value) && return 0.0
    total = zero(value[firstindex(value)])
    for element in value
        total += element
    end
    return total
end

to_constrained(::VectorLogitTransform, value::AbstractVector) = map(v -> inv(one(v) + exp(-v)), value)
to_unconstrained(::VectorLogitTransform, value::AbstractVector) = map(v -> log(v) - log1p(-v), value)
function logabsdetjac(::VectorLogitTransform, value::AbstractVector)
    isempty(value) && return 0.0
    total = zero(value[firstindex(value)])
    for element in value
        constrained = inv(one(element) + exp(-element))
        total += log(constrained) + log1p(-constrained)
    end
    return total
end

_bounded_sigmoid(value) = inv(one(value) + exp(-value))
to_constrained(transform::BoundedTransform, value) =
    transform.lower + (transform.upper - transform.lower) * _bounded_sigmoid(value)
function to_unconstrained(transform::BoundedTransform, value)
    scaled = (value - transform.lower) / (transform.upper - transform.lower)
    return log(scaled) - log1p(-scaled)
end
function logabsdetjac(transform::BoundedTransform, value)
    sigmoid = _bounded_sigmoid(value)
    return log(transform.upper - transform.lower) + log(sigmoid) + log1p(-sigmoid)
end

to_constrained(transform::LowerBoundedTransform, value) = transform.lower + exp(value)
to_unconstrained(transform::LowerBoundedTransform, value) = log(value - transform.lower)
logabsdetjac(::LowerBoundedTransform, value) = value

to_constrained(transform::UpperBoundedTransform, value) = transform.upper - exp(value)
to_unconstrained(transform::UpperBoundedTransform, value) = log(transform.upper - value)
logabsdetjac(::UpperBoundedTransform, value) = value

function _simplex_logabsdet(constrained::AbstractVector)
    total = zero(eltype(constrained))
    for value in constrained
        total += log(value)
    end
    return total
end

function _to_constrained_simplex!(
    destination::AbstractVector,
    transform::SimplexTransform,
    values::AbstractVector,
)
    length(values) == transform.size - 1 ||
        throw(DimensionMismatch("expected $(transform.size - 1) unconstrained simplex values, got $(length(values))"))
    length(destination) == transform.size ||
        throw(DimensionMismatch("expected simplex destination of length $(transform.size), got $(length(destination))"))

    max_value = zero(promote_type(eltype(destination), eltype(values)))
    for value in values
        max_value = max(max_value, value)
    end

    total = exp(-max_value)
    for index in eachindex(values)
        shifted = exp(values[index] - max_value)
        destination[index] = shifted
        total += shifted
    end
    destination[end] = exp(-max_value)
    for index in eachindex(destination)
        destination[index] /= total
    end
    return destination
end

function _to_unconstrained_simplex!(
    destination::AbstractVector,
    transform::SimplexTransform,
    values::AbstractVector,
)
    length(values) == transform.size ||
        throw(DimensionMismatch("expected simplex values of length $(transform.size), got $(length(values))"))
    length(destination) == transform.size - 1 ||
        throw(DimensionMismatch("expected simplex unconstrained destination of length $(transform.size - 1), got $(length(destination))"))

    last_value = values[end]
    last_value > zero(last_value) || throw(ArgumentError("simplex values must be strictly positive"))
    total = zero(last_value)
    for value in values
        value > zero(value) || throw(ArgumentError("simplex values must be strictly positive"))
        total += value
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * transform.size * 16 ||
        throw(ArgumentError("simplex values must sum to 1"))

    log_last = log(last_value)
    for index in eachindex(destination)
        destination[index] = log(values[index]) - log_last
    end
    return destination
end

function to_constrained(transform::SimplexTransform, value::AbstractVector)
    destination = similar(collect(value), transform.size)
    return _to_constrained_simplex!(destination, transform, value)
end

function to_unconstrained(transform::SimplexTransform, value::AbstractVector)
    destination = similar(collect(value), transform.size - 1)
    return _to_unconstrained_simplex!(destination, transform, value)
end

function logabsdetjac(transform::SimplexTransform, value::AbstractVector)
    constrained = to_constrained(transform, value)
    return _simplex_logabsdet(constrained)
end

function _slot_parameter_values(params::AbstractVector, slot::ParameterSlotSpec)
    indices = parametervalueindices(slot)
    if slot.value_length == 1
        return params[first(indices)]
    end
    return collect(view(params, indices))
end

function _write_slot_value!(destination::AbstractVector, slot::ParameterSlotSpec, value)
    indices = parametervalueindices(slot)
    if slot.value_length == 1
        destination[first(indices)] = value
        return destination
    end

    if value isa AbstractVector || value isa Tuple
        length(value) == slot.value_length ||
            throw(DimensionMismatch("expected $(slot.value_length) values for choice $(slot.binding), got $(length(value))"))
        for (index, item) in zip(indices, value)
            destination[index] = Float64(item)
        end
        return destination
    end

    throw(ArgumentError("parameter slot $(slot.binding) expected a vector-like value, got $(typeof(value))"))
end

function _transform_slot_to_constrained!(
    destination::AbstractVector,
    slot::ParameterSlotSpec,
    params::AbstractVector,
)
    if slot.transform isa IdentityTransform
        destination[slot.value_index] = params[slot.index]
        return zero(params[slot.index])
    elseif slot.transform isa VectorIdentityTransform
        copyto!(view(destination, parametervalueindices(slot)), view(params, parameterindices(slot)))
        return zero(params[first(parameterindices(slot))])
    elseif slot.transform isa VectorLogTransform
        logabsdet = zero(params[first(parameterindices(slot))])
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            unconstrained_value = params[parameter_index]
            destination[value_index] = exp(unconstrained_value)
            logabsdet += unconstrained_value
        end
        return logabsdet
    elseif slot.transform isa VectorLogitTransform
        logabsdet = zero(params[first(parameterindices(slot))])
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            unconstrained_value = params[parameter_index]
            constrained_value = _bounded_sigmoid(unconstrained_value)
            destination[value_index] = constrained_value
            logabsdet += log(constrained_value) + log1p(-constrained_value)
        end
        return logabsdet
    elseif slot.transform isa LogTransform
        unconstrained_value = params[slot.index]
        destination[slot.value_index] = exp(unconstrained_value)
        return unconstrained_value
    elseif slot.transform isa LogitTransform
        unconstrained_value = params[slot.index]
        constrained_value = to_constrained(slot.transform, unconstrained_value)
        destination[slot.value_index] = constrained_value
        return logabsdetjac(slot.transform, unconstrained_value)
    elseif slot.transform isa BoundedTransform ||
           slot.transform isa LowerBoundedTransform ||
           slot.transform isa UpperBoundedTransform
        unconstrained_value = params[slot.index]
        constrained_value = to_constrained(slot.transform, unconstrained_value)
        destination[slot.value_index] = constrained_value
        return logabsdetjac(slot.transform, unconstrained_value)
    elseif slot.transform isa SimplexTransform
        unconstrained = view(params, parameterindices(slot))
        constrained = view(destination, parametervalueindices(slot))
        _to_constrained_simplex!(constrained, slot.transform, unconstrained)
        return _simplex_logabsdet(constrained)
    end

    throw(ArgumentError("unsupported parameter transform $(typeof(slot.transform))"))
end

function _transform_slot_to_unconstrained!(
    destination::AbstractVector,
    slot::ParameterSlotSpec,
    params::AbstractVector,
)
    if slot.transform isa IdentityTransform
        destination[slot.index] = params[slot.value_index]
    elseif slot.transform isa VectorIdentityTransform
        copyto!(view(destination, parameterindices(slot)), view(params, parametervalueindices(slot)))
    elseif slot.transform isa VectorLogTransform
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            destination[parameter_index] = log(params[value_index])
        end
    elseif slot.transform isa VectorLogitTransform
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            constrained_value = params[value_index]
            destination[parameter_index] = log(constrained_value) - log1p(-constrained_value)
        end
    elseif slot.transform isa LogTransform
        destination[slot.index] = log(params[slot.value_index])
    elseif slot.transform isa LogitTransform
        destination[slot.index] = to_unconstrained(slot.transform, params[slot.value_index])
    elseif slot.transform isa BoundedTransform ||
           slot.transform isa LowerBoundedTransform ||
           slot.transform isa UpperBoundedTransform
        destination[slot.index] = to_unconstrained(slot.transform, params[slot.value_index])
    elseif slot.transform isa SimplexTransform
        unconstrained = view(destination, parameterindices(slot))
        constrained = view(params, parametervalueindices(slot))
        _to_unconstrained_simplex!(unconstrained, slot.transform, constrained)
    else
        throw(ArgumentError("unsupported parameter transform $(typeof(slot.transform))"))
    end
    return destination
end

function parameterchoicemap(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametervaluecount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    cm = ChoiceMap()
    for slot in layout.slots
        _pushchoice!(cm, _static_address(slot.address), _slot_parameter_values(params, slot))
    end
    return cm
end

function parameter_vector(trace::TeaTrace)
    layout = parameterlayout(trace.model)
    params = Vector{Float64}(undef, parametervaluecount(layout))
    for slot in layout.slots
        _write_slot_value!(params, slot, trace[_static_address(slot.address)])
    end
    return params
end

function transform_to_constrained(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, parametervaluecount(layout))
    _transform_to_constrained!(constrained, model, params)
    return constrained
end

function _transform_to_constrained!(destination::AbstractVector, model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))
    constrained_expected = parametervaluecount(layout)
    length(destination) == constrained_expected ||
        throw(DimensionMismatch("expected constrained destination of length $constrained_expected, got $(length(destination))"))

    for slot in layout.slots
        _transform_slot_to_constrained!(destination, slot, params)
    end
    return destination
end

function transform_to_constrained_with_logabsdet(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, parametervaluecount(layout))
    logabsdet = expected == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        logabsdet += _transform_slot_to_constrained!(constrained, slot, params)
    end
    return constrained, logabsdet
end

function transform_to_unconstrained(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametervaluecount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    unconstrained = similar(params, parametercount(layout))
    for slot in layout.slots
        _transform_slot_to_unconstrained!(unconstrained, slot, params)
    end
    return unconstrained
end

transform_to_unconstrained(trace::TeaTrace) = transform_to_unconstrained(trace.model, parameter_vector(trace))

function initialparameters(model::TeaModel, args::Tuple=(); rng::AbstractRNG=Random.default_rng())
    trace, _ = generate(model, args, choicemap(); rng=rng)
    return parameter_vector(trace)
end
