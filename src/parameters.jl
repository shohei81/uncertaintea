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
to_constrained(::LogTransform, value) = exp(value)
to_unconstrained(::LogTransform, value) = log(value)
logabsdetjac(::IdentityTransform, value) = zero(value)
logabsdetjac(::LogTransform, value) = value

function parameterchoicemap(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    cm = ChoiceMap()
    for slot in layout.slots
        _pushchoice!(cm, _static_address(slot.address), params[slot.index])
    end
    return cm
end

function parameter_vector(trace::TeaTrace)
    layout = parameterlayout(trace.model)
    params = Vector{Float64}(undef, parametercount(layout))
    for slot in layout.slots
        params[slot.index] = Float64(trace[_static_address(slot.address)])
    end
    return params
end

function transform_to_constrained(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, expected)
    for slot in layout.slots
        constrained[slot.index] = to_constrained(slot.transform, params[slot.index])
    end
    return constrained
end

function transform_to_constrained_with_logabsdet(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, expected)
    logabsdet = expected == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        unconstrained_value = params[slot.index]
        constrained[slot.index] = to_constrained(slot.transform, unconstrained_value)
        logabsdet += logabsdetjac(slot.transform, unconstrained_value)
    end
    return constrained, logabsdet
end

function transform_to_unconstrained(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    unconstrained = similar(params, expected)
    for slot in layout.slots
        unconstrained[slot.index] = to_unconstrained(slot.transform, params[slot.index])
    end
    return unconstrained
end

transform_to_unconstrained(trace::TeaTrace) = transform_to_unconstrained(trace.model, parameter_vector(trace))

function initialparameters(model::TeaModel, args::Tuple=(); rng::AbstractRNG=Random.default_rng())
    trace, _ = generate(model, args, choicemap(); rng=rng)
    return parameter_vector(trace)
end
