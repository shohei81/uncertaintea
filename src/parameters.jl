function _static_address(address::AddressSpec)
    parts = Any[]
    for part in address.parts
        part isa AddressLiteralPart || throw(ArgumentError("parameter vectors require static addresses, got $address"))
        push!(parts, part.value)
    end
    return Tuple(parts)
end

to_constrained(::IdentityTransform, value) = Float64(value)
to_unconstrained(::IdentityTransform, value) = Float64(value)
to_constrained(::LogTransform, value) = exp(Float64(value))
to_unconstrained(::LogTransform, value) = log(Float64(value))

function parameterchoicemap(model::TeaModel, params::AbstractVector)
    plan = executionplan(model)
    expected = parametercount(plan.parameter_layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    cm = ChoiceMap()
    for step in plan.steps
        slot = step.parameter_slot
        isnothing(slot) && continue
        _pushchoice!(cm, _static_address(step.address), params[slot])
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

    constrained = Vector{Float64}(undef, expected)
    for slot in layout.slots
        constrained[slot.index] = to_constrained(slot.transform, params[slot.index])
    end
    return constrained
end

function transform_to_unconstrained(model::TeaModel, params::AbstractVector)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    unconstrained = Vector{Float64}(undef, expected)
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
