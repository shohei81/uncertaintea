abstract type AbstractAddressPart end

struct AddressLiteralPart <: AbstractAddressPart
    value::Any
end

struct AddressDynamicPart <: AbstractAddressPart
    value::Any
end

struct AddressSpec
    parts::Tuple{Vararg{AbstractAddressPart}}
end

abstract type AbstractChoiceRhsSpec end

struct DistributionSpec <: AbstractChoiceRhsSpec
    family::Symbol
    arguments::Vector{Any}
end

struct GenerativeCallSpec <: AbstractChoiceRhsSpec
    callee::Any
    arguments::Vector{Any}
end

struct RawChoiceRhsSpec <: AbstractChoiceRhsSpec
    expr::Any
end

struct LoopScopeSpec
    iterator::Symbol
    iterable::Any
    shape_specialized::Bool
end

struct ChoiceSpec
    binding::Union{Nothing,Symbol}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
end

struct ModelSpec
    name::Symbol
    mode::Symbol
    arguments::Vector{Symbol}
    choices::Vector{ChoiceSpec}
    shape_specialized::Bool
end

function modelspec(model)
    return model.spec
end

isstaticaddress(address::AddressSpec) = all(part -> part isa AddressLiteralPart, address.parts)
isaddresstemplate(address::AddressSpec) = !isstaticaddress(address)
isrepeatedchoice(choice::ChoiceSpec) = !isempty(choice.scopes)
hasrepeatedchoices(spec::ModelSpec) = any(isrepeatedchoice, spec.choices)

function Base.show(io::IO, part::AddressLiteralPart)
    print(io, repr(part.value))
end

function Base.show(io::IO, part::AddressDynamicPart)
    print(io, "\$", repr(part.value))
end

function Base.show(io::IO, address::AddressSpec)
    print(io, "AddressSpec(")
    for (idx, part) in enumerate(address.parts)
        idx > 1 && print(io, " => ")
        show(io, part)
    end
    print(io, ")")
end

function Base.show(io::IO, spec::ChoiceSpec)
    print(io, "ChoiceSpec(")
    isnothing(spec.binding) || print(io, "binding=", spec.binding, ", ")
    print(io, "address=")
    show(io, spec.address)
    print(io, ", rhs=", nameof(typeof(spec.rhs)))
    isempty(spec.scopes) || print(io, ", scopes=", length(spec.scopes))
    print(io, ")")
end

function Base.show(io::IO, spec::ModelSpec)
    print(io, "ModelSpec(", spec.name, ", mode=", spec.mode, ", choices=", length(spec.choices), ")")
end
