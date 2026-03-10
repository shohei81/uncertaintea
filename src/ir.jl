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

abstract type AbstractParameterTransform end

struct IdentityTransform <: AbstractParameterTransform end
struct LogTransform <: AbstractParameterTransform end

struct ChoiceSpec
    binding::Union{Nothing,Symbol}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
end

struct ParameterSlotSpec
    choice_index::Int
    binding::Symbol
    address::AddressSpec
    index::Int
    transform::AbstractParameterTransform
end

struct ParameterLayout
    slots::Vector{ParameterSlotSpec}
end

struct ChoicePlanStep
    choice_index::Int
    binding::Union{Nothing,Symbol}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
    parameter_slot::Union{Nothing,Int}
end

struct ExecutionPlan
    model_name::Symbol
    steps::Vector{ChoicePlanStep}
    parameter_layout::ParameterLayout
end

struct ModelSpec
    name::Symbol
    mode::Symbol
    arguments::Vector{Symbol}
    choices::Vector{ChoiceSpec}
    shape_specialized::Bool
    parameter_layout::ParameterLayout
    execution_plan::ExecutionPlan
end

function ModelSpec(
    name::Symbol,
    mode::Symbol,
    arguments,
    choices::Vector{ChoiceSpec},
    shape_specialized::Bool,
    parameter_layout::ParameterLayout,
)
    argument_symbols = Symbol[arg for arg in arguments]
    plan = build_execution_plan(name, choices, parameter_layout)
    return ModelSpec(name, mode, argument_symbols, choices, shape_specialized, parameter_layout, plan)
end

function modelspec(model)
    return model.spec
end

function parameterlayout(model)
    return model.spec.parameter_layout
end

function executionplan(model)
    return model.spec.execution_plan
end

isstaticaddress(address::AddressSpec) = all(part -> part isa AddressLiteralPart, address.parts)
isaddresstemplate(address::AddressSpec) = !isstaticaddress(address)
isrepeatedchoice(choice::ChoiceSpec) = !isempty(choice.scopes)
hasrepeatedchoices(spec::ModelSpec) = any(isrepeatedchoice, spec.choices)
parametercount(layout::ParameterLayout) = length(layout.slots)

function _parameter_slot_index(layout::ParameterLayout, choice_index::Int)
    for slot in layout.slots
        if slot.choice_index == choice_index
            return slot.index
        end
    end
    return nothing
end

function build_execution_plan(name::Symbol, choices::Vector{ChoiceSpec}, layout::ParameterLayout)
    steps = ChoicePlanStep[]
    for (choice_index, choice) in enumerate(choices)
        push!(
            steps,
            ChoicePlanStep(
                choice_index,
                choice.binding,
                choice.address,
                choice.rhs,
                choice.scopes,
                _parameter_slot_index(layout, choice_index),
            ),
        )
    end
    return ExecutionPlan(name, steps, layout)
end

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

function Base.show(io::IO, spec::ParameterSlotSpec)
    print(
        io,
        "ParameterSlotSpec(index=",
        spec.index,
        ", binding=",
        spec.binding,
        ", choice=",
        spec.choice_index,
        ", transform=",
        nameof(typeof(spec.transform)),
        ")",
    )
end

function Base.show(io::IO, layout::ParameterLayout)
    print(io, "ParameterLayout(", length(layout.slots), " slots)")
end

function Base.show(io::IO, step::ChoicePlanStep)
    print(io, "ChoicePlanStep(choice=", step.choice_index)
    isnothing(step.parameter_slot) || print(io, ", parameter_slot=", step.parameter_slot)
    isempty(step.scopes) || print(io, ", scopes=", length(step.scopes))
    print(io, ")")
end

function Base.show(io::IO, plan::ExecutionPlan)
    print(io, "ExecutionPlan(", plan.model_name, ", steps=", length(plan.steps), ")")
end

function Base.show(io::IO, spec::ModelSpec)
    print(io, "ModelSpec(", spec.name, ", mode=", spec.mode, ", choices=", length(spec.choices), ")")
end
