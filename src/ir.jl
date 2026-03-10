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

abstract type AbstractPlanStep end

struct ChoicePlanStep <: AbstractPlanStep
    choice_index::Int
    binding::Union{Nothing,Symbol}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
    parameter_slot::Union{Nothing,Int}
end

struct DeterministicPlanStep <: AbstractPlanStep
    binding::Symbol
    expr::Any
end

struct LoopPlanStep <: AbstractPlanStep
    iterator::Symbol
    iterable::Any
    body::Vector{AbstractPlanStep}
end

struct ExecutionPlan
    model_name::Symbol
    steps::Vector{AbstractPlanStep}
    parameter_layout::ParameterLayout
end

struct ModelSpec
    name::Symbol
    mode::Symbol
    arguments::Vector{Symbol}
    choices::Vector{ChoiceSpec}
    shape_specialized::Bool
    parameter_layout::ParameterLayout
    return_expr::Any
    execution_plan::ExecutionPlan
end

function ModelSpec(
    name::Symbol,
    mode::Symbol,
    arguments,
    choices::Vector{ChoiceSpec},
    shape_specialized::Bool,
    parameter_layout::ParameterLayout,
    return_expr::Any,
)
    argument_symbols = Symbol[arg for arg in arguments]
    plan = ExecutionPlan(name, AbstractPlanStep[], parameter_layout)
    return ModelSpec(name, mode, argument_symbols, choices, shape_specialized, parameter_layout, return_expr, plan)
end

function ModelSpec(
    name::Symbol,
    mode::Symbol,
    arguments,
    choices::Vector{ChoiceSpec},
    shape_specialized::Bool,
    parameter_layout::ParameterLayout,
    return_expr::Any,
    plan_steps,
)
    argument_symbols = Symbol[arg for arg in arguments]
    raw_steps = AbstractPlanStep[step for step in plan_steps]
    plan = build_execution_plan(name, raw_steps, parameter_layout, return_expr)
    return ModelSpec(name, mode, argument_symbols, choices, shape_specialized, plan.parameter_layout, return_expr, plan)
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

function _substitute_expr(expr, substitutions::Dict{Symbol,Any})
    if expr isa QuoteNode
        return expr
    elseif expr isa Symbol
        return get(substitutions, expr, expr)
    elseif expr isa Expr
        return Expr(expr.head, map(arg -> _substitute_expr(arg, substitutions), expr.args)...)
    elseif expr isa Tuple
        return tuple((_substitute_expr(arg, substitutions) for arg in expr)...)
    end
    return expr
end

function _substitute_address(address::AddressSpec, substitutions::Dict{Symbol,Any})
    parts = map(address.parts) do part
        if part isa AddressLiteralPart
            return part
        end
        return AddressDynamicPart(_substitute_expr(part.value, substitutions))
    end
    return AddressSpec(tuple(parts...))
end

function _prefix_address(prefix::AddressSpec, address::AddressSpec)
    return AddressSpec((prefix.parts..., address.parts...))
end

function _collect_bound_symbols!(steps::Vector{AbstractPlanStep}, bindings::Set{Symbol}, iterators::Set{Symbol})
    for step in steps
        if step isa ChoicePlanStep
            isnothing(step.binding) || push!(bindings, step.binding)
        elseif step isa DeterministicPlanStep
            push!(bindings, step.binding)
        elseif step isa LoopPlanStep
            push!(iterators, step.iterator)
            _collect_bound_symbols!(step.body, bindings, iterators)
        end
    end
    return nothing
end

function _substitute_rhs(rhs::DistributionSpec, substitutions::Dict{Symbol,Any})
    return DistributionSpec(rhs.family, Any[_substitute_expr(arg, substitutions) for arg in rhs.arguments])
end

function _substitute_rhs(rhs::GenerativeCallSpec, substitutions::Dict{Symbol,Any})
    callee = rhs.callee isa TeaModel ? rhs.callee : _substitute_expr(rhs.callee, substitutions)
    arguments = Any[_substitute_expr(arg, substitutions) for arg in rhs.arguments]
    return GenerativeCallSpec(callee, arguments)
end

_substitute_rhs(rhs::RawChoiceRhsSpec, substitutions::Dict{Symbol,Any}) = RawChoiceRhsSpec(_substitute_expr(rhs.expr, substitutions))

function _substitute_loop_scopes(scopes::Vector{LoopScopeSpec}, substitutions::Dict{Symbol,Any})
    replaced = LoopScopeSpec[]
    for scope in scopes
        iterator = get(substitutions, scope.iterator, scope.iterator)
        iterator isa Symbol || throw(ArgumentError("loop iterator substitution must stay a Symbol"))
        push!(replaced, LoopScopeSpec(iterator, _substitute_expr(scope.iterable, substitutions), scope.shape_specialized))
    end
    return replaced
end

function _substitute_step(step::ChoicePlanStep, substitutions::Dict{Symbol,Any}; prefix::Union{Nothing,AddressSpec}=nothing, parameter_slot=nothing)
    address = _substitute_address(step.address, substitutions)
    prefixed = isnothing(prefix) ? address : _prefix_address(prefix, address)
    binding = isnothing(step.binding) ? nothing : get(substitutions, step.binding, step.binding)
    if !isnothing(binding) && !(binding isa Symbol)
        throw(ArgumentError("choice binding substitution must stay a Symbol"))
    end
    return ChoicePlanStep(
        step.choice_index,
        binding,
        prefixed,
        _substitute_rhs(step.rhs, substitutions),
        _substitute_loop_scopes(step.scopes, substitutions),
        parameter_slot,
    )
end

function _substitute_step(step::DeterministicPlanStep, substitutions::Dict{Symbol,Any}; prefix::Union{Nothing,AddressSpec}=nothing, parameter_slot=nothing)
    binding = get(substitutions, step.binding, step.binding)
    binding isa Symbol || throw(ArgumentError("deterministic binding substitution must stay a Symbol"))
    return DeterministicPlanStep(binding, _substitute_expr(step.expr, substitutions))
end

function _substitute_step(step::LoopPlanStep, substitutions::Dict{Symbol,Any}; prefix::Union{Nothing,AddressSpec}=nothing, parameter_slot=nothing)
    iterator = get(substitutions, step.iterator, step.iterator)
    iterator isa Symbol || throw(ArgumentError("loop iterator substitution must stay a Symbol"))
    body = AbstractPlanStep[_substitute_step(inner, substitutions; prefix=prefix, parameter_slot=nothing) for inner in step.body]
    return LoopPlanStep(iterator, _substitute_expr(step.iterable, substitutions), body)
end

function _wrap_steps_with_scopes(steps::Vector{AbstractPlanStep}, scopes::Vector{LoopScopeSpec})
    result = steps
    for scope in reverse(scopes)
        result = AbstractPlanStep[LoopPlanStep(scope.iterator, scope.iterable, result)]
    end
    return result
end

function _merge_loop_steps(steps::Vector{AbstractPlanStep})
    merged = AbstractPlanStep[]
    for step in steps
        if step isa LoopPlanStep
            body = _merge_loop_steps(step.body)
            if !isempty(merged) && merged[end] isa LoopPlanStep
                previous = merged[end]
                if previous.iterator == step.iterator && previous.iterable == step.iterable
                    merged[end] = LoopPlanStep(previous.iterator, previous.iterable, vcat(previous.body, body))
                    continue
                end
            end
            push!(merged, LoopPlanStep(step.iterator, step.iterable, body))
        else
            push!(merged, step)
        end
    end
    return merged
end

function _parameter_transform(rhs::DistributionSpec)
    if rhs.family === :normal
        return IdentityTransform()
    elseif rhs.family === :lognormal
        return LogTransform()
    end
    return nothing
end

_parameter_transform(::AbstractChoiceRhsSpec) = nothing

function _parameterize_step(
    step::ChoicePlanStep,
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
)
    step_index = step_counter[]
    step_counter[] += 1
    transform = isnothing(step.binding) ? nothing : _parameter_transform(step.rhs)

    if isnothing(transform) || !isempty(step.scopes) || !isstaticaddress(step.address)
        return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, nothing)
    end

    slot_index = slot_counter[]
    slot_counter[] += 1
    push!(slots, ParameterSlotSpec(step_index, step.binding, step.address, slot_index, transform))
    return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, slot_index)
end

function _parameterize_plan_steps(
    steps::Vector{AbstractPlanStep},
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
)
    parameterized = AbstractPlanStep[]
    for step in steps
        if step isa ChoicePlanStep
            push!(parameterized, _parameterize_step(step, slots, step_counter, slot_counter))
        elseif step isa DeterministicPlanStep
            push!(parameterized, step)
        elseif step isa LoopPlanStep
            body = _parameterize_plan_steps(step.body, slots, step_counter, slot_counter)
            push!(parameterized, LoopPlanStep(step.iterator, step.iterable, body))
        else
            throw(ArgumentError("unsupported plan step in parameterization: $(typeof(step))"))
        end
    end
    return parameterized
end

function _assign_parameter_layout(steps::Vector{AbstractPlanStep})
    slots = ParameterSlotSpec[]
    step_counter = Ref(1)
    slot_counter = Ref(1)
    parameterized = _parameterize_plan_steps(steps, slots, step_counter, slot_counter)
    return parameterized, ParameterLayout(slots)
end

function _inline_plan_steps(steps::Vector{AbstractPlanStep})
    expanded = AbstractPlanStep[]
    for step in steps
        append!(expanded, _inline_plan_step(step))
    end
    return _merge_loop_steps(expanded)
end

function _inline_plan_step(step::DeterministicPlanStep)
    return AbstractPlanStep[step]
end

function _inline_plan_step(step::LoopPlanStep)
    body = _inline_plan_steps(step.body)
    return AbstractPlanStep[LoopPlanStep(step.iterator, step.iterable, body)]
end

function _inline_plan_step(step::ChoicePlanStep)
    if step.rhs isa DistributionSpec || step.rhs isa RawChoiceRhsSpec
        return _wrap_steps_with_scopes(AbstractPlanStep[ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, LoopScopeSpec[], step.parameter_slot)], step.scopes)
    elseif step.rhs isa GenerativeCallSpec
        callee = step.rhs.callee
        callee isa TeaModel || throw(ArgumentError("generative call inlining requires a TeaModel callee, got $(typeof(callee))"))

        callee_spec = modelspec(callee)
        substitutions = Dict{Symbol,Any}()
        for (argname, argexpr) in zip(callee_spec.arguments, step.rhs.arguments)
            substitutions[argname] = argexpr
        end

        bound_symbols = Set{Symbol}()
        iterators = Set{Symbol}()
        _collect_bound_symbols!(executionplan(callee).steps, bound_symbols, iterators)
        for sym in union(bound_symbols, iterators)
            substitutions[sym] = gensym(sym)
        end

        inlined = AbstractPlanStep[]
        for inner_step in executionplan(callee).steps
            substituted = _substitute_step(inner_step, substitutions; prefix=step.address, parameter_slot=nothing)
            append!(inlined, _inline_plan_step(substituted))
        end

        if !isnothing(step.binding)
            return_expr = _substitute_expr(callee_spec.return_expr, substitutions)
            push!(inlined, DeterministicPlanStep(step.binding, return_expr))
        end

        return _wrap_steps_with_scopes(inlined, step.scopes)
    end

    throw(ArgumentError("unsupported choice RHS in execution-plan inlining: $(typeof(step.rhs))"))
end

function build_execution_plan(name::Symbol, raw_steps::Vector{AbstractPlanStep}, layout::ParameterLayout, return_expr::Any)
    if isempty(raw_steps)
        return ExecutionPlan(name, AbstractPlanStep[], layout)
    end

    steps = _inline_plan_steps(raw_steps)
    parameterized_steps, parameterized_layout = _assign_parameter_layout(steps)
    return ExecutionPlan(name, parameterized_steps, parameterized_layout)
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

function Base.show(io::IO, step::DeterministicPlanStep)
    print(io, "DeterministicPlanStep(binding=", step.binding, ")")
end

function Base.show(io::IO, step::LoopPlanStep)
    print(io, "LoopPlanStep(iterator=", step.iterator, ", body=", length(step.body), ")")
end

function Base.show(io::IO, plan::ExecutionPlan)
    print(io, "ExecutionPlan(", plan.model_name, ", steps=", length(plan.steps), ")")
end

function Base.show(io::IO, spec::ModelSpec)
    print(io, "ModelSpec(", spec.name, ", mode=", spec.mode, ", choices=", length(spec.choices), ")")
end
