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
struct VectorIdentityTransform <: AbstractParameterTransform
    size::Int

    function VectorIdentityTransform(size::Int)
        size >= 1 || throw(ArgumentError("vector identity transform requires size >= 1"))
        return new(size)
    end
end
struct LogTransform <: AbstractParameterTransform end
struct LogitTransform <: AbstractParameterTransform end
struct SimplexTransform <: AbstractParameterTransform
    size::Int

    function SimplexTransform(size::Int)
        size >= 2 || throw(ArgumentError("simplex transform requires size >= 2"))
        return new(size)
    end
end

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
    dimension::Int
    value_index::Int
    value_length::Int
    transform::AbstractParameterTransform
end

struct ParameterLayout
    slots::Vector{ParameterSlotSpec}
    parameter_count::Int
    value_count::Int
end

struct EnvironmentLayout
    symbols::Vector{Symbol}
    slot_by_symbol::Dict{Symbol,Int}
    argument_slots::Vector{Int}
end

abstract type AbstractPlanStep end

struct ChoicePlanStep <: AbstractPlanStep
    choice_index::Int
    binding::Union{Nothing,Symbol}
    binding_slot::Union{Nothing,Int}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
    parameter_slot::Union{Nothing,Int}
end

function ChoicePlanStep(choice_index::Int, binding, address::AddressSpec, rhs::AbstractChoiceRhsSpec, scopes, parameter_slot)
    normalized_scopes = LoopScopeSpec[scope for scope in scopes]
    return ChoicePlanStep(choice_index, binding, nothing, address, rhs, normalized_scopes, parameter_slot)
end

struct DeterministicPlanStep <: AbstractPlanStep
    binding::Symbol
    binding_slot::Union{Nothing,Int}
    expr::Any
end

DeterministicPlanStep(binding::Symbol, expr::Any) = DeterministicPlanStep(binding, nothing, expr)

struct LoopPlanStep <: AbstractPlanStep
    iterator::Symbol
    iterator_slot::Union{Nothing,Int}
    iterable::Any
    body::Vector{AbstractPlanStep}
end

function LoopPlanStep(iterator::Symbol, iterable::Any, body)
    normalized_body = AbstractPlanStep[step for step in body]
    return LoopPlanStep(iterator, nothing, iterable, normalized_body)
end

struct ExecutionPlan
    model_name::Symbol
    steps::Vector{AbstractPlanStep}
    parameter_layout::ParameterLayout
    environment_layout::EnvironmentLayout
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
    environment_layout = EnvironmentLayout(copy(argument_symbols), Dict(arg => idx for (idx, arg) in enumerate(argument_symbols)), collect(eachindex(argument_symbols)))
    plan = ExecutionPlan(name, AbstractPlanStep[], parameter_layout, environment_layout)
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
    plan = build_execution_plan(name, argument_symbols, raw_steps, parameter_layout, return_expr)
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
parametercount(layout::ParameterLayout) = layout.parameter_count
parametervaluecount(layout::ParameterLayout) = layout.value_count

parameterindices(slot::ParameterSlotSpec) = slot.index:(slot.index + slot.dimension - 1)
parametervalueindices(slot::ParameterSlotSpec) = slot.value_index:(slot.value_index + slot.value_length - 1)
isscalarparameterslot(slot::ParameterSlotSpec) = slot.dimension == 1 && slot.value_length == 1

function _parameter_slot_index(layout::ParameterLayout, choice_index::Int)
    for (slot_index, slot) in enumerate(layout.slots)
        if slot.choice_index == choice_index
            return slot_index
        end
    end
    return nothing
end

function _push_environment_symbol!(symbols::Vector{Symbol}, seen::Set{Symbol}, symbol::Symbol)
    symbol in seen && return nothing
    push!(symbols, symbol)
    push!(seen, symbol)
    return nothing
end

function _collect_environment_symbols!(steps::Vector{AbstractPlanStep}, symbols::Vector{Symbol}, seen::Set{Symbol})
    for step in steps
        if step isa ChoicePlanStep
            isnothing(step.binding) || _push_environment_symbol!(symbols, seen, step.binding)
        elseif step isa DeterministicPlanStep
            _push_environment_symbol!(symbols, seen, step.binding)
        elseif step isa LoopPlanStep
            _push_environment_symbol!(symbols, seen, step.iterator)
            _collect_environment_symbols!(step.body, symbols, seen)
        end
    end
    return nothing
end

function _build_environment_layout(arguments::Vector{Symbol}, steps::Vector{AbstractPlanStep})
    symbols = Symbol[]
    seen = Set{Symbol}()
    for argument in arguments
        _push_environment_symbol!(symbols, seen, argument)
    end
    _collect_environment_symbols!(steps, symbols, seen)

    slot_by_symbol = Dict{Symbol,Int}()
    for (idx, symbol) in enumerate(symbols)
        slot_by_symbol[symbol] = idx
    end
    argument_slots = Int[slot_by_symbol[argument] for argument in arguments]
    return EnvironmentLayout(symbols, slot_by_symbol, argument_slots)
end

function _annotate_environment_slots(step::ChoicePlanStep, layout::EnvironmentLayout)
    binding_slot = isnothing(step.binding) ? nothing : layout.slot_by_symbol[step.binding]
    return ChoicePlanStep(step.choice_index, step.binding, binding_slot, step.address, step.rhs, step.scopes, step.parameter_slot)
end

function _annotate_environment_slots(step::DeterministicPlanStep, layout::EnvironmentLayout)
    return DeterministicPlanStep(step.binding, layout.slot_by_symbol[step.binding], step.expr)
end

function _annotate_environment_slots(step::LoopPlanStep, layout::EnvironmentLayout)
    body = AbstractPlanStep[_annotate_environment_slots(inner, layout) for inner in step.body]
    return LoopPlanStep(step.iterator, layout.slot_by_symbol[step.iterator], step.iterable, body)
end

function _annotate_environment_slots(steps::Vector{AbstractPlanStep}, layout::EnvironmentLayout)
    return AbstractPlanStep[_annotate_environment_slots(step, layout) for step in steps]
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
    if rhs.family === :normal || rhs.family === :laplace || rhs.family === :studentt
        return IdentityTransform()
    elseif rhs.family === :mvnormal
        size = _mvnormal_static_size(rhs.arguments)
        isnothing(size) || return VectorIdentityTransform(size)
    elseif rhs.family === :lognormal || rhs.family === :exponential || rhs.family === :gamma ||
           rhs.family === :inversegamma || rhs.family === :weibull
        return LogTransform()
    elseif rhs.family === :beta
        return LogitTransform()
    elseif rhs.family === :dirichlet
        size = _dirichlet_static_size(rhs.arguments)
        isnothing(size) || return SimplexTransform(size)
    end
    return nothing
end

_parameter_transform(::AbstractChoiceRhsSpec) = nothing

_parameter_dimensions(::IdentityTransform) = (1, 1)
_parameter_dimensions(transform::VectorIdentityTransform) = (transform.size, transform.size)
_parameter_dimensions(::LogTransform) = (1, 1)
_parameter_dimensions(::LogitTransform) = (1, 1)
_parameter_dimensions(transform::SimplexTransform) = (transform.size - 1, transform.size)

function _static_length(expr)
    if expr isa Expr
        if expr.head == :vect || expr.head == :tuple
            return length(expr.args)
        end
    elseif expr isa QuoteNode
        value = expr.value
        if value isa Tuple || value isa AbstractVector
            return length(value)
        end
    elseif expr isa Tuple || expr isa AbstractVector
        return length(expr)
    end
    return nothing
end

function _dirichlet_static_size(arguments::Vector)
    isempty(arguments) && return nothing
    if length(arguments) == 1
        return _static_length(arguments[1])
    end
    return length(arguments)
end

function _mvnormal_static_size(arguments::Vector)
    length(arguments) == 2 || return nothing
    mu_size = _static_length(arguments[1])
    sigma_size = _static_length(arguments[2])
    if !isnothing(mu_size) && !isnothing(sigma_size)
        mu_size == sigma_size || throw(ArgumentError("mvnormal requires mean and scale vectors with the same static length"))
        return mu_size
    end
    return something(mu_size, sigma_size)
end

function _parameterize_step(
    step::ChoicePlanStep,
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    step_index = step_counter[]
    step_counter[] += 1
    transform = isnothing(step.binding) ? nothing : _parameter_transform(step.rhs)

    if isnothing(transform) || !isempty(step.scopes) || !isstaticaddress(step.address)
        return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, nothing)
    end

    slot_index = slot_counter[]
    slot_counter[] += 1
    dimension, value_length = _parameter_dimensions(transform)
    parameter_index = parameter_counter[]
    value_index = value_counter[]
    parameter_counter[] += dimension
    value_counter[] += value_length
    push!(
        slots,
        ParameterSlotSpec(
            step_index,
            step.binding,
            step.address,
            parameter_index,
            dimension,
            value_index,
            value_length,
            transform,
        ),
    )
    return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, slot_index)
end

function _parameterize_plan_steps(
    steps::Vector{AbstractPlanStep},
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    parameterized = AbstractPlanStep[]
    for step in steps
        if step isa ChoicePlanStep
            push!(
                parameterized,
                _parameterize_step(
                    step,
                    slots,
                    step_counter,
                    slot_counter,
                    parameter_counter,
                    value_counter,
                ),
            )
        elseif step isa DeterministicPlanStep
            push!(parameterized, step)
        elseif step isa LoopPlanStep
            body = _parameterize_plan_steps(
                step.body,
                slots,
                step_counter,
                slot_counter,
                parameter_counter,
                value_counter,
            )
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
    parameter_counter = Ref(1)
    value_counter = Ref(1)
    parameterized = _parameterize_plan_steps(
        steps,
        slots,
        step_counter,
        slot_counter,
        parameter_counter,
        value_counter,
    )
    return parameterized, ParameterLayout(slots, parameter_counter[] - 1, value_counter[] - 1)
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

function build_execution_plan(name::Symbol, arguments::Vector{Symbol}, raw_steps::Vector{AbstractPlanStep}, layout::ParameterLayout, return_expr::Any)
    if isempty(raw_steps)
        environment_layout = _build_environment_layout(arguments, AbstractPlanStep[])
        return ExecutionPlan(name, AbstractPlanStep[], layout, environment_layout)
    end

    steps = _inline_plan_steps(raw_steps)
    parameterized_steps, parameterized_layout = _assign_parameter_layout(steps)
    environment_layout = _build_environment_layout(arguments, parameterized_steps)
    annotated_steps = _annotate_environment_slots(parameterized_steps, environment_layout)
    return ExecutionPlan(name, annotated_steps, parameterized_layout, environment_layout)
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
        ", dimension=",
        spec.dimension,
        ", value_index=",
        spec.value_index,
        ", value_length=",
        spec.value_length,
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
    print(
        io,
        "ParameterLayout(",
        length(layout.slots),
        " slots, parameters=",
        layout.parameter_count,
        ", values=",
        layout.value_count,
        ")",
    )
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
