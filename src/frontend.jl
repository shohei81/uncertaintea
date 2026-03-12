function _qualify(name::Symbol)
    return GlobalRef(@__MODULE__, name)
end

function _argument_name(arg)
    if arg isa Symbol
        return arg
    elseif arg isa Expr && arg.head == :(::)
        return _argument_name(arg.args[1])
    elseif arg isa Expr && arg.head == :kw
        return _argument_name(arg.args[1])
    end

    throw(ArgumentError("@tea could not determine the argument name for $arg"))
end

function _tea_mode(mode_expr)
    if mode_expr === :static
        return _qualify(:STATIC_MODE)
    elseif mode_expr === :dynamic
        return _qualify(:DYNAMIC_MODE)
    else
        throw(ArgumentError("@tea only supports `static` or `dynamic` modes"))
    end
end

function _qualify_builtin_distribution(name)
    if name in (
        :normal,
        :lognormal,
        :laplace,
        :exponential,
        :gamma,
        :inversegamma,
        :weibull,
        :beta,
        :bernoulli,
        :binomial,
        :geometric,
        :negativebinomial,
        :poisson,
        :studentt,
        :categorical,
    )
        return _qualify(name)
    end
    return name
end

function _rewrite_tea_expr(expr, ctxsym)
    if !(expr isa Expr)
        return expr
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        lhs = expr.args[2]
        rhs = _rewrite_tea_expr(expr.args[3], ctxsym)

        if lhs isa Symbol
            return Expr(:(=), lhs, Expr(:call, _qualify(:choice), ctxsym, QuoteNode(lhs), rhs))
        elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
            return Expr(:call, _qualify(:choice), ctxsym, lhs.args[1], rhs)
        else
            throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
        end
    end

    rewritten_args = Any[_rewrite_tea_expr(arg, ctxsym) for arg in expr.args]
    if expr.head == :call && !isempty(rewritten_args) && rewritten_args[1] isa Symbol
        rewritten_args[1] = _qualify_builtin_distribution(rewritten_args[1])
    end
    return Expr(expr.head, rewritten_args...)
end

function _address_part_expr(value, literal::Bool)
    if literal
        return :($(_qualify(:AddressLiteralPart))($(QuoteNode(value))))
    end
    return :($(_qualify(:AddressDynamicPart))($(QuoteNode(value))))
end

function _append_address_parts!(parts, expr; symbol_literal::Bool=false)
    if expr isa QuoteNode
        push!(parts, _address_part_expr(expr.value, true))
        return parts
    elseif expr isa Symbol
        push!(parts, _address_part_expr(expr, symbol_literal))
        return parts
    elseif expr isa Expr && expr.head == :call && !isempty(expr.args) && expr.args[1] === :(=>)
        _append_address_parts!(parts, expr.args[2]; symbol_literal=symbol_literal)
        _append_address_parts!(parts, expr.args[3]; symbol_literal=symbol_literal)
        return parts
    elseif expr isa Number || expr isa String || expr isa Char
        push!(parts, _address_part_expr(expr, true))
        return parts
    end

    push!(parts, _address_part_expr(expr, false))
    return parts
end

function _address_spec_expr(lhs)
    parts = Expr[]
    if lhs isa Symbol
        _append_address_parts!(parts, lhs; symbol_literal=true)
    elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
        _append_address_parts!(parts, lhs.args[1]; symbol_literal=false)
    else
        throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
    end

    tuple_expr = Expr(:tuple, parts...)
    return :($(_qualify(:AddressSpec))($tuple_expr))
end

function _choice_spec_expr(expr)
    return _choice_spec_expr(expr, (), nothing)
end

function _choice_spec_expr(expr, loop_scopes, binding_override=nothing)
    expr isa Expr && expr.head == :call && expr.args[1] === :~ ||
        throw(ArgumentError("expected a choice expression, got $expr"))

    lhs = expr.args[2]
    rhs = expr.args[3]
    binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override
    binding = isnothing(binding_symbol) ? :(nothing) : QuoteNode(binding_symbol)
    address = _address_spec_expr(lhs)
    rhs_spec = _rhs_spec_expr(rhs)
    scopes_expr = Expr(:vect, map(_loop_scope_spec_expr, loop_scopes)...)
    return :($(_qualify(:ChoiceSpec))($binding, $address, $rhs_spec, $scopes_expr))
end

function _rhs_spec_expr(rhs)
    if rhs isa Expr && rhs.head == :call && !isempty(rhs.args)
        callee = rhs.args[1]
        arguments = Expr(:vect, map(QuoteNode, rhs.args[2:end])...)

        if callee isa Symbol &&
           callee in (
            :normal,
            :lognormal,
            :laplace,
            :exponential,
            :gamma,
            :inversegamma,
            :weibull,
            :beta,
            :bernoulli,
            :binomial,
            :geometric,
            :negativebinomial,
            :poisson,
            :studentt,
            :categorical,
            :mvnormal,
        )
            return :($(_qualify(:DistributionSpec))($(QuoteNode(callee)), $arguments))
        end

        return :($(_qualify(:GenerativeCallSpec))($callee, $arguments))
    end

    return :($(_qualify(:RawChoiceRhsSpec))($(QuoteNode(rhs))))
end

function _loop_scope_spec_expr(scope)
    iterator, iterable = scope
    return :($(_qualify(:LoopScopeSpec))(
        $(QuoteNode(iterator)),
        $(QuoteNode(iterable)),
        $(_expr_has_dynamic_content(iterable)),
    ))
end

function _parameter_layout_expr(choice_nodes)
    slot_exprs = Expr[]
    slot_lookup = Dict{Int,Int}()
    slot_index = 1

    for (choice_index, node) in enumerate(choice_nodes)
        choice_expr, loop_scopes, binding_override = node
        lhs = choice_expr.args[2]
        rhs = choice_expr.args[3]
        binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override

        if !isnothing(binding_symbol) && isempty(loop_scopes) && _supports_parameter_slot(rhs)
            address = _address_spec_expr(lhs)
            transform = _parameter_transform_expr(rhs)
            push!(slot_exprs, :($(_qualify(:ParameterSlotSpec))(
                $choice_index,
                $(QuoteNode(binding_symbol)),
                $address,
                $slot_index,
                $transform,
            )))
            slot_lookup[choice_index] = slot_index
            slot_index += 1
        end
    end

    return :($(_qualify(:ParameterLayout))($(Expr(:vect, slot_exprs...)))), slot_lookup
end

function _supported_distribution_family(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] isa Symbol || return nothing
    family = rhs.args[1]
    family in (:normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma, :weibull, :beta, :studentt) ||
        return nothing
    return family
end

function _supports_parameter_slot(rhs)
    return !isnothing(_supported_distribution_family(rhs))
end

function _parameter_transform_expr(rhs)
    family = _supported_distribution_family(rhs)
    isnothing(family) && throw(ArgumentError("unsupported parameter transform for $rhs"))

    if family === :normal || family === :laplace
        return :($(_qualify(:IdentityTransform))())
    elseif family === :lognormal || family === :exponential || family === :gamma ||
           family === :inversegamma || family === :weibull
        return :($(_qualify(:LogTransform))())
    elseif family === :beta
        return :($(_qualify(:LogitTransform))())
    elseif family === :studentt
        return :($(_qualify(:IdentityTransform))())
    end

    throw(ArgumentError("unsupported parameter transform family $family"))
end

function _address_has_dynamic_parts(lhs)
    if lhs isa Symbol
        return false
    elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
        return _address_expr_has_dynamic_parts(lhs.args[1])
    end

    throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
end

function _address_expr_has_dynamic_parts(expr)
    if expr isa QuoteNode
        return false
    elseif expr isa Symbol
        return true
    elseif expr isa Expr && expr.head == :call && !isempty(expr.args) && expr.args[1] === :(=>)
        return _address_expr_has_dynamic_parts(expr.args[2]) || _address_expr_has_dynamic_parts(expr.args[3])
    elseif expr isa Number || expr isa String || expr isa Char
        return false
    end

    return true
end

function _expr_has_dynamic_content(expr)
    if expr isa QuoteNode
        return false
    elseif expr isa Symbol
        return true
    elseif expr isa Number || expr isa String || expr isa Char
        return false
    elseif expr isa Expr
        if expr.head == :call
            start = 2
            if !isempty(expr.args) && expr.args[1] isa Symbol && expr.args[1] in (:, :+, :-, :*, :/, :%, :^, :(=>))
                start = 2
            else
                start = 1
            end
            for idx in start:length(expr.args)
                _expr_has_dynamic_content(expr.args[idx]) && return true
            end
            return false
        end

        return any(_expr_has_dynamic_content, expr.args)
    end

    return true
end

function _parse_loop_scope(iteration)
    if iteration isa Expr && iteration.head in (:(=), :in)
        iterator = iteration.args[1]
        iterator isa Symbol || throw(ArgumentError("@tea only supports simple loop iterators, got $iterator"))
        return iterator, iteration.args[2]
    end

    throw(ArgumentError("@tea only supports simple `for x in xs` loops in static analysis"))
end

function _collect_choice_spec_exprs!(expr, specs::Vector{Tuple{Expr,Tuple}})
    return _collect_choice_spec_exprs!(expr, specs, ())
end

function _collect_choice_spec_exprs!(expr, specs::Vector{Tuple{Expr,Tuple}}, loop_scopes::Tuple)
    if !(expr isa Expr)
        return specs
    end

    if expr.head == :for && length(expr.args) == 2
        scope = _parse_loop_scope(expr.args[1])
        return _collect_choice_spec_exprs!(expr.args[2], specs, (loop_scopes..., scope))
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        push!(specs, (expr, loop_scopes, nothing))
        return specs
    end

    for arg in expr.args
        _collect_choice_spec_exprs!(arg, specs, loop_scopes)
    end

    return specs
end

function _collect_plan_nodes!(expr, nodes::Vector{Tuple}, loop_scopes::Tuple=())
    if !(expr isa Expr)
        return nodes
    end

    if expr.head == :block
        for arg in expr.args
            arg isa LineNumberNode && continue
            _collect_plan_nodes!(arg, nodes, loop_scopes)
        end
        return nodes
    end

    if expr.head == :for && length(expr.args) == 2
        scope = _parse_loop_scope(expr.args[1])
        _collect_plan_nodes!(expr.args[2], nodes, (loop_scopes..., scope))
        return nodes
    end

    if expr.head == :(=) && length(expr.args) == 2
        lhs, rhs = expr.args
        if rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :~
            binding_override = lhs isa Symbol ? lhs : nothing
            push!(nodes, (:choice, rhs, loop_scopes, binding_override))
        elseif lhs isa Symbol && isempty(loop_scopes)
            push!(nodes, (:deterministic, lhs, rhs))
        end
        return nodes
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        push!(nodes, (:choice, expr, loop_scopes, nothing))
        return nodes
    end

    for arg in expr.args
        _collect_plan_nodes!(arg, nodes, loop_scopes)
    end

    return nodes
end

function _execution_plan_steps_expr(plan_nodes, slot_lookup)
    step_exprs = Expr[]
    choice_index = 0

    for node in plan_nodes
        if node[1] === :choice
            choice_index += 1
            choice_expr = node[2]
            loop_scopes = node[3]
            binding_override = node[4]
            lhs = choice_expr.args[2]
            binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override
            binding = isnothing(binding_symbol) ? :(nothing) : QuoteNode(binding_symbol)
            address = _address_spec_expr(lhs)
            rhs_spec = _rhs_spec_expr(choice_expr.args[3])
            scopes_expr = Expr(:vect, map(_loop_scope_spec_expr, loop_scopes)...)
            slot_expr = haskey(slot_lookup, choice_index) ? slot_lookup[choice_index] : :(nothing)
            push!(step_exprs, :($(_qualify(:ChoicePlanStep))(
                $choice_index,
                $binding,
                $address,
                $rhs_spec,
                $scopes_expr,
                $slot_expr,
            )))
        elseif node[1] === :deterministic
            push!(step_exprs, :($(_qualify(:DeterministicPlanStep))(
                $(QuoteNode(node[2])),
                $(QuoteNode(node[3])),
            )))
        end
    end

    return Expr(:vect, step_exprs...)
end

function _model_spec_expr(mode_expr, signature, body)
    name = signature.args[1]
    arguments = [_argument_name(arg) for arg in signature.args[2:end]]
    argument_expr = Expr(:vect, map(QuoteNode, arguments)...)
    plan_nodes = Tuple[]
    _collect_plan_nodes!(body, plan_nodes)
    choice_nodes = Tuple{Expr,Tuple,Any}[ (node[2], node[3], node[4]) for node in plan_nodes if node[1] === :choice ]
    choice_exprs = map(node -> _choice_spec_expr(node[1], node[2], node[3]), choice_nodes)
    choices_expr = Expr(:vect, choice_exprs...)
    shape_specialized = any(node -> _address_has_dynamic_parts(node[1].args[2]) || any(scope -> _expr_has_dynamic_content(scope[2]), node[2]), choice_nodes)
    parameter_layout_expr, slot_lookup = _parameter_layout_expr(choice_nodes)
    plan_steps_expr = _execution_plan_steps_expr(plan_nodes, slot_lookup)
    return_expr = _return_expr_expr(body)

    return :($(_qualify(:ModelSpec))(
        $(QuoteNode(name)),
        $(QuoteNode(mode_expr)),
        $argument_expr,
        $choices_expr,
        $shape_specialized,
        $parameter_layout_expr,
        $return_expr,
        $plan_steps_expr,
    ))
end

function _return_expr_expr(body)
    body isa Expr && body.head == :block || return QuoteNode(body)

    for arg in reverse(body.args)
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :return
            return QuoteNode(arg.args[1])
        end
        return QuoteNode(arg)
    end

    return QuoteNode(nothing)
end

function _expand_tea(mode_expr, definition)
    definition isa Expr && definition.head == :function ||
        throw(ArgumentError("@tea expects a function definition"))

    signature = definition.args[1]
    body = definition.args[2]

    signature isa Expr && signature.head == :call ||
        throw(ArgumentError("@tea currently supports standard function definitions only"))

    name = signature.args[1]
    name isa Symbol || throw(ArgumentError("@tea expects a plain function name"))

    ctxsym = gensym(:tea_ctx)
    impl_name = gensym(name)
    impl_signature = Expr(:call, impl_name, ctxsym, signature.args[2:end]...)
    rewritten_body = _rewrite_tea_expr(body, ctxsym)
    mode = _tea_mode(mode_expr)
    spec_expr = _model_spec_expr(mode_expr, signature, body)
    function_definition = Expr(:function, impl_signature, rewritten_body)
    binding_definition = :($name = $(_qualify(:TeaModel))($mode, $(QuoteNode(name)), $impl_name, $spec_expr))

    return esc(Expr(:block, function_definition, binding_definition))
end

macro tea(args...)
    if length(args) == 1
        return _expand_tea(:dynamic, args[1])
    elseif length(args) == 2
        return _expand_tea(args[1], args[2])
    else
        throw(ArgumentError("@tea expects `@tea function ... end` or `@tea static function ... end`"))
    end
end
