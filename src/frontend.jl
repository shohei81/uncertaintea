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

    rewritten_args = map(arg -> _rewrite_tea_expr(arg, ctxsym), expr.args)
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
    expr isa Expr && expr.head == :call && expr.args[1] === :~ ||
        throw(ArgumentError("expected a choice expression, got $expr"))

    lhs = expr.args[2]
    rhs = expr.args[3]
    binding = lhs isa Symbol ? QuoteNode(lhs) : :(nothing)
    address = _address_spec_expr(lhs)
    rhs_spec = _rhs_spec_expr(rhs)
    return :($(_qualify(:ChoiceSpec))($binding, $address, $rhs_spec))
end

function _rhs_spec_expr(rhs)
    if rhs isa Expr && rhs.head == :call && !isempty(rhs.args)
        callee = rhs.args[1]
        arguments = Expr(:vect, map(QuoteNode, rhs.args[2:end])...)

        if callee isa Symbol && callee in (:normal, :lognormal, :bernoulli, :categorical, :mvnormal)
            return :($(_qualify(:DistributionSpec))($(QuoteNode(callee)), $arguments))
        end

        return :($(_qualify(:GenerativeCallSpec))($(QuoteNode(callee)), $arguments))
    end

    return :($(_qualify(:RawChoiceRhsSpec))($(QuoteNode(rhs))))
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

function _collect_choice_spec_exprs!(expr, specs::Vector{Expr})
    if !(expr isa Expr)
        return specs
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        push!(specs, expr)
        return specs
    end

    for arg in expr.args
        _collect_choice_spec_exprs!(arg, specs)
    end

    return specs
end

function _model_spec_expr(mode_expr, signature, body)
    name = signature.args[1]
    arguments = [_argument_name(arg) for arg in signature.args[2:end]]
    argument_expr = Expr(:vect, map(QuoteNode, arguments)...)
    choice_nodes = Expr[]
    _collect_choice_spec_exprs!(body, choice_nodes)
    choice_exprs = map(_choice_spec_expr, choice_nodes)
    choices_expr = Expr(:vect, choice_exprs...)
    shape_specialized = any(choice -> _address_has_dynamic_parts(choice.args[2]), choice_nodes)

    return :($(_qualify(:ModelSpec))(
        $(QuoteNode(name)),
        $(QuoteNode(mode_expr)),
        $argument_expr,
        $choices_expr,
        $shape_specialized,
    ))
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
