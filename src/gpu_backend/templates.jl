function gpu_backend_argument_signature(argument_declarations)
    return join((string(argument) for argument in argument_declarations), ", ")
end

function gpu_backend_stub_source_lines(
    module_symbol::Symbol,
    entry_symbol::Symbol,
    argument_declarations;
    preamble_lines=(),
    body_lines=("    return nothing",),
)
    lines = String[string("module ", module_symbol)]
    append!(lines, string.(preamble_lines))
    push!(
        lines,
        string(
            "function ",
            entry_symbol,
            "(",
            gpu_backend_argument_signature(argument_declarations),
            ")",
        ),
    )
    append!(lines, string.(body_lines))
    push!(lines, "end")
    push!(lines, "end")
    return Tuple(lines)
end

function gpu_backend_stub_source_blob(
    module_symbol::Symbol,
    entry_symbol::Symbol,
    argument_declarations;
    preamble_lines=(),
    body_lines=("    return nothing",),
)
    return join(
        gpu_backend_stub_source_lines(
            module_symbol,
            entry_symbol,
            argument_declarations;
            preamble_lines=preamble_lines,
            body_lines=body_lines,
        ),
        "\n",
    )
end

function gpu_backend_stage_preamble_lines(
    stage_kind::Symbol,
    target::Symbol;
    metadata_lines=(),
    preamble_lines=(),
)
    return (
        string("const STAGE_KIND = :", stage_kind),
        string("const TARGET_POLICY = :", target),
        (string(line) for line in metadata_lines)...,
        (string(line) for line in preamble_lines)...,
    )
end

function gpu_backend_stage_source_lines(
    module_symbol::Symbol,
    entry_symbol::Symbol,
    argument_declarations,
    stage_kind::Symbol,
    target::Symbol;
    metadata_lines=(),
    preamble_lines=(),
    body_lines=("    return nothing",),
)
    return gpu_backend_stub_source_lines(
        module_symbol,
        entry_symbol,
        argument_declarations;
        preamble_lines=gpu_backend_stage_preamble_lines(
            stage_kind,
            target;
            metadata_lines=metadata_lines,
            preamble_lines=preamble_lines,
        ),
        body_lines=body_lines,
    )
end

function gpu_backend_stage_source_blob(
    module_symbol::Symbol,
    entry_symbol::Symbol,
    argument_declarations,
    stage_kind::Symbol,
    target::Symbol;
    metadata_lines=(),
    preamble_lines=(),
    body_lines=("    return nothing",),
)
    return join(
        gpu_backend_stage_source_lines(
            module_symbol,
            entry_symbol,
            argument_declarations,
            stage_kind,
            target;
            metadata_lines=metadata_lines,
            preamble_lines=preamble_lines,
            body_lines=body_lines,
        ),
        "\n",
    )
end
