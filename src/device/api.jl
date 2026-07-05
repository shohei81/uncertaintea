# Public, allocating convenience entry point for the device logjoint path.

"""
    device_batched_logjoint(model, params, args=(), constraints=choicemap();
                            backend=KernelAbstractions.CPU(),
                            precision=float(eltype(params))) -> Vector

Device-resident batched (unconstrained) logjoint via KernelAbstractions. `params` is
a `parameter_count × batch` matrix of unconstrained parameters; the returned vector
holds the per-column unconstrained logjoint (prior + likelihood + transform
log-abs-det), matching `batched_logjoint_unconstrained` on the CPU.

Constructs a fresh `DeviceBatchedWorkspace` each call; use `DeviceBatchedWorkspace` +
`device_batched_logjoint!` to reuse device buffers and staged observations. Throws an
`ArgumentError` (pointing at `device_lowering_report`) if the model is not
representable on the device.
"""
function device_batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap();
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=float(eltype(params)),
)
    workspace = DeviceBatchedWorkspace(
        model,
        size(params, 2);
        backend=backend,
        precision=precision,
        args=args,
        constraints=constraints,
    )
    return device_batched_logjoint!(workspace, params)
end
