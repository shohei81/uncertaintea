function _batched_backend_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
)
    element_type = float(eltype(params))
    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    backend_plan = workspace.backend_plan
    isnothing(backend_plan) && return nothing
    _backend_gradient_supported(backend_plan) || return nothing

    cache = BatchedBackendGradientCache(
        workspace,
        zeros(element_type, size(params, 1), length(workspace.environment.layout.symbols), size(params, 2)),
        Matrix{element_type}[],
        batch_args,
        batch_constraints,
        _backend_gradient_seed_rows(workspace.layout),
        IdDict{Any,Any}(),
    )
    totals = _batched_totals_buffer!(workspace, size(params, 2), element_type)
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(totals, gradient_buffer, model, cache, params)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    return cache
end

function _batched_gradient_column!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    destination::AbstractVector,
    seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    objective = theta -> _logjoint_unconstrained_with_workspace!(model, workspace, theta, args, constraints)
    config = ForwardDiff.GradientConfig(objective, seed)
    ForwardDiff.gradient!(destination, objective, seed, config)
    return destination
end

function _batched_gradient_column_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
    batch_index::Int,
)
    column_constraints = _batched_constraints(batch_constraints, batch_index)
    workspace = BatchedLogjointWorkspace(model, column_constraints)
    objective = BatchedGradientObjective(
        model,
        workspace,
        _batched_args(batch_args, batch_index),
        column_constraints,
    )
    seed = collect(view(params, :, batch_index))
    config = ForwardDiff.GradientConfig(objective, seed)
    buffer = view(gradient_buffer, :, batch_index)
    return BatchedGradientColumnCache(objective, config, buffer)
end

function _batched_flat_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
)
    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    objective = BatchedFlatGradientObjective(
        model,
        workspace,
        batch_args,
        batch_constraints,
        size(params, 1),
        size(params, 2),
    )
    seed = collect(vec(params))
    probe = _batched_totals_buffer!(workspace, size(params, 2), eltype(params))
    try
        _batched_logjoint_unconstrained_with_workspace!(probe, model, workspace, params, batch_args, batch_constraints)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    config = ForwardDiff.GradientConfig(objective, seed)
    return BatchedFlatGradientCache(objective, config, vec(gradient_buffer))
end

function BatchedLogjointGradientCache(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_unconstrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    parameter_count = size(params, 1)
    gradient_buffer = Matrix{float(eltype(params))}(undef, parameter_count, batch_size)
    if batch_size == 0
        return BatchedLogjointGradientCache(model, Any[], nothing, nothing, gradient_buffer, parameter_count, batch_size)
    end

    backend_cache = _batched_backend_gradient_cache(model, gradient_buffer, params, batch_args, batch_constraints)
    if !isnothing(backend_cache)
        return BatchedLogjointGradientCache(model, Any[], backend_cache, nothing, gradient_buffer, parameter_count, batch_size)
    end

    flat_cache =
        isnothing(_backend_execution_plan(model)) ? nothing :
        _batched_flat_gradient_cache(model, gradient_buffer, params, batch_args, batch_constraints)
    if !isnothing(flat_cache)
        return BatchedLogjointGradientCache(model, Any[], nothing, flat_cache, gradient_buffer, parameter_count, batch_size)
    end

    first_cache = _batched_gradient_column_cache(model, gradient_buffer, params, batch_args, batch_constraints, 1)
    column_caches = Vector{typeof(first_cache)}(undef, batch_size)
    column_caches[1] = first_cache
    for batch_index = 2:batch_size
        column_caches[batch_index] = _batched_gradient_column_cache(
            model,
            gradient_buffer,
            params,
            batch_args,
            batch_constraints,
            batch_index,
        )
    end

    return BatchedLogjointGradientCache(model, column_caches, nothing, nothing, gradient_buffer, parameter_count, batch_size)
end

# A cached analytic backend gradient can hit a runtime capability gap the
# construction-time probe could not see (a marginalize branch whose suffix
# becomes unevaluable for an ignored column at the CURRENT parameters):
# recompute that call per column, which evaluates only what each column needs
# (the scalar workspace path itself retries on the compiled plan). Later calls
# keep the analytic tier.
function _batched_backend_gradient_or_columns!(
    totals::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    backend_cache = cache.backend_cache
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(
            totals,
            cache.gradient_buffer,
            cache.model,
            backend_cache,
            params,
        )
        return totals, cache.gradient_buffer
    catch err
        err isa BatchedBackendFallback || rethrow()
    end
    for batch_index = 1:cache.batch_size
        column_args = _batched_args(backend_cache.args, batch_index)
        column_constraints = _batched_constraints(backend_cache.constraints, batch_index)
        seed = collect(view(params, :, batch_index))
        _batched_gradient_column!(
            cache.model,
            backend_cache.workspace,
            view(cache.gradient_buffer, :, batch_index),
            seed,
            column_args,
            column_constraints,
        )
        totals[batch_index] = _logjoint_unconstrained_with_workspace!(
            cache.model,
            backend_cache.workspace,
            seed,
            column_args,
            column_constraints,
        )
    end
    return totals, cache.gradient_buffer
end

function batched_logjoint_gradient_unconstrained!(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    if !isnothing(cache.backend_cache)
        totals = _batched_totals_buffer!(cache.backend_cache.workspace, cache.batch_size, eltype(cache.gradient_buffer))
        _batched_backend_gradient_or_columns!(totals, cache, params)
        return cache.gradient_buffer
    end

    if !isnothing(cache.flat_cache)
        ForwardDiff.gradient!(cache.flat_cache.flat_buffer, cache.flat_cache.objective, vec(params), cache.flat_cache.config)
        return cache.gradient_buffer
    end

    for batch_index = 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        ForwardDiff.gradient!(column_cache.buffer, column_cache.objective, view(params, :, batch_index), column_cache.config)
    end
    return cache.gradient_buffer
end

function _batched_logjoint_unconstrained_from_gradient_cache!(
    destination::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    length(destination) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batched values, got $(length(destination))"))

    if !isnothing(cache.backend_cache)
        return _batched_logjoint_unconstrained_with_workspace!(
            destination,
            cache.model,
            cache.backend_cache.workspace,
            params,
            cache.backend_cache.args,
            cache.backend_cache.constraints,
        )
    end

    if !isnothing(cache.flat_cache)
        objective = cache.flat_cache.objective
        return _batched_logjoint_unconstrained_with_workspace!(
            destination,
            cache.model,
            objective.workspace,
            params,
            objective.args,
            objective.constraints,
        )
    end

    for batch_index = 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        objective = column_cache.objective
        destination[batch_index] = _logjoint_unconstrained_with_workspace!(
            cache.model,
            objective.workspace,
            view(params, :, batch_index),
            objective.args,
            objective.constraints,
        )
    end
    return destination
end

function _batched_logjoint_and_gradient_unconstrained!(
    destination::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    if !isnothing(cache.backend_cache)
        _batched_backend_gradient_or_columns!(destination, cache, params)
        return destination, cache.gradient_buffer
    end

    batched_logjoint_gradient_unconstrained!(cache, params)
    _batched_logjoint_unconstrained_from_gradient_cache!(destination, cache, params)
    return destination, cache.gradient_buffer
end

function batched_logjoint_gradient_unconstrained(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    return copy(batched_logjoint_gradient_unconstrained!(cache, params))
end

function batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_constrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return float(eltype(params))[]

    # The backend totals and environment buffers inherit the params element type
    # (issue #92): an integer-typed constrained matrix would make the totals
    # integer and hit InexactError when a float log-density is accumulated.
    # Promote silently so integer observation/parameter matrices just work.
    eltype(params) <: AbstractFloat || (params = float.(params))

    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    # the backend plan for a dependent-transform model scores in z space, so
    # the CONSTRAINED entry point must use the per-column reference instead
    if !isnothing(workspace.backend_plan) && !_has_dependent_transforms(workspace.layout)
        try
            return _logjoint_with_batched_backend!(workspace, params, batch_args, batch_constraints)
        catch err
            if !(err isa BatchedBackendFallback)
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint!(workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_unconstrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return float(eltype(params))[]

    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    values = Vector{float(eltype(params))}(undef, batch_size)
    return _batched_logjoint_unconstrained_with_workspace!(values, model, workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    cache = BatchedLogjointGradientCache(model, params, args, constraints)
    return batched_logjoint_gradient_unconstrained(cache, params)
end
