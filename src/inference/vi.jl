struct ADVIResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    location::Vector{Float64}
    log_scale::Vector{Float64}
    best_location::Vector{Float64}
    best_log_scale::Vector{Float64}
    elbo_history::Vector{Float64}
    gradient_norm_history::Vector{Float64}
    best_elbo::Float64
    num_particles::Int
    learning_rate::Float64
    gradient_backend::Symbol
    guide::Symbol
    # :fullrank -> the strict lower triangle of the Cholesky factor (the
    # diagonal lives in log_scale); :lowrank -> the d x k factor B of
    # Sigma = D^2 + B B'; nothing for :meanfield.
    scale_factor::Union{Nothing,Matrix{Float64}}
    best_scale_factor::Union{Nothing,Matrix{Float64}}
end

# Mean-field compatibility constructor (also used by the device ADVI path).
function ADVIResult(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    location::Vector{Float64},
    log_scale::Vector{Float64},
    best_location::Vector{Float64},
    best_log_scale::Vector{Float64},
    elbo_history::Vector{Float64},
    gradient_norm_history::Vector{Float64},
    best_elbo::Float64,
    num_particles::Int,
    learning_rate::Float64,
    gradient_backend::Symbol,
)
    return ADVIResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        best_location,
        best_log_scale,
        elbo_history,
        gradient_norm_history,
        best_elbo,
        num_particles,
        learning_rate,
        gradient_backend,
        :meanfield,
        nothing,
        nothing,
    )
end

function _advi_gradient_backend(cache::BatchedLogjointGradientCache)
    !isnothing(cache.backend_cache) && return :backend_native
    !isnothing(cache.flat_cache) && return :flat_forwarddiff
    return :column_forwarddiff
end

function _resolve_unconstrained_point(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    point,
    rng::AbstractRNG,
    name::AbstractString,
)
    if isnothing(point)
        return _initial_hmc_position(model, args, constraints, nothing, rng)
    end

    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    constrained_total = parametervaluecount(layout)
    if length(point) == parameter_total
        return Float64[value for value in point]
    elseif length(point) == constrained_total
        return transform_to_unconstrained(model, Float64[value for value in point], args, constraints)
    end

    throw(
        DimensionMismatch(
            "expected $name to have length $parameter_total in unconstrained space or $constrained_total in constrained space, got $(length(point))",
        ),
    )
end

function _resolve_scale_vector(name::AbstractString, scale, parameter_total::Int)
    if scale isa Real
        return fill(Float64(scale), parameter_total)
    end

    length(scale) == parameter_total ||
        throw(DimensionMismatch("expected $name to have length $parameter_total, got $(length(scale))"))
    return Float64[value for value in scale]
end

function _draw_gaussian_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    rng::AbstractRNG,
)
    size(destination) == size(noise) ||
        throw(DimensionMismatch("expected Gaussian particle destination and noise matrices to have matching shapes"))
    size(destination, 1) == length(location) == length(log_scale) ||
        throw(
            DimensionMismatch(
                "expected Gaussian particle matrices with $(length(location)) rows, got $(size(destination, 1))",
            ),
        )

    for parameter_index in eachindex(location, log_scale)
        scale = exp(log_scale[parameter_index])
        for particle_index in axes(destination, 2)
            epsilon = randn(rng)
            noise[parameter_index, particle_index] = epsilon
            destination[parameter_index, particle_index] =
                location[parameter_index] + scale * epsilon
        end
    end
    return destination
end

function _gaussian_entropy(log_scale::AbstractVector)
    entropy = 0.5 * length(log_scale) * (1.0 + log(2.0 * pi))
    for value in log_scale
        entropy += value
    end
    return entropy
end

function _gaussian_logdensity!(
    destination::AbstractVector,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
)
    size(noise, 2) == length(destination) ||
        throw(
            DimensionMismatch(
                "expected Gaussian log-density destination of length $(size(noise, 2)), got $(length(destination))",
            ),
        )

    normalizer = 0.5 * size(noise, 1) * log(2.0 * pi)
    for value in log_scale
        normalizer += value
    end

    for particle_index in eachindex(destination)
        squared_norm = 0.0
        for parameter_index in axes(noise, 1)
            epsilon = noise[parameter_index, particle_index]
            squared_norm += epsilon * epsilon
        end
        destination[particle_index] = -normalizer - 0.5 * squared_norm
    end
    return destination
end

# --- structured guides (:fullrank / :lowrank) ---------------------------------
#
# Both guides keep the reparameterization estimator and the closed-form
# entropy of the mean-field path. :fullrank draws theta = mu + L eps with L
# lower triangular (diagonal exp(log_scale), strict lower triangle in
# `factor`), so the entropy is the same sum-of-log-diagonals. :lowrank draws
# theta = mu + D eps1 + B eps2 with Sigma = D^2 + B B'; its entropy needs
# logdet(Sigma), computed through the k x k matrix M = I + B' D^-2 B.

function _draw_fullrank_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    rng::AbstractRNG,
)
    Random.randn!(rng, noise)
    for particle_index in axes(destination, 2)
        for parameter_index in eachindex(location)
            value =
                location[parameter_index] +
                exp(log_scale[parameter_index]) * noise[parameter_index, particle_index]
            for lower_index = 1:(parameter_index-1)
                value += factor[parameter_index, lower_index] * noise[lower_index, particle_index]
            end
            destination[parameter_index, particle_index] = value
        end
    end
    return destination
end

function _draw_lowrank_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    rank_noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    rng::AbstractRNG,
)
    Random.randn!(rng, noise)
    Random.randn!(rng, rank_noise)
    for particle_index in axes(destination, 2)
        for parameter_index in eachindex(location)
            value =
                location[parameter_index] +
                exp(log_scale[parameter_index]) * noise[parameter_index, particle_index]
            for rank_index in axes(factor, 2)
                value += factor[parameter_index, rank_index] * rank_noise[rank_index, particle_index]
            end
            destination[parameter_index, particle_index] = value
        end
    end
    return destination
end

# Sigma^-1 B (via Woodbury: D^-2 B M^-1), the intermediate W = D^-2 B, and
# logdet(M) with M = I + B' D^-2 B, shared by the lowrank entropy value and
# its gradients.
function _lowrank_entropy_terms(log_scale::AbstractVector, factor::AbstractMatrix)
    W = similar(factor)
    for rank_index in axes(factor, 2), parameter_index in axes(factor, 1)
        W[parameter_index, rank_index] =
            factor[parameter_index, rank_index] * exp(-2.0 * log_scale[parameter_index])
    end
    M = Matrix{Float64}(I, size(factor, 2), size(factor, 2))
    mul!(M, transpose(factor), W, 1.0, 1.0)
    chol = cholesky(Symmetric(M))
    return W / chol, W, logdet(chol)
end

function _fullrank_entropy(log_scale::AbstractVector)
    return _gaussian_entropy(log_scale)
end

function _lowrank_entropy(log_scale::AbstractVector, factor::AbstractMatrix)
    _, _, logdet_M = _lowrank_entropy_terms(log_scale, factor)
    return _gaussian_entropy(log_scale) + 0.5 * logdet_M
end

function _accumulate_fullrank_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
    particle_valid::AbstractVector{Bool},
    num_valid_particles::Int,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    fill!(factor_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        particle_valid[particle_index] || continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += gradient
            log_scale_gradient[parameter_index] += gradient * noise[parameter_index, particle_index]
            for lower_index = 1:(parameter_index-1)
                factor_gradient[parameter_index, lower_index] +=
                    gradient * noise[lower_index, particle_index]
            end
        end
    end
    for parameter_index in eachindex(location_gradient)
        location_gradient[parameter_index] /= num_valid_particles
        # reparameterization term plus the d/dlog_scale of the entropy (= 1)
        log_scale_gradient[parameter_index] =
            1.0 +
            exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index] /
            num_valid_particles
    end
    factor_gradient ./= num_valid_particles
    return nothing
end

function _accumulate_lowrank_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    rank_noise::AbstractMatrix,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    particle_valid::AbstractVector{Bool},
    num_valid_particles::Int,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    fill!(factor_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        particle_valid[particle_index] || continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += gradient
            log_scale_gradient[parameter_index] += gradient * noise[parameter_index, particle_index]
            for rank_index in axes(factor, 2)
                factor_gradient[parameter_index, rank_index] +=
                    gradient * rank_noise[rank_index, particle_index]
            end
        end
    end
    sigma_inv_factor, W, _ = _lowrank_entropy_terms(log_scale, factor)
    for parameter_index in eachindex(location_gradient)
        location_gradient[parameter_index] /= num_valid_particles
        # (Sigma^-1)_ii = exp(-2w_i) - sum_l C[i,l] W[i,l]  (Woodbury diagonal)
        sigma_inv_diagonal = exp(-2.0 * log_scale[parameter_index])
        for rank_index in axes(factor, 2)
            sigma_inv_diagonal -=
                sigma_inv_factor[parameter_index, rank_index] * W[parameter_index, rank_index]
        end
        log_scale_gradient[parameter_index] =
            exp(2.0 * log_scale[parameter_index]) * sigma_inv_diagonal +
            exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index] /
            num_valid_particles
    end
    for rank_index in axes(factor, 2), parameter_index in eachindex(location_gradient)
        factor_gradient[parameter_index, rank_index] =
            factor_gradient[parameter_index, rank_index] / num_valid_particles +
            sigma_inv_factor[parameter_index, rank_index]
    end
    return nothing
end

# Three-block variant of _clip_advi_gradients! for the structured guides.
function _clip_advi_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradient_clip::Real,
)
    gradient_norm = 0.0
    for value in location_gradient
        gradient_norm += value * value
    end
    for value in log_scale_gradient
        gradient_norm += value * value
    end
    for value in factor_gradient
        gradient_norm += value * value
    end
    gradient_norm = sqrt(gradient_norm)

    if isfinite(gradient_clip) && gradient_norm > gradient_clip
        scale = gradient_clip / gradient_norm
        location_gradient .*= scale
        log_scale_gradient .*= scale
        factor_gradient .*= scale
        return Float64(gradient_clip)
    end

    return gradient_norm
end

"""
    variational_covariance(result::ADVIResult; use_best=true) -> Matrix{Float64}

The covariance of the fitted variational approximation in unconstrained
space: diagonal for `:meanfield`, `L L'` for `:fullrank`, `D^2 + B B'` for
`:lowrank`.
"""
function variational_covariance(result::ADVIResult; use_best::Bool=true)
    log_scale = use_best ? result.best_log_scale : result.log_scale
    factor = use_best ? result.best_scale_factor : result.scale_factor
    if result.guide === :meanfield
        return Matrix(Diagonal(exp.(2.0 .* log_scale)))
    elseif result.guide === :fullrank
        cholesky_factor = copy(factor)
        for parameter_index in eachindex(log_scale)
            cholesky_factor[parameter_index, parameter_index] = exp(log_scale[parameter_index])
        end
        return cholesky_factor * transpose(cholesky_factor)
    end
    return Matrix(Diagonal(exp.(2.0 .* log_scale))) + factor * transpose(factor)
end

function _batched_transform_to_constrained!(
    destination::AbstractMatrix,
    model::TeaModel,
    params::AbstractMatrix,
)
    size(destination, 2) == size(params, 2) ||
        throw(
            DimensionMismatch(
                "expected constrained particle destination with $(size(params, 2)) columns, got $(size(destination, 2))",
            ),
        )

    for particle_index in axes(params, 2)
        _transform_to_constrained!(view(destination, :, particle_index), model, view(params, :, particle_index))
    end
    return destination
end

# Signature-aware column transform (#95 PR-6): the unconstrained columns hold
# the conditioned latent set, so reconstruction reads observations from the
# constraints rather than the syntactic default layout.
function _signature_batched_transform_to_constrained!(
    destination::AbstractMatrix,
    model::TeaModel,
    params::AbstractMatrix,
    args::Tuple,
    constraints::ChoiceMap,
)
    size(destination, 2) == size(params, 2) ||
        throw(
            DimensionMismatch(
                "expected constrained particle destination with $(size(params, 2)) columns, got $(size(destination, 2))",
            ),
        )

    for particle_index in axes(params, 2)
        destination[:, particle_index] =
            transform_to_constrained(model, collect(view(params, :, particle_index)), args, constraints)
    end
    return destination
end

function _clip_advi_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    gradient_clip::Real,
)
    gradient_norm = 0.0
    for value in location_gradient
        gradient_norm += value * value
    end
    for value in log_scale_gradient
        gradient_norm += value * value
    end
    gradient_norm = sqrt(gradient_norm)

    if isfinite(gradient_clip) && gradient_norm > gradient_clip
        scale = gradient_clip / gradient_norm
        for index in eachindex(location_gradient)
            location_gradient[index] *= scale
            log_scale_gradient[index] *= scale
        end
        return Float64(gradient_clip)
    end

    return gradient_norm
end

function _adam_ascent_step!(
    parameters::AbstractArray,
    first_moment::AbstractArray,
    second_moment::AbstractArray,
    gradient::AbstractArray,
    iteration::Int,
    learning_rate::Float64,
    beta1::Float64,
    beta2::Float64,
    epsilon::Float64,
)
    beta1_correction = 1.0 - beta1^iteration
    beta2_correction = 1.0 - beta2^iteration
    for index in eachindex(parameters, first_moment, second_moment, gradient)
        grad = gradient[index]
        first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * grad
        second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * grad * grad
        mhat = first_moment[index] / beta1_correction
        vhat = second_moment[index] / beta2_correction
        parameters[index] += learning_rate * mhat / (sqrt(vhat) + epsilon)
    end
    return parameters
end

function variational_mean(
    result::ADVIResult;
    space::Symbol=:constrained,
    use_best::Bool=true,
)
    parameters = use_best ? result.best_location : result.location
    if space === :unconstrained
        return copy(parameters)
    elseif space === :constrained
        return transform_to_constrained(result.model, parameters, result.args, result.constraints)
    end

    throw(ArgumentError("variational space must be :constrained or :unconstrained"))
end

function variational_samples(
    result::ADVIResult;
    num_samples::Int,
    space::Symbol=:constrained,
    use_best::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
)
    num_samples > 0 || throw(ArgumentError("variational_samples requires num_samples > 0"))

    location = use_best ? result.best_location : result.location
    log_scale = use_best ? result.best_log_scale : result.log_scale
    factor = use_best ? result.best_scale_factor : result.scale_factor
    unconstrained = Matrix{Float64}(undef, length(location), num_samples)
    noise = similar(unconstrained)
    if result.guide === :fullrank
        _draw_fullrank_particles!(unconstrained, noise, location, log_scale, factor, rng)
    elseif result.guide === :lowrank
        rank_noise = Matrix{Float64}(undef, size(factor, 2), num_samples)
        _draw_lowrank_particles!(unconstrained, noise, rank_noise, location, log_scale, factor, rng)
    else
        _draw_gaussian_particles!(unconstrained, noise, location, log_scale, rng)
    end

    if space === :unconstrained
        return unconstrained
    elseif space === :constrained
        layout = _conditioned_parameter_layout(result.model, result.constraints)
        constrained = Matrix{Float64}(undef, parametervaluecount(layout), num_samples)
        _signature_batched_transform_to_constrained!(
            constrained,
            result.model,
            unconstrained,
            result.args,
            result.constraints,
        )
        return constrained
    end

    throw(ArgumentError("variational space must be :constrained or :unconstrained"))
end

function batched_advi(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_steps::Int,
    num_particles::Int=32,
    learning_rate::Real=0.05,
    initial_params=nothing,
    initial_log_scale=-1.0,
    guide::Symbol=:meanfield,
    lowrank_rank::Int=1,
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
    gradient_clip::Real=Inf,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    guide in (:meanfield, :fullrank, :lowrank) ||
        throw(ArgumentError("batched_advi guide must be :meanfield, :fullrank, or :lowrank, got $guide"))
    if backend !== nothing
        guide === :meanfield ||
            throw(ArgumentError("batched_advi guide=$guide is CPU-only; the device path supports :meanfield"))
    end
    if backend !== nothing
        # Device-resident ADVI inner loop. RNG stays host-side; results are
        # statistically equivalent to the CPU path (untouched when backend===nothing).
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_advi `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
        device_precision = precision === nothing ? default_device_precision(backend) : precision
        return _run_device_batched_advi(
            model, args, constraints;
            num_steps=num_steps,
            num_particles=num_particles,
            learning_rate=learning_rate,
            initial_params=initial_params,
            initial_log_scale=initial_log_scale,
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
            gradient_clip=gradient_clip,
            callback=callback,
            callback_every=callback_every,
            backend=backend,
            precision=device_precision,
            rng=rng,
        )
    end

    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_advi requires at least one parameterized latent choice"))
    num_steps > 0 || throw(ArgumentError("batched_advi requires num_steps > 0"))
    num_particles > 0 || throw(ArgumentError("batched_advi requires num_particles > 0"))
    learning_rate > 0 || throw(ArgumentError("batched_advi requires learning_rate > 0"))
    0 <= beta1 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta1 < 1"))
    0 <= beta2 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta2 < 1"))
    adam_epsilon > 0 || throw(ArgumentError("batched_advi requires adam_epsilon > 0"))
    gradient_clip > 0 || throw(ArgumentError("batched_advi requires gradient_clip > 0"))

    location = _resolve_unconstrained_point(model, args, constraints, initial_params, rng, "initial_params")
    log_scale = _resolve_scale_vector("initial_log_scale", initial_log_scale, parameter_total)
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    values = Vector{Float64}(undef, num_particles)
    # Shape the gradient cache WITHOUT consuming the RNG: the cache only needs the
    # particle matrix shape and a finite representative point, not a real draw. The
    # first genuine draw happens inside the loop below, so a same-seed device run
    # (which has no pre-loop draw) sees an identical RNG stream (issue #108).
    particles .= location
    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    gradient_backend = _advi_gradient_backend(cache)

    # Structured-guide state: a zero factor makes both guides start at the
    # mean-field initialization.
    factor = nothing
    factor_gradient = nothing
    factor_m1 = nothing
    factor_m2 = nothing
    rank_noise = nothing
    if guide === :fullrank
        factor = zeros(Float64, parameter_total, parameter_total)
    elseif guide === :lowrank
        1 <= lowrank_rank <= parameter_total || throw(
            ArgumentError(
                "batched_advi requires 1 <= lowrank_rank <= $(parameter_total), got $lowrank_rank",
            ),
        )
        factor = zeros(Float64, parameter_total, lowrank_rank)
        rank_noise = Matrix{Float64}(undef, lowrank_rank, num_particles)
    end
    if !isnothing(factor)
        factor_gradient = zero(factor)
        factor_m1 = zero(factor)
        factor_m2 = zero(factor)
    end

    location_gradient = zeros(Float64, parameter_total)
    log_scale_gradient = zeros(Float64, parameter_total)
    location_m1 = zeros(Float64, parameter_total)
    location_m2 = zeros(Float64, parameter_total)
    log_scale_m1 = zeros(Float64, parameter_total)
    log_scale_m2 = zeros(Float64, parameter_total)
    elbo_history = Vector{Float64}(undef, num_steps)
    gradient_norm_history = Vector{Float64}(undef, num_steps)
    particle_valid = Vector{Bool}(undef, num_particles)
    best_location = copy(location)
    best_log_scale = copy(log_scale)
    best_factor = isnothing(factor) ? nothing : copy(factor)
    best_elbo = -Inf

    learning_rate_f64 = Float64(learning_rate)
    beta1_f64 = Float64(beta1)
    beta2_f64 = Float64(beta2)
    adam_epsilon_f64 = Float64(adam_epsilon)
    gradient_clip_f64 = Float64(gradient_clip)

    for iteration = 1:num_steps
        if guide === :fullrank
            _draw_fullrank_particles!(particles, noise, location, log_scale, factor, rng)
        elseif guide === :lowrank
            _draw_lowrank_particles!(particles, noise, rank_noise, location, log_scale, factor, rng)
        else
            _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
        end
        _batched_logjoint_and_gradient_unconstrained!(values, cache, particles)
        num_valid_particles = 0
        for particle_index = 1:num_particles
            particle_valid[particle_index] =
                isfinite(values[particle_index]) &&
                all(isfinite, view(cache.gradient_buffer, :, particle_index))
            num_valid_particles += particle_valid[particle_index]
        end
        num_valid_particles > 0 ||
            throw(ArgumentError("batched_advi encountered only non-finite unconstrained logjoint values or gradients"))
        num_valid_particles == num_particles ||
            @warn "batched_advi skipped particles with a non-finite logjoint or gradient" maxlog = 1

        if guide === :fullrank
            _accumulate_fullrank_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                cache.gradient_buffer,
                noise,
                log_scale,
                particle_valid,
                num_valid_particles,
            )
        elseif guide === :lowrank
            _accumulate_lowrank_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                cache.gradient_buffer,
                noise,
                rank_noise,
                log_scale,
                factor,
                particle_valid,
                num_valid_particles,
            )
        else
            fill!(location_gradient, 0.0)
            fill!(log_scale_gradient, 1.0)
            for parameter_index = 1:parameter_total
                scale = exp(log_scale[parameter_index])
                mean_gradient = 0.0
                mean_scale_gradient = 0.0
                for particle_index = 1:num_particles
                    particle_valid[particle_index] || continue
                    target_gradient = cache.gradient_buffer[parameter_index, particle_index]
                    mean_gradient += target_gradient
                    mean_scale_gradient += target_gradient * scale * noise[parameter_index, particle_index]
                end
                location_gradient[parameter_index] = mean_gradient / num_valid_particles
                log_scale_gradient[parameter_index] += mean_scale_gradient / num_valid_particles
            end
        end

        gradient_norm_history[iteration] =
            isnothing(factor_gradient) ?
            _clip_advi_gradients!(location_gradient, log_scale_gradient, gradient_clip_f64) :
            _clip_advi_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                gradient_clip_f64,
            )
        elbo_total = 0.0
        for particle_index = 1:num_particles
            particle_valid[particle_index] || continue
            elbo_total += values[particle_index]
        end
        entropy =
            guide === :lowrank ? _lowrank_entropy(log_scale, factor) : _gaussian_entropy(log_scale)
        elbo = elbo_total / num_valid_particles + entropy
        elbo_history[iteration] = elbo
        if elbo > best_elbo
            best_elbo = elbo
            copyto!(best_location, location)
            copyto!(best_log_scale, log_scale)
            isnothing(best_factor) || copyto!(best_factor, factor)
        end

        _adam_ascent_step!(
            location,
            location_m1,
            location_m2,
            location_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        _adam_ascent_step!(
            log_scale,
            log_scale_m1,
            log_scale_m2,
            log_scale_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        isnothing(factor) || _adam_ascent_step!(
            factor,
            factor_m1,
            factor_m2,
            factor_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        isnothing(callback) || _invoke_progress_callback(
            callback, callback_every, :step, iteration, num_steps, NaN, 0)
    end

    return ADVIResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        best_location,
        best_log_scale,
        elbo_history,
        gradient_norm_history,
        best_elbo,
        num_particles,
        learning_rate_f64,
        gradient_backend,
        guide,
        factor,
        best_factor,
    )
end
