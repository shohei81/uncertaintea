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

    layout = parameterlayout(model)
    parameter_total = parametercount(layout)
    constrained_total = parametervaluecount(layout)
    if length(point) == parameter_total
        return Float64[value for value in point]
    elseif length(point) == constrained_total
        return transform_to_unconstrained(model, Float64[value for value in point])
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
    parameters::AbstractVector,
    first_moment::AbstractVector,
    second_moment::AbstractVector,
    gradient::AbstractVector,
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
        return transform_to_constrained(result.model, parameters)
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
    unconstrained = Matrix{Float64}(undef, length(location), num_samples)
    noise = similar(unconstrained)
    _draw_gaussian_particles!(unconstrained, noise, location, log_scale, rng)

    if space === :unconstrained
        return unconstrained
    elseif space === :constrained
        constrained = Matrix{Float64}(undef, parametervaluecount(parameterlayout(result.model)), num_samples)
        _batched_transform_to_constrained!(constrained, result.model, unconstrained)
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
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
    gradient_clip::Real=Inf,
    rng::AbstractRNG=Random.default_rng(),
)
    layout = parameterlayout(model)
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
    _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    gradient_backend = _advi_gradient_backend(cache)

    location_gradient = zeros(Float64, parameter_total)
    log_scale_gradient = zeros(Float64, parameter_total)
    location_m1 = zeros(Float64, parameter_total)
    location_m2 = zeros(Float64, parameter_total)
    log_scale_m1 = zeros(Float64, parameter_total)
    log_scale_m2 = zeros(Float64, parameter_total)
    elbo_history = Vector{Float64}(undef, num_steps)
    gradient_norm_history = Vector{Float64}(undef, num_steps)
    best_location = copy(location)
    best_log_scale = copy(log_scale)
    best_elbo = -Inf

    learning_rate_f64 = Float64(learning_rate)
    beta1_f64 = Float64(beta1)
    beta2_f64 = Float64(beta2)
    adam_epsilon_f64 = Float64(adam_epsilon)
    gradient_clip_f64 = Float64(gradient_clip)

    for iteration in 1:num_steps
        _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
        _batched_logjoint_and_gradient_unconstrained!(values, cache, particles)
        all(isfinite, values) || throw(ArgumentError("batched_advi encountered a non-finite unconstrained logjoint value"))
        all(isfinite, cache.gradient_buffer) ||
            throw(ArgumentError("batched_advi encountered a non-finite unconstrained gradient"))

        fill!(location_gradient, 0.0)
        fill!(log_scale_gradient, 1.0)
        for parameter_index in 1:parameter_total
            scale = exp(log_scale[parameter_index])
            mean_gradient = 0.0
            mean_scale_gradient = 0.0
            for particle_index in 1:num_particles
                target_gradient = cache.gradient_buffer[parameter_index, particle_index]
                mean_gradient += target_gradient
                mean_scale_gradient += target_gradient * scale * noise[parameter_index, particle_index]
            end
            location_gradient[parameter_index] = mean_gradient / num_particles
            log_scale_gradient[parameter_index] += mean_scale_gradient / num_particles
        end

        gradient_norm_history[iteration] = _clip_advi_gradients!(
            location_gradient,
            log_scale_gradient,
            gradient_clip_f64,
        )
        elbo = sum(values) / num_particles + _gaussian_entropy(log_scale)
        elbo_history[iteration] = elbo
        if elbo > best_elbo
            best_elbo = elbo
            copyto!(best_location, location)
            copyto!(best_log_scale, log_scale)
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
    )
end
