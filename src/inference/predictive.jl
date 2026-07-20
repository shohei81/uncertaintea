# Prior / posterior predictive sampling.
#
# CPU-only by design: each predictive draw runs the dynamic `generate` path so
# that the model's observation/predictive addresses are re-sampled given a set
# of latent parameters.

struct PredictiveDraws
    draws::Vector{ChoiceMap}
end

Base.length(pd::PredictiveDraws) = length(pd.draws)

# Values of a single address collected across every predictive draw. ChoiceMap
# getindex already normalizes the address, so any address spelling works here.
function Base.getindex(pd::PredictiveDraws, address)
    return [draw[address] for draw in pd.draws]
end

# Union of the addresses present in the first draw. Every draw of a given model
# exercises the same predictive addresses, so the first draw is representative.
function addresses(pd::PredictiveDraws)
    isempty(pd.draws) && return Address[]
    first_draw = pd.draws[1]
    result = Address[]
    for entry in first_draw
        push!(result, first(entry))
    end
    return result
end

function Base.show(io::IO, pd::PredictiveDraws)
    print(io, "PredictiveDraws(", length(pd.draws), " draws)")
end

# Shared per-draw kernel: for each latent parameter vector, constrain the model
# to those latents, run `generate`, and keep only the addresses that are NOT
# latent parameters (i.e. the predictive / observation addresses).
function _predictive_from_param_columns(
    model::TeaModel,
    args::Tuple,
    constrained_columns,
    rng::AbstractRNG,
)
    draws = ChoiceMap[]
    for params in constrained_columns
        constraint_cm = parameterchoicemap(model, params)
        trace, _ = generate(model, args, constraint_cm; rng=rng)
        draw = ChoiceMap()
        for entry in trace.choices
            address = first(entry)
            haskey(constraint_cm, address) || _pushchoice!(draw, address, last(entry))
        end
        push!(draws, draw)
    end
    return PredictiveDraws(draws)
end

# Selection of `num_draws` indices out of `total` available draws. When fewer
# than the total are requested the selection is spread evenly across the range;
# the count is capped at the number of available draws.
function _even_draw_indices(total::Int, num_draws::Int)
    total > 0 || throw(ArgumentError("predict requires at least one available draw"))
    num_draws > 0 || throw(ArgumentError("predict requires num_draws > 0"))
    count = min(num_draws, total)
    count == total && return collect(1:total)
    return [round(Int, value) for value in range(1, total; length=count)]
end

# Posterior predictive from pooled HMC/NUTS chains.
function predict(
    model::TeaModel,
    args::Tuple,
    chains::HMCChains;
    num_draws::Int=sum(size(chain.constrained_samples, 2) for chain in chains.chains),
    rng::AbstractRNG=Random.default_rng(),
)
    isempty(chains.chains) && throw(ArgumentError("predict requires at least one chain"))
    pooled = reduce(hcat, (chain.constrained_samples for chain in chains.chains))
    total = size(pooled, 2)
    indices = _even_draw_indices(total, num_draws)
    columns = (view(pooled, :, index) for index in indices)
    return _predictive_from_param_columns(model, args, columns, rng)
end

# Posterior predictive from importance sampling / SIR results: resample particle
# indices in proportion to the normalized weights (systematic), then run the
# per-draw predictive kernel on the resampled constrained particles.
function predict(
    model::TeaModel,
    args::Tuple,
    result::ImportanceSamplingResult;
    num_draws::Int=size(result.constrained_particles, 2),
    rng::AbstractRNG=Random.default_rng(),
)
    num_draws > 0 || throw(ArgumentError("predict requires num_draws > 0"))
    ancestors = _systematic_resample_indices(result.normalized_weights, num_draws, rng)
    columns = (view(result.constrained_particles, :, ancestor) for ancestor in ancestors)
    return _predictive_from_param_columns(model, args, columns, rng)
end

function predict(
    model::TeaModel,
    args::Tuple,
    result::SMCResult;
    num_draws::Int=size(result.importance.constrained_particles, 2),
    rng::AbstractRNG=Random.default_rng(),
)
    return predict(model, args, result.importance; num_draws=num_draws, rng=rng)
end

# Posterior predictive from SIR results: the SIR step already resampled
# `num_samples` particles (`result.ancestors`), so draw from those selected
# particles instead of re-resampling the importance population.
function predict(
    model::TeaModel,
    args::Tuple,
    result::SIRResult;
    num_draws::Int=size(result.constrained_samples, 2),
    rng::AbstractRNG=Random.default_rng(),
)
    num_draws > 0 || throw(ArgumentError("predict requires num_draws > 0"))
    total = size(result.constrained_samples, 2)
    indices = _even_draw_indices(total, num_draws)
    columns = (view(result.constrained_samples, :, index) for index in indices)
    return _predictive_from_param_columns(model, args, columns, rng)
end

# Prior predictive: unconstrained `generate` per draw, keeping ALL addresses.
function predict(
    model::TeaModel,
    args::Tuple=();
    num_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
)
    num_draws > 0 || throw(ArgumentError("predict requires num_draws > 0"))
    draws = ChoiceMap[]
    for _ = 1:num_draws
        trace, _ = generate(model, args, choicemap(); rng=rng)
        draw = ChoiceMap()
        for entry in trace.choices
            _pushchoice!(draw, first(entry), last(entry))
        end
        push!(draws, draw)
    end
    return PredictiveDraws(draws)
end
