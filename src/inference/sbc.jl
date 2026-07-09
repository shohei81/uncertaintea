# Simulation-based calibration (Talts et al. 2018): draw parameters from the
# prior, simulate data given them, run posterior inference, and rank the true
# parameter within the posterior draws. Under correct inference every rank
# statistic is uniform on 0:num_posterior_draws, so non-uniform ranks expose
# sampler bias that pointwise logpdf/gradient tests structurally cannot (e.g.
# a miscalibrated mass-matrix adaptation).

struct SBCResult
    parameter_names::Vector{String}
    # num_parameter_values x num_simulations; each entry in 0:num_posterior_draws
    ranks::Matrix{Int}
    num_posterior_draws::Int
    pvalues::Vector{Float64}
    warnings::Vector{String}
end

has_warnings(result::SBCResult) = !isempty(result.warnings)

function Base.show(io::IO, ::MIME"text/plain", result::SBCResult)
    println(
        io,
        "SBCResult: ",
        size(result.ranks, 2),
        " simulations x ",
        result.num_posterior_draws,
        " posterior draws",
    )
    for (name, pvalue) in zip(result.parameter_names, result.pvalues)
        println(io, "  ", name, ": rank-uniformity p = ", round(pvalue; sigdigits=3))
    end
    if has_warnings(result)
        println(io, "  warnings:")
        for warning in result.warnings
            println(io, "    - ", warning)
        end
    else
        print(io, "  no warnings")
    end
    return nothing
end

_chisq_ccdf(dof::Int, x::Real) = gamma_inc(dof / 2, x / 2)[2]

_sbc_bin(rank::Int, num_bins::Int, num_posterior_draws::Int) =
    min(num_bins, fld(rank * num_bins, num_posterior_draws + 1) + 1)

# Chi-squared uniformity p-value for one parameter's rank statistics. The L+1
# possible ranks are binned equal-width, but when num_bins does not divide
# L+1 the bins hold different numbers of possible ranks, so each bin's
# expected count is proportional to the ranks it can receive (a uniform
# expected count would false-alarm on perfectly calibrated ranks).
function _sbc_uniformity_pvalue(ranks::AbstractVector{Int}, num_posterior_draws::Int, num_bins::Int)
    num_bins = min(num_bins, num_posterior_draws + 1)
    possible = zeros(Int, num_bins)
    for rank = 0:num_posterior_draws
        possible[_sbc_bin(rank, num_bins, num_posterior_draws)] += 1
    end
    counts = zeros(Int, num_bins)
    for rank in ranks
        counts[_sbc_bin(rank, num_bins, num_posterior_draws)] += 1
    end
    total = length(ranks)
    statistic = 0.0
    for bin = 1:num_bins
        expected = total * possible[bin] / (num_posterior_draws + 1)
        statistic += (counts[bin] - expected)^2 / expected
    end
    return _chisq_ccdf(num_bins - 1, statistic)
end

"""
    sbc(model, args=(); num_simulations, num_posterior_draws, num_warmup,
        thin=1, num_bins=..., warn_threshold=1e-3, rng=Random.default_rng(),
        observation_addresses=nothing, nuts_kwargs...) -> SBCResult

Simulation-based calibration of NUTS on `model`: for each simulation, draw
parameters and data jointly from the prior, condition on the simulated data,
sample `num_posterior_draws * thin` posterior draws (keeping every `thin`-th
to reduce autocorrelation), and record the rank of the true parameter within
the kept draws. Correct inference makes each rank uniform on
`0:num_posterior_draws`; per-parameter chi-squared uniformity p-values below
`warn_threshold` produce warnings (see `has_warnings`).

`observation_addresses` defaults to every choice address in a prior trace
that is not a latent parameter slot. Remaining keyword arguments are
forwarded to [`nuts`](@ref). Runtime scales with
`num_simulations * (num_warmup + num_posterior_draws * thin)`; keep the fast
suite variant small and use `bench/sbc_validation.jl` for release-grade runs.
"""
function sbc(
    model::TeaModel,
    args::Tuple=();
    num_simulations::Int,
    num_posterior_draws::Int,
    num_warmup::Int,
    thin::Int=1,
    num_bins::Int=max(2, min(20, cld(num_simulations, 5))),
    warn_threshold::Real=1e-3,
    rng::AbstractRNG=Random.default_rng(),
    observation_addresses::Union{Nothing,AbstractVector}=nothing,
    nuts_kwargs...,
)
    num_simulations > 0 || throw(ArgumentError("sbc requires num_simulations > 0"))
    num_posterior_draws > 0 || throw(ArgumentError("sbc requires num_posterior_draws > 0"))
    thin > 0 || throw(ArgumentError("sbc requires thin > 0"))
    num_bins > 1 || throw(ArgumentError("sbc requires num_bins > 1"))

    layout = parameterlayout(model)
    num_values = parametervaluecount(layout)
    num_values > 0 || throw(ArgumentError("sbc requires a model with at least one latent parameter"))
    data_addresses = isnothing(observation_addresses) ? nothing : collect(Any, observation_addresses)

    ranks = Matrix{Int}(undef, num_values, num_simulations)
    for simulation = 1:num_simulations
        prior_trace, _ = generate(model, args, choicemap(); rng=rng)
        truth = parameter_vector(prior_trace)
        if isnothing(data_addresses)
            # observations = every trace address that is not a latent slot
            latent_map = parameterchoicemap(model, truth)
            data_addresses = Any[
                first(entry) for entry in prior_trace.choices.entries if !haskey(latent_map, first(entry))
            ]
            isempty(data_addresses) &&
                throw(ArgumentError("sbc requires at least one observation address to condition on"))
        end
        data = choicemap((address, prior_trace[address]) for address in data_addresses)
        chain = nuts(
            model,
            args,
            data;
            num_samples=num_posterior_draws * thin,
            num_warmup=num_warmup,
            rng=rng,
            nuts_kwargs...,
        )
        draws = view(chain.constrained_samples, :, thin:thin:(num_posterior_draws*thin))
        for value_index = 1:num_values
            ranks[value_index, simulation] = count(<(truth[value_index]), view(draws, value_index, :))
        end
    end

    names = _export_parameter_names(model, :constrained)
    pvalues = [
        _sbc_uniformity_pvalue(view(ranks, value_index, :), num_posterior_draws, num_bins) for
        value_index = 1:num_values
    ]
    warnings = String[]
    for (name, pvalue) in zip(names, pvalues)
        pvalue < warn_threshold && push!(
            warnings,
            "rank distribution for $name deviates from uniform (p = $(round(pvalue; sigdigits=3)))",
        )
    end
    return SBCResult(names, ranks, num_posterior_draws, pvalues, warnings)
end
