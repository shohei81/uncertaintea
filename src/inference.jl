struct HMCMassAdaptationWindowSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    pooled_samples::Int
    weight_sum::Float64
    effective_count::Float64
    mean_weight::Float64
    clip_scale_start::Float64
    clip_scale_end::Float64
    updated::Bool
    mass_mean::Float64
    mass_min::Float64
    mass_max::Float64
end

struct HMCMassAdaptationSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    chains::Int
    num_updated::Int
    mean_pooled_samples::Float64
    mean_weight_sum::Float64
    mean_effective_count::Float64
    min_effective_count::Float64
    max_effective_count::Float64
    mean_weight::Float64
    mean_clip_scale_end::Float64
    mean_mass::Float64
    min_mass::Float64
    max_mass::Float64
end

struct HMCDiagnosticsSummary
    acceptance_rate::Float64
    divergence_rate::Float64
    step_sizes::Vector{Float64}
    mean_step_size::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationSummary}
end

struct HMCChain
    sampler::Symbol
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    acceptance_stats::Vector{Float64}
    energies::Vector{Float64}
    energy_errors::Vector{Float64}
    accepted::BitVector
    divergent::BitVector
    step_size::Float64
    mass_matrix::Vector{Float64}
    num_leapfrog_steps::Int
    max_tree_depth::Int
    tree_depths::Vector{Int}
    integration_steps::Vector{Int}
    target_accept::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationWindowSummary}
end

struct HMCChains{A,C}
    model::TeaModel
    args::A
    constraints::C
    chains::Vector{HMCChain}
end

struct HMCParameterSummary
    index::Int
    binding::Symbol
    address::Any
    mean::Float64
    sd::Float64
    quantiles::Vector{Float64}
    rhat::Float64
    ess::Float64
end

struct HMCSummary
    model::TeaModel
    space::Symbol
    quantile_probs::Vector{Float64}
    diagnostics::HMCDiagnosticsSummary
    parameters::Vector{HMCParameterSummary}
end

mutable struct DualAveragingState
    target_accept::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    mu::Float64
    log_step_size::Float64
    log_step_size_avg::Float64
    hbar::Float64
    iteration::Int
end

mutable struct RunningVarianceState
    mean::Vector{Float64}
    m2::Vector{Float64}
    clipped_sample::Vector{Float64}
    window_length::Int
    count::Int
    weight_sum::Float64
    weight_square_sum::Float64
end

const _RUNNING_VARIANCE_CLIP_START = 4
const _RUNNING_VARIANCE_CLIP_SCALE_EARLY = 8.0
const _RUNNING_VARIANCE_CLIP_SCALE_LATE = 5.0
const _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY = 1.0
const _RUNNING_VARIANCE_FLOOR = 1e-3

struct WarmupSchedule
    initial_buffer::Int
    slow_window_ends::Vector{Int}
    terminal_buffer::Int
end

mutable struct BatchedHMCWorkspace
    logjoint_workspace::BatchedLogjointWorkspace
    gradient_cache::BatchedLogjointGradientCache
    current_gradient::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    momentum::Matrix{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    valid::BitVector
    current_hamiltonian::Vector{Float64}
    proposed_hamiltonian::Vector{Float64}
    proposed_logjoint::Vector{Float64}
    log_accept_ratio::Vector{Float64}
    energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    accepted_step::BitVector
    divergent_step::BitVector
    mass_adaptation_weights::Vector{Float64}
    constrained_position::Matrix{Float64}
    sqrt_inverse_mass_matrix::Vector{Float64}
end

function BatchedHMCWorkspace(
    model::TeaModel,
    position::AbstractMatrix,
    args=(),
    constraints=choicemap(),
    inverse_mass_matrix::AbstractVector=ones(size(position, 1)),
)
    num_params, num_chains = size(position)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    length(inverse_mass_matrix) == num_params ||
        throw(DimensionMismatch("expected inverse mass matrix of length $num_params, got $(length(inverse_mass_matrix))"))

    return BatchedHMCWorkspace(
        BatchedLogjointWorkspace(model),
        BatchedLogjointGradientCache(model, position, batch_args, batch_constraints),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        sqrt.(Float64.(inverse_mass_matrix)),
    )
end

Base.length(chain::HMCChain) = size(chain.unconstrained_samples, 2)
Base.length(chains::HMCChains) = length(chains.chains)
Base.length(summary::HMCSummary) = length(summary.parameters)
Base.getindex(chains::HMCChains, index::Int) = chains.chains[index]
Base.getindex(summary::HMCSummary, index::Int) = summary.parameters[index]
Base.firstindex(chains::HMCChains) = firstindex(chains.chains)
Base.firstindex(summary::HMCSummary) = firstindex(summary.parameters)
Base.lastindex(chains::HMCChains) = lastindex(chains.chains)
Base.lastindex(summary::HMCSummary) = lastindex(summary.parameters)
Base.iterate(chains::HMCChains, state...) = iterate(chains.chains, state...)
Base.iterate(summary::HMCSummary, state...) = iterate(summary.parameters, state...)

_sampler_label(chain::HMCChain) = uppercase(String(chain.sampler))

function _sampler_label(chains::HMCChains)
    isempty(chains.chains) && return "HMC"
    sampler = first(chains.chains).sampler
    return all(chain -> chain.sampler === sampler, chains.chains) ? uppercase(String(sampler)) : "MIXED"
end

function _summary_float(value::Real; digits::Int=4)
    if isnan(value)
        return "NaN"
    elseif !isfinite(value)
        return value > 0 ? "Inf" : "-Inf"
    end
    return string(round(Float64(value); digits=digits))
end

function _show_mass_adaptation_summary_line(io::IO, summary::HMCMassAdaptationSummary; indent::AbstractString="")
    print(
        io,
        indent,
        "window ",
        summary.window_index,
        " [",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        "]",
        " updated=",
        summary.num_updated,
        "/",
        summary.chains,
        " eff=",
        _summary_float(summary.mean_effective_count; digits=2),
        " mass=",
        _summary_float(summary.mean_mass),
        " clip=",
        _summary_float(summary.mean_clip_scale_end; digits=2),
    )
end

function Base.show(io::IO, summary::HMCMassAdaptationWindowSummary)
    print(
        io,
        "HMCMassAdaptationWindowSummary(window=",
        summary.window_index,
        ", iterations=",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        ", effective_count=",
        round(summary.effective_count; digits=2),
        ", updated=",
        summary.updated,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationWindowSummary)
    println(io, "HMCMassAdaptationWindowSummary")
    println(io, "  window: ", summary.window_index)
    println(io, "  iterations: ", summary.iteration_start, ":", summary.iteration_end)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples: ", summary.pooled_samples)
    println(io, "  effective_count: ", _summary_float(summary.effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(io, "  clip_scale: ", _summary_float(summary.clip_scale_start; digits=2), " -> ", _summary_float(summary.clip_scale_end; digits=2))
    println(io, "  updated: ", summary.updated)
    print(io, "  mass: mean=", _summary_float(summary.mass_mean), " min=", _summary_float(summary.mass_min), " max=", _summary_float(summary.mass_max))
end

function Base.show(io::IO, summary::HMCMassAdaptationSummary)
    print(
        io,
        "HMCMassAdaptationSummary(window=",
        summary.window_index,
        ", chains=",
        summary.chains,
        ", effective_count=",
        round(summary.mean_effective_count; digits=2),
        ", updated=",
        summary.num_updated,
        "/",
        summary.chains,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationSummary)
    println(io, "HMCMassAdaptationSummary")
    println(io, "  window: ", summary.window_index, " (", summary.iteration_start, ":", summary.iteration_end, ")")
    println(io, "  chains: ", summary.chains, " updated=", summary.num_updated, "/", summary.chains)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples_mean: ", _summary_float(summary.mean_pooled_samples; digits=2))
    println(io, "  weight_sum_mean: ", _summary_float(summary.mean_weight_sum; digits=2))
    println(io, "  effective_count: mean=", _summary_float(summary.mean_effective_count; digits=2),
        " min=", _summary_float(summary.min_effective_count; digits=2),
        " max=", _summary_float(summary.max_effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(io, "  clip_scale_end_mean: ", _summary_float(summary.mean_clip_scale_end; digits=2))
    print(io, "  mass: mean=", _summary_float(summary.mean_mass), " min=", _summary_float(summary.min_mass), " max=", _summary_float(summary.max_mass))
end

function Base.show(io::IO, diagnostics::HMCDiagnosticsSummary)
    print(
        io,
        "HMCDiagnosticsSummary(acceptance_rate=",
        round(diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(diagnostics.mass_adaptation_windows),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", diagnostics::HMCDiagnosticsSummary)
    println(io, "HMCDiagnosticsSummary")
    println(io, "  acceptance_rate: ", _summary_float(diagnostics.acceptance_rate; digits=3))
    println(io, "  divergence_rate: ", _summary_float(diagnostics.divergence_rate; digits=3))
    println(
        io,
        "  step_size: mean=",
        _summary_float(diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : minimum(diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : maximum(diagnostics.step_sizes)),
    )
    if isempty(diagnostics.mass_adaptation_windows)
        print(io, "  mass_adaptation_windows: none")
        return nothing
    end
    println(io, "  mass_adaptation_windows:")
    for summary in diagnostics.mass_adaptation_windows
        _show_mass_adaptation_summary_line(io, summary; indent="    ")
        println(io)
    end
    return nothing
end

function Base.show(io::IO, chain::HMCChain)
    print(io, "HMCChain(", lowercase(String(chain.sampler)), ", ", chain.model.name)
    print(io, ", samples=", length(chain))
    print(io, ", acceptance_rate=", round(acceptancerate(chain); digits=3))
    print(io, ", divergences=", count(identity, chain.divergent))
    print(io, ", step_size=", round(chain.step_size; digits=4))
    if chain.sampler === :nuts
        print(io, ", max_tree_depth=", chain.max_tree_depth)
    else
        print(io, ", num_leapfrog_steps=", chain.num_leapfrog_steps)
    end
    print(io, ", mass_windows=", length(chain.mass_adaptation_windows), ")")
end

function Base.show(io::IO, chains::HMCChains)
    print(
        io,
        "HMCChains(",
        lowercase(_sampler_label(chains)),
        ", ",
        chains.model.name,
        ", chains=",
        length(chains),
        ", samples=",
        numsamples(chains),
        ", acceptance_rate=",
        round(acceptancerate(chains); digits=3),
        ", divergences=",
        sum(count(identity, chain.divergent) for chain in chains.chains),
        ")",
    )
end

function Base.show(io::IO, summary::HMCSummary)
    print(
        io,
        "HMCSummary(",
        summary.model.name,
        ", space=",
        summary.space,
        ", parameters=",
        length(summary),
        ", acceptance_rate=",
        round(summary.diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(summary.diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(summary.diagnostics.mass_adaptation_windows),
        ", quantiles=",
        summary.quantile_probs,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCSummary)
    println(io, "HMCSummary(", summary.model.name, ")")
    println(io, "  space: ", summary.space)
    println(io, "  quantiles: ", summary.quantile_probs)
    println(io, "  parameters: ", length(summary))
    max_parameters = min(length(summary.parameters), 5)
    for parameter_index in 1:max_parameters
        parameter = summary.parameters[parameter_index]
        mid_quantile = parameter.quantiles[cld(length(parameter.quantiles), 2)]
        println(
            io,
            "    ",
            parameter.binding,
            " @ ",
            parameter.address,
            ": mean=",
            _summary_float(parameter.mean),
            " sd=",
            _summary_float(parameter.sd),
            " median=",
            _summary_float(mid_quantile),
            " rhat=",
            _summary_float(parameter.rhat; digits=3),
            " ess=",
            _summary_float(parameter.ess; digits=1),
        )
    end
    if length(summary.parameters) > max_parameters
        println(io, "    ... ", length(summary.parameters) - max_parameters, " more parameters")
    end
    println(io, "  diagnostics:")
    print(io, "    acceptance_rate: ", _summary_float(summary.diagnostics.acceptance_rate; digits=3))
    println(io)
    print(io, "    divergence_rate: ", _summary_float(summary.diagnostics.divergence_rate; digits=3))
    println(io)
    print(
        io,
        "    step_size: mean=",
        _summary_float(summary.diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : minimum(summary.diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : maximum(summary.diagnostics.step_sizes)),
    )
    println(io)
    if isempty(summary.diagnostics.mass_adaptation_windows)
        print(io, "    mass_adaptation_windows: none")
        return nothing
    end
    println(io, "    mass_adaptation_windows:")
    for window_summary in summary.diagnostics.mass_adaptation_windows
        _show_mass_adaptation_summary_line(io, window_summary; indent="      ")
        println(io)
    end
    return nothing
end

function acceptancerate(chain::HMCChain)
    isempty(chain.acceptance_stats) && return 0.0
    return _sample_mean(chain.acceptance_stats)
end

function acceptancerate(chains::HMCChains)
    total_samples = sum(length(chain.acceptance_stats) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(sum(chain.acceptance_stats) for chain in chains.chains) / total_samples
end

function divergencerate(chain::HMCChain)
    isempty(chain.divergent) && return 0.0
    return count(identity, chain.divergent) / length(chain.divergent)
end

function divergencerate(chains::HMCChains)
    total_samples = sum(length(chain.divergent) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(count(identity, chain.divergent) for chain in chains.chains) / total_samples
end

function acceptancerate(summary::HMCSummary)
    return summary.diagnostics.acceptance_rate
end

function divergencerate(summary::HMCSummary)
    return summary.diagnostics.divergence_rate
end

function massadaptationwindows(chain::HMCChain)
    return chain.mass_adaptation_windows
end

function massadaptationwindows(chains::HMCChains)
    return [chain.mass_adaptation_windows for chain in chains.chains]
end

function massadaptationwindows(summary::HMCSummary)
    return summary.diagnostics.mass_adaptation_windows
end

function treedepths(chain::HMCChain)
    return chain.tree_depths
end

function treedepths(chains::HMCChains)
    return [chain.tree_depths for chain in chains.chains]
end

function integrationsteps(chain::HMCChain)
    return chain.integration_steps
end

function integrationsteps(chains::HMCChains)
    return [chain.integration_steps for chain in chains.chains]
end

function _diagnostic_space_samples(chain::HMCChain, space::Symbol)
    if space === :constrained
        return chain.constrained_samples
    elseif space === :unconstrained
        return chain.unconstrained_samples
    end

    throw(ArgumentError("diagnostic space must be :constrained or :unconstrained"))
end

function nchains(chains::HMCChains)
    return length(chains)
end

function numsamples(chains::HMCChains)
    isempty(chains.chains) && return 0
    return length(first(chains.chains))
end

function _summary_address(address::AddressSpec)
    parts = Any[part isa AddressLiteralPart ? part.value : part.value for part in address.parts]
    return length(parts) == 1 ? first(parts) : Tuple(parts)
end

function _validate_hmc_diagnostics(chains::HMCChains, space::Symbol)
    length(chains) >= 2 || throw(ArgumentError("multi-chain diagnostics require at least 2 chains"))
    num_samples = numsamples(chains)
    num_samples >= 4 || throw(ArgumentError("multi-chain diagnostics require at least 4 samples per chain"))

    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params = size(first_samples, 1)
    for chain in chains.chains
        chain_samples = _diagnostic_space_samples(chain, space)
        size(chain_samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must have the same parameter dimension"))
        size(chain_samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must have the same number of samples"))
    end

    return num_params, num_samples
end

function _validate_summary_quantiles(quantile_probs)
    isempty(quantile_probs) && throw(ArgumentError("summary quantiles must be non-empty"))
    probabilities = Float64[Float64(prob) for prob in quantile_probs]
    for prob in probabilities
        0.0 <= prob <= 1.0 || throw(ArgumentError("summary quantiles must lie in [0, 1]"))
    end
    return probabilities
end

function _mass_adaptation_diagnostics(chains::HMCChains)
    groups = Dict{NTuple{4, Int}, Vector{HMCMassAdaptationWindowSummary}}()
    for chain in chains.chains
        for window in chain.mass_adaptation_windows
            key = (window.window_index, window.iteration_start, window.iteration_end, window.window_length)
            push!(get!(groups, key, HMCMassAdaptationWindowSummary[]), window)
        end
    end

    summaries = HMCMassAdaptationSummary[]
    for key in sort!(collect(keys(groups)); by=identity)
        windows = groups[key]
        push!(
            summaries,
            HMCMassAdaptationSummary(
                key[1],
                key[2],
                key[3],
                key[4],
                length(windows),
                count(window -> window.updated, windows),
                _sample_mean([window.pooled_samples for window in windows]),
                _sample_mean([window.weight_sum for window in windows]),
                _sample_mean([window.effective_count for window in windows]),
                minimum(window.effective_count for window in windows),
                maximum(window.effective_count for window in windows),
                _sample_mean([window.mean_weight for window in windows]),
                _sample_mean([window.clip_scale_end for window in windows]),
                _sample_mean([window.mass_mean for window in windows]),
                minimum(window.mass_min for window in windows),
                maximum(window.mass_max for window in windows),
            ),
        )
    end
    return summaries
end

function _diagnostics_summary(chains::HMCChains)
    step_sizes = Float64[chain.step_size for chain in chains.chains]
    return HMCDiagnosticsSummary(
        acceptancerate(chains),
        divergencerate(chains),
        step_sizes,
        isempty(step_sizes) ? 0.0 : _sample_mean(step_sizes),
        _mass_adaptation_diagnostics(chains),
    )
end

function _sample_mean(values::AbstractVector)
    return sum(values) / length(values)
end

function _sample_variance(values::AbstractVector, mean_value::Real=_sample_mean(values))
    length(values) > 1 || return 0.0
    return sum((value - mean_value)^2 for value in values) / (length(values) - 1)
end

function _sample_sd(values::AbstractVector, mean_value::Real=_sample_mean(values))
    return sqrt(max(_sample_variance(values, mean_value), 0.0))
end

function _pooled_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    pooled = Vector{Float64}(undef, length(chains) * num_samples)
    offset = 1
    for chain in chains.chains
        samples = _diagnostic_space_samples(chain, space)
        pooled[offset:(offset + num_samples - 1)] = samples[parameter_index, :]
        offset += num_samples
    end
    return pooled
end

function _quantile(sorted_values::AbstractVector, probability::Float64)
    num_values = length(sorted_values)
    num_values == 0 && throw(ArgumentError("quantile requires at least one value"))
    num_values == 1 && return Float64(sorted_values[1])

    position = 1 + (num_values - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return Float64(sorted_values[lower])
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]
end

function _quantiles(values::AbstractVector, probabilities::AbstractVector{Float64})
    sorted_values = sort(collect(values))
    return Float64[_quantile(sorted_values, probability) for probability in probabilities]
end

function _split_chain_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    split_samples = fld(num_samples, 2)
    even_samples = 2 * split_samples
    split_draws = Matrix{Float64}(undef, 2 * length(chains), split_samples)

    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        split_draws[2 * chain_index - 1, :] = samples[parameter_index, 1:split_samples]
        split_draws[2 * chain_index, :] = samples[parameter_index, split_samples + 1:even_samples]
    end

    return split_draws
end

function _chain_draw_statistics(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means = Vector{Float64}(undef, num_chains)
    chain_variances = Vector{Float64}(undef, num_chains)
    for chain_index in 1:num_chains
        chain_draws = view(draws, chain_index, :)
        chain_means[chain_index] = _sample_mean(chain_draws)
        chain_variances[chain_index] = _sample_variance(chain_draws, chain_means[chain_index])
    end

    within_variance = _sample_mean(chain_variances)
    between_variance = num_samples > 1 ? num_samples * _sample_variance(chain_means) : 0.0
    var_plus = ((num_samples - 1) / num_samples) * within_variance + between_variance / num_samples
    return chain_means, chain_variances, within_variance, between_variance, var_plus
end

function _split_rhat(draws::AbstractMatrix)
    _, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    if within_variance == 0
        return var_plus == 0 ? 1.0 : Inf
    end

    return sqrt(max(var_plus / within_variance, 1.0))
end

function _autocovariance(draws::AbstractVector, lag::Int, mean_value::Real)
    num_samples = length(draws)
    total = 0.0
    for index in 1:(num_samples - lag)
        total += (draws[index] - mean_value) * (draws[index + lag] - mean_value)
    end
    return total / num_samples
end

function _split_ess(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    total_draws = num_chains * num_samples

    if within_variance == 0 && var_plus == 0
        return Float64(total_draws)
    elseif var_plus <= 0
        return 0.0
    end

    pair_sums = Float64[]
    autocovariance_means = Vector{Float64}(undef, num_chains)
    for pair_start in 0:2:(num_samples - 1)
        pair_sum = 0.0
        for lag in pair_start:min(pair_start + 1, num_samples - 1)
            for chain_index in 1:num_chains
                autocovariance_means[chain_index] = _autocovariance(view(draws, chain_index, :), lag, chain_means[chain_index])
            end
            mean_autocovariance = _sample_mean(autocovariance_means)
            rho_hat = lag == 0 ? 1.0 : 1 - (within_variance - mean_autocovariance) / var_plus
            pair_sum += min(rho_hat, 1.0)
        end

        pair_sum > 0 || break
        push!(pair_sums, pair_sum)
    end

    for index in 2:length(pair_sums)
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    end

    tau_hat = -1 + 2 * sum(pair_sums)
    tau_hat = max(tau_hat, 1.0)
    return min(Float64(total_draws), Float64(total_draws) / tau_hat)
end

function rhat(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_rhat(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function ess(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_ess(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function summarize(chains::HMCChains; space::Symbol=:constrained, quantiles=(0.05, 0.5, 0.95))
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    quantile_probs = _validate_summary_quantiles(quantiles)
    rhats = rhat(chains; space=space)
    ess_values = ess(chains; space=space)
    diagnostics = _diagnostics_summary(chains)
    layout = parameterlayout(chains.model)
    parametercount(layout) == num_params ||
        throw(DimensionMismatch("summary expected $num_params parameters in layout, got $(parametercount(layout))"))

    parameters = Vector{HMCParameterSummary}(undef, num_params)
    for slot in layout.slots
        draws = _pooled_parameter_draws(chains, slot.index, space)
        mean_value = _sample_mean(draws)
        parameters[slot.index] = HMCParameterSummary(
            slot.index,
            slot.binding,
            _summary_address(slot.address),
            mean_value,
            _sample_sd(draws, mean_value),
            _quantiles(draws, quantile_probs),
            rhats[slot.index],
            ess_values[slot.index],
        )
    end

    return HMCSummary(chains.model, space, quantile_probs, diagnostics, parameters)
end

function _validate_hmc_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("HMC requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("HMC requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("HMC requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("HMC requires step_size > 0"))
    num_leapfrog_steps > 0 || throw(ArgumentError("HMC requires num_leapfrog_steps > 0"))
    0 < target_accept < 1 || throw(ArgumentError("HMC requires 0 < target_accept < 1"))
    divergence_threshold > 0 || throw(ArgumentError("HMC requires divergence_threshold > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("HMC requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("HMC requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_nuts_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    max_delta_energy::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("NUTS requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("NUTS requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("NUTS requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("NUTS requires step_size > 0"))
    max_tree_depth > 0 || throw(ArgumentError("NUTS requires max_tree_depth > 0"))
    0 < target_accept < 1 || throw(ArgumentError("NUTS requires 0 < target_accept < 1"))
    max_delta_energy > 0 || throw(ArgumentError("NUTS requires max_delta_energy > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("NUTS requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("NUTS requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_hmc_chains_arguments(num_chains::Int, sampler_name::AbstractString="HMC")
    num_chains > 0 || throw(ArgumentError("$sampler_name requires num_chains > 0"))
    return nothing
end

function _validate_batched_hmc_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains)
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _initial_hmc_position(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params,
    rng::AbstractRNG,
)
    if isnothing(initial_params)
        trace, _ = generate(model, args, constraints; rng=rng)
        return transform_to_unconstrained(trace)
    end

    expected = parametercount(parameterlayout(model))
    length(initial_params) == expected || throw(DimensionMismatch("expected $expected initial parameters, got $(length(initial_params))"))
    return Float64[value for value in initial_params]
end

function _sample_momentum(rng::AbstractRNG, inverse_mass_matrix::AbstractVector)
    return randn(rng, length(inverse_mass_matrix)) ./ sqrt.(inverse_mass_matrix)
end

function _initial_batched_hmc_positions(
    model::TeaModel,
    args,
    constraints,
    initial_params,
    rng::AbstractRNG,
    num_params::Int,
    num_chains::Int,
)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    seeds = rand(rng, UInt, num_chains)

    for chain_index in 1:num_chains
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chain_rng = MersenneTwister(seeds[chain_index])
        positions[:, chain_index] = _initial_hmc_position(
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            chain_initial_params,
            chain_rng,
        )
    end

    return positions
end

function _sample_batched_momentum(
    rng::AbstractRNG,
    inverse_mass_matrix::AbstractVector,
    num_chains::Int,
)
    momentum = Matrix{Float64}(undef, length(inverse_mass_matrix), num_chains)
    _sample_batched_momentum!(momentum, rng, sqrt.(Float64.(inverse_mass_matrix)))
    return momentum
end

function _sample_batched_momentum!(
    destination::AbstractMatrix,
    rng::AbstractRNG,
    sqrt_inverse_mass_matrix::AbstractVector,
)
    size(destination, 1) == length(sqrt_inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(sqrt_inverse_mass_matrix)) rows, got $(size(destination, 1))"))

    for chain_index in axes(destination, 2)
        for parameter_index in eachindex(sqrt_inverse_mass_matrix)
            destination[parameter_index, chain_index] =
                randn(rng) / sqrt_inverse_mass_matrix[parameter_index]
        end
    end
    return destination
end

function _update_sqrt_inverse_mass_matrix!(
    destination::AbstractVector,
    inverse_mass_matrix::AbstractVector,
)
    length(destination) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected inverse mass matrix of length $(length(destination)), got $(length(inverse_mass_matrix))"))
    for index in eachindex(destination, inverse_mass_matrix)
        destination[index] = sqrt(Float64(inverse_mass_matrix[index]))
    end
    return destination
end

function _batched_kinetic_energy(
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    energy = Vector{Float64}(undef, size(momentum, 2))
    return _batched_kinetic_energy!(energy, momentum, inverse_mass_matrix)
end

function _batched_kinetic_energy!(
    destination::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    size(momentum, 1) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(inverse_mass_matrix)) rows, got $(size(momentum, 1))"))
    size(momentum, 2) == length(destination) ||
        throw(DimensionMismatch("expected kinetic-energy destination of length $(size(momentum, 2)), got $(length(destination))"))

    for chain_index in axes(momentum, 2)
        kinetic_energy = 0.0
        for parameter_index in eachindex(inverse_mass_matrix)
            momentum_value = momentum[parameter_index, chain_index]
            kinetic_energy += momentum_value^2 * inverse_mass_matrix[parameter_index]
        end
        destination[chain_index] = kinetic_energy / 2
    end
    return destination
end

function _batched_hamiltonian(
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    hamiltonian = Vector{Float64}(undef, length(logjoint_values))
    return _batched_hamiltonian!(hamiltonian, logjoint_values, momentum, inverse_mass_matrix)
end

function _batched_hamiltonian!(
    destination::AbstractVector,
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    length(logjoint_values) == length(destination) ||
        throw(DimensionMismatch("expected hamiltonian inputs of length $(length(destination)), got $(length(logjoint_values))"))

    _batched_kinetic_energy!(destination, momentum, inverse_mass_matrix)
    for chain_index in eachindex(destination)
        destination[chain_index] -= Float64(logjoint_values[chain_index])
    end
    return destination
end

function _batched_acceptance_probability(log_accept_ratio::AbstractVector)
    probabilities = Vector{Float64}(undef, length(log_accept_ratio))
    return _batched_acceptance_probability!(probabilities, log_accept_ratio)
end

function _batched_acceptance_probability!(
    destination::AbstractVector,
    log_accept_ratio::AbstractVector,
)
    length(destination) == length(log_accept_ratio) ||
        throw(DimensionMismatch("expected acceptance-probability destination of length $(length(log_accept_ratio)), got $(length(destination))"))

    for index in eachindex(log_accept_ratio)
        destination[index] = _acceptance_probability(log_accept_ratio[index])
    end
    return destination
end

function _mean_acceptance_probability(accept_prob::AbstractVector)
    isempty(accept_prob) && return 0.0
    return sum(accept_prob) / length(accept_prob)
end

function _mean_batched_adaptation_probability(
    accept_prob::AbstractVector,
    divergent::AbstractVector,
)
    length(accept_prob) == length(divergent) ||
        throw(DimensionMismatch("expected acceptance and divergence vectors of matching length, got $(length(accept_prob)) and $(length(divergent))"))
    isempty(accept_prob) && return 0.0

    total = 0.0
    for index in eachindex(accept_prob, divergent)
        total += divergent[index] ? 0.0 : accept_prob[index]
    end
    return total / length(accept_prob)
end

function _batched_leapfrog!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = workspace.proposal_position
    p = workspace.proposal_momentum
    proposed_gradient = workspace.proposal_gradient
    valid = workspace.valid
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(proposed_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(proposed_gradient))"))

    copyto!(q, position)
    copyto!(p, workspace.momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index in 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step in 1:num_leapfrog_steps
        for chain_index in 1:num_chains
            valid[chain_index] || continue
            q[:, chain_index] .+= step_size .* (inverse_mass_matrix .* p[:, chain_index])
        end

        if leapfrog_step < num_leapfrog_steps
            gradient = batched_logjoint_gradient_unconstrained!(workspace.gradient_cache, q)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, gradient = _batched_logjoint_and_gradient_unconstrained!(
                workspace.proposed_logjoint,
                workspace.gradient_cache,
                q,
            )
            copyto!(proposed_gradient, gradient)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(proposed_gradient, :, chain_index)) || !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index in 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* proposed_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, workspace.proposed_logjoint, proposed_gradient, valid
end

function _leapfrog(
    model::TeaModel,
    position::Vector{Float64},
    momentum::Vector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = copy(position)
    p = copy(momentum)

    gradient = _logjoint_gradient!(gradient_cache, q)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step in 1:num_leapfrog_steps
        q .+= step_size .* (inverse_mass_matrix .* p)
        gradient = _logjoint_gradient!(gradient_cache, q)
        all(isfinite, gradient) || return nothing

        if leapfrog_step < num_leapfrog_steps
            p .+= step_size .* gradient
        end
    end

    p .+= (step_size / 2) .* gradient
    p .*= -1

    proposed_logjoint = logjoint_unconstrained(model, q, args, constraints)
    isfinite(proposed_logjoint) || return nothing
    return q, p, proposed_logjoint
end

function _kinetic_energy(momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return sum((momentum .^ 2) .* inverse_mass_matrix) / 2
end

function _hamiltonian(logjoint_value::Float64, momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return -logjoint_value + _kinetic_energy(momentum, inverse_mass_matrix)
end

struct NUTSState
    position::Vector{Float64}
    momentum::Vector{Float64}
    logjoint::Float64
    gradient::Vector{Float64}
end

struct NUTSSubtree
    left::NUTSState
    right::NUTSState
    proposal::NUTSState
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    turning::Bool
    divergent::Bool
end

function _copy_nuts_state(state::NUTSState)
    return NUTSState(copy(state.position), copy(state.momentum), state.logjoint, copy(state.gradient))
end

function _logaddexp(x::Float64, y::Float64)
    if x == -Inf
        return y
    elseif y == -Inf
        return x
    end
    high = max(x, y)
    return high + log1p(exp(min(x, y) - high))
end

function _leapfrog_step(
    model::TeaModel,
    state::NUTSState,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
)
    q = copy(state.position)
    p = copy(state.momentum)
    p .+= (step_size / 2) .* state.gradient
    q .+= step_size .* (inverse_mass_matrix .* p)
    proposed_logjoint = logjoint_unconstrained(model, q, args, constraints)
    isfinite(proposed_logjoint) || return nothing
    proposed_gradient = _logjoint_gradient!(gradient_cache, q)
    all(isfinite, proposed_gradient) || return nothing
    gradient = copy(proposed_gradient)
    p .+= (step_size / 2) .* gradient
    return NUTSState(q, p, proposed_logjoint, gradient)
end

function _is_turning(
    left_position::AbstractVector,
    right_position::AbstractVector,
    left_momentum::AbstractVector,
    right_momentum::AbstractVector,
)
    delta = right_position .- left_position
    return dot(delta, left_momentum) <= 0 || dot(delta, right_momentum) <= 0
end

function _build_nuts_subtree(
    model::TeaModel,
    start_state::NUTSState,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    direction::Int,
    depth::Int,
    initial_hamiltonian::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    left = _copy_nuts_state(start_state)
    right = _copy_nuts_state(start_state)
    proposal = _copy_nuts_state(start_state)
    current = start_state
    log_weight = -Inf
    accept_stat_sum = 0.0
    accept_stat_count = 0
    integration_steps = 0
    turning = false
    divergent = false

    for _ in 1:(1 << depth)
        next_state = _leapfrog_step(
            model,
            current,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            direction * step_size,
        )
        if isnothing(next_state)
            divergent = true
            break
        end
        current = next_state
        integration_steps += 1
        if direction < 0
            left = _copy_nuts_state(next_state)
        else
            right = _copy_nuts_state(next_state)
        end

        proposed_hamiltonian = _hamiltonian(next_state.logjoint, next_state.momentum, inverse_mass_matrix)
        delta_energy = proposed_hamiltonian - initial_hamiltonian
        if !isfinite(delta_energy) || delta_energy > max_delta_energy
            divergent = true
            break
        end

        accept_stat_sum += min(1.0, exp(min(0.0, initial_hamiltonian - proposed_hamiltonian)))
        accept_stat_count += 1
        candidate_log_weight = -proposed_hamiltonian
        combined_log_weight = _logaddexp(log_weight, candidate_log_weight)
        if !isfinite(log_weight) || log(rand(rng)) < candidate_log_weight - combined_log_weight
            proposal = _copy_nuts_state(next_state)
        end
        log_weight = combined_log_weight

        turning = _is_turning(left.position, right.position, left.momentum, right.momentum)
        turning && break
    end

    return NUTSSubtree(
        left,
        right,
        proposal,
        log_weight,
        accept_stat_sum,
        accept_stat_count,
        integration_steps,
        turning,
        divergent,
    )
end

function _nuts_proposal(
    model::TeaModel,
    position::Vector{Float64},
    current_logjoint::Float64,
    current_gradient::Vector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    initial_momentum = _sample_momentum(rng, inverse_mass_matrix)
    initial_state = NUTSState(copy(position), initial_momentum, current_logjoint, copy(current_gradient))
    left = _copy_nuts_state(initial_state)
    right = _copy_nuts_state(initial_state)
    proposal = _copy_nuts_state(initial_state)
    initial_hamiltonian = _hamiltonian(initial_state.logjoint, initial_state.momentum, inverse_mass_matrix)
    log_weight = -initial_hamiltonian
    accept_stat_sum = 0.0
    accept_stat_count = 0
    integration_steps = 0
    tree_depth = 0
    divergent = false
    turning = false

    while tree_depth < max_tree_depth && !divergent && !turning
        direction = rand(rng, Bool) ? 1 : -1
        subtree = _build_nuts_subtree(
            model,
            direction < 0 ? left : right,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
            direction,
            tree_depth,
            initial_hamiltonian,
            max_delta_energy,
            rng,
        )
        tree_depth += 1

        if subtree.integration_steps == 0
            divergent = subtree.divergent
            break
        end

        if direction < 0
            left = subtree.left
        else
            right = subtree.right
        end

        integration_steps += subtree.integration_steps
        accept_stat_sum += subtree.accept_stat_sum
        accept_stat_count += subtree.accept_stat_count
        if isfinite(subtree.log_weight)
            combined_log_weight = _logaddexp(log_weight, subtree.log_weight)
            if log(rand(rng)) < subtree.log_weight - combined_log_weight
                proposal = _copy_nuts_state(subtree.proposal)
            end
            log_weight = combined_log_weight
        end

        divergent = subtree.divergent
        turning = subtree.turning || _is_turning(left.position, right.position, left.momentum, right.momentum)
    end

    accept_stat = accept_stat_count == 0 ? 0.0 : accept_stat_sum / accept_stat_count
    proposed_hamiltonian = _hamiltonian(proposal.logjoint, proposal.momentum, inverse_mass_matrix)
    energy_error = proposed_hamiltonian - initial_hamiltonian
    moved = any(proposal.position .!= position)
    return proposal, accept_stat, tree_depth, integration_steps, proposed_hamiltonian, energy_error, divergent, moved
end

function _dual_averaging_state(step_size::Float64, target_accept::Float64)
    log_step_size = log(step_size)
    return DualAveragingState(
        target_accept,
        0.05,
        10.0,
        0.75,
        log(10 * step_size),
        log_step_size,
        log_step_size,
        0.0,
        0,
    )
end

function _update_step_size!(state::DualAveragingState, accept_prob::Float64)
    state.iteration += 1
    eta = 1 / (state.iteration + state.t0)
    state.hbar = (1 - eta) * state.hbar + eta * (state.target_accept - accept_prob)
    state.log_step_size = state.mu - sqrt(state.iteration) / state.gamma * state.hbar
    eta_bar = state.iteration^(-state.kappa)
    state.log_step_size_avg = eta_bar * state.log_step_size + (1 - eta_bar) * state.log_step_size_avg
    return exp(state.log_step_size)
end

function _final_step_size(state::DualAveragingState)
    return exp(state.log_step_size_avg)
end

function _running_variance_state(num_params::Int, window_length::Int=_RUNNING_VARIANCE_CLIP_START + 16)
    window_length > 0 || throw(ArgumentError("running variance state requires window_length > 0"))
    return RunningVarianceState(
        zeros(num_params),
        zeros(num_params),
        zeros(num_params),
        window_length,
        0,
        0.0,
        0.0,
    )
end

function _warmup_window_length(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    window_start = window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
    return schedule.slow_window_ends[window_index] - window_start + 1
end

function _warmup_window_start(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    return window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
end

function _running_variance_window_progress(state::RunningVarianceState)
    if state.window_length <= _RUNNING_VARIANCE_CLIP_START
        return 1.0
    end
    return min(
        max(state.count - _RUNNING_VARIANCE_CLIP_START, 0) /
        (state.window_length - _RUNNING_VARIANCE_CLIP_START),
        1.0,
    )
end

function _running_variance_clip_scale(state::RunningVarianceState)
    state.count <= _RUNNING_VARIANCE_CLIP_START && return _RUNNING_VARIANCE_CLIP_SCALE_EARLY
    progress = _running_variance_window_progress(state)
    return _RUNNING_VARIANCE_CLIP_SCALE_EARLY +
        (_RUNNING_VARIANCE_CLIP_SCALE_LATE - _RUNNING_VARIANCE_CLIP_SCALE_EARLY) * progress
end

function _running_variance_sample!(
    state::RunningVarianceState,
    sample::AbstractVector,
)
    clipped_sample = state.clipped_sample
    if state.count < _RUNNING_VARIANCE_CLIP_START
        copyto!(clipped_sample, sample)
        return clipped_sample
    end

    clip_scale = _running_variance_clip_scale(state)
    @inbounds for index in eachindex(clipped_sample, sample, state.mean, state.m2)
        variance = state.m2[index] / max(state.count - 1, 1)
        bound = clip_scale * sqrt(max(variance, _RUNNING_VARIANCE_FLOOR))
        delta = sample[index] - state.mean[index]
        clipped_sample[index] = state.mean[index] + clamp(delta, -bound, bound)
    end
    return clipped_sample
end

function _warmup_schedule(num_warmup::Int)
    num_warmup < 0 && throw(ArgumentError("warmup schedule requires num_warmup >= 0"))
    num_warmup == 0 && return WarmupSchedule(0, Int[], 0)

    if num_warmup < 20
        initial_buffer = min(5, max(num_warmup - 1, 0))
        if initial_buffer == num_warmup
            return WarmupSchedule(num_warmup, Int[], 0)
        end
        return WarmupSchedule(initial_buffer, [num_warmup], 0)
    end

    initial_buffer = min(max(5, fld(num_warmup, 10)), num_warmup)
    terminal_buffer = min(max(5, fld(num_warmup, 10)), max(num_warmup - initial_buffer, 0))
    slow_budget = num_warmup - initial_buffer - terminal_buffer
    if slow_budget <= 0
        return WarmupSchedule(num_warmup, Int[], 0)
    end

    window_ends = Int[]
    next_window_size = min(25, slow_budget)
    window_start = initial_buffer + 1
    remaining = slow_budget
    while remaining > 0
        window_size = remaining <= fld(3 * next_window_size, 2) ? remaining : next_window_size
        window_end = window_start + window_size - 1
        push!(window_ends, window_end)
        remaining -= window_size
        window_start = window_end + 1
        next_window_size *= 2
    end

    return WarmupSchedule(initial_buffer, window_ends, terminal_buffer)
end

function _running_variance_effective_count(state::RunningVarianceState)
    state.weight_square_sum <= 0 && return 0.0
    return state.weight_sum^2 / state.weight_square_sum
end

function _mass_adaptation_window_summary(
    schedule::WarmupSchedule,
    window_index::Int,
    state::RunningVarianceState,
    inverse_mass_matrix::AbstractVector,
    updated::Bool,
)
    mass_matrix = 1 ./ inverse_mass_matrix
    pooled_samples = state.count
    return HMCMassAdaptationWindowSummary(
        window_index,
        _warmup_window_start(schedule, window_index),
        schedule.slow_window_ends[window_index],
        state.window_length,
        pooled_samples,
        state.weight_sum,
        _running_variance_effective_count(state),
        pooled_samples == 0 ? 0.0 : state.weight_sum / pooled_samples,
        _RUNNING_VARIANCE_CLIP_SCALE_EARLY,
        _running_variance_clip_scale(state),
        updated,
        sum(mass_matrix) / length(mass_matrix),
        minimum(mass_matrix),
        maximum(mass_matrix),
    )
end

function _update_running_variance!(
    state::RunningVarianceState,
    sample::AbstractVector,
    weight::Real,
)
    weight_value = Float64(weight)
    0 <= weight_value <= 1 || throw(ArgumentError("running variance weight must lie in [0, 1], got $weight"))
    iszero(weight_value) && return nothing

    update_sample = _running_variance_sample!(state, sample)
    state.count += 1
    new_weight_sum = state.weight_sum + weight_value
    @inbounds for index in eachindex(update_sample, state.mean, state.m2)
        delta = update_sample[index] - state.mean[index]
        state.mean[index] += (weight_value / new_weight_sum) * delta
        delta2 = update_sample[index] - state.mean[index]
        state.m2[index] += weight_value * delta * delta2
    end
    state.weight_sum = new_weight_sum
    state.weight_square_sum += weight_value^2
    return nothing
end

function _update_running_variance!(state::RunningVarianceState, sample::AbstractVector)
    _update_running_variance!(state, sample, 1.0)
    return nothing
end

function _update_running_variance!(
    state::RunningVarianceState,
    samples::AbstractMatrix,
    weights::AbstractVector,
)
    size(samples, 2) == length(weights) ||
        throw(DimensionMismatch("expected $(size(samples, 2)) running variance weights, got $(length(weights))"))
    for column_index in axes(samples, 2)
        _update_running_variance!(state, view(samples, :, column_index), weights[column_index])
    end
    return nothing
end

function _inverse_mass_matrix(state::RunningVarianceState, regularization::Float64)
    effective_count = _running_variance_effective_count(state)
    if effective_count < 2
        return ones(length(state.mean))
    end

    variance_denom = state.weight_sum - state.weight_square_sum / state.weight_sum
    variance_denom <= 0 && return ones(length(state.mean))
    variance = state.m2 ./ variance_denom
    shrinkage = effective_count / (effective_count + 5.0)
    regularized_variance = shrinkage .* variance .+ (1 - shrinkage)
    return 1 ./ max.(regularized_variance, regularization)
end

function _mass_adaptation_weight(
    state::RunningVarianceState,
    accepted_step::Bool,
    accept_prob::Real,
    divergent_step::Bool,
)
    divergent_step && return 0.0
    accepted_step && return 1.0
    if !isfinite(accept_prob)
        return 0.0
    end
    progress = _running_variance_window_progress(state)
    rejection_weight = clamp(Float64(accept_prob), 0.0, 1.0)
    return _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY +
        (rejection_weight - _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY) * progress
end

function _mass_adaptation_weights!(
    state::RunningVarianceState,
    destination::AbstractVector,
    accepted_step::AbstractVector,
    accept_prob::AbstractVector,
    divergent_step::AbstractVector,
)
    length(destination) == length(accepted_step) == length(accept_prob) == length(divergent_step) ||
        throw(DimensionMismatch("mass adaptation weights require matching vector lengths"))
    @inbounds for index in eachindex(destination, accepted_step, accept_prob, divergent_step)
        destination[index] = _mass_adaptation_weight(
            state,
            accepted_step[index],
            accept_prob[index],
            divergent_step[index],
        )
    end
    return destination
end

function _acceptance_probability(log_accept_ratio::Float64)
    if !isfinite(log_accept_ratio)
        return 0.0
    elseif log_accept_ratio >= 0
        return 1.0
    end
    return exp(log_accept_ratio)
end

function _proposal_log_accept_ratio(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
)
    isnothing(proposal) && return -Inf
    _, proposed_momentum, proposed_logjoint = proposal
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    return current_hamiltonian - proposed_hamiltonian
end

function _proposal_diagnostics(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
    divergence_threshold::Float64,
)
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    if isnothing(proposal)
        return current_hamiltonian, Inf, true
    end

    _, proposed_momentum, proposed_logjoint = proposal
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    energy_error = proposed_hamiltonian - current_hamiltonian
    divergent = !isfinite(energy_error) || abs(energy_error) > divergence_threshold
    return proposed_hamiltonian, energy_error, divergent
end

function _find_reasonable_batched_step_size(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_logjoint::AbstractVector,
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    target_accept = 0.5

    for _ in 0:20
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        _, proposal_momentum, proposed_logjoint, _, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)
        for chain_index in eachindex(current_logjoint)
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error) || abs(energy_error) > divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        mean_accept_prob = _mean_batched_adaptation_probability(accept_prob, divergent_step)
        direction = mean_accept_prob > target_accept ? 1.0 : -1.0
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        if (direction > 0 && mean_accept_prob <= target_accept) ||
           (direction < 0 && mean_accept_prob >= target_accept)
            break
        end
        reasonable_step_size = next_step_size
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _find_reasonable_step_size(
    model::TeaModel,
    position::Vector{Float64},
    current_logjoint::Float64,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    log_target = log(0.5)
    momentum = _sample_momentum(rng, inverse_mass_matrix)
    proposal = _leapfrog(
        model,
        position,
        momentum,
        gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        reasonable_step_size,
        1,
    )
    log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
    direction = log_accept_ratio > log_target ? 1.0 : -1.0

    for _ in 1:20
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        reasonable_step_size = next_step_size
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        if (direction > 0 && log_accept_ratio <= log_target) ||
           (direction < 0 && log_accept_ratio >= log_target)
            break
        end
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _chain_initial_params(initial_params, chain_index::Int, num_params::Int, num_chains::Int)
    if isnothing(initial_params)
        return nothing
    elseif initial_params isa AbstractMatrix
        size(initial_params) == (num_params, num_chains) ||
            throw(DimensionMismatch("expected initial_params matrix of size ($num_params, $num_chains), got $(size(initial_params))"))
        return collect(Float64, view(initial_params, :, chain_index))
    elseif initial_params isa AbstractVector && !isempty(initial_params) && first(initial_params) isa AbstractVector
        length(initial_params) == num_chains ||
            throw(DimensionMismatch("expected $num_chains initial parameter vectors, got $(length(initial_params))"))
        chain_params = initial_params[chain_index]
        length(chain_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters for chain $chain_index, got $(length(chain_params))"))
        return Float64[value for value in chain_params]
    elseif initial_params isa AbstractVector
        length(initial_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters, got $(length(initial_params))"))
        return Float64[value for value in initial_params]
    end

    throw(ArgumentError("unsupported initial_params container for multi-chain HMC"))
end

function hmc(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial HMC parameters produced a non-finite unconstrained logjoint"))
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints)

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    inverse_mass_matrix = ones(num_params)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
            rng,
        )
    end
    dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        accepted_step = false
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        accept_prob = _acceptance_probability(log_accept_ratio)
        proposal_energy, energy_error, divergent_step = _proposal_diagnostics(
            current_logjoint,
            momentum,
            proposal,
            inverse_mass_matrix,
            hmc_divergence_threshold,
        )
        sample_energy = isnothing(proposal) ? _hamiltonian(current_logjoint, momentum, inverse_mass_matrix) : proposal_energy
        if !isnothing(proposal)
            proposed_position, _, proposed_logjoint = proposal

            if log(rand(rng)) < min(0.0, log_accept_ratio)
                position = proposed_position
                current_logjoint = proposed_logjoint
                accepted_step = true
            end
        end

        if !accepted_step
            sample_energy = _hamiltonian(current_logjoint, momentum, inverse_mass_matrix)
        end

        if iteration <= num_warmup
            if adapt_step_size
                hmc_step_size = _update_step_size!(dual_state, accept_prob)
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _update_running_variance!(
                    variance_state,
                    position,
                    _mass_adaptation_weight(variance_state, accepted_step, accept_prob, divergent_step),
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        hmc_step_size = _find_reasonable_step_size(
                            model,
                            position,
                            current_logjoint,
                            gradient_cache,
                            inverse_mass_matrix,
                            args,
                            constraints,
                            hmc_step_size,
                            rng,
                        )
                        dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    hmc_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_prob
            energies[sample_index] = sample_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = accepted_step
            divergent[sample_index] = divergent_step
        end
    end

    return HMCChain(
        :hmc,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        hmc_step_size,
        1 ./ inverse_mass_matrix,
        num_leapfrog_steps,
        0,
        zeros(Int, num_samples),
        fill(num_leapfrog_steps, num_samples),
        hmc_target_accept,
        mass_adaptation_windows,
    )
end

function nuts(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_nuts_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        max_delta_energy,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial NUTS parameters produced a non-finite unconstrained logjoint"))
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints)
    current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial NUTS parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    tree_depths = zeros(Int, num_samples)
    integration_steps_per_sample = zeros(Int, num_samples)
    total_iterations = num_warmup + num_samples
    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_max_delta_energy = Float64(max_delta_energy)
    inverse_mass_matrix = ones(num_params)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        nuts_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            nuts_step_size,
            rng,
        )
    end
    dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        proposal, accept_stat, tree_depth, integration_steps_used, proposal_energy, energy_error, divergent_step, moved_step =
            _nuts_proposal(
                model,
                position,
                current_logjoint,
                current_gradient,
                gradient_cache,
                inverse_mass_matrix,
                args,
                constraints,
                nuts_step_size,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )

        if moved_step
            position = proposal.position
            current_logjoint = proposal.logjoint
            current_gradient = proposal.gradient
        end

        if iteration <= num_warmup
            if adapt_step_size
                nuts_step_size = _update_step_size!(dual_state, accept_stat)
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _update_running_variance!(
                    variance_state,
                    position,
                    _mass_adaptation_weight(variance_state, false, accept_stat, divergent_step),
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        nuts_step_size = _find_reasonable_step_size(
                            model,
                            position,
                            current_logjoint,
                            gradient_cache,
                            inverse_mass_matrix,
                            args,
                            constraints,
                            nuts_step_size,
                            rng,
                        )
                        dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    nuts_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_stat
            energies[sample_index] = proposal_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = moved_step
            divergent[sample_index] = divergent_step
            tree_depths[sample_index] = tree_depth
            integration_steps_per_sample[sample_index] = integration_steps_used
        end
    end

    return HMCChain(
        :nuts,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        nuts_step_size,
        1 ./ inverse_mass_matrix,
        0,
        max_tree_depth,
        tree_depths,
        integration_steps_per_sample,
        nuts_target_accept,
        mass_adaptation_windows,
    )
end

function hmc_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains)
    num_params = parametercount(parameterlayout(model))
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)

    for chain_index in 1:num_chains
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chains[chain_index] = hmc(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            divergence_threshold=divergence_threshold,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

function nuts_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    num_params = parametercount(parameterlayout(model))
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)

    for chain_index in 1:num_chains
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chains[chain_index] = nuts(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            max_delta_energy=max_delta_energy,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

function batched_hmc(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_batched_hmc_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
        args,
        constraints,
    )

    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    position = _initial_batched_hmc_positions(
        model,
        batch_args,
        batch_constraints,
        initial_params,
        rng,
        num_params,
        num_chains,
    )
    inverse_mass_matrix = ones(num_params)
    workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint,
        workspace.gradient_cache,
        position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_batched_step_size(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            hmc_divergence_threshold,
            rng,
        )
    end
    dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

        for chain_index in 1:num_chains
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error[chain_index] = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error[chain_index]) ||
                    abs(energy_error[chain_index]) > hmc_divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain_index in 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                copyto!(view(position, :, chain_index), view(proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        if iteration <= num_warmup
            if adapt_step_size
                hmc_step_size = _update_step_size!(
                    dual_state,
                    _mean_batched_adaptation_probability(accept_prob, divergent_step),
                )
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _mass_adaptation_weights!(
                    variance_state,
                    workspace.mass_adaptation_weights,
                    accepted_step,
                    accept_prob,
                    divergent_step,
                )
                _update_running_variance!(
                    variance_state,
                    position,
                    workspace.mass_adaptation_weights,
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        hmc_step_size = _find_reasonable_batched_step_size(
                            workspace,
                            model,
                            position,
                            current_logjoint,
                            current_gradient,
                            inverse_mass_matrix,
                            batch_args,
                            batch_constraints,
                            hmc_step_size,
                            hmc_divergence_threshold,
                            rng,
                        )
                        dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    hmc_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index in 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _transform_to_constrained!(
                    view(workspace.constrained_position, :, chain_index),
                    model,
                    view(position, :, chain_index),
                )
                copyto!(
                    view(constrained_samples, :, sample_index, chain_index),
                    view(workspace.constrained_position, :, chain_index),
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = accept_prob[chain_index]
                energies[sample_index, chain_index] =
                    accepted_step[chain_index] ? proposed_hamiltonian[chain_index] : current_hamiltonian[chain_index]
                energy_errors[sample_index, chain_index] = energy_error[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
            end
        end
    end

    mass_matrix = 1 ./ inverse_mass_matrix
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index in 1:num_chains
        chains[chain_index] = HMCChain(
            :hmc,
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(acceptance_stats[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            hmc_step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            fill(num_leapfrog_steps, num_samples),
            hmc_target_accept,
            copy(mass_adaptation_windows),
        )
    end

    return HMCChains(model, args, constraints, chains)
end
