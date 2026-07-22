# Ecosystem interop: export HMCChains into the draw-major array layout used by
# ArviZ / MCMCChains, plus zero-dependency dictionary and (via a package
# extension) native MCMCChains.Chains construction.

# Display names for each parameter in the requested space, matching the
# per-component convention that `summarize` uses (see
# `_summary_parameter_entries` in diagnostics.jl): scalar slots keep their
# binding name (e.g. "sigma"), vector slots get a bracketed component index
# (e.g. "v[1]").
# Signature-aware naming (#95 PR-6): sampler output is sized by the conditioning
# signature of the chain's constraints, so its display names must come from that
# same layout, not the syntactic default `parameterlayout(model)`.
function _export_parameter_names(model::TeaModel, constraints::ChoiceMap, space::Symbol)
    return _export_parameter_names(_conditioned_parameter_layout(model, constraints), space)
end

function _export_parameter_names(model::TeaModel, space::Symbol)
    return _export_parameter_names(parameterlayout(model), space)
end

function _export_parameter_names(layout::ParameterLayout, space::Symbol)
    entries = _summary_parameter_entries(layout, space)
    num_params = length(entries)
    names = Vector{String}(undef, num_params)
    for slot in layout.slots
        indices = space === :constrained ? parametervalueindices(slot) : parameterindices(slot)
        component_count = length(indices)
        for (offset, parameter_index) in enumerate(indices)
            names[parameter_index] = component_count == 1 ?
                                     String(slot.binding) : string(slot.binding, "[", offset, "]")
        end
    end
    return names
end

"""
    parameter_names(chains::HMCChains; space=:constrained) -> Vector{String}

Return per-parameter display names in the requested `space` (`:constrained` or
`:unconstrained`), ordered to match the third dimension of [`posterior_array`](@ref).
"""
function parameter_names(chains::HMCChains; space::Symbol=:constrained)
    return _export_parameter_names(chains.model, chains.constraints, space)
end

"""
    posterior_array(chains::HMCChains; space=:constrained) -> Array{Float64,3}

Return posterior draws shaped `(num_samples, num_chains, num_params)` following
the ArviZ / MCMCChains draw-major convention. `space` selects `:constrained`
(default) or `:unconstrained` samples.
"""
function posterior_array(chains::HMCChains; space::Symbol=:constrained)
    isempty(chains.chains) && throw(ArgumentError("posterior_array requires at least one chain"))
    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params, num_samples = size(first_samples)
    num_chains = length(chains.chains)
    result = Array{Float64,3}(undef, num_samples, num_chains, num_params)
    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        size(samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must share the same parameter dimension"))
        size(samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must share the same number of samples"))
        for p = 1:num_params, s = 1:num_samples
            result[s, chain_index, p] = samples[p, s]
        end
    end
    return result
end

"""
    to_arviz_dict(chains::HMCChains) -> Dict{String,Any}

Return an ArviZ-style nested dictionary with a `"posterior"` group (one
`(num_samples, num_chains)` matrix per constrained parameter) and a
`"sample_stats"` group (`"diverging"`, `"energy"`, `"tree_depth"`,
`"acceptance_rate"`, `"lp"`).
"""
function to_arviz_dict(chains::HMCChains)
    isempty(chains.chains) && throw(ArgumentError("to_arviz_dict requires at least one chain"))
    names = parameter_names(chains; space=:constrained)
    draws = posterior_array(chains; space=:constrained)
    num_samples, num_chains, num_params = size(draws)

    posterior = Dict{String,Any}()
    for p = 1:num_params
        posterior[names[p]] = Array{Float64,2}(draws[:, :, p])
    end

    diverging = Array{Bool,2}(undef, num_samples, num_chains)
    energy = Array{Float64,2}(undef, num_samples, num_chains)
    tree_depth = Array{Float64,2}(undef, num_samples, num_chains)
    acceptance_rate = Array{Float64,2}(undef, num_samples, num_chains)
    lp = Array{Float64,2}(undef, num_samples, num_chains)
    for (chain_index, chain) in enumerate(chains.chains)
        for s = 1:num_samples
            diverging[s, chain_index] = chain.divergent[s]
            energy[s, chain_index] = chain.energies[s]
            tree_depth[s, chain_index] = chain.tree_depths[s]
            acceptance_rate[s, chain_index] = chain.acceptance_stats[s]
            lp[s, chain_index] = chain.logjoint_values[s]
        end
    end

    sample_stats = Dict{String,Any}(
        "diverging" => diverging,
        "energy" => energy,
        "tree_depth" => tree_depth,
        "acceptance_rate" => acceptance_rate,
        "lp" => lp,
    )

    return Dict{String,Any}("posterior" => posterior, "sample_stats" => sample_stats)
end

# Canonical package-extension pattern: the core declares the function with no
# methods; the UncertainTeaMCMCChainsExt extension (loaded when MCMCChains.jl is
# present) adds the `::HMCChains` method. Calling without the extension loaded
# raises a MethodError, which is the intended "load MCMCChains.jl" signal.
"""
    to_mcmcchains(chains::HMCChains; space=:constrained)

Convert `chains` into an `MCMCChains.Chains` object. Requires the optional
`MCMCChains` dependency to be loaded (which activates the package extension).
"""
function to_mcmcchains end
