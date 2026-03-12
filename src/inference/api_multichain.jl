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

