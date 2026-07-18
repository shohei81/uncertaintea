# MH-within-Gibbs for discrete latent sites (docs/mh-within-gibbs.md, issue
# #13 track 2): alternate symmetric single-site Metropolis-Hastings updates on
# discrete latents with NUTS transitions on the continuous block. Conditioning
# flows through ONE mutable ChoiceMap shared by reference with the NUTS target
# and gradient cache, so upserting a discrete value needs no rebuild.

const _GIBBS_DISCRETE_FAMILIES = (:bernoulli, :categorical, :poisson, :geometric, :negativebinomial, :binomial)

struct GibbsSite
    address::Any
    kind::Symbol      # :flip (bernoulli) | :uniform_other (categorical) | :walk (integer supports)
    lower::Int        # static lower support bound for :walk pre-rejection
    categories::Int   # K for :uniform_other
end

# Sites are matched against the INLINED execution plan's choice steps, not
# the raw spec: submodel calls are expanded there with prefixed addresses
# ((:sub, :z) for a child latent), and loop-scoped or dynamic address parts
# match as templates (literal parts must agree, dynamic parts match anything).
function _gibbs_collect_choice_steps!(collected::Vector{ChoicePlanStep}, steps)
    for step in steps
        if step isa ChoicePlanStep
            push!(collected, step)
        elseif step isa LoopPlanStep
            _gibbs_collect_choice_steps!(collected, step.body)
        end
    end
    return collected
end

function _gibbs_step_matches(step::ChoicePlanStep, address::Tuple; exact::Bool)
    parts = step.address.parts
    length(parts) == length(address) || return false
    for index in eachindex(address)
        part = parts[index]
        if part isa AddressLiteralPart
            part.value == address[index] || return false
        elseif exact
            return false
        end
    end
    return true
end

# Classification of a matched step for one site: whether it stays
# marginalized, the family, and the proposal-relevant support size.
function _gibbs_step_classification(step::ChoicePlanStep, address::Tuple)
    step.rhs isa DistributionSpec || throw(
        ArgumentError(
            "gibbs found the latent choice $address but could not match it to a distribution " *
            "choice in the model's execution plan",
        ),
    )
    step.rhs.marginalize === :enumerate && return (true, :none, 0)
    family = step.rhs.family
    categories = 0
    if family === :categorical
        probabilities = step.rhs.arguments[1]
        if probabilities isa Expr && probabilities.head == :vect
            categories = length(probabilities.args)
        end
    end
    return (false, family, categories)
end

# Classify one candidate latent address: `nothing` for marginalize=:enumerate
# sites (they stay marginalized inside the logjoint), a GibbsSite for discrete
# families, a loud error otherwise. An exact all-literal address match wins
# outright; otherwise every template match (dynamic parts as wildcards) must
# agree on the classification, because a wildcard template can shadow an
# unrelated concrete site.
function _gibbs_site(choice_steps::Vector{ChoicePlanStep}, address::Tuple)
    matched_step = nothing
    for step in choice_steps
        if _gibbs_step_matches(step, address; exact=true)
            matched_step = step
            break
        end
    end
    classification = if !isnothing(matched_step)
        _gibbs_step_classification(matched_step, address)
    else
        candidates = [step for step in choice_steps if _gibbs_step_matches(step, address; exact=false)]
        isempty(candidates) && throw(
            ArgumentError(
                "gibbs could not match the latent choice $address to any choice in the model's " *
                "execution plan (a dynamic address expression that normalizes to multiple parts, " *
                "e.g. a Pair-valued address argument, cannot be matched; use static or " *
                "single-part addresses for Gibbs sites)",
            ),
        )
        classifications = unique(_gibbs_step_classification(step, address) for step in candidates)
        length(classifications) == 1 || throw(
            ArgumentError(
                "gibbs matched the latent choice $address to multiple template choices with " *
                "conflicting classifications $classifications; disambiguate the addresses",
            ),
        )
        classifications[1]
    end

    marginalized, family, categories = classification
    marginalized && return nothing
    family in _GIBBS_DISCRETE_FAMILIES || throw(
        ArgumentError(
            "gibbs requires every non-observed slotless choice to be a finite- or integer-support " *
            "discrete latent; choice $address has family `$family` (continuous slotless latents " *
            "have no sampler)",
        ),
    )
    if family === :bernoulli
        return GibbsSite(address, :flip, 0, 0)
    elseif family === :categorical
        # a singleton support can never move: self-transition instead of
        # sampling from an empty proposal range
        categories == 1 && return GibbsSite(address, :fixed, 1, 1)
        categories > 1 && return GibbsSite(address, :uniform_other, 1, categories)
        # non-literal probability vector: the support size is unknown at this
        # point, so fall back to the +-1 walk (out-of-range proposals reject
        # through scoring)
        return GibbsSite(address, :walk, 1, 0)
    end
    return GibbsSite(address, :walk, 0, 0)
end

# Discrete Gibbs sites = prior-trace addresses minus continuous parameter
# slots minus observations minus marginalized sites (the SBC observation
# detection pattern, complemented).
function _gibbs_discrete_sites(model::TeaModel, trace::TeaTrace, constraints::ChoiceMap)
    latent_map = parameterchoicemap(model, parameter_vector(trace))
    choice_steps = _gibbs_collect_choice_steps!(ChoicePlanStep[], executionplan(model).steps)
    sites = GibbsSite[]
    for entry in trace.choices.entries
        address = first(entry)
        haskey(latent_map, address) && continue
        haskey(constraints, address) && continue
        site = _gibbs_site(choice_steps, address)
        isnothing(site) || push!(sites, site)
    end
    return sites
end

function _gibbs_propose(rng::AbstractRNG, site::GibbsSite, value)
    if site.kind === :flip
        current = value isa Bool ? value : value != 0
        return !current
    elseif site.kind === :uniform_other
        shifted = Int(value) + rand(rng, 1:(site.categories-1))
        return shifted > site.categories ? shifted - site.categories : shifted
    end
    return Int(value) + rand(rng, (-1, 1))
end

"""
    gibbs(model, args=(), constraints=choicemap(); num_samples, kwargs...)

MH-within-Gibbs for models with discrete latent sites (docs/mh-within-gibbs.md):
each iteration sweeps the discrete sites with symmetric single-site
Metropolis-Hastings updates, then advances the continuous block with one NUTS
transition conditioned on the current discrete values. Discrete sites are
discovered automatically (every non-observed choice without a parameter slot
that is not `marginalize=:enumerate`); a model without discrete sites reduces
to plain NUTS, and a model without continuous slots to pure single-site MH.

Returns a [`GibbsChain`](@ref) with the continuous samples and per-site
discrete samples. The NUTS keyword arguments mirror [`nuts`](@ref).
"""
function gibbs(
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
    metric::Symbol=:diag,
    rng::AbstractRNG=Random.default_rng(),
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    num_samples >= 1 || throw(ArgumentError("num_samples must be positive"))
    num_warmup >= 0 || throw(ArgumentError("num_warmup must be non-negative"))
    num_params = parametercount(parameterlayout(model))
    constrained_num_params = parametervaluecount(parameterlayout(model))
    has_continuous = num_params > 0

    # one prior trace seeds BOTH the discrete values and (absent
    # initial_params) the continuous position, so the initial state is a
    # coherent draw from the prior conditioned on the observations
    trace, _ = generate(model, args, constraints; rng=rng)
    sites = _gibbs_discrete_sites(model, trace, constraints)

    # the persistent conditioning map: observations plus the CURRENT discrete
    # values, mutated in place so the NUTS target/gradient cache (which hold
    # it by reference) always see the current conditioning
    merged = choicemap((first(entry), last(entry)) for entry in constraints.entries)
    for site in sites
        _pushchoice!(merged, site.address, trace.choices[site.address])
    end

    position = if isnothing(initial_params)
        has_continuous ? transform_to_unconstrained(trace) : Float64[]
    else
        _initial_hmc_position(model, args, merged, initial_params, rng)
    end
    current_logjoint = logjoint_unconstrained(model, position, args, merged)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial gibbs state produced a non-finite unconstrained logjoint"))

    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_max_delta_energy = Float64(max_delta_energy)
    if has_continuous
        gradient_cache = _logjoint_gradient_cache(model, position, args, merged)
        current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
        all(isfinite, current_gradient) ||
            throw(ArgumentError("initial gibbs state produced a non-finite unconstrained gradient"))
        inverse_mass_matrix = ones(num_params)
        if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
            nuts_step_size = _find_reasonable_step_size(
                model,
                position,
                current_logjoint,
                gradient_cache,
                inverse_mass_matrix,
                args,
                merged,
                nuts_step_size,
                rng,
            )
        end
        driver = WarmupDriver(
            num_params,
            num_warmup,
            nuts_step_size,
            nuts_target_accept;
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            metric=metric,
        )
        refind = ScalarStepSizeSearch(model, gradient_cache, args, merged, rng, position, current_logjoint)
        nuts_target = ModelDensityTarget(model, args, merged, gradient_cache)
    end

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, constrained_num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = zeros(num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    discrete_samples = Matrix{Int}(undef, length(sites), num_samples)
    discrete_accepted = zeros(Int, length(sites))

    total_iterations = num_warmup + num_samples
    sample_index = 0
    for iteration = 1:total_iterations
        # --- discrete pass: symmetric single-site MH ----------------------
        for (site_index, site) in enumerate(sites)
            site.kind === :fixed && continue
            current_value = merged[site.address]
            proposed = _gibbs_propose(rng, site, current_value)
            # static-lower-bound pre-rejection (the compiled scorer would
            # bind the invalid value and a suffix consumer may throw)
            site.kind === :walk && proposed < site.lower && continue
            _pushchoice!(merged, site.address, proposed)
            # a proposal the model cannot evaluate is a zero-density region
            # (dynamic support bounds); the CURRENT state's scoring is never
            # caught, so genuine model errors still surface loudly
            proposal_logjoint = try
                logjoint_unconstrained(model, position, args, merged)
            catch
                -Inf
            end
            if log(rand(rng)) < proposal_logjoint - current_logjoint
                current_logjoint = proposal_logjoint
                discrete_accepted[site_index] += 1
            else
                _pushchoice!(merged, site.address, current_value)
            end
        end

        # --- continuous pass: one NUTS transition -------------------------
        divergent_step = false
        moved_step = false
        accept_stat = 0.0
        if has_continuous
            # the discrete pass changed the conditioning under the unchanged
            # position, so the cached logjoint/gradient must be refreshed
            current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
            nuts_step_size = driver.step_size
            inverse_mass_matrix = _driver_metric(driver)
            proposal, accept_stat, _, _, _, _, divergent_step, moved_step = _nuts_proposal(
                nuts_target,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
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
                refind.position = position
                refind.current_logjoint = current_logjoint
                mass_weight = _mass_adaptation_weight(driver.variance_state, false, accept_stat, divergent_step)
                warmup_update!(driver, iteration, accept_stat, position, mass_weight, refind)
                iteration == num_warmup && warmup_finalize!(driver)
            end
        end

        if iteration > num_warmup
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] =
                has_continuous ? transform_to_constrained(model, position, args) : Float64[]
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_stat
            accepted[sample_index] = moved_step
            divergent[sample_index] = divergent_step
            for (site_index, site) in enumerate(sites)
                discrete_samples[site_index, sample_index] = Int(merged[site.address])
            end
        end
    end

    return GibbsChain(
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        accepted,
        divergent,
        has_continuous ? driver.step_size : nuts_step_size,
        has_continuous ? copy(driver.inverse_mass_matrix) : Float64[],
        Any[site.address for site in sites],
        discrete_samples,
        discrete_accepted ./ total_iterations,
    )
end
