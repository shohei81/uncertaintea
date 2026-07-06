    # PR 39: per-chain warmup adaptation for the batched samplers.
    # Contract: batched_hmc / batched_nuts accept per_chain_adaptation (default
    # false). When false the run is bitwise identical to the shared-driver path;
    # when true each chain owns a WarmupDriver, adapting its own step size and
    # diagonal inverse mass matrix. The batched integrators / momentum sampling /
    # Hamiltonians gain per-chain overloads (step-size vector + inverse-mass
    # matrix, one column per chain). The progress callback reports the mean of the
    # per-chain step sizes.

    @tea static function pca_gaussian_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    pca_gaussian_constraints = choicemap((:y, 0.3f0))

    # Fixed-seed regression: per_chain_adaptation=false reproduces the shared
    # driver's adapted step size / mass matrix exactly (values captured from the
    # code prior to this change; these match the part33 shared-driver regression).
    pca_reg = batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=30,
        per_chain_adaptation=false,
        rng=MersenneTwister(404),
    )
    for pca_reg_chain in pca_reg.chains
        # Re-pinned after the batched-NUTS merge-cohort stale-select fix
        # (PR 6.4); still matches the part33 shared-driver regression.
        @test pca_reg_chain.step_size ≈ 1.2168785742992647 atol = 1e-12
        @test length(pca_reg_chain.mass_matrix) == 1
        @test pca_reg_chain.mass_matrix[1] ≈ 0.5636619744202114 atol = 1e-12
    end

    # Per-chain divergence: two chains facing different observation scales via a
    # per-chain args vector. sigma=0.1 yields a tight posterior on mu; sigma=10.0
    # yields a wide one. _validate_batched_args accepts a vector of tuples.
    @tea static function pca_scale_model(sigma_arg)
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, sigma_arg)
        return mu
    end

    pca_scale_constraints = choicemap((:y, 0.0f0))
    pca_scale_args = [(0.1f0,), (10.0f0,)]

    pca_scale = batched_nuts(
        pca_scale_model,
        pca_scale_args,
        pca_scale_constraints;
        num_chains=2,
        num_samples=50,
        num_warmup=150,
        per_chain_adaptation=true,
        rng=MersenneTwister(2024),
    )
    pca_step_tight = pca_scale.chains[1].step_size
    pca_step_wide = pca_scale.chains[2].step_size
    pca_mass_tight = pca_scale.chains[1].mass_matrix[1]
    pca_mass_wide = pca_scale.chains[2].mass_matrix[1]
    # Adapted step sizes must differ by at least 2x across the two scales.
    @test max(pca_step_tight, pca_step_wide) / min(pca_step_tight, pca_step_wide) >= 2.0
    # The wider posterior (chain 2, sigma=10) must carry the larger mass entry.
    @test pca_mass_wide > pca_mass_tight

    # Statistical sanity: per-chain mode on the shared gaussian recovers the
    # posterior mean (0.15) and mixes (rhat < 1.2) over 2 chains x 200 draws.
    pca_stat = batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=2,
        num_samples=200,
        num_warmup=200,
        per_chain_adaptation=true,
        rng=MersenneTwister(20260705),
    )
    pca_stat_summary = summarize(pca_stat)
    @test pca_stat_summary[1].mean ≈ 0.15 atol = 0.1
    @test pca_stat_summary[1].rhat < 1.2

    # batched_hmc per-chain smoke: adapts distinct per-chain step sizes too.
    pca_hmc = batched_hmc(
        pca_scale_model,
        pca_scale_args,
        pca_scale_constraints;
        num_chains=2,
        num_samples=50,
        num_warmup=150,
        per_chain_adaptation=true,
        rng=MersenneTwister(99),
    )
    @test pca_hmc.chains[1].step_size != pca_hmc.chains[2].step_size
    @test pca_hmc.chains[2].mass_matrix[1] > pca_hmc.chains[1].mass_matrix[1]

    # Callback still fires in per-chain mode with the documented phase fields; the
    # reported step_size is the mean of the per-chain step sizes.
    pca_cb_events = NamedTuple[]
    batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=2,
        num_samples=20,
        num_warmup=20,
        per_chain_adaptation=true,
        callback=info -> push!(pca_cb_events, info),
        callback_every=10,
        rng=MersenneTwister(313),
    )
    @test !isempty(pca_cb_events)
    @test all(e -> e.phase === :warmup || e.phase === :sample, pca_cb_events)
    @test all(e -> haskey(e, :step_size) && haskey(e, :divergences), pca_cb_events)
    @test all(e -> isfinite(e.step_size), pca_cb_events)
