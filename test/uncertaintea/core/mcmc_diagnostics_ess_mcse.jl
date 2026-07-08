@testset "mcmc_diagnostics_ess_mcse" begin
    @tea static function diag_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    diag_constraints = choicemap((:y, 0.4f0))

    diag_chains = hmc_chains(
        diag_model,
        (),
        diag_constraints;
        num_chains=4,
        num_samples=300,
        num_warmup=300,
        rng=MersenneTwister(2024),
    )

    diag_summary = summarize(diag_chains)
    diag_param = diag_summary.parameters[1]

    # MCSE of the mean is exactly sd * sqrt(1 / ess_bulk).
    @test diag_param.mcse_mean ≈ diag_param.sd * sqrt(1 / diag_param.ess) atol = 1e-12
    @test isfinite(diag_param.mcse_mean)

    # Tail-ESS is positive and finite for a well-mixed parameter.
    @test diag_param.ess_tail > 0
    @test isfinite(diag_param.ess_tail)

    # One MCSE per requested quantile.
    @test length(diag_param.mcse_quantiles) == length(diag_summary.quantile_probs)
    @test all(value -> isfinite(value) && value >= 0, diag_param.mcse_quantiles)

    # Per-chain statistics are opt-in.
    @test diag_param.per_chain_means === nothing
    @test diag_param.per_chain_sds === nothing

    diag_summary_pc = summarize(diag_chains; per_chain=true)
    diag_param_pc = diag_summary_pc.parameters[1]
    @test length(diag_param_pc.per_chain_means) == nchains(diag_chains)
    @test length(diag_param_pc.per_chain_sds) == nchains(diag_chains)
    @test sum(diag_param_pc.per_chain_means) / length(diag_param_pc.per_chain_means) ≈ diag_param_pc.mean atol = 1e-8

    # E-BFMI formula: hand-computed on [1.0, 3.0, 2.0].
    # numerator = (3-1)^2 + (2-3)^2 = 5 ; denominator = (1-2)^2 + (3-2)^2 + (2-2)^2 = 2 ; ratio = 2.5
    @test UncertainTea._ebfmi([1.0, 3.0, 2.0]) ≈ 2.5 atol = 1e-12
    @test isnan(UncertainTea._ebfmi([1.0]))
    @test isnan(UncertainTea._ebfmi([5.0, 5.0, 5.0]))

    # check_diagnostics returns well-shaped fields for the clean run.
    diag_clean_warnings = check_diagnostics(diag_chains)
    @test length(diag_clean_warnings.num_divergent) == nchains(diag_chains)
    @test length(diag_clean_warnings.ebfmi) == nchains(diag_chains)
    @test length(diag_clean_warnings.treedepth_hits) == nchains(diag_chains)
    @test all(==(0), diag_clean_warnings.treedepth_hits)
    @test isempty(diag_clean_warnings.high_rhat_parameters)
    @test isempty(diag_clean_warnings.low_ess_parameters)
    @test has_warnings(diag_clean_warnings) == false

    # Warnings triggering: large step, no adaptation forces divergences.
    diag_div_chains = hmc_chains(
        diag_model,
        (),
        diag_constraints;
        num_chains=4,
        num_samples=200,
        num_warmup=0,
        step_size=16.0,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        rng=MersenneTwister(7),
    )
    diag_div_summary = summarize(diag_div_chains)
    @test any(count -> count > 0, diag_div_summary.warnings.num_divergent)
    @test has_warnings(diag_div_summary.warnings)

    # The text/plain show renders a warnings section for the divergent run.
    diag_div_show = sprint(show, MIME("text/plain"), diag_div_summary)
    @test occursin("warnings", diag_div_show)
end
