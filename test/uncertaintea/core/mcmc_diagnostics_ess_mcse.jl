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

    # Synthetic chains with hand-picked draws/energies, reusing diag_model's
    # single-parameter layout (issue #102/#103/#107 regressions).
    function diag_make_chain(samples::Matrix{Float64}, energies::Vector{Float64})
        n = size(samples, 2)
        return UncertainTea.HMCChain(
            :nuts,
            diag_model,
            (),
            diag_constraints,
            samples,
            samples,
            zeros(n),
            ones(n),
            energies,
            zeros(n),
            trues(n),
            falses(n),
            0.1,
            ones(size(samples, 1)),
            0,
            10,
            zeros(Int, n),
            zeros(Int, n),
            0.8,
            UncertainTea.HMCMassAdaptationWindowSummary[],
            nothing,
        )
    end
    function diag_make_chains(sample_mats; energies=nothing)
        chain_energies = energies === nothing ? [zeros(size(mat, 2)) for mat in sample_mats] : energies
        return UncertainTea.HMCChains(
            diag_model,
            (),
            diag_constraints,
            [diag_make_chain(mat, chain_energies[i]) for (i, mat) in enumerate(sample_mats)],
        )
    end

    # mcse_quantiles magnitude oracle (issue #102): for 4x4000 iid N(0,1)
    # draws the MCSE of a quantile is sqrt(p(1-p)/N) / phi(z_p) — one standard
    # error, not a 95% CI half-width (which would be ~1.96x larger).
    diag_iid_rng = MersenneTwister(20260719)
    diag_iid_chains = diag_make_chains([reshape(randn(diag_iid_rng, 4000), 1, 4000) for _ = 1:4])
    diag_iid_param = summarize(diag_iid_chains).parameters[1]
    diag_phi(z) = exp(-z^2 / 2) / sqrt(2pi)
    diag_mcse_median_analytic = 1 / (2 * diag_phi(0.0) * sqrt(16000))    # ~0.00991
    diag_mcse_q05_analytic = sqrt(0.05 * 0.95 / 16000) / diag_phi(-1.6448536269514722)  # ~0.0167
    @test 0.6 * diag_mcse_median_analytic < diag_iid_param.mcse_quantiles[2] < 1.5 * diag_mcse_median_analytic
    @test 0.6 * diag_mcse_q05_analytic < diag_iid_param.mcse_quantiles[1] < 1.5 * diag_mcse_q05_analytic

    # Constant (stuck) chains report NaN ESS/Rhat and warn (issue #103),
    # following the Stan/posterior convention instead of ESS=N, Rhat=1.
    for stuck_value in (0.0, 1.0)
        diag_stuck_chains = diag_make_chains([fill(stuck_value, 1, 1000) for _ = 1:4])
        @test all(isnan, ess(diag_stuck_chains))
        @test all(isnan, rhat(diag_stuck_chains))
        diag_stuck_warnings = check_diagnostics(diag_stuck_chains)
        @test has_warnings(diag_stuck_warnings)
        @test !isempty(diag_stuck_warnings.low_ess_parameters)
        @test !isempty(diag_stuck_warnings.high_rhat_parameters)
        # summarize and its show must tolerate the NaN diagnostics.
        diag_stuck_summary = summarize(diag_stuck_chains)
        @test isnan(diag_stuck_summary.parameters[1].mcse_mean)
        @test all(isnan, diag_stuck_summary.parameters[1].mcse_quantiles)
        @test occursin("warnings", sprint(show, MIME("text/plain"), diag_stuck_summary))
    end

    # ebfmi_threshold kwarg parameterizes the E-BFMI warning (issue #107).
    # Step energies (one jump of 10 in 1000 samples) give E-BFMI = 0.004.
    diag_ebfmi_rng = MersenneTwister(7)
    diag_step_energies = vcat(zeros(500), fill(10.0, 500))
    diag_ebfmi_chains = diag_make_chains(
        [reshape(randn(diag_ebfmi_rng, 1000), 1, 1000) for _ = 1:4];
        energies=[copy(diag_step_energies) for _ = 1:4],
    )
    diag_ebfmi_default = check_diagnostics(diag_ebfmi_chains)
    @test all(value -> isapprox(value, 0.004; atol=1e-12), diag_ebfmi_default.ebfmi)
    @test diag_ebfmi_default.ebfmi_threshold == 0.3
    @test has_warnings(diag_ebfmi_default)
    diag_ebfmi_loose = check_diagnostics(diag_ebfmi_chains; ebfmi_threshold=0.001)
    @test diag_ebfmi_loose.ebfmi_threshold == 0.001
    @test !has_warnings(diag_ebfmi_loose)
    diag_ebfmi_strict = check_diagnostics(diag_ebfmi_chains; ebfmi_threshold=0.99)
    @test has_warnings(diag_ebfmi_strict)
    # The rendered warning section reports the threshold actually used.
    diag_strict_show = sprint(UncertainTea._show_sampler_warnings, diag_ebfmi_strict)
    @test occursin("low E-BFMI (< 0.99)", diag_strict_show)
end
