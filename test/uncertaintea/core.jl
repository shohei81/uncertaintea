# Every file below is wrapped in its own named @testset and depends only on
# ../fixtures.jl, so any file can be run standalone (include fixtures.jl first)
# and the include order is free.
#
# UNCERTAINTEA_TEST_GROUP selects a CI shard (.github/workflows/ci.yml); the
# default "all" runs everything, so a plain local `Pkg.test()` is unchanged.
# Groups are balanced by measured wall-clock time; put a new file in the
# topically matching group and rebalance only if one group grows far beyond
# the others. sampling.jl belongs to the "inference" group (see runtests.jl).
include("fixtures.jl")

core_test_files = [
    # (file, group)
    ("dsl_static_model_semantics.jl", "dsl"),
    ("batched_logjoint_and_gradient.jl", "dsl"),
    ("hmc_and_nuts_workspace.jl", "dsl"),
    ("nuts_scheduler_and_backend.jl", "dsl"),
    ("tuple_and_loop_addresses.jl", "dsl"),
    ("custom_distribution_registration.jl", "dsl"),
    ("reparam_scaffolding.jl", "dsl"),
    ("discrete_enum_scaffolding.jl", "dsl"),
    ("dist_exponential_poisson.jl", "dist"),
    ("dist_gamma_studentt.jl", "dist"),
    ("dist_beta_categorical.jl", "dist"),
    ("dist_inversegamma_weibull_binomial.jl", "dist"),
    ("dist_laplace_geometric_negbinom.jl", "dist"),
    ("dist_dirichlet.jl", "dist"),
    ("dist_mvnormal_diag.jl", "dist"),
    ("dist_truncated.jl", "dist"),
    ("dist_mixture.jl", "dist"),
    ("dist_mvnormal_dense.jl", "dist"),
    ("dist_lkj_cholesky.jl", "dist"),
    ("dist_integer_params.jl", "dist"),
    ("dist_bernoulli.jl", "dist"),
    ("vector_backend_sampler.jl", "backend-device"),
    ("batched_scoring_eltype_f32.jl", "backend-device"),
    ("vectorized_obs_iid_latents.jl", "backend-device"),
    ("backend_native_families.jl", "backend-device"),
    ("device_lowering_parity.jl", "backend-device"),
    ("device_gradient_dual.jl", "backend-device"),
    ("device_hmc_advi.jl", "backend-device"),
    ("device_masked_nuts.jl", "backend-device"),
    ("gradient_crosscheck.jl", "crosscheck"),
    ("batched_advi_particle.jl", "inference"),
    ("tempered_batched_smc.jl", "inference"),
    ("proposal_diagnostics_overflow.jl", "inference"),
    ("integrator_nuts_proposal.jl", "inference"),
    ("nuts_uturn_turning.jl", "inference"),
    ("nuts_fixed_step_moments.jl", "inference"),
    ("warmup_driver_regression.jl", "inference"),
    ("mcmc_diagnostics_ess_mcse.jl", "inference"),
    ("predictive_sampling_smc_resampling.jl", "inference"),
    ("waic_psis_loo.jl", "inference"),
    ("map_laplace_approximation.jl", "inference"),
    ("per_chain_warmup_batched.jl", "inference"),
    ("dense_mass_matrix_single_chain.jl", "inference"),
    ("masked_batched_nuts.jl", "inference"),
    ("sbc_calibration.jl", "inference"),
    ("advi_structured_guides.jl", "inference"),
    ("pathfinder_init.jl", "inference"),
    ("reparam_noncentered_cpu.jl", "inference"),
    ("discrete_enum_cpu.jl", "inference"),
    ("mh_within_gibbs.jl", "inference"),
]

let registered = Set(first.(core_test_files)), on_disk = Set(f for f in readdir(joinpath(@__DIR__, "core")) if endswith(f, ".jl"))

    unregistered = sort!(collect(setdiff(on_disk, registered)))
    isempty(unregistered) ||
        error("Test files not registered in core_test_files (so they would never run): $unregistered")
    missing_files = sort!(collect(setdiff(registered, on_disk)))
    isempty(missing_files) ||
        error("core_test_files entries with no file on disk: $missing_files")
end

test_group = get(ENV, "UNCERTAINTEA_TEST_GROUP", "all")
known_test_groups = ("all", "dsl", "dist", "backend-device", "inference", "crosscheck")
test_group in known_test_groups ||
    error("Unknown UNCERTAINTEA_TEST_GROUP=\"$test_group\"; expected one of $known_test_groups")

for (file, group) in core_test_files
    if test_group == "all" || test_group == group
        include("core/$file")
    end
end
