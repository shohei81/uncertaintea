# Issue #142: allocation-free batched leapfrog. The per-chain broadcast-slice
# updates in src/inference/integrator.jl (e.g. `p[:, c] .+= h .* g[:, c]`) were
# replaced with explicit @inbounds loops that keep the SAME arithmetic order,
# so seeded batched samplers must stay bitwise identical. The checksums below
# were snapshotted from the pre-#142 broadcast implementation (this branch,
# before editing src/inference/integrator.jl / nuts/tree_dynamics.jl) — any
# drift means the rewrite changed arithmetic, not just allocations.
#
# Regenerate (against a known-good tree) with:
#   UNCERTAINTEA_AFLF_PRINT=1 julia --project=. -e '
#     using Test, Random, UncertainTea;
#     include("test/uncertaintea/fixtures.jl");
#     include("test/uncertaintea/core/allocfree_leapfrog_bitwise.jl")'

# Gauss shape from the cross-PPL benchmark (bench/crossppl/julia/models.jl):
# 2 latents, `n` loop-addressed observations.
@tea static function aflf_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# Eight schools, noncentered (Rubin 1981) with a log-scale tau prior (as in
# the ncc_* fixtures) plus a small floor: a positive-support tau (benchmark
# truncatedstudentt, or bare exp) can underflow to exactly 0.0 on extreme
# warmup proposals and batched scoring then throws ("normal requires
# sigma > 0", pre-existing on main) — that robustness issue is orthogonal to
# the integrator arithmetic this file pins down.
@tea static function aflf_eight_schools_noncentered(sigma)
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.5)
    tau = 0.001 + exp(log_tau)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

# Deterministic gauss data: 1000 observations from a seeded stream.
const aflf_gauss_n = 1000
aflf_gauss_ys = let rng = MersenneTwister(20260723)
    0.7 .+ 1.3 .* randn(rng, aflf_gauss_n)
end
aflf_gauss_constraints = choicemap(((:y => i, aflf_gauss_ys[i]) for i = 1:aflf_gauss_n)...)

aflf_schools_y = Float64[28, 8, -3, 7, -1, 1, 18, 12]
aflf_schools_sigma = Float64[15, 10, 16, 11, 9, 11, 10, 18]
aflf_schools_constraints = choicemap(((:y => i, aflf_schools_y[i]) for i = 1:8)...)

# Order-sensitive, bitwise-sensitive fold over the raw Float64 bit patterns.
function aflf_fold(h::UInt64, values::AbstractArray{Float64})
    for x in values
        h = (h ⊻ reinterpret(UInt64, x)) * 0x2545f4914f6cdd1d
    end
    return h
end

function aflf_checksum(result)
    h = 0x9e3779b97f4a7c15
    for chain in result.chains
        h = aflf_fold(h, chain.unconstrained_samples)
        h = aflf_fold(h, chain.logjoint_values)
    end
    return h
end

function aflf_nuts(model, args, cons, seed; num_chains, tree_strategy, per_chain_adaptation=false)
    return batched_nuts(
        model,
        args,
        cons;
        num_chains,
        num_samples=10,
        num_warmup=15,
        max_tree_depth=6,
        tree_strategy,
        per_chain_adaptation,
        rng=MersenneTwister(seed),
    )
end

function aflf_hmc(model, args, cons, seed; num_chains, per_chain_adaptation=false)
    return batched_hmc(
        model,
        args,
        cons;
        num_chains,
        num_samples=10,
        num_warmup=15,
        num_leapfrog_steps=8,
        per_chain_adaptation,
        rng=MersenneTwister(seed),
    )
end

aflf_gauss_args = (aflf_gauss_n,)
aflf_schools_args = (aflf_schools_sigma,)

# name => runner. NUTS covers batched_leapfrog_step_to! (shared + per-chain
# overloads, both tree strategies); HMC covers batched_leapfrog_trajectory!
# (shared + per-chain overloads).
aflf_configs = [
    (
        "gauss_hybrid_4c",
        () -> aflf_nuts(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 142; num_chains=4, tree_strategy=:hybrid),
    ),
    (
        "gauss_hybrid_64c",
        () -> aflf_nuts(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 143; num_chains=64, tree_strategy=:hybrid),
    ),
    (
        "gauss_masked_4c",
        () -> aflf_nuts(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 144; num_chains=4, tree_strategy=:masked),
    ),
    (
        "gauss_masked_64c",
        () -> aflf_nuts(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 145; num_chains=64, tree_strategy=:masked),
    ),
    (
        "schools_hybrid_4c",
        () -> aflf_nuts(
            aflf_eight_schools_noncentered,
            aflf_schools_args,
            aflf_schools_constraints,
            146;
            num_chains=4,
            tree_strategy=:hybrid,
        ),
    ),
    (
        "schools_hybrid_64c",
        () -> aflf_nuts(
            aflf_eight_schools_noncentered,
            aflf_schools_args,
            aflf_schools_constraints,
            147;
            num_chains=64,
            tree_strategy=:hybrid,
        ),
    ),
    (
        "schools_masked_4c",
        () -> aflf_nuts(
            aflf_eight_schools_noncentered,
            aflf_schools_args,
            aflf_schools_constraints,
            148;
            num_chains=4,
            tree_strategy=:masked,
        ),
    ),
    (
        "schools_masked_64c",
        () -> aflf_nuts(
            aflf_eight_schools_noncentered,
            aflf_schools_args,
            aflf_schools_constraints,
            149;
            num_chains=64,
            tree_strategy=:masked,
        ),
    ),
    (
        "gauss_hybrid_4c_perchain",
        () -> aflf_nuts(
            aflf_gauss,
            aflf_gauss_args,
            aflf_gauss_constraints,
            150;
            num_chains=4,
            tree_strategy=:hybrid,
            per_chain_adaptation=true,
        ),
    ),
    (
        "schools_masked_4c_perchain",
        () -> aflf_nuts(
            aflf_eight_schools_noncentered,
            aflf_schools_args,
            aflf_schools_constraints,
            151;
            num_chains=4,
            tree_strategy=:masked,
            per_chain_adaptation=true,
        ),
    ),
    ("gauss_hmc_4c", () -> aflf_hmc(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 152; num_chains=4)),
    (
        "gauss_hmc_4c_perchain",
        () -> aflf_hmc(aflf_gauss, aflf_gauss_args, aflf_gauss_constraints, 153; num_chains=4, per_chain_adaptation=true),
    ),
]

# Snapshotted from the pre-#142 broadcast-slice implementation (see header).
aflf_expected = Dict{String,UInt64}(
    "gauss_hybrid_4c" => 0x45c390abfff09ce6,
    "gauss_hybrid_64c" => 0x8b5eaf8de84609bd,
    "gauss_masked_4c" => 0xa983e12c2e3e9d2b,
    "gauss_masked_64c" => 0x0275cbe52b1d40fa,
    "schools_hybrid_4c" => 0x6f3d0c1cf5ad64ca,
    "schools_hybrid_64c" => 0x93d6648989e57209,
    "schools_masked_4c" => 0x7d0bbdf54c395561,
    "schools_masked_64c" => 0xa4fa8770b2e28ff9,
    "gauss_hybrid_4c_perchain" => 0xe5715dabe97bda12,
    "schools_masked_4c_perchain" => 0x474e9cd5c1c76e4b,
    "gauss_hmc_4c" => 0x267d725888dd6f31,
    "gauss_hmc_4c_perchain" => 0xc5ba15e4fc4f241a,
)

if get(ENV, "UNCERTAINTEA_AFLF_PRINT", "") == "1"
    println("aflf_expected = Dict{String,UInt64}(")
    for (name, runner) in aflf_configs
        println("    \"", name, "\" => 0x", string(aflf_checksum(runner()), base=16, pad=16), ",")
    end
    println(")")
else
    @testset "allocfree_leapfrog_bitwise" begin
        for (name, runner) in aflf_configs
            @test aflf_checksum(runner()) == aflf_expected[name]
        end
    end
end

# --- @allocated regression: the integrator bookkeeping itself must not
# allocate. A trivial standard-normal batched target isolates the integrator
# from model-gradient allocations.
struct AflfQuadraticTarget <: UncertainTea.AbstractBatchedDensityTarget end

function UncertainTea.batched_target_gradient!(
    gradient_destination::AbstractMatrix,
    ::AflfQuadraticTarget,
    positions::AbstractMatrix,
)
    @inbounds for chain in axes(positions, 2), row in axes(positions, 1)
        gradient_destination[row, chain] = -positions[row, chain]
    end
    return gradient_destination
end

function UncertainTea.batched_target_logdensity_and_gradient!(
    values_destination::AbstractVector,
    gradient_destination::AbstractMatrix,
    ::AflfQuadraticTarget,
    positions::AbstractMatrix,
)
    @inbounds for chain in axes(positions, 2)
        acc = 0.0
        for row in axes(positions, 1)
            acc += positions[row, chain]^2
            gradient_destination[row, chain] = -positions[row, chain]
        end
        values_destination[chain] = -acc / 2
    end
    return values_destination, gradient_destination
end

function aflf_step_workspace(num_params, num_chains)
    rng = MersenneTwister(7)
    position = randn(rng, num_params, num_chains)
    momentum = randn(rng, num_params, num_chains)
    gradient = -position
    return (
        destination_position=Matrix{Float64}(undef, num_params, num_chains),
        destination_momentum=Matrix{Float64}(undef, num_params, num_chains),
        destination_gradient=Matrix{Float64}(undef, num_params, num_chains),
        destination_logjoint=Vector{Float64}(undef, num_chains),
        valid=Vector{Bool}(undef, num_chains),
        position=position,
        momentum=momentum,
        gradient=gradient,
        target=AflfQuadraticTarget(),
        inverse_mass_matrix=fill(0.8, num_params),
        inverse_mass_matrices=fill(0.8, num_params, num_chains),
        step_size=0.05,
        step_sizes=fill(0.05, num_chains),
        direction=[isodd(chain) ? 1 : -1 for chain = 1:num_chains],
        active=trues(num_chains),
    )
end

function aflf_step_to_allocs(w)
    return @allocated UncertainTea.batched_leapfrog_step_to!(
        w.destination_position, w.destination_momentum, w.destination_gradient,
        w.destination_logjoint, w.valid, w.position, w.momentum, w.gradient,
        w.target, w.inverse_mass_matrix, w.step_size, w.direction, w.active,
    )
end

function aflf_step_to_perchain_allocs(w)
    return @allocated UncertainTea.batched_leapfrog_step_to!(
        w.destination_position, w.destination_momentum, w.destination_gradient,
        w.destination_logjoint, w.valid, w.position, w.momentum, w.gradient,
        w.target, w.inverse_mass_matrices, w.step_sizes, w.direction, w.active,
    )
end

function aflf_trajectory_allocs(w)
    return @allocated UncertainTea.batched_leapfrog_trajectory!(
        w.destination_position, w.destination_momentum, w.destination_gradient,
        w.destination_logjoint, w.valid, w.position, w.momentum, w.gradient,
        w.target, w.inverse_mass_matrix, w.step_size, 4,
    )
end

function aflf_trajectory_perchain_allocs(w)
    return @allocated UncertainTea.batched_leapfrog_trajectory!(
        w.destination_position, w.destination_momentum, w.destination_gradient,
        w.destination_logjoint, w.valid, w.position, w.momentum, w.gradient,
        w.target, w.inverse_mass_matrices, w.step_sizes, 4,
    )
end

if get(ENV, "UNCERTAINTEA_AFLF_PRINT", "") != "1"
    @testset "allocfree_leapfrog_allocations" begin
        w = aflf_step_workspace(2, 32)
        for measure in
            (aflf_step_to_allocs, aflf_step_to_perchain_allocs, aflf_trajectory_allocs, aflf_trajectory_perchain_allocs)
            measure(w) # warm up compilation
            @test measure(w) == 0
        end
    end
end
