# Issue #142: allocation-free batched leapfrog. The per-chain broadcast-slice
# updates in src/inference/integrator.jl (e.g. `p[:, c] .+= h .* g[:, c]`) were
# replaced with explicit loops that keep the SAME elementwise arithmetic, so
# integration must stay bitwise identical. This file pins that down with a
# same-process A/B seam: reference copies of the ORIGINAL broadcast-slice
# integrators live below, both implementations run on identical inputs, and
# every output is compared bit-for-bit. (An earlier revision hardcoded golden
# checksums of seeded batched_nuts draws snapshotted on macOS/Julia 1.12; they
# do not transfer across platforms/Julia versions — Linux/Julia 1.10 CI
# produced different floating-point streams — so cross-implementation equality
# is asserted within one process instead, which is platform-independent by
# construction.)
#
# Also covered: the per-iteration buffer reuse in
# _initialize_batched_nuts_continuations! (sqrt inverse-mass + all-chains
# mask), seeded determinism of batched_nuts with the new workspace buffers,
# and @allocated == 0 regressions for all four batched leapfrog entry points.

# Eight schools, noncentered (Rubin 1981) with a log-scale tau prior (as in
# the ncc_* fixtures) plus a small floor: a positive-support tau can underflow
# to exactly 0.0 on extreme warmup proposals and batched scoring then throws
# ("normal requires sigma > 0", pre-existing robustness issue) — orthogonal to
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

aflf_schools_y = Float64[28, 8, -3, 7, -1, 1, 18, 12]
aflf_schools_sigma = Float64[15, 10, 16, 11, 9, 11, 10, 18]
aflf_schools_constraints = choicemap(((:y => i, aflf_schools_y[i]) for i = 1:8)...)

# --- Trivial standard-normal batched target: deterministic, allocation-free,
# and shared by both implementations so gradients are identical by identity.
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

# --- Reference implementations: the pre-#142 broadcast-slice integrator
# bodies, copied verbatim from src/inference/integrator.jl at fe22bde
# (dimension guards dropped; every arithmetic statement kept literally).

function aflf_reference_trajectory!(
    q, p, destination_gradient, destination_logjoint, valid,
    position, momentum, current_gradient,
    target, inverse_mass_matrix::AbstractVector, step_size::Float64, num_steps::Int,
)
    num_chains = size(q, 2)
    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index = 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step = 1:num_steps
        for chain_index = 1:num_chains
            valid[chain_index] || continue
            q[:, chain_index] .+= step_size .* (inverse_mass_matrix .* p[:, chain_index])
        end

        if leapfrog_step < num_steps
            gradient = UncertainTea.batched_target_gradient!(destination_gradient, target, q)
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, _ = UncertainTea.batched_target_logdensity_and_gradient!(
                destination_logjoint, destination_gradient, target, q,
            )
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(destination_gradient, :, chain_index)) ||
                   !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index = 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* destination_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

function aflf_reference_trajectory!(
    q, p, destination_gradient, destination_logjoint, valid,
    position, momentum, current_gradient,
    target, inverse_mass_matrices::AbstractMatrix, step_sizes::AbstractVector, num_steps::Int,
)
    num_chains = size(q, 2)
    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index = 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            step_size = step_sizes[chain_index]
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step = 1:num_steps
        for chain_index = 1:num_chains
            valid[chain_index] || continue
            step_size = step_sizes[chain_index]
            q[:, chain_index] .+= step_size .* (view(inverse_mass_matrices, :, chain_index) .* p[:, chain_index])
        end

        if leapfrog_step < num_steps
            gradient = UncertainTea.batched_target_gradient!(destination_gradient, target, q)
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    step_size = step_sizes[chain_index]
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, _ = UncertainTea.batched_target_logdensity_and_gradient!(
                destination_logjoint, destination_gradient, target, q,
            )
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(destination_gradient, :, chain_index)) ||
                   !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index = 1:num_chains
        valid[chain_index] || continue
        step_size = step_sizes[chain_index]
        p[:, chain_index] .+= (step_size / 2) .* destination_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

function aflf_reference_step_to!(
    q, p, destination_gradient, destination_logjoint, valid,
    position, momentum, gradient,
    target, inverse_mass_matrix::AbstractVector, step_size::Float64, direction, active,
)
    num_chains = size(position, 2)
    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    for chain_index = 1:num_chains
        active[chain_index] || continue
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_size
        p[:, chain_index] .+= (signed_step / 2) .* gradient[:, chain_index]
        q[:, chain_index] .+= signed_step .* (inverse_mass_matrix .* p[:, chain_index])
    end

    proposed_logjoint, _ = UncertainTea.batched_target_logdensity_and_gradient!(
        destination_logjoint, destination_gradient, target, q,
    )
    for chain_index = 1:num_chains
        valid[chain_index] || continue
        if !isfinite(proposed_logjoint[chain_index]) ||
           !all(isfinite, view(destination_gradient, :, chain_index))
            valid[chain_index] = false
        else
            signed_step = direction[chain_index] * step_size
            p[:, chain_index] .+= (signed_step / 2) .* destination_gradient[:, chain_index]
        end
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

function aflf_reference_step_to!(
    q, p, destination_gradient, destination_logjoint, valid,
    position, momentum, gradient,
    target, inverse_mass_matrices::AbstractMatrix, step_sizes::AbstractVector, direction, active,
)
    num_chains = size(position, 2)
    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    for chain_index = 1:num_chains
        active[chain_index] || continue
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_sizes[chain_index]
        p[:, chain_index] .+= (signed_step / 2) .* gradient[:, chain_index]
        q[:, chain_index] .+= signed_step .* (view(inverse_mass_matrices, :, chain_index) .* p[:, chain_index])
    end

    proposed_logjoint, _ = UncertainTea.batched_target_logdensity_and_gradient!(
        destination_logjoint, destination_gradient, target, q,
    )
    for chain_index = 1:num_chains
        valid[chain_index] || continue
        if !isfinite(proposed_logjoint[chain_index]) ||
           !all(isfinite, view(destination_gradient, :, chain_index))
            valid[chain_index] = false
        else
            signed_step = direction[chain_index] * step_sizes[chain_index]
            p[:, chain_index] .+= (signed_step / 2) .* destination_gradient[:, chain_index]
        end
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

# --- A/B harness -----------------------------------------------------------

# Bit-level equality: distinguishes 0.0 from -0.0 and compares NaN payloads,
# which `==` on Float64 arrays does not.
aflf_same_bits(a::AbstractArray{Float64}, b::AbstractArray{Float64}) =
    size(a) == size(b) && reinterpret(UInt64, vec(Array(a))) == reinterpret(UInt64, vec(Array(b)))

struct AflfBuffers
    q::Matrix{Float64}
    p::Matrix{Float64}
    gradient::Matrix{Float64}
    logjoint::Vector{Float64}
    valid::Vector{Bool}
end

AflfBuffers(num_params, num_chains) = AflfBuffers(
    fill(NaN, num_params, num_chains),
    fill(NaN, num_params, num_chains),
    fill(NaN, num_params, num_chains),
    fill(NaN, num_chains),
    fill(false, num_chains),
)

function aflf_scenario_inputs(seed, num_params, num_chains; nonfinite_columns=Int[])
    rng = MersenneTwister(seed)
    position = randn(rng, num_params, num_chains)
    momentum = randn(rng, num_params, num_chains)
    gradient = randn(rng, num_params, num_chains)
    for (offset, chain) in enumerate(nonfinite_columns)
        gradient[1+offset%num_params, chain] = isodd(offset) ? NaN : -Inf
    end
    inverse_mass_matrix = 0.5 .+ rand(rng, num_params)
    inverse_mass_matrices = 0.5 .+ rand(rng, num_params, num_chains)
    step_sizes = 0.01 .+ 0.2 .* rand(rng, num_chains)
    direction = [rand(rng, Bool) ? 1 : -1 for _ = 1:num_chains]
    active = [rand(rng) < 0.8 for _ = 1:num_chains]
    any(active) || (active[1] = true)
    return (; position, momentum, gradient, inverse_mass_matrix, inverse_mass_matrices, step_sizes, direction, active)
end

function aflf_compare(production::AflfBuffers, reference::AflfBuffers)
    @test aflf_same_bits(production.q, reference.q)
    @test aflf_same_bits(production.p, reference.p)
    @test aflf_same_bits(production.gradient, reference.gradient)
    @test aflf_same_bits(production.logjoint, reference.logjoint)
    @test production.valid == reference.valid
end

@testset "allocfree_leapfrog_ab_bitwise" begin
    target = AflfQuadraticTarget()
    scenarios = [
        (seed=1, num_params=1, num_chains=1, nonfinite_columns=Int[]),
        (seed=2, num_params=3, num_chains=5, nonfinite_columns=Int[]),
        (seed=3, num_params=2, num_chains=64, nonfinite_columns=[7, 20]),
        (seed=4, num_params=7, num_chains=17, nonfinite_columns=[1, 17]),
    ]
    for scenario in scenarios, num_steps in (1, 3, 8)
        inputs = aflf_scenario_inputs(scenario.seed, scenario.num_params, scenario.num_chains;
            nonfinite_columns=scenario.nonfinite_columns)
        a = AflfBuffers(scenario.num_params, scenario.num_chains)
        b = AflfBuffers(scenario.num_params, scenario.num_chains)

        # shared trajectory
        UncertainTea.batched_leapfrog_trajectory!(
            a.q, a.p, a.gradient, a.logjoint, a.valid,
            inputs.position, inputs.momentum, copy(inputs.gradient),
            target, inputs.inverse_mass_matrix, 0.05, num_steps,
        )
        aflf_reference_trajectory!(
            b.q, b.p, b.gradient, b.logjoint, b.valid,
            inputs.position, inputs.momentum, copy(inputs.gradient),
            target, inputs.inverse_mass_matrix, 0.05, num_steps,
        )
        aflf_compare(a, b)

        # per-chain trajectory
        UncertainTea.batched_leapfrog_trajectory!(
            a.q, a.p, a.gradient, a.logjoint, a.valid,
            inputs.position, inputs.momentum, copy(inputs.gradient),
            target, inputs.inverse_mass_matrices, inputs.step_sizes, num_steps,
        )
        aflf_reference_trajectory!(
            b.q, b.p, b.gradient, b.logjoint, b.valid,
            inputs.position, inputs.momentum, copy(inputs.gradient),
            target, inputs.inverse_mass_matrices, inputs.step_sizes, num_steps,
        )
        aflf_compare(a, b)
    end

    for scenario in scenarios
        inputs = aflf_scenario_inputs(scenario.seed + 100, scenario.num_params, scenario.num_chains;
            nonfinite_columns=scenario.nonfinite_columns)
        a = AflfBuffers(scenario.num_params, scenario.num_chains)
        b = AflfBuffers(scenario.num_params, scenario.num_chains)

        # shared single step
        UncertainTea.batched_leapfrog_step_to!(
            a.q, a.p, a.gradient, a.logjoint, a.valid,
            inputs.position, inputs.momentum, inputs.gradient,
            target, inputs.inverse_mass_matrix, 0.05, inputs.direction, inputs.active,
        )
        aflf_reference_step_to!(
            b.q, b.p, b.gradient, b.logjoint, b.valid,
            inputs.position, inputs.momentum, inputs.gradient,
            target, inputs.inverse_mass_matrix, 0.05, inputs.direction, inputs.active,
        )
        aflf_compare(a, b)

        # per-chain single step
        UncertainTea.batched_leapfrog_step_to!(
            a.q, a.p, a.gradient, a.logjoint, a.valid,
            inputs.position, inputs.momentum, inputs.gradient,
            target, inputs.inverse_mass_matrices, inputs.step_sizes, inputs.direction, inputs.active,
        )
        aflf_reference_step_to!(
            b.q, b.p, b.gradient, b.logjoint, b.valid,
            inputs.position, inputs.momentum, inputs.gradient,
            target, inputs.inverse_mass_matrices, inputs.step_sizes, inputs.direction, inputs.active,
        )
        aflf_compare(a, b)
    end
end

# --- Single-chain drift A/B: `_apply_mass_drift!` (diagonal loop) vs the
# original `q .+= step .* _mass_drift(imm, p)` spelling.
@testset "allocfree_leapfrog_single_chain_drift" begin
    rng = MersenneTwister(11)
    for num_params in (1, 4, 33)
        q0 = randn(rng, num_params)
        p0 = randn(rng, num_params)
        imm = 0.5 .+ rand(rng, num_params)
        production = copy(q0)
        UncertainTea._apply_mass_drift!(production, p0, imm, 0.07)
        reference = copy(q0)
        reference .+= 0.07 .* UncertainTea._mass_drift(imm, p0)
        @test aflf_same_bits(production, reference)
    end
end

# --- Per-iteration NUTS buffers: same values as the old per-draw allocations.
@testset "allocfree_nuts_iteration_buffers" begin
    num_chains = 6
    workspace = UncertainTea.BatchedNUTSWorkspace(
        aflf_eight_schools_noncentered,
        zeros(10, num_chains),
        (aflf_schools_sigma,),
        aflf_schools_constraints,
        5,
    )

    rng = MersenneTwister(21)
    imm_vector = 0.5 .+ rand(rng, 10)
    shared = UncertainTea._batched_nuts_sqrt_inverse_mass!(workspace, imm_vector)
    @test shared === workspace.sqrt_inverse_mass
    @test aflf_same_bits(shared, sqrt.(Float64.(imm_vector)))

    imm_matrix = 0.5 .+ rand(rng, 10, num_chains)
    per_chain = UncertainTea._batched_nuts_sqrt_inverse_mass!(workspace, imm_matrix)
    @test per_chain === workspace.sqrt_inverse_mass_columns
    @test aflf_same_bits(per_chain, sqrt.(Float64.(imm_matrix)))

    mask = UncertainTea._all_chains_active!(workspace)
    @test mask === workspace.all_chains_active
    @test mask == trues(num_chains)
    workspace.all_chains_active[3] = false
    @test UncertainTea._all_chains_active!(workspace) == trues(num_chains)
end

# --- Seeded determinism: batched_nuts with the reused workspace buffers must
# reproduce itself exactly under a fixed seed (same-process comparison, so it
# holds on any platform/Julia version).
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

function aflf_nuts(seed; tree_strategy, per_chain_adaptation=false)
    return batched_nuts(
        aflf_eight_schools_noncentered,
        (aflf_schools_sigma,),
        aflf_schools_constraints;
        num_chains=4,
        num_samples=10,
        num_warmup=15,
        max_tree_depth=6,
        tree_strategy,
        per_chain_adaptation,
        rng=MersenneTwister(seed),
    )
end

@testset "allocfree_leapfrog_seeded_determinism" begin
    for (seed, strategy, per_chain) in
        ((146, :hybrid, false), (148, :masked, false), (151, :hybrid, true))
        first_run = aflf_checksum(aflf_nuts(seed; tree_strategy=strategy, per_chain_adaptation=per_chain))
        second_run = aflf_checksum(aflf_nuts(seed; tree_strategy=strategy, per_chain_adaptation=per_chain))
        @test first_run == second_run
    end
end

# --- @allocated regression: the integrator bookkeeping itself must not
# allocate (the quadratic mock target isolates it from model gradients).
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

@testset "allocfree_leapfrog_allocations" begin
    w = aflf_step_workspace(2, 32)
    for measure in
        (aflf_step_to_allocs, aflf_step_to_perchain_allocs, aflf_trajectory_allocs, aflf_trajectory_perchain_allocs)
        measure(w) # warm up compilation
        @test measure(w) == 0
    end
end
