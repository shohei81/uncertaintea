# Device-resident batched HMC inner loop.
#
# The RNG stays HOST-side: momenta and the accept/reject uniforms are generated on
# the host (matching the CPU `batched_hmc` draw shape) and uploaded. Everything
# else -- the leapfrog integration, per-column validity, and the Hamiltonians --
# runs in KernelAbstractions kernels over device arrays. Per-iteration downloads
# are limited to a handful of small (`num_chains`) vectors during sampling; during
# warmup the position/gradient matrices are ALSO downloaded so the host-side dual
# averaging + running-variance mass adaptation can proceed (see the sync-point
# comment block in `_run_device_batched_hmc`).
#
# Diagonal mass only (`inverse_mass::Vector`); per-chain adaptation is rejected by
# the driver before we get here. Everything is `T`-generic so the CPU reference
# backend and a GPU backend (e.g. Metal at Float32) share one code path.

# ---- kernels -------------------------------------------------------------------

# Momentum (half- or full-) kick: p += factor * grad, over valid chains only.
# `ndrange = (num_params, num_chains)`.
@kernel function _device_hmc_kick!(p, @Const(grad), @Const(valid), factor)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(valid[b]) != 0x00
        @inbounds p[pidx, b] += factor * grad[pidx, b]
    end
end

# Position drift: q += step_size * (inverse_mass .* p), over valid chains only.
@kernel function _device_hmc_drift!(q, @Const(p), @Const(inverse_mass), @Const(valid), step_size)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(valid[b]) != 0x00
        @inbounds q[pidx, b] += step_size * inverse_mass[pidx] * p[pidx, b]
    end
end

# Final half-kick + momentum flip: p = -(p + half_step * grad), over valid chains.
@kernel function _device_hmc_final_halfkick!(p, @Const(grad), @Const(valid), half_step)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(valid[b]) != 0x00
        @inbounds p[pidx, b] = -(p[pidx, b] + half_step * grad[pidx, b])
    end
end

# Per-chain validity seed: valid[b] = all finite over the current gradient column.
# `ndrange = num_chains`, in-thread loop over parameters.
@kernel function _device_hmc_validity_init!(valid, @Const(grad), num_params::Int)
    b = @index(Global)
    ok = true
    for pidx = 1:num_params
        ok &= isfinite(@inbounds grad[pidx, b])
    end
    @inbounds valid[b] = ok ? 0x01 : 0x00
end

# Per-chain validity fold: valid[b] &= all finite over the gradient column.
@kernel function _device_hmc_validity_update!(valid, @Const(grad), num_params::Int)
    b = @index(Global)
    if @inbounds(valid[b]) != 0x00
        ok = true
        for pidx = 1:num_params
            ok &= isfinite(@inbounds grad[pidx, b])
        end
        @inbounds valid[b] = ok ? 0x01 : 0x00
    end
end

# Final-step validity fold: also require the proposed logjoint to be finite.
@kernel function _device_hmc_validity_update_final!(valid, @Const(grad), @Const(logjoint), num_params::Int)
    b = @index(Global)
    if @inbounds(valid[b]) != 0x00
        ok = isfinite(@inbounds logjoint[b])
        for pidx = 1:num_params
            ok &= isfinite(@inbounds grad[pidx, b])
        end
        @inbounds valid[b] = ok ? 0x01 : 0x00
    end
end

# Per-chain Hamiltonian: H[b] = 0.5 * sum_p (momentum^2 * inverse_mass) - logjoint[b].
@kernel function _device_hmc_hamiltonian!(hamiltonian, @Const(momentum), @Const(inverse_mass), @Const(logjoint), num_params::Int)
    b = @index(Global)
    kinetic = zero(eltype(hamiltonian))
    for pidx = 1:num_params
        m = @inbounds momentum[pidx, b]
        kinetic += m * m * @inbounds(inverse_mass[pidx])
    end
    @inbounds hamiltonian[b] = kinetic / 2 - logjoint[b]
end

# Accept-mask column update: for accepted chains copy the proposal position/gradient
# columns into the current buffers and adopt the proposed logjoint.
@kernel function _device_hmc_accept_columns!(
    position,
    current_gradient,
    current_logjoint,
    @Const(proposal_position),
    @Const(proposal_gradient),
    @Const(proposed_logjoint),
    @Const(accept),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(accept[b]) != 0x00
        @inbounds position[pidx, b] = proposal_position[pidx, b]
        @inbounds current_gradient[pidx, b] = proposal_gradient[pidx, b]
        if pidx == 1
            @inbounds current_logjoint[b] = proposed_logjoint[b]
        end
    end
end

# ---- device HMC workspace ------------------------------------------------------

# Wraps a `DeviceBatchedWorkspace` (whose `params_device`/`gradients_device`/
# `totals_device` serve as the proposal position / gradient / logjoint buffers) and
# adds the HMC-specific device buffers. Diagonal mass only.
mutable struct DeviceHMCWorkspace{T,B<:KernelAbstractions.Backend}
    inner::DeviceBatchedWorkspace{T}
    backend::B
    num_params::Int
    num_chains::Int
    position::Any            # P x C  current (accepted) position
    momentum::Any            # P x C  uploaded initial momentum
    working_momentum::Any    # P x C  momentum integrated in place
    current_gradient::Any    # P x C  gradient at the current position
    inverse_mass::Any        # P      diagonal inverse mass
    valid::Any               # C      UInt8 per-chain validity
    accept_mask::Any         # C      UInt8 per-chain accept flag
    current_logjoint::Any    # C      logjoint at the current position
    current_hamiltonian::Any # C
    proposed_hamiltonian::Any # C
end

function DeviceHMCWorkspace(
    model::TeaModel,
    num_chains::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
)
    inner = DeviceBatchedWorkspace(
        model, num_chains; backend=backend, precision=precision, args=args, constraints=constraints,
    )
    _device_ensure_gradient_buffers!(inner)
    T = precision
    P = inner.parameter_count
    C = inner.batch_size
    mat() = KernelAbstractions.allocate(backend, T, P, C)
    vec_t() = KernelAbstractions.allocate(backend, T, C)
    return DeviceHMCWorkspace{T,typeof(backend)}(
        inner,
        backend,
        P,
        C,
        mat(),
        mat(),
        mat(),
        mat(),
        KernelAbstractions.allocate(backend, T, P),
        KernelAbstractions.allocate(backend, UInt8, C),
        KernelAbstractions.allocate(backend, UInt8, C),
        vec_t(),
        vec_t(),
        vec_t(),
    )
end

# Launch the fused device gradient kernel against `inner.params_device` in place
# (no upload, no download, no synchronize): the working position is already resident.
function _device_launch_gradient!(inner::DeviceBatchedWorkspace)
    kernel = _device_gradient_kernel!(inner.backend)
    kernel(
        inner.totals_device,
        inner.gradients_device,
        inner.plan,
        inner.params_device,
        inner.observed_device,
        inner.grad_slots_device,
        inner.trip_counts_device,
        inner.loop_starts_device;
        ndrange=(inner.parameter_count, inner.batch_size),
    )
    return nothing
end

# Device leapfrog trajectory (matches `batched_leapfrog_trajectory!` step for step):
# initial half-kick from `current_gradient`, then K (drift, gradient, kick) steps
# with a final half-kick + momentum flip. Reads `ws.position`/`ws.momentum`, leaves
# the proposal in `ws.inner.params_device` (position), `ws.working_momentum`
# (momentum), `ws.inner.gradients_device`/`ws.inner.totals_device`
# (proposal gradient / logjoint), and `ws.valid`.
function _device_leapfrog_integrate!(ws::DeviceHMCWorkspace{T}, step_size::Real, num_steps::Int) where {T}
    be = ws.backend
    P = ws.num_params
    C = ws.num_chains
    inner = ws.inner
    q = inner.params_device
    grad = inner.gradients_device
    logj = inner.totals_device
    p = ws.working_momentum
    h = convert(T, step_size)
    half = convert(T, step_size / 2)

    copyto!(q, ws.position)
    copyto!(p, ws.momentum)

    _device_hmc_validity_init!(be)(ws.valid, ws.current_gradient, P; ndrange=C)
    _device_hmc_kick!(be)(p, ws.current_gradient, ws.valid, half; ndrange=(P, C))

    for leapfrog_step = 1:num_steps
        _device_hmc_drift!(be)(q, p, ws.inverse_mass, ws.valid, h; ndrange=(P, C))
        _device_launch_gradient!(inner)
        if leapfrog_step < num_steps
            _device_hmc_validity_update!(be)(ws.valid, grad, P; ndrange=C)
            _device_hmc_kick!(be)(p, grad, ws.valid, h; ndrange=(P, C))
        else
            _device_hmc_validity_update_final!(be)(ws.valid, grad, logj, P; ndrange=C)
        end
    end

    _device_hmc_final_halfkick!(be)(p, grad, ws.valid, half; ndrange=(P, C))
    KernelAbstractions.synchronize(be)
    return nothing
end

# ---- device HMC driver ---------------------------------------------------------

# Device-resident shared-adaptation batched HMC. Mirrors the shared-mode CPU
# `batched_hmc` loop, but the per-iteration leapfrog / Hamiltonians / accept-column
# update run on the device. Warmup dual averaging + running-variance mass adaptation
# stay on the HOST (they need accept statistics and positions).
#
# SYNC POINTS (per iteration, num_chains = C, num_params = P):
#   uploads : inverse_mass (P), momentum (P x C), accept_mask (C UInt8)
#   downloads (always): valid (C), current_hamiltonian (C), proposed_hamiltonian (C)
#   downloads (sampling iters): position (P x C), current_logjoint (C)
#   downloads (warmup iters, for host adaptation + step-size re-search): position,
#             current_gradient (both P x C), current_logjoint (C)
# Kernel launches per iteration: 1 current-Hamiltonian + [validity_init + 1 kick +
#   K*(drift + gradient + (kick|final-validity)) + final-halfkick] + 1
#   proposed-Hamiltonian + 1 accept-columns  =  2*K + 5 KA launches (K = leapfrog
#   steps; the gradient launch is itself the fused device gradient kernel).
function _run_device_batched_hmc(
    model::TeaModel,
    args,
    constraints;
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    initial_params,
    target_accept::Real,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    find_reasonable_step_size::Bool,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    callback,
    callback_every::Int,
    backend::KernelAbstractions.Backend,
    precision::Type,
    rng::AbstractRNG,
)
    T = precision
    num_params = parametercount(parameterlayout(model))
    constrained_num_params = parametervaluecount(parameterlayout(model))
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
        constrained_num_params,
        num_chains,
    )

    # Build the device workspace first: this is where an unsupported model raises the
    # ArgumentError pointing back at `device_lowering_report`.
    ws = DeviceHMCWorkspace(
        model, num_chains; backend=backend, precision=precision, args=args, constraints=constraints,
    )

    # Host workspace: used only for the initial finite check, the reasonable
    # step-size search, and the window-end step-size re-search (all Float64).
    inverse_mass_matrix = ones(num_params)
    host_workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = host_workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint, host_workspace.gradient_cache, position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained gradient"))

    # Seed the device buffers with the initial state.
    copyto!(ws.position, convert(Array{T}, position))
    copyto!(ws.current_gradient, convert(Array{T}, current_gradient))
    copyto!(ws.current_logjoint, convert(Array{T}, current_logjoint))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
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

    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_batched_step_size(
            host_workspace,
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
    driver = WarmupDriver(
        num_params,
        num_warmup,
        hmc_step_size,
        hmc_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    # The re-search reads these host mirrors, which we refresh (download) each warmup
    # iteration; passing the same array objects keeps the search current.
    refind = BatchedStepSizeSearch(
        host_workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        hmc_divergence_threshold,
        rng,
    )

    # Host staging + download buffers (allocated once).
    sqrt_inverse_mass = Vector{Float64}(undef, num_params)
    host_momentum = Matrix{Float64}(undef, num_params, num_chains)
    momentum_upload = Matrix{T}(undef, num_params, num_chains)
    inverse_mass_upload = Vector{T}(undef, num_params)
    host_valid = Vector{UInt8}(undef, num_chains)
    host_current_ham = Vector{T}(undef, num_chains)
    host_proposed_ham = Vector{T}(undef, num_chains)
    host_accept_mask = Vector{UInt8}(undef, num_chains)
    position_download = Matrix{T}(undef, num_params, num_chains)
    gradient_download = Matrix{T}(undef, num_params, num_chains)
    logjoint_download = Vector{T}(undef, num_chains)

    accept_prob = Vector{Float64}(undef, num_chains)
    accepted_step = falses(num_chains)
    divergent_step = falses(num_chains)
    energy_error_vec = Vector{Float64}(undef, num_chains)
    current_ham_f64 = Vector{Float64}(undef, num_chains)
    proposed_ham_f64 = Vector{Float64}(undef, num_chains)
    mass_adaptation_weights = Vector{Float64}(undef, num_chains)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        hmc_step_size = driver.step_size
        inverse_mass_matrix = driver.inverse_mass_matrix
        inverse_mass_upload .= inverse_mass_matrix
        copyto!(ws.inverse_mass, inverse_mass_upload)

        _update_sqrt_inverse_mass_matrix!(sqrt_inverse_mass, inverse_mass_matrix)
        _sample_batched_momentum!(host_momentum, rng, sqrt_inverse_mass)
        momentum_upload .= host_momentum
        copyto!(ws.momentum, momentum_upload)

        _device_hmc_hamiltonian!(ws.backend)(
            ws.current_hamiltonian, ws.momentum, ws.inverse_mass, ws.current_logjoint, num_params; ndrange=num_chains,
        )
        _device_leapfrog_integrate!(ws, hmc_step_size, num_leapfrog_steps)
        _device_hmc_hamiltonian!(ws.backend)(
            ws.proposed_hamiltonian, ws.working_momentum, ws.inverse_mass, ws.inner.totals_device, num_params; ndrange=num_chains,
        )
        KernelAbstractions.synchronize(ws.backend)

        copyto!(host_valid, ws.valid)
        copyto!(host_current_ham, ws.current_hamiltonian)
        copyto!(host_proposed_ham, ws.proposed_hamiltonian)

        fill!(accepted_step, false)
        fill!(divergent_step, true)
        for chain_index = 1:num_chains
            if host_valid[chain_index] != 0x00
                current_ham = Float64(host_current_ham[chain_index])
                proposed_ham = Float64(host_proposed_ham[chain_index])
                current_ham_f64[chain_index] = current_ham
                proposed_ham_f64[chain_index] = proposed_ham
                log_accept_ratio = current_ham - proposed_ham
                energy_error_vec[chain_index] = proposed_ham - current_ham
                divergent_step[chain_index] =
                    !isfinite(energy_error_vec[chain_index]) ||
                    energy_error_vec[chain_index] > hmc_divergence_threshold
                accept_prob[chain_index] = _acceptance_probability(log_accept_ratio)
                if log(rand(rng)) < min(0.0, log_accept_ratio)
                    accepted_step[chain_index] = true
                end
            else
                current_ham_f64[chain_index] = Float64(host_current_ham[chain_index])
                proposed_ham_f64[chain_index] = Inf
                energy_error_vec[chain_index] = Inf
                accept_prob[chain_index] = 0.0
            end
            host_accept_mask[chain_index] = accepted_step[chain_index] ? 0x01 : 0x00
        end

        copyto!(ws.accept_mask, host_accept_mask)
        _device_hmc_accept_columns!(ws.backend)(
            ws.position,
            ws.current_gradient,
            ws.current_logjoint,
            ws.inner.params_device,
            ws.inner.gradients_device,
            ws.inner.totals_device,
            ws.accept_mask;
            ndrange=(num_params, num_chains),
        )
        KernelAbstractions.synchronize(ws.backend)

        cumulative_divergences += count(divergent_step)

        # Download the (accepted) position + logjoint for host bookkeeping. During
        # warmup also grab the gradient so the step-size re-search sees fresh state.
        copyto!(position_download, ws.position)
        copyto!(logjoint_download, ws.current_logjoint)
        for chain_index = 1:num_chains
            for parameter_index = 1:num_params
                position[parameter_index, chain_index] = Float64(position_download[parameter_index, chain_index])
            end
            current_logjoint[chain_index] = Float64(logjoint_download[chain_index])
        end
        if iteration <= num_warmup
            copyto!(gradient_download, ws.current_gradient)
            for chain_index = 1:num_chains
                for parameter_index = 1:num_params
                    current_gradient[parameter_index, chain_index] =
                        Float64(gradient_download[parameter_index, chain_index])
                end
            end
        end

        if iteration <= num_warmup
            _mass_adaptation_weights!(
                driver.variance_state,
                mass_adaptation_weights,
                accepted_step,
                accept_prob,
                divergent_step,
            )
            accept_statistic = _mean_batched_adaptation_probability(accept_prob, divergent_step)
            warmup_update!(
                driver,
                iteration,
                accept_statistic,
                position,
                mass_adaptation_weights,
                refind,
            )
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, hmc_step_size, cumulative_divergences)
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index = 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _transform_to_constrained!(
                    view(host_workspace.constrained_position, :, chain_index),
                    model,
                    view(position, :, chain_index),
                )
                copyto!(
                    view(constrained_samples, :, sample_index, chain_index),
                    view(host_workspace.constrained_position, :, chain_index),
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = accept_prob[chain_index]
                energies[sample_index, chain_index] =
                    accepted_step[chain_index] ? proposed_ham_f64[chain_index] : current_ham_f64[chain_index]
                energy_errors[sample_index, chain_index] = energy_error_vec[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, hmc_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
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
            driver.step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            fill(num_leapfrog_steps, num_samples),
            hmc_target_accept,
            copy(driver.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end
