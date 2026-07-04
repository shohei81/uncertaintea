function leapfrog_step!(
    destination::NUTSState,
    target::AbstractDensityTarget,
    state::NUTSState,
    inverse_mass_matrix::Vector{Float64},
    step_size::Float64,
)
    q = destination.position
    p = destination.momentum
    gradient = destination.gradient
    copyto!(q, state.position)
    copyto!(p, state.momentum)
    p .+= (step_size / 2) .* state.gradient
    q .+= step_size .* (inverse_mass_matrix .* p)
    value, proposed_gradient = target_logdensity_and_gradient!(target, q)
    isfinite(value) || return false
    all(isfinite, proposed_gradient) || return false
    copyto!(gradient, proposed_gradient)
    p .+= (step_size / 2) .* gradient
    destination.logjoint = value
    return true
end

function leapfrog_trajectory(
    target::AbstractDensityTarget,
    position::AbstractVector{Float64},
    momentum::AbstractVector{Float64},
    inverse_mass_matrix::Vector{Float64},
    step_size::Float64,
    num_steps::Int,
)
    q = copy(position)
    p = copy(momentum)

    gradient = target_gradient!(target, q)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step in 1:num_steps
        q .+= step_size .* (inverse_mass_matrix .* p)
        gradient = target_gradient!(target, q)
        all(isfinite, gradient) || return nothing

        if leapfrog_step < num_steps
            p .+= step_size .* gradient
        end
    end

    p .+= (step_size / 2) .* gradient
    p .*= -1

    value = target_logdensity(target, q)
    isfinite(value) || return nothing
    return q, p, value
end

function batched_leapfrog_trajectory!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    current_gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    num_steps::Int,
)
    q = destination_position
    p = destination_momentum
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(destination_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index in 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step in 1:num_steps
        for chain_index in 1:num_chains
            valid[chain_index] || continue
            q[:, chain_index] .+= step_size .* (inverse_mass_matrix .* p[:, chain_index])
        end

        if leapfrog_step < num_steps
            gradient = batched_target_gradient!(destination_gradient, target, q)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
                destination_logjoint,
                destination_gradient,
                target,
                q,
            )
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(destination_gradient, :, chain_index)) ||
                   !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index in 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* destination_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

function batched_leapfrog_step_to!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    q = destination_position
    p = destination_momentum
    num_chains = size(position, 2)
    size(position) == size(momentum) == size(gradient) ||
        throw(DimensionMismatch("batched leapfrog step requires matching position, momentum, and gradient matrices"))
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(destination_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"))
    length(destination_logjoint) == num_chains == length(valid) ||
        throw(DimensionMismatch("expected batched leapfrog vectors of length $num_chains"))
    length(direction) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog directions, got $(length(direction))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog activity flags, got $(length(active))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    for chain_index in 1:num_chains
        active[chain_index] || continue
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_size
        p[:, chain_index] .+= (signed_step / 2) .* gradient[:, chain_index]
        q[:, chain_index] .+= signed_step .* (inverse_mass_matrix .* p[:, chain_index])
    end

    proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
        destination_logjoint,
        destination_gradient,
        target,
        q,
    )
    for chain_index in 1:num_chains
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
