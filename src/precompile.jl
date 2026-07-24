# PrecompileTools workload for the inference hot paths (issue #155).
#
# A fresh session used to pay ~14 s of model-independent JIT on the first
# `batched_nuts`/`nuts_chains` call. This workload runs the hot entry points on
# tiny toy models at precompile time so that shared infrastructure lands in the
# package image. The remaining per-model specialization cost (~7-11 s per new
# `TeaModel{M,F,S}`) is the other half of #155 and is out of scope here.
#
# Rules for this file:
# - tiny data and tiny iteration counts: the whole workload must stay well
#   under 30 s of precompile time
# - fixed seeds everywhere: precompilation must be deterministic
# - no device/Metal code: device paths stay out of the workload
# - no side effects: nothing printed, nothing written

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @tea static function _pc_gauss()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    # One latent per transform family used by the plan steps: lognormal and
    # gamma (log transform), beta (logit transform), plus a bernoulli
    # observation.
    @tea static function _pc_transforms()
        sigma ~ lognormal(0.0f0, 0.5f0)
        rate ~ gamma(2.0f0, 1.0f0)
        p ~ beta(2.0f0, 2.0f0)
        {:y} ~ normal(0.0f0, sigma)
        {:z} ~ bernoulli(p)
        return rate
    end

    @tea static function _pc_loop(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:n
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    @tea static function _pc_broadcast(xs)
        slope ~ normal(0.0, 1.0)
        sigma ~ lognormal(0.0, 0.5)
        {:y} ~ normal.(slope .* xs, sigma)
        return slope
    end

    @tea static function _pc_iid()
        eps ~ iid(normal(0.0f0, 1.0f0), 3)
        return eps
    end

    @tea static function _pc_noncentered()
        tau ~ lognormal(0.0f0, 0.5f0)
        theta ~ normal(0.0f0, tau; reparam=:noncentered)
        {:y} ~ normal(theta, 1.0f0)
        return theta
    end

    _pc_gauss_cm = choicemap((:y, 0.3f0))
    _pc_transforms_cm = choicemap((:y, 0.4f0), (:z, true))
    _pc_loop_cm = choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0))
    _pc_xs = [0.0, 0.5, 1.0]
    _pc_broadcast_cm = choicemap(:y => [0.1, 0.6, 0.9])
    _pc_noncentered_cm = choicemap((:y, 0.5f0))

    @compile_workload begin
        # Single-trace entry points.
        _pc_trace, _ = generate(_pc_gauss, (), _pc_gauss_cm; rng=MersenneTwister(1))
        logjoint(_pc_gauss, parameter_vector(_pc_trace), (), _pc_gauss_cm)
        logjoint_gradient_unconstrained(_pc_gauss, [0.1], (), _pc_gauss_cm)

        # Batched scoring, shared and per-chain constraints.
        _pc_batch = reshape([-0.4, 0.0, 0.6, 1.0], 1, 4)
        batched_logjoint_gradient_unconstrained(_pc_gauss, _pc_batch, (), _pc_gauss_cm)
        batched_logjoint_gradient_unconstrained(_pc_gauss, _pc_batch, (), [_pc_gauss_cm for _ = 1:4])

        # Samplers: single chain, host multichain, and both batched tree
        # strategies.
        nuts(_pc_gauss, (), _pc_gauss_cm; num_samples=5, num_warmup=5, rng=MersenneTwister(2))
        nuts_chains(_pc_gauss, (), _pc_gauss_cm; num_chains=2, num_samples=5, num_warmup=5, rng=MersenneTwister(3))
        batched_nuts(
            _pc_gauss,
            (),
            _pc_gauss_cm;
            num_chains=4,
            num_samples=5,
            num_warmup=5,
            tree_strategy=:hybrid,
            rng=MersenneTwister(4),
        )
        batched_nuts(
            _pc_gauss,
            (),
            _pc_gauss_cm;
            num_chains=4,
            num_samples=5,
            num_warmup=5,
            tree_strategy=:masked,
            rng=MersenneTwister(5),
        )

        # One model per distribution/transform family in the plan steps.
        generate(_pc_transforms, (), _pc_transforms_cm; rng=MersenneTwister(6))
        logjoint(_pc_transforms, [0.8, 1.5, 0.4], (), _pc_transforms_cm)
        logjoint_gradient_unconstrained(_pc_transforms, [-0.2, 0.4, 0.1], (), _pc_transforms_cm)
        batched_logjoint_gradient_unconstrained(
            _pc_transforms,
            reshape([-0.2, 0.4, 0.1, 0.0, 0.2, -0.1], 3, 2),
            (),
            _pc_transforms_cm,
        )

        # Loop-addressed observations. The batched gradient covers the
        # loop-step scorers; the single-trace ForwardDiff path for loop plans
        # is left to the per-model half of #155.
        logjoint(_pc_loop, [0.2], (2,), _pc_loop_cm)
        batched_logjoint_gradient_unconstrained(_pc_loop, reshape([-0.1, 0.3], 1, 2), (2,), _pc_loop_cm)

        # Broadcast-normal observations.
        logjoint(_pc_broadcast, [0.5, 0.9], (_pc_xs,), _pc_broadcast_cm)
        batched_logjoint_gradient_unconstrained(
            _pc_broadcast,
            reshape([0.5, -0.1, 0.2, 0.0], 2, 2),
            (_pc_xs,),
            _pc_broadcast_cm,
        )

        # iid latent vector.
        generate(_pc_iid, (); rng=MersenneTwister(7))
        batched_logjoint_gradient_unconstrained(_pc_iid, reshape([0.1, -0.2, 0.3, 0.0, 0.1, -0.1], 3, 2), ())

        # reparam=:noncentered latent.
        generate(_pc_noncentered, (), _pc_noncentered_cm; rng=MersenneTwister(8))
        logjoint_gradient_unconstrained(_pc_noncentered, [0.1, 0.2], (), _pc_noncentered_cm)
        batched_logjoint_gradient_unconstrained(
            _pc_noncentered,
            reshape([0.1, 0.2, -0.1, 0.3], 2, 2),
            (),
            _pc_noncentered_cm,
        )
    end
end
