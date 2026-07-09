# PR-2 scaffolding for reparam=:noncentered (docs/noncentered-reparam.md):
# the frontend parses and validates the flag, the IR carries it, and every
# execution path that cannot honor it yet rejects it honestly. Semantics land
# in PR-3/4/5.

@tea static function reparam_flagged_model()
    mu ~ normal(0.0, 1.0)
    log_tau ~ normal(0.0, 0.5)
    tau = exp(log_tau)
    theta ~ normal(mu, tau; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@tea static function reparam_centered_model()
    mu ~ normal(0.0, 1.0)
    theta ~ normal(mu, 1.0; reparam=:centered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# an explicit :centered flag must behave exactly like no flag even for
# families whose macro-time layout pass inspects the positional arguments
@tea static function reparam_centered_truncated_model()
    x ~ truncatednormal(0.0, 1.0, -1.0, 2.0; reparam=:centered)
    {:y} ~ normal(x, 0.5)
    return x
end

@testset "reparam_scaffolding" begin
    @testset "reparam_spec_and_layout" begin
        spec = modelspec(reparam_flagged_model)
        theta_choice = spec.choices[findfirst(c -> c.binding === :theta, spec.choices)]
        @test theta_choice.rhs isa DistributionSpec
        @test theta_choice.rhs.reparam === :noncentered
        @test theta_choice.rhs.arguments == Any[:mu, :tau]

        layout = parameterlayout(reparam_flagged_model)
        @test parametercount(layout) == 3
        theta_slot = layout.slots[findfirst(slot -> slot.binding === :theta, layout.slots)]
        @test theta_slot.transform isa UncertainTea.NoncenteredTransform

        # an explicit :centered flag behaves exactly like no flag
        centered_layout = parameterlayout(reparam_centered_model)
        @test all(slot -> slot.transform isa IdentityTransform, centered_layout.slots)
        constraints = choicemap((:y, 0.4))
        trace, _ = generate(reparam_centered_model, (), constraints; rng=MersenneTwister(2))
        @test isfinite(logjoint_unconstrained(reparam_centered_model, transform_to_unconstrained(trace), (), constraints))

        # ... including for families whose layout pass reads positional
        # arguments (static truncation bounds)
        truncated_layout = parameterlayout(reparam_centered_truncated_model)
        @test truncated_layout.slots[1].transform isa BoundedTransform
        truncated_trace, _ =
            generate(reparam_centered_truncated_model, (), constraints; rng=MersenneTwister(3))
        @test isfinite(
            logjoint_unconstrained(
                reparam_centered_truncated_model,
                transform_to_unconstrained(truncated_trace),
                (),
                constraints,
            ),
        )
    end

    @testset "reparam_runtime_path_stays_centered" begin
        constraints = choicemap((:y, 0.4))
        trace, logw = generate(reparam_flagged_model, (), constraints; rng=MersenneTwister(7))
        @test isfinite(logw)
        @test haskey(trace.choices, :theta)
        # assess scores the same centered joint density
        full = choicemap(
            (:mu, trace[:mu]),
            (:log_tau, trace[:log_tau]),
            (:theta, trace[:theta]),
            (:y, 0.4),
        )
        manual =
            UncertainTea.logpdf(normal(0.0, 1.0), trace[:mu]) +
            UncertainTea.logpdf(normal(0.0, 0.5), trace[:log_tau]) +
            UncertainTea.logpdf(normal(trace[:mu], exp(trace[:log_tau])), trace[:theta]) +
            UncertainTea.logpdf(normal(trace[:theta], 0.5), 0.4)
        @test assess(reparam_flagged_model, (), full) ≈ manual atol = 1e-8
    end

    @testset "reparam_honest_rejections" begin
        # CPU semantics landed in PR-3 (see reparam_noncentered_cpu.jl); the
        # batched/backend paths stay honestly rejected until PR-4
        constraints = choicemap((:y, 0.4))
        report = backend_report(reparam_flagged_model)
        @test report.supported == false
        @test any(occursin("reparam=:noncentered", issue) for issue in report.issues)
        @test_throws ErrorException batched_logjoint_unconstrained(
            reparam_flagged_model,
            zeros(3, 2),
            (),
            constraints,
        )
    end

    @testset "reparam_macro_validation" begin
        # macro-expansion errors under @eval arrive wrapped in LoadError
        # unknown keyword on a distribution call (previously silently corrupted)
        @test_throws LoadError @eval @tea static function reparam_bad_kw_model()
            x ~ normal(0.0, 1.0; scale=2.0)
            return x
        end
        # non-literal / unknown reparam value
        @test_throws LoadError @eval @tea static function reparam_bad_value_model()
            x ~ normal(0.0, 1.0; reparam=:bogus)
            return x
        end
        # ineligible family
        @test_throws LoadError @eval @tea static function reparam_bad_family_model()
            x ~ gamma(2.0, 2.0; reparam=:noncentered)
            return x
        end
        # iid latents are planned but not supported yet
        @test_throws LoadError @eval @tea static function reparam_iid_model()
            x ~ iid(normal(0.0, 1.0), 4; reparam=:noncentered)
            return x
        end
        # nested reparam (e.g. on a mixture component) would diverge between
        # the runtime body and the static spec
        @test_throws LoadError @eval @tea static function reparam_nested_model()
            x ~ mixture((0.5, 0.5), normal(0.0, 1.0; reparam=:noncentered), normal(2.0, 1.0))
            return x
        end
        # observation-shaped site gets no parameter slot
        @test_throws ArgumentError @eval @tea static function reparam_obs_model()
            x ~ normal(0.0, 1.0)
            {:y} ~ normal(x, 1.0; reparam=:noncentered)
            return x
        end
    end
end
