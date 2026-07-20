@testset "dsl_contract_guards" begin
    # --- issue #68: branchful control flow is rejected in static models -----
    # The linear execution plan has no IR for `if`; before the guard the
    # frontend silently appended BOTH branches to the plan and miscompiled
    # logjoint. Macro-expansion errors under @eval arrive wrapped in LoadError.
    @testset "static branchful control flow rejected at macro time" begin
        @test_throws LoadError @eval @tea static function cfg_if_model(flag)
            if flag
                mu = 0.0
            else
                mu = 10.0
            end
            {:y} ~ normal(mu, 1.0)
        end

        @test_throws LoadError @eval @tea static function cfg_elseif_model(k)
            if k == 1
                mu = 0.0
            elseif k == 2
                mu = 1.0
            end
            {:y} ~ normal(mu, 1.0)
        end

        @test_throws LoadError @eval @tea static function cfg_ternary_model(flag)
            mu = flag ? 0.0 : 10.0
            {:y} ~ normal(mu, 1.0)
        end

        # nested occurrences (inside loops and begin blocks) are caught too
        @test_throws LoadError @eval @tea static function cfg_nested_if_model(n)
            for i = 1:n
                begin
                    if i > 1
                        {:y => i} ~ normal(0.0, 1.0)
                    end
                end
            end
        end

        # short-circuit && / || that conditionally executes a choice or an
        # assignment hits the same linearization path
        @test_throws LoadError @eval @tea static function cfg_shortcircuit_choice_model(flag)
            flag && ({:y} ~ normal(0.0, 1.0))
        end
        @test_throws LoadError @eval @tea static function cfg_shortcircuit_assign_model(flag)
            flag || (mu = 10.0)
            {:y} ~ normal(0.0, 1.0)
        end

        # value-level short-circuit without a choice/assignment inside stays
        # accepted at macro time (it is not silently linearized; the compiled
        # plan rejects the expression loudly if it ever needs to evaluate it)
        @tea static function cfg_shortcircuit_value_model(a)
            mu = ifelse((a > 0.0) && (a < 1.0), a, 0.0)
            {:y} ~ normal(mu, 1.0)
        end
        trace, logw = generate(cfg_shortcircuit_value_model, (0.5,), choicemap((:y, 0.3)))
        @test isfinite(logw)

        # ifelse(cond, a, b) is the supported deterministic value selection
        @tea static function cfg_ifelse_model(flag)
            mu = ifelse(flag, 0.0, 10.0)
            {:y} ~ normal(mu, 1.0)
        end
        cm = choicemap((:y, 0.0))
        @test assess(cfg_ifelse_model, (true,), cm) ≈
              logjoint(cfg_ifelse_model, Float64[], (true,), cm) atol = 1e-12
        @test assess(cfg_ifelse_model, (false,), cm) ≈
              logjoint(cfg_ifelse_model, Float64[], (false,), cm) atol = 1e-12
        @test assess(cfg_ifelse_model, (true,), cm) != assess(cfg_ifelse_model, (false,), cm)
    end

    @testset "dynamic branchful models: runtime works, compiled scoring rejects" begin
        @tea dynamic function cfg_dynamic_if_model(flag)
            if flag
                mu = 0.0
            else
                mu = 10.0
            end
            {:y} ~ normal(mu, 1.0)
        end
        cm = choicemap((:y, 0.0))
        # generate/assess execute the real body, so each flag scores its own branch
        @test assess(cfg_dynamic_if_model, (true,), cm) ≈
              UncertainTea.logpdf(normal(0.0, 1.0), 0.0) atol = 1e-12
        @test assess(cfg_dynamic_if_model, (false,), cm) ≈
              UncertainTea.logpdf(normal(10.0, 1.0), 0.0) atol = 1e-12
        # the linearized plan cannot represent the branch: compiled scoring must refuse
        @test_throws ArgumentError logjoint(cfg_dynamic_if_model, Float64[], (true,), cm)
        report = backend_report(cfg_dynamic_if_model)
        @test !report.supported
    end

    # --- issue #79: duplicate choice addresses -------------------------------
    @testset "duplicate static addresses rejected at model construction" begin
        @test_throws ArgumentError @eval @tea static function dup_static_model()
            {:x} ~ normal(0.0, 1.0)
            {:x} ~ normal(10.0, 1.0)
        end

        # implicit-binding duplicates too
        @test_throws ArgumentError @eval @tea static function dup_binding_model()
            x ~ normal(0.0, 1.0)
            x ~ normal(10.0, 1.0)
        end

        # submodel inlining that lands two choices on the same prefixed address
        @eval @tea static function dup_inner_model()
            z ~ normal(0.0, 1.0)
            return z
        end
        @test_throws ArgumentError @eval @tea static function dup_submodel_model()
            a = ({:s} ~ dup_inner_model())
            b = ({:s} ~ dup_inner_model())
            return a + b
        end
    end

    @testset "duplicate loop-generated addresses rejected at execution" begin
        @tea static function dup_loop_model(n)
            for i = 1:n
                {:x => 1} ~ normal(0.0, 1.0)
            end
        end
        # a single iteration is fine ...
        trace, _ = generate(dup_loop_model, (1,); rng=MersenneTwister(7))
        @test length(trace) == 1
        # ... but a second visit to the same address must error, not overwrite
        @test_throws ArgumentError generate(dup_loop_model, (2,); rng=MersenneTwister(7))
        @test_throws ArgumentError assess(
            dup_loop_model,
            (2,),
            choicemap((:x => 1, 0.5)),
        )
    end

    @testset "user-side ChoiceMap updates keep overwrite semantics" begin
        # only in-model duplicate RECORDING errors; explicit user-side updates
        # (constructing constraints, overriding entries) still overwrite
        cm = choicemap((:y, 0.1), (:y, 0.2))
        @test length(cm) == 1
        @test cm[:y] == 0.2
    end

    # --- issue #89: default @tea arguments work uniformly --------------------
    @testset "default arguments agree between generate and compiled scoring" begin
        @tea static function defarg_model(mu=2.0)
            x ~ normal(mu, 1.0)
        end
        trace, _ = generate(defarg_model; rng=MersenneTwister(11))
        @test logjoint(defarg_model, [trace[:x]]) ≈
              logjoint(defarg_model, [trace[:x]], (2.0,)) atol = 1e-12
        @test logjoint_unconstrained(defarg_model, [trace[:x]]) ≈
              logjoint_unconstrained(defarg_model, [trace[:x]], (2.0,)) atol = 1e-12
        @test assess(defarg_model, (), choicemap((:x, trace[:x]))) ≈
              logjoint(defarg_model, [trace[:x]]) atol = 1e-12

        # defaults may reference earlier arguments (Julia default-arg semantics)
        @tea static function defarg_chain_model(a, b=a + 1.0)
            x ~ normal(a * b, 1.0)
        end
        trace2, _ = generate(defarg_chain_model, (2.0,); rng=MersenneTwister(12))
        @test logjoint(defarg_chain_model, [trace2[:x]], (2.0,)) ≈
              logjoint(defarg_chain_model, [trace2[:x]], (2.0, 3.0)) atol = 1e-12

        # missing REQUIRED arguments still fail loudly
        @test_throws DimensionMismatch logjoint(defarg_chain_model, [trace2[:x]])
        # too many arguments fail loudly too
        @test_throws DimensionMismatch logjoint(defarg_model, [trace[:x]], (2.0, 3.0))

        # batched entry points fill defaults the same way
        params = reshape([trace[:x], trace[:x]], 1, 2)
        @test batched_logjoint(defarg_model, params) ≈
              batched_logjoint(defarg_model, params, (2.0,)) atol = 1e-12
        @test batched_logjoint_unconstrained(defarg_model, params, [(), ()]) ≈
              batched_logjoint_unconstrained(defarg_model, params, [(2.0,), (2.0,)]) atol = 1e-12

        # pointwise/observation paths fill defaults too
        @tea static function defarg_obs_model(mu=2.0)
            x ~ normal(mu, 1.0)
            {:y} ~ normal(x, 1.0)
        end
        @test observation_addresses(defarg_obs_model, (), choicemap((:y, 0.3))) == Any[(:y,)]
    end

    # --- issue #106: choicemap tuple-of-pairs --------------------------------
    @testset "choicemap tuple-of-pairs parses as entries" begin
        cm = choicemap((:a => 1.0, :b => 2.0))
        @test length(cm) == 2
        @test cm[:a] == 1.0
        @test cm[:b] == 2.0

        # consistent with the vector and longer-tuple container forms
        @test length(choicemap([:a => 1.0, :b => 2.0])) == 2
        @test length(choicemap((:a => 1.0, :b => 2.0, :c => 3.0))) == 2 + 1

        # a 2-tuple of Pairs as ONE entry among several is ambiguous: error
        @test_throws ArgumentError choicemap((:a => 1.0, :b => 2.0), (:c => 3.0, :d => 4.0))

        # the documented single-entry forms keep working
        @test choicemap((:y, 0.3))[:y] == 0.3
        @test choicemap((:y => 1, 0.5))[:y=>1] == 0.5
        @test length(choicemap((:y => i, Float64(i)) for i = 1:3)) == 3
    end
end
