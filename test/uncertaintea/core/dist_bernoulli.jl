# Issue #85: bernoulli support is Bool or a numeric 0/1; every other value
# scores -Inf on the CPU, backend, and device paths alike (previously any
# nonzero value scored as true).

@testset "dist_bernoulli_support" begin
    bern_lp(x) = UncertainTea.logpdf(bernoulli(0.3), x)
    @test bern_lp(true) ≈ log(0.3) atol = 1e-12
    @test bern_lp(false) ≈ log(0.7) atol = 1e-12
    @test bern_lp(1) ≈ log(0.3) atol = 1e-12
    @test bern_lp(0) ≈ log(0.7) atol = 1e-12
    @test bern_lp(1.0) ≈ log(0.3) atol = 1e-12
    @test bern_lp(0.0) ≈ log(0.7) atol = 1e-12
    @test bern_lp(2) == -Inf
    @test bern_lp(-1) == -Inf
    @test bern_lp(0.5) == -Inf
    @test bern_lp(NaN) == -Inf
    @test bern_lp(:yes) == -Inf

    bern_backend_lp(x) = UncertainTea._backend_bernoulli_logpdf(0.3, x)
    @test bern_backend_lp(true) ≈ log(0.3) atol = 1e-12
    @test bern_backend_lp(0) ≈ log(0.7) atol = 1e-12
    @test bern_backend_lp(1.0) ≈ log(0.3) atol = 1e-12
    @test bern_backend_lp(2) == -Inf
    @test bern_backend_lp(-1) == -Inf
    @test bern_backend_lp(0.5) == -Inf

    # the device mirror sees float-normalized values and stays exception-free
    bern_device_lp(x) = UncertainTea._device_bernoulli_logpdf(0.3, x)
    @test bern_device_lp(1.0) ≈ log(0.3) atol = 1e-12
    @test bern_device_lp(0.0) ≈ log(0.7) atol = 1e-12
    @test bern_device_lp(2.0) == -Inf
    @test bern_device_lp(-1.0) == -Inf
    @test bern_device_lp(0.5) == -Inf

    # model-level: constraining a bernoulli choice to an unsupported value
    # scores -Inf instead of silently reading it as true
    @tea static function bern_support_model()
        {:y} ~ bernoulli(0.3)
    end
    @test assess(bern_support_model, (), choicemap((:y, 2))) == -Inf
    @test assess(bern_support_model, (), choicemap((:y, true))) ≈ log(0.3) atol = 1e-12
end
