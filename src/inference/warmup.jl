# Shared warmup orchestration for the HMC/NUTS sampler drivers.
# Owns dual-averaging step-size adaptation and windowed running-variance mass
# adaptation so the single-chain and batched drivers share one state machine.

# `metric_kind` is :diag (default; the run is bitwise identical to the legacy
# path and `dense_covariance_state`/`dense_metric` stay nothing) or :dense. In
# dense mode the diagonal `variance_state` is still maintained verbatim (it drives
# the mass-adaptation weights, window bookkeeping, and the reported diagonal
# `mass_matrix`), and a parallel `dense_covariance_state` accumulates the full
# covariance from the same clipped samples to produce the `DenseMetric` used by
# the integrator.
mutable struct WarmupDriver
    warmup_schedule::WarmupSchedule
    dual_state::DualAveragingState
    variance_state::RunningVarianceState
    inverse_mass_matrix::Vector{Float64}
    step_size::Float64
    mass_window_index::Int
    mass_adaptation_windows::Vector{HMCMassAdaptationWindowSummary}
    num_params::Int
    num_warmup::Int
    adapt_step_size::Bool
    adapt_mass_matrix::Bool
    target_accept::Float64
    mass_matrix_regularization::Float64
    mass_matrix_min_samples::Int
    metric_kind::Symbol
    dense_covariance_state::Union{Nothing,DenseRunningCovarianceState}
    dense_metric::Union{Nothing,DenseMetric}
end

function WarmupDriver(
    num_params::Int,
    num_warmup::Int,
    initial_step_size::Real,
    target_accept::Real;
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    metric::Symbol=:diag,
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    step_size = Float64(initial_step_size)
    accept = Float64(target_accept)
    warmup_schedule = _warmup_schedule(num_warmup)
    dual_state = _dual_averaging_state(step_size, accept)
    initial_window_length =
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
        _warmup_window_length(warmup_schedule, 1)
    variance_state = _running_variance_state(num_params, initial_window_length)
    if metric === :dense
        dense_covariance_state = _dense_running_covariance_state(num_params, initial_window_length)
        dense_metric = DenseMetric(Matrix{Float64}(I, num_params, num_params))
    else
        dense_covariance_state = nothing
        dense_metric = nothing
    end
    return WarmupDriver(
        warmup_schedule,
        dual_state,
        variance_state,
        ones(num_params),
        step_size,
        1,
        HMCMassAdaptationWindowSummary[],
        num_params,
        num_warmup,
        adapt_step_size,
        adapt_mass_matrix,
        accept,
        Float64(mass_matrix_regularization),
        mass_matrix_min_samples,
        metric,
        dense_covariance_state,
        dense_metric,
    )
end

# The metric object the sampling loop threads through the integrator: the raw
# diagonal Vector for :diag (bitwise-identical legacy path) or the DenseMetric
# for :dense.
_driver_metric(driver::WarmupDriver) =
    driver.metric_kind === :dense ? driver.dense_metric : driver.inverse_mass_matrix

# Callable step-size re-search structs. They stand in for the closures the four
# drivers used, invoked only at window ends. Kept mutable so single-chain callers
# can refresh position/logjoint before each warmup step; batched callers hold the
# in-place-mutated buffers directly.

mutable struct ScalarStepSizeSearch{M,C,A,K,R<:AbstractRNG}
    model::M
    gradient_cache::C
    args::A
    constraints::K
    rng::R
    position::Vector{Float64}
    current_logjoint::Float64
end

function (search::ScalarStepSizeSearch)(step_size::Float64, inverse_mass_matrix::Union{Vector{Float64},MassMetric})
    return _find_reasonable_step_size(
        search.model,
        search.position,
        search.current_logjoint,
        search.gradient_cache,
        inverse_mass_matrix,
        search.args,
        search.constraints,
        step_size,
        search.rng,
    )
end

mutable struct BatchedStepSizeSearch{M,A,K,R<:AbstractRNG}
    workspace::Union{Nothing,BatchedHMCWorkspace}
    model::M
    position::Matrix{Float64}
    current_logjoint::Vector{Float64}
    current_gradient::Matrix{Float64}
    args::A
    constraints::K
    energy_bound::Float64
    rng::R
end

function (search::BatchedStepSizeSearch)(step_size::Float64, inverse_mass_matrix::Vector{Float64})
    workspace = search.workspace
    if workspace === nothing
        workspace = BatchedHMCWorkspace(
            search.model,
            search.position,
            search.args,
            search.constraints,
            inverse_mass_matrix,
        )
        search.workspace = workspace
    end
    return _find_reasonable_batched_step_size(
        workspace,
        search.model,
        search.position,
        search.current_logjoint,
        search.current_gradient,
        inverse_mass_matrix,
        search.args,
        search.constraints,
        step_size,
        search.energy_bound,
        search.rng,
    )
end

# One warmup iteration. `accept_statistic` drives dual averaging; `mass_weights`
# are the caller-computed per-sample weights (scalar for single chain, vector for
# batched) accumulated over `positions` (Vector or Matrix). `refind` is invoked
# only at window ends, exactly where the original drivers re-ran the search.
function warmup_update!(
    driver::WarmupDriver,
    iteration::Int,
    accept_statistic::Float64,
    positions,
    mass_weights,
    refind,
)
    if driver.adapt_step_size
        driver.step_size = _update_step_size!(driver.dual_state, accept_statistic)
    end

    schedule = driver.warmup_schedule
    if driver.adapt_mass_matrix &&
       driver.mass_window_index <= length(schedule.slow_window_ends) &&
       iteration > schedule.initial_buffer
        _update_running_variance!(driver.variance_state, positions, mass_weights)
        if driver.metric_kind === :dense
            _update_dense_covariance!(driver.dense_covariance_state, positions, mass_weights)
        end
        if iteration == schedule.slow_window_ends[driver.mass_window_index]
            mass_updated = false
            if _running_variance_effective_count(driver.variance_state) >= driver.mass_matrix_min_samples
                driver.inverse_mass_matrix =
                    _inverse_mass_matrix(driver.variance_state, driver.mass_matrix_regularization)
                if driver.metric_kind === :dense
                    driver.dense_metric = DenseMetric(
                        _dense_inverse_mass_matrix(driver.dense_covariance_state, driver.mass_matrix_regularization),
                    )
                end
                mass_updated = true
            end
            push!(
                driver.mass_adaptation_windows,
                _mass_adaptation_window_summary(
                    schedule,
                    driver.mass_window_index,
                    driver.variance_state,
                    driver.inverse_mass_matrix,
                    mass_updated,
                ),
            )
            driver.mass_window_index += 1
            if driver.mass_window_index <= length(schedule.slow_window_ends)
                next_window_length = _warmup_window_length(schedule, driver.mass_window_index)
                driver.variance_state = _running_variance_state(driver.num_params, next_window_length)
                if driver.metric_kind === :dense
                    driver.dense_covariance_state =
                        _dense_running_covariance_state(driver.num_params, next_window_length)
                end
            else
                driver.variance_state = _running_variance_state(driver.num_params)
                if driver.metric_kind === :dense
                    driver.dense_covariance_state = _dense_running_covariance_state(driver.num_params)
                end
            end
            if driver.adapt_step_size && iteration < driver.num_warmup
                driver.step_size = refind(driver.step_size, _driver_metric(driver))
                driver.dual_state = _dual_averaging_state(driver.step_size, driver.target_accept)
            end
        end
    end

    return driver.step_size
end

# Applied once, at iteration == num_warmup, after the last warmup_update!.
function warmup_finalize!(driver::WarmupDriver)
    if driver.adapt_step_size
        driver.step_size = _final_step_size(driver.dual_state)
    end
    if driver.adapt_mass_matrix &&
       _running_variance_effective_count(driver.variance_state) >= driver.mass_matrix_min_samples
        driver.inverse_mass_matrix =
            _inverse_mass_matrix(driver.variance_state, driver.mass_matrix_regularization)
        if driver.metric_kind === :dense
            driver.dense_metric = DenseMetric(
                _dense_inverse_mass_matrix(driver.dense_covariance_state, driver.mass_matrix_regularization),
            )
        end
    end
    return driver.step_size
end
