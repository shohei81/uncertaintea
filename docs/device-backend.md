# Device backend: device-resident batched logjoint

This is the first phase of *real* on-device execution (as opposed to source-code
emission). It lowers a static, backend-lowerable model to an `isbits`
`DeviceExecutionPlan` and evaluates a batched, unconstrained logjoint inside a
single fused [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernel, with one thread per batch column. The same code path runs on
`KernelAbstractions.CPU()` (the authoritative reference target) and on any GPU
backend; a lightweight package extension declares `Float32` as Metal's precision.

## Status

Supported today:

- **Distributions:** `normal`, `lognormal`, `exponential`, `gamma`, `laplace`,
  `bernoulli`, `poisson`, `beta` (device-safe, `T`-generic log-densities with a
  pure-Julia Lanczos `loggamma`, no exceptions, out-of-support -> `-Inf`).
- **Latent parameter transforms:** `Identity`, `Log`, `Logit` (scalar). Vector
  transforms (`Simplex`/`VectorIdentity`, i.e. `dirichlet`/`mvnormal` latents) are
  reported as unsupported.
- **Structure:** scalar latent priors, numeric deterministic assignments, and
  single (non-nested) unit-range loops with observed choices. Observed values are
  resolved on the host into a dense `observed[row, col]` matrix during one-time
  staging; addresses are fully erased from the device plan.

The entry points are `device_lowering_report(model)` (returns
`(supported, issues)`), `device_batched_logjoint(model, params, args, constraints)`
(allocating convenience), and `DeviceBatchedWorkspace` + `device_batched_logjoint!`
(buffer/staging reuse). Inputs are **unconstrained** parameters and the result folds
the transform log-abs-det in-kernel, so the authoritative counterpart is
`batched_logjoint_unconstrained`. Unsupported models raise a clear `ArgumentError`
pointing back at `device_lowering_report`; the CPU backend remains authoritative.

## What follows

Device-resident gradients (reverse-mode over the same fused kernel) and full
HMC/NUTS integration loops build on this device logjoint; they are out of scope
for this phase.
