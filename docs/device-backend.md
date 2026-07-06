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

## Device-resident inner loops (HMC + ADVI)

`batched_hmc` and `batched_advi` accept a `backend::KernelAbstractions.Backend`
(plus optional `precision`) keyword. When `backend === nothing` the existing CPU
path is used unchanged (bitwise identical to before). When a backend is supplied,
the per-iteration inner loop runs on the device:

- **HMC:** momenta and the accept/reject uniforms are drawn on the HOST (matching
  the CPU draw shape) and uploaded; the leapfrog integration, per-column validity
  folds, and the Hamiltonians run in KernelAbstractions kernels
  (`src/device/hmc_kernels.jl`); accept/reject decisions happen on the host from two
  downloaded length-`num_chains` Hamiltonian vectors plus the validity mask, and the
  accepted columns are copied on-device via an accept-mask kernel. Warmup dual
  averaging and running-variance mass adaptation stay on the host (they need accept
  statistics and positions), so warmup iterations additionally download the
  position/gradient matrices. Diagonal mass only; `per_chain_adaptation=true` with a
  backend raises an `ArgumentError`.
- **ADVI:** the host draws the reparameterized particles and uploads them; the fused
  device gradient kernel scores them and a reduction kernel
  (`src/device/advi_kernels.jl`) produces the mean location gradient and the
  noise-weighted scale-gradient accumulator on-device, so only two length-P vectors
  (plus the per-particle values) return each step. Adam + ELBO bookkeeping stay on
  the host.

Because the RNG stays host-side but gradients are computed via device forward-mode
duals, device results are **statistically equivalent** to the CPU path, not bitwise
identical. Unsupported models raise the same `device_lowering_report`-pointing
`ArgumentError`.

## Masked doubling status (batched NUTS)

`batched_nuts(...; tree_strategy=:masked)` runs the mask-based iterative-doubling
tree builder (`src/inference/nuts/masked_doubling.jl`): all chains advance through
the same doubling round in lockstep with active masks, so every leapfrog step is
one full-width batched gradient call -- the exact shape the device gradient kernel
consumes. Per-chain tree bookkeeping (log weights, U-turn state, proposal
selection, directions) stays in small host arrays. The path is CPU-validated
(statistically equivalent to the default `:hybrid` strategy; deterministic under a
seed) and benchmarked in `bench/nuts_masked_bench.jl`; wiring its inner leapfrog
loop onto the device backend is the next step.

## What follows

Device-resident NUTS integration (dynamic trajectory lengths) and on-device warmup
adaptation build on these inner loops; they are out of scope for this phase.
