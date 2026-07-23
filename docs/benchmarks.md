# Cross-PPL Benchmarks

Reproducible comparison of UncertainTea against NumPyro (JAX) and Stan
(CmdStan) on correctness first, then performance — produced by the
`bench/crossppl/` harness (issue #121). Turing.jl and Gen.jl are planned
follow-up targets.

**Never quote a timing from this document without its correctness gate.**
Every headline number below comes from a run whose draws passed the shared
gate — rank-normalized split R-hat < 1.01 on every parameter, posterior means
and 5%/95% quantiles within 4 combined MCSEs of the CmdStan reference —
computed by ONE implementation (ArviZ 0.x) over every framework's raw draws.
Rows marked FAIL keep their timings visible for context only.

## Provenance

- **Date:** 2026-07-23
- **Hardware:** Apple M4 (10 cores, 32 GB), Metal 4; macOS 26.5.1. All
  frameworks measured natively on this one machine.
- **Software:** Julia 1.12.2, UncertainTea @ `6df3064` (branch
  `bench-crossppl-121`), NumPyro 0.19 / JAX 0.9 (pinned in
  `bench/crossppl/python/uv.lock`), CmdStan 2.36.0, Python 3.12.
- **Sampler settings:** NUTS everywhere, `target_accept=0.8`,
  `max_tree_depth=10`, diagonal metric, framework-default warmup schedules.
  Correctness pass: 4 chains × 1000 warmup + 1000 draws × 3 repetitions
  (± is std over repetitions). Scaling sweep: 200 warmup + 500 draws at
  64–4096 chains.
- **Metric:** min-over-parameters bulk ESS per second of pure sampling time.
  Warmup and compile/TTFX are excluded and reported in their own columns.

## Key findings

1. **Sampler quality is competitive; per-observation cost is not.** On
   eight-schools-noncentered (8 observations) UncertainTea matches NumPyro's
   ESS/s despite running chains sequentially. On models with 500–1000
   loop-addressed observations it falls 9–100× behind NumPyro and 350–680×
   behind Stan — the gap scales with observation count, pointing at
   per-observation scoring overhead (issue #138), not at NUTS itself.
2. **The batched paths fail the correctness gate at ≥64 chains in the
   default configuration.** Prior-draw initialization plus shared step-size
   adaptation permanently strands the ~6% of chains whose initial `s` lands
   in the steep small-scale region (divergence rate ≈ P(prior s < 0.4) =
   6.2%; pinning the init gives div = 0 and reference-matching posteriors).
   Issue #137 tracks the fix; until then UncertainTea has no gate-passing
   default-configuration scaling numbers to report.
3. **With the #137 workaround (pinned init, labelled rows below), the
   device story is: Metal overtakes the 10-core CPU backend at ~4096 chains
   (1.2×), but NumPyro's vectorized CPU sweep is still 2.5–10× ahead of
   both at every measured chain count.** The GPU-native claim needs #137 and
   #138 fixed before it can be demonstrated honestly.

Issues found while building the harness: #134 (broadcast-normal observations
don't lower to the device path — the DSL's flagship GPU form), #135
(covariate indexing `xs[i]` doesn't lower — GLMs can't ride the device
path), #136 (`nuts_chains` runs chains sequentially), #137, #138 (above).

## Correctness pass (4 chains × 1000 warmup + 1000 draws)

### eight_schools_noncentered — all PASS

| framework | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|
| stan | PASS | 114,139 ± 13,223 | 93,461 | 0.019 | 0.012 | 3.5 | 0.00025 |
| uncertaintea-batched-cpu | PASS | 3,743 ± 180 | 3,824 | 0.335 | 0.42 | 6.6 | 0.0013 |
| uncertaintea-cpu | PASS | 3,354 ± 210 | 3,135 | 0.376 | 0.485 | 4.0 | 0.00075 |
| numpyro-parallel | PASS | 3,330 ± 380 | 2,715 | 0.685 | 0.736 | 1.5 | 0.00017 |

### eight_schools_centered — all FAIL (funnel; expected)

Every framework, Stan included, exceeds R-hat 1.01 with 2–3% divergences at
`target_accept=0.8` — the canonical centered-parameterization pathology. The
gate rejecting all four implementations equally is evidence it works; this
model stays in the suite as the honesty check.

### logistic (N=500, D=8)

| framework | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|
| stan | PASS | 56,304 ± 3,040 | 30,727 | 0.088 | 0.082 | 3.8 | 0 |
| numpyro-parallel | PASS | 9,804 ± 590 | 5,290 | 0.523 | 0.64 | 1.3 | 0 |
| uncertaintea-cpu | PASS | 83 ± 6 | 73.9 | 24.5 | 24.2 | 51.5 | 0 |
| uncertaintea-batched-cpu | FAIL (q95 z=4.1, marginal) | 50.2 | 50.8 | 34.1 | 29 | 63.1 | 0 |

### gauss (mean/scale, N=1000)

| framework | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|
| stan | PASS | 293,495 ± 22,022 | 200,218 | 0.012 | 0.009 | 0.13 | 0 |
| numpyro-parallel | PASS | 7,676 ± 730 | 5,849 | 0.417 | 0.543 | 0.97 | 0 |
| uncertaintea-batched-cpu | PASS | 848 ± 50 | 759 | 1.85 | 2.02 | 7.5 | 0 |
| uncertaintea-cpu | PASS | 173 ± 3.2 | 192 | 6.95 | 7.41 | 16.6 | 0 |

Chain-parallelism differences (documented, reflected in wall-clock): Stan
forks 4 processes, NumPyro `parallel` uses 4 XLA devices, UncertainTea `cpu`
is sequential (#136) and `batched-cpu` vectorized single-threaded.

## Scaling sweep (gauss, 200 warmup + 500 draws, 64–4096 chains)

Default-configuration UncertainTea rows all FAIL the gate (issue #137,
finding 2 above; they ran with 200 draws before the failure was diagnosed
and are kept only as evidence). The `-pinned-init` rows apply the #137
workaround — every chain initialized at the posterior mode — and PASS; they
are diagnostic numbers, not default-configuration results, and NumPyro rows
below run its default init.

| framework | chains | correct | min bulk ESS/s | sampling s | warmup s | div rate |
|---|---|---|---|---|---|---|
| numpyro-vectorized (f32) | 64 | PASS | 19,938 ± 1,392 | 1.21 | 1.11 | 0 |
| numpyro-vectorized (f32) | 512 | PASS | 42,069 ± 1,762 | 4.76 | 2.94 | 0 |
| numpyro-vectorized (f32) | 4096 | PASS | 9,188 | 171 | 68.9 | 0 |
| uncertaintea-batched-cpu-ka-pinned-init (f64) | 64 | PASS | 4,247 ± 170 | 3.26 | 1.67 | 0 |
| uncertaintea-batched-cpu-ka-pinned-init (f64) | 512 | PASS | 3,910 ± 360 | 27.7 | 11.9 | 0 |
| uncertaintea-batched-cpu-ka-pinned-init (f64) | 4096 | PASS | 2,975 | 281 | 111 | 0 |
| uncertaintea-batched-metal-pinned-init (f32) | 64 | PASS | 1,418 ± 25 | 9.57 | 6.08 | 0 |
| uncertaintea-batched-metal-pinned-init (f32) | 512 | PASS | 3,064 ± 160 | 35 | 15.6 | 0 |
| uncertaintea-batched-metal-pinned-init (f32) | 4096 | PASS | 3,648 | 228 | 97.2 | 0 |

(`batched-cpu-ka` = masked tree strategy on the KernelAbstractions CPU
backend with `julia -t auto`, i.e. all 10 cores — the fair CPU counterpart
to the Metal leg. The single-threaded `batched-cpu` hybrid path is 5–8×
slower than `-ka` and its 16384-chain point would take hours, so the sweep
caps at 4096.)

Full tables including the FAIL rows: `bench/crossppl/results/summary.md`
(regenerated by `./run_all.sh analyze`; raw per-run draws and timings are
gitignored but reproducible).

## Models

| model | shape | exercises |
|---|---|---|
| `eight_schools_centered` | hierarchical, funnel | divergence behaviour, gate honesty |
| `eight_schools_noncentered` | hierarchical | `reparam=:noncentered`, `iid` latents |
| `logistic` | GLM, N=500, D=8 | loop-addressed discrete observations |
| `gauss` | mean/scale, N=1000 | device path; chain-count scaling sweep |

Identical joint densities across frameworks; priors in
`bench/crossppl/julia/models.jl` and `bench/crossppl/python/stan/*.stan`.
A discrete-latent model (`marginalize=:enumerate` vs Stan's
hand-marginalization) and an `lkjcholesky` model are planned additions;
the scaling model is a loop-addressed gaussian rather than a regression
because of #134/#135.

## Methodology notes

- Sampling time is isolated from warmup per framework: UncertainTea runs
  (warmup, 1 draw) and (warmup, S draws) from the same RNG seed (identical
  warmup trajectories) and differences them; NumPyro's `warmup()`/`run()`
  are timed separately; CmdStan's own per-chain elapsed report is used (max
  over concurrently running chains).
- All checked-in numbers are native macOS, one machine, so the GPU/CPU
  crossover axis is not distorted by a VM. `bench/crossppl/docker/`
  provides a single pinned Linux environment for portable CPU-only reruns.
- Precision differs by leg and is disclosed per row (Metal requires f32;
  the NumPyro sweep matches it; everything else is f64).

## Refresh procedure

```bash
cd bench/crossppl
./run_all.sh cpu && ./run_all.sh metal && ./run_all.sh pinned && ./run_all.sh analyze
```

Update this document from `results/summary.md` with the date, hardware, and
commit; keep the correctness-gate framing intact.
