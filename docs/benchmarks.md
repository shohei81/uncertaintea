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

## Provenance (current)

- **Date:** 2026-07-24 (post-optimization; supersedes the 2026-07-23 baseline
  preserved at the bottom of this file).
- **Hardware:** Apple M4 (10 cores, 32 GB), Metal 4; macOS 26.5.1. All
  frameworks measured natively on this one machine.
- **Software:** Julia 1.12.2, UncertainTea @ `bfca904` (main, after the first
  performance wave: PRs #164/#165/#166/#167/#168/#169/#171/#172/#173/#170),
  NumPyro 0.19 / JAX 0.9 (pinned in `bench/crossppl/python/uv.lock`),
  CmdStan 2.36.0, Python 3.12.
- **Sampler settings:** NUTS everywhere, `target_accept=0.8`,
  `max_tree_depth=10`, diagonal metric, framework-default warmup schedules.
  Correctness pass: 4 chains × 1000 warmup + 1000 draws × 3 repetitions
  (± is std over repetitions). Scaling sweep: 200 warmup + 500 draws at
  64–4096 chains (3 reps ≤512, 1 rep at 4096).
- **Metric:** min-over-parameters bulk ESS per second of pure sampling time.
  Warmup and compile/TTFX are excluded and reported in their own columns.

## Key findings (current)

1. **On the many-chains vectorized workload UncertainTea now leads NumPyro
   and is second only to single-chain Stan.** On the `gauss` scaling sweep the
   host batched CPU backend hits 53,430 bulk ESS/s at 512 chains and 36,879 at
   4096, versus NumPyro-vectorized's 28,378 and 7,687 — all gate-passing. This
   is the payoff of the batched-gradient rework (observed-loop fast path #166,
   sufficient-statistics fusion #146) plus per-chain adaptation becoming the
   host default (#137), which together took the 512-chain leg from a
   gate-failing 72 s to a passing 2.0 s. Only Stan's single-chain C++
   (171,372 ESS/s on this 2-parameter model) is faster.
2. **Metal now overtakes both at 4096 chains.** The device masked path still
   runs a host gradient every iteration (#151, not yet fixed), but that
   gradient inherited the #146 fusion, so Metal's 4096-chain sampling dropped
   from 228 s to 36.3 s → 41,215 ESS/s, ahead of the 10-core CPU backend
   (36,879) and NumPyro (7,687). This is the first honest "GPU-native
   overtakes CPU" crossover in the suite. Fixing #151 (move the per-iteration
   leaf onto the device) should widen it substantially and lower the
   crossover chain count.
3. **Correctness still requires per-chain adaptation, which is default only on
   the host paths.** The stranding failure (#137: prior-draw init + shared
   step size permanently strands ~6% of chains, divergence rate ≈
   P(prior s < 0.4) = 6.2%) is fixed by default on `batched-cpu`, but the
   KernelAbstractions CPU (`batched-cpu-ka`) and Metal paths are treated as
   device backends where per-chain step-size adaptation is still deferred
   (#137 device part) — so their default-config rows FAIL, and the Metal
   scaling numbers above come from the `-pinned-init` diagnostic workaround.
   Warmup cost rose with the per-chain default (#158 tracks recovering it via
   pooled-mass adaptation).
4. **GLMs still trail.** `logistic` batched-cpu passes the gate now (156
   ESS/s, was gate-marginal) but stays far behind NumPyro (4,185) because the
   bernoulli-logit + covariate observation does not lower to the analytic
   batched path yet (#150/#134/#135). Single-chain `logistic` improved 83 →
   406 ESS/s from the interpreter rework (#145).

Open issues from the audit still shaping these numbers: #150/#134/#135 (GLM /
device lowering), #151/#152/#153 (device engineering), #137-device + #158
(per-chain/pooled adaptation on device), #144 (generated type-stable scorer,
the remaining single-chain gap vs Stan).

## Scaling sweep — gauss (mean/scale, N=1000; 200 warmup + 500 draws)

| framework | chains | precision | correct | min bulk ESS/s | sampling s | warmup s | div |
|---|---|---|---|---|---|---|---|
| stan (single chain, 1000 draws) | 4 | f64 | PASS | 171,372 ± 32,742 | 0.021 | 0.015 | 0 |
| uncertaintea-batched-cpu | 4 | f64 | PASS | 38,708 ± 2,955 | 0.05 | 0.086 | 0 |
| uncertaintea-batched-cpu | 64 | f64 | PASS | 57,519 ± 17,018 | 0.26 | 0.62 | 0 |
| uncertaintea-batched-cpu | 512 | f64 | PASS | **53,430 ± 5,336** | 2.05 | 4.68 | 0 |
| uncertaintea-batched-cpu | 4096 | f64 | PASS | **36,879** | 23.2 | 38.7 | 0 |
| numpyro-vectorized | 64 | f32 | PASS | 14,279 ± 1,205 | 1.70 | 1.63 | 0 |
| numpyro-vectorized | 512 | f32 | PASS | 28,378 ± 1,773 | 7.08 | 4.23 | 0 |
| numpyro-vectorized | 4096 | f32 | PASS | 7,687 | 205 | 93.6 | 0 |
| uncertaintea-batched-metal-pinned-init | 64 | f32 | PASS | 2,975 ± 290 | 8.29 | 6.21 | 0 |
| uncertaintea-batched-metal-pinned-init | 512 | f32 | PASS | 14,712 ± 3,959 | 13.9 | 6.32 | 0 |
| uncertaintea-batched-metal-pinned-init | 4096 | f32 | PASS | **41,215** | 36.3 | 13.4 | 0 |
| uncertaintea-batched-cpu-ka | 64 | f64 | FAIL (#137 dev) | 85.8 | 3.1 | 1.48 | 0.057 |
| uncertaintea-batched-cpu-ka | 512 | f64 | FAIL (#137 dev) | 43.2 | 37.3 | 13.5 | 0.07 |
| uncertaintea-batched-metal | 64 | f32 | FAIL (#137 dev) | 9.53 | 29.8 | 15.7 | 0.057 |
| uncertaintea-batched-metal | 512 | f32 | FAIL (#137 dev) | 24.8 | 66.7 | 21.2 | 0.07 |

`-pinned-init` = the #137 diagnostic workaround (every chain initialized at
the posterior mode); those rows are not default-configuration results. The
default-config `-ka` and Metal rows FAIL because per-chain adaptation is not
yet the device default (#137 device part); their timings are context only.
`batched-cpu-ka` is also slow here because the masked KernelAbstractions path
does not use the fused analytic gradient the host `batched-cpu` path got — on
this shape `batched-cpu` now dominates it.

## Correctness pass (4 chains × 1000 warmup + 1000 draws)

### eight_schools_noncentered — all PASS

| framework | min bulk ESS/s | sampling s | div rate |
|---|---|---|---|
| stan | 52,567 ± 3,823 | 0.02 | 0.0002 |
| uncertaintea-cpu | 9,554 ± 2,393 | — | ~0 |
| numpyro-parallel | 1,978 ± 120 | — | ~0 |
| uncertaintea-batched-cpu | 1,640 ± 330 | — | ~0 |

Single-chain UncertainTea improved 3,354 → 9,554 ESS/s from the biased-merge
tree change (#159, +54% ESS/gradient) and the interpreter rework (#145). The
batched path's 4-chain number is lower because per-chain adaptation (now
default) spends more warmup per chain at tiny chain counts — the batched path
is built for the many-chains regime above, not 4 chains.

### logistic (N=500, D=8) — all PASS

| framework | min bulk ESS/s | div rate |
|---|---|---|
| stan | 23,628 ± 1,951 | 0 |
| numpyro-parallel | 4,185 ± 500 | 0 |
| uncertaintea-cpu | 406 ± 32 | 0 |
| uncertaintea-batched-cpu | 156 ± 6 | 0 |

GLM gap persists (no analytic/​device lowering yet — #150/#134/#135);
single-chain improved 83 → 406 from #145.

### eight_schools_centered — all FAIL (funnel; expected)

Every framework, Stan included, exceeds R-hat 1.01 with 2–3% divergences at
`target_accept=0.8` — the canonical centered-parameterization pathology. The
gate rejecting all four implementations equally is evidence it works; this
model stays in the suite as the honesty check.

## What changed since the 2026-07-23 baseline

| leg | 2026-07-23 | 2026-07-24 | driver |
|---|---|---|---|
| gauss batched-cpu 512 chains | 72 s / FAIL | 2.0 s / **PASS**, 53k ESS/s | #146, #166, #137, #142 |
| gauss batched-cpu 4096 chains | 763 s / FAIL | 23.2 s / **PASS**, 37k ESS/s | same |
| gauss Metal 4096 (pinned) | 228 s, 3.6k ESS/s | 36.3 s, **41k ESS/s** | #146 (via the #151 host gradient) |
| gauss cpu (single chain) | 173 ESS/s | 509 ESS/s | #145, #159 |
| eight-schools-nc cpu | 3,354 ESS/s | 9,554 ESS/s | #145, #159 |
| logistic cpu | 83 ESS/s | 406 ESS/s | #145 |

## Models

| model | shape | exercises |
|---|---|---|
| `eight_schools_centered` | hierarchical, funnel | divergence behaviour, gate honesty |
| `eight_schools_noncentered` | hierarchical | `reparam=:noncentered`, `iid` latents |
| `logistic` | GLM, N=500, D=8 | loop-addressed discrete observations |
| `gauss` | mean/scale, N=1000 | device path; chain-count scaling sweep |

Identical joint densities across frameworks; priors in
`bench/crossppl/julia/models.jl` and `bench/crossppl/python/stan/*.stan`. A
discrete-latent model (`marginalize=:enumerate` vs Stan's hand-marginalization)
and an `lkjcholesky` model are planned additions; the scaling model is a
loop-addressed gaussian rather than a regression because of #134/#135.

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

Update the "current" sections above from `results/summary.md` with the date,
hardware, and commit; move the previous numbers into the history table; keep
the correctness-gate framing intact.

---

## Baseline archive — 2026-07-23 (pre-optimization, UncertainTea @ `6df3064`)

The original first-cut measurement, kept for the history table above. At that
commit UncertainTea's batched paths failed the gate at ≥64 chains (#137
undiagnosed as default), the batched CPU gradient re-fetched observations per
call (#138), and Metal ran an unfused host gradient every iteration (#151):
gauss batched-cpu 512 chains took 72 s (FAIL), Metal 4096 chains 228 s at
3,648 ESS/s (pinned), single-chain gauss 173 ESS/s, and NumPyro-vectorized led
the gate-passing scaling sweep at every chain count. See the git history of
this file (`docs/benchmarks.md` at `6df3064`) for the full original tables.
