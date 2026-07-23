# Cross-PPL benchmark summary

Common diagnostics: ArviZ rank-normalized split R-hat and bulk/tail ESS computed identically for every framework. Correctness gate: R-hat < 1.01, mean/quantile agreement with the Stan reference within 4.0 combined MCSEs.

## eight_schools_centered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | FAIL | 168 ± 88 | 86.1 | 0.605 | 0.724 | 1.43 | 0.021 |
| stan | 4 | 1000 | float64 | FAIL | 5,084 ± 2,586 | 4,363 | 0.0317 | 0.03 | 2.97 | 0.027 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | FAIL | 165 ± 41 | 208 | 0.982 | 1.24 | 6.76 | 0.0049 |
| uncertaintea-cpu | 4 | 1000 | Float64 | FAIL | 156 ± 24 | 74.8 | 0.538 | 1.12 | 5.27 | 0.013 |

Correctness failures (timings above are reported for context but MUST NOT be quoted):
- numpyro-parallel__chains4: mu: rhat=1.0131
- numpyro-parallel__chains4: mu: rhat=1.0620
- numpyro-parallel__chains4: tau: rhat=1.0235
- numpyro-parallel__chains4: tau: rhat=1.0391
- numpyro-parallel__chains4: tau: rhat=1.0991
- numpyro-parallel__chains4: theta[1]: rhat=1.0186
- numpyro-parallel__chains4: theta[2]: rhat=1.0266
- numpyro-parallel__chains4: theta[3]: rhat=1.0339
- numpyro-parallel__chains4: theta[4]: rhat=1.0115
- numpyro-parallel__chains4: theta[4]: rhat=1.0324
- numpyro-parallel__chains4: theta[5]: q95 z=6.8 (11.895 vs ref 10.350)
- numpyro-parallel__chains4: theta[5]: rhat=1.0440
- numpyro-parallel__chains4: theta[6]: rhat=1.0351
- numpyro-parallel__chains4: theta[7]: rhat=1.0209
- numpyro-parallel__chains4: theta[8]: rhat=1.0248
- stan__chains4: mu: rhat=1.0118
- stan__chains4: tau: rhat=1.0131
- stan__chains4: tau: rhat=1.0159
- stan__chains4: tau: rhat=1.0168
- stan__chains4: theta[1]: rhat=1.0115
- stan__chains4: theta[7]: rhat=1.0109
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0147
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0157
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0111
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0227
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0464
- uncertaintea-batched-cpu__chains4: theta[1]: rhat=1.0115
- uncertaintea-batched-cpu__chains4: theta[1]: rhat=1.0292
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0103
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0143
- uncertaintea-batched-cpu__chains4: theta[3]: rhat=1.0125
- uncertaintea-batched-cpu__chains4: theta[4]: rhat=1.0178
- uncertaintea-batched-cpu__chains4: theta[5]: rhat=1.0103
- uncertaintea-batched-cpu__chains4: theta[6]: rhat=1.0102
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0146
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0154
- uncertaintea-batched-cpu__chains4: theta[8]: rhat=1.0168
- uncertaintea-cpu__chains4: mu: rhat=1.0215
- uncertaintea-cpu__chains4: mu: rhat=1.0240
- uncertaintea-cpu__chains4: mu: rhat=1.0246
- uncertaintea-cpu__chains4: tau: rhat=1.0195
- uncertaintea-cpu__chains4: tau: rhat=1.0488
- uncertaintea-cpu__chains4: tau: rhat=1.0692
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0141
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0197
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0434
- uncertaintea-cpu__chains4: theta[2]: rhat=1.0129
- uncertaintea-cpu__chains4: theta[2]: rhat=1.0138
- uncertaintea-cpu__chains4: theta[2]: rhat=1.0143
- uncertaintea-cpu__chains4: theta[3]: rhat=1.0113
- uncertaintea-cpu__chains4: theta[3]: rhat=1.0164
- uncertaintea-cpu__chains4: theta[4]: rhat=1.0101
- uncertaintea-cpu__chains4: theta[4]: rhat=1.0124
- uncertaintea-cpu__chains4: theta[4]: rhat=1.0153
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0124
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0187
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0224
- uncertaintea-cpu__chains4: theta[6]: rhat=1.0102
- uncertaintea-cpu__chains4: theta[6]: rhat=1.0128
- uncertaintea-cpu__chains4: theta[6]: rhat=1.0134
- uncertaintea-cpu__chains4: theta[7]: rhat=1.0185
- uncertaintea-cpu__chains4: theta[7]: rhat=1.0195
- uncertaintea-cpu__chains4: theta[7]: rhat=1.0287
- uncertaintea-cpu__chains4: theta[8]: rhat=1.0106
- uncertaintea-cpu__chains4: theta[8]: rhat=1.0166
- uncertaintea-cpu__chains4: theta[8]: rhat=1.0181

## eight_schools_noncentered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 3,330 ± 3.8e+02 | 2,715 | 0.685 | 0.736 | 1.48 | 0.00017 |
| stan | 4 | 1000 | float64 | PASS | 114,139 ± 13,223 | 93,461 | 0.019 | 0.0117 | 3.51 | 0.00025 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 3,743 ± 1.8e+02 | 3,824 | 0.335 | 0.42 | 6.57 | 0.0013 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 3,354 ± 2.1e+02 | 3,135 | 0.376 | 0.485 | 3.95 | 0.00075 |

## gauss

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 7,676 ± 7.3e+02 | 5,849 | 0.417 | 0.543 | 0.965 | 0 |
| numpyro-vectorized | 64 | 500 | float32 | PASS | 19,938 ± 1,392 | 17,126 | 1.21 | 1.11 | 3.32 | 0 |
| numpyro-vectorized | 512 | 500 | float32 | PASS | 42,069 ± 1,762 | 36,235 | 4.76 | 2.94 | 8.01 | 0 |
| numpyro-vectorized | 4096 | 500 | float32 | PASS | 9,188 ± 0 | 8,032 | 171 | 68.9 | 181 | 0 |
| stan | 4 | 1000 | float64 | PASS | 293,495 ± 22,022 | 200,218 | 0.012 | 0.009 | 0.131 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 848 ± 50 | 759 | 1.85 | 2.02 | 7.51 | 0 |
| uncertaintea-batched-cpu | 64 | 200 | Float64 | FAIL | 38.6 ± 2.9 | 9.94 | 6.89 | 8.33 | 14.9 | 0.057 |
| uncertaintea-batched-cpu | 512 | 200 | Float64 | FAIL | 22.7 ± 3.9 | 7.49 | 72.1 | 68.2 | 161 | 0.07 |
| uncertaintea-batched-cpu | 4096 | 200 | Float64 | FAIL | 17 ± 0 | 5.59 | 763 | 584 | 1,266 | 0.067 |
| uncertaintea-batched-cpu-ka | 64 | 200 | Float64 | FAIL | 171 ± 21 | 42.6 | 1.65 | 1.96 | 7.79 | 0.052 |
| uncertaintea-batched-cpu-ka | 512 | 200 | Float64 | FAIL | 114 ± 15 | 37.1 | 14.5 | 30.3 | 93.1 | 0.067 |
| uncertaintea-batched-cpu-ka | 4096 | 200 | Float64 | FAIL | 143 ± 0 | 47.3 | 90.3 | 99.2 | 198 | 0.068 |
| uncertaintea-batched-cpu-ka-pinned-init | 64 | 500 | Float64 | PASS | 4,247 ± 1.7e+02 | 4,134 | 3.26 | 1.67 | 9.01 | 0 |
| uncertaintea-batched-cpu-ka-pinned-init | 512 | 500 | Float64 | PASS | 3,910 ± 3.6e+02 | 3,834 | 27.7 | 11.9 | 41.9 | 0 |
| uncertaintea-batched-cpu-ka-pinned-init | 4096 | 500 | Float64 | PASS | 2,975 ± 0 | 3,019 | 281 | 111 | 400 | 0 |
| uncertaintea-batched-metal | 64 | 200 | Float32 | FAIL | 64.2 ± 29 | 16.9 | 5.37 | 7.68 | 25.2 | 0.047 |
| uncertaintea-batched-metal | 512 | 200 | Float32 | FAIL | 90.1 ± 12 | 30.2 | 17.7 | 18.7 | 51.6 | 0.071 |
| uncertaintea-batched-metal | 4096 | 200 | Float32 | FAIL | 149 ± 0 | 49.9 | 85.6 | 98.7 | 188 | 0.069 |
| uncertaintea-batched-metal-pinned-init | 64 | 500 | Float32 | PASS | 1,418 ± 25 | 1,385 | 9.57 | 6.08 | 31.8 | 0 |
| uncertaintea-batched-metal-pinned-init | 512 | 500 | Float32 | PASS | 3,064 ± 1.6e+02 | 3,050 | 35 | 15.6 | 64.4 | 0 |
| uncertaintea-batched-metal-pinned-init | 4096 | 500 | Float32 | PASS | 3,648 ± 0 | 3,704 | 228 | 97.2 | 321 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 173 ± 3.2 | 192 | 6.95 | 7.41 | 16.6 | 0 |

Correctness failures (timings above are reported for context but MUST NOT be quoted):
- uncertaintea-batched-cpu__chains64: mu: rhat=1.1457
- uncertaintea-batched-cpu__chains64: mu: rhat=1.1635
- uncertaintea-batched-cpu__chains64: mu: rhat=1.1732
- uncertaintea-batched-cpu__chains64: s: rhat=1.1378
- uncertaintea-batched-cpu__chains64: s: rhat=1.1733
- uncertaintea-batched-cpu__chains64: s: rhat=1.1737
- uncertaintea-batched-cpu__chains512: mu: mean z=4.1 (0.427 vs ref 0.495)
- uncertaintea-batched-cpu__chains512: mu: rhat=1.2123
- uncertaintea-batched-cpu__chains512: mu: rhat=1.2366
- uncertaintea-batched-cpu__chains512: mu: rhat=1.2521
- uncertaintea-batched-cpu__chains512: s: mean z=6.1 (1.127 vs ref 1.184)
- uncertaintea-batched-cpu__chains512: s: mean z=6.3 (1.122 vs ref 1.184)
- uncertaintea-batched-cpu__chains512: s: mean z=6.6 (1.115 vs ref 1.184)
- uncertaintea-batched-cpu__chains512: s: q05 z=14.3 (0.315 vs ref 1.140)
- uncertaintea-batched-cpu__chains512: s: q05 z=5.6 (0.379 vs ref 1.140)
- uncertaintea-batched-cpu__chains512: s: q05 z=9.2 (0.365 vs ref 1.140)
- uncertaintea-batched-cpu__chains512: s: rhat=1.1814
- uncertaintea-batched-cpu__chains512: s: rhat=1.1895
- uncertaintea-batched-cpu__chains512: s: rhat=1.2060
- uncertaintea-batched-cpu__chains4096: mu: mean z=9.5 (0.439 vs ref 0.495)
- uncertaintea-batched-cpu__chains4096: mu: q95 z=4.1 (0.562 vs ref 0.555)
- uncertaintea-batched-cpu__chains4096: mu: rhat=1.2269
- uncertaintea-batched-cpu__chains4096: s: mean z=17.4 (1.124 vs ref 1.184)
- uncertaintea-batched-cpu__chains4096: s: q05 z=36.3 (0.384 vs ref 1.140)
- uncertaintea-batched-cpu__chains4096: s: rhat=1.1872
- uncertaintea-batched-cpu-ka__chains64: mu: rhat=1.1383
- uncertaintea-batched-cpu-ka__chains64: mu: rhat=1.1443
- uncertaintea-batched-cpu-ka__chains64: mu: rhat=1.1603
- uncertaintea-batched-cpu-ka__chains64: s: rhat=1.1362
- uncertaintea-batched-cpu-ka__chains64: s: rhat=1.1375
- uncertaintea-batched-cpu-ka__chains64: s: rhat=1.1705
- uncertaintea-batched-cpu-ka__chains512: mu: mean z=4.1 (0.427 vs ref 0.495)
- uncertaintea-batched-cpu-ka__chains512: mu: rhat=1.2125
- uncertaintea-batched-cpu-ka__chains512: mu: rhat=1.2192
- uncertaintea-batched-cpu-ka__chains512: mu: rhat=1.2274
- uncertaintea-batched-cpu-ka__chains512: mu: rhat=1.2509
- uncertaintea-batched-cpu-ka__chains512: s: mean z=6.0 (1.127 vs ref 1.184)
- uncertaintea-batched-cpu-ka__chains512: s: mean z=6.1 (1.125 vs ref 1.184)
- uncertaintea-batched-cpu-ka__chains512: s: mean z=6.1 (1.127 vs ref 1.184)
- uncertaintea-batched-cpu-ka__chains512: s: mean z=6.6 (1.115 vs ref 1.184)
- uncertaintea-batched-cpu-ka__chains512: s: q05 z=14.3 (0.315 vs ref 1.140)
- uncertaintea-batched-cpu-ka__chains512: s: q05 z=5.6 (0.379 vs ref 1.140)
- uncertaintea-batched-cpu-ka__chains512: s: q05 z=7.0 (0.361 vs ref 1.140)
- uncertaintea-batched-cpu-ka__chains512: s: q05 z=7.2 (0.369 vs ref 1.140)
- uncertaintea-batched-cpu-ka__chains512: s: rhat=1.1763
- uncertaintea-batched-cpu-ka__chains512: s: rhat=1.1800
- uncertaintea-batched-cpu-ka__chains512: s: rhat=1.1808
- uncertaintea-batched-cpu-ka__chains512: s: rhat=1.2062
- uncertaintea-batched-cpu-ka__chains4096: mu: mean z=9.6 (0.438 vs ref 0.495)
- uncertaintea-batched-cpu-ka__chains4096: mu: q95 z=4.1 (0.562 vs ref 0.555)
- uncertaintea-batched-cpu-ka__chains4096: mu: rhat=1.2285
- uncertaintea-batched-cpu-ka__chains4096: s: mean z=17.5 (1.123 vs ref 1.184)
- uncertaintea-batched-cpu-ka__chains4096: s: q05 z=36.1 (0.382 vs ref 1.140)
- uncertaintea-batched-cpu-ka__chains4096: s: rhat=1.1887
- uncertaintea-batched-metal__chains64: mu: rhat=1.1085
- uncertaintea-batched-metal__chains64: mu: rhat=1.1422
- uncertaintea-batched-metal__chains64: mu: rhat=1.1649
- uncertaintea-batched-metal__chains64: s: rhat=1.1018
- uncertaintea-batched-metal__chains64: s: rhat=1.1371
- uncertaintea-batched-metal__chains64: s: rhat=1.1731
- uncertaintea-batched-metal__chains512: mu: mean z=4.3 (0.420 vs ref 0.495)
- uncertaintea-batched-metal__chains512: mu: rhat=1.2124
- uncertaintea-batched-metal__chains512: mu: rhat=1.2369
- uncertaintea-batched-metal__chains512: mu: rhat=1.2616
- uncertaintea-batched-metal__chains512: s: mean z=6.1 (1.127 vs ref 1.184)
- uncertaintea-batched-metal__chains512: s: mean z=6.3 (1.122 vs ref 1.184)
- uncertaintea-batched-metal__chains512: s: mean z=6.8 (1.112 vs ref 1.184)
- uncertaintea-batched-metal__chains512: s: q05 z=14.3 (0.315 vs ref 1.140)
- uncertaintea-batched-metal__chains512: s: q05 z=5.6 (0.379 vs ref 1.140)
- uncertaintea-batched-metal__chains512: s: q05 z=9.2 (0.365 vs ref 1.140)
- uncertaintea-batched-metal__chains512: s: rhat=1.1810
- uncertaintea-batched-metal__chains512: s: rhat=1.1888
- uncertaintea-batched-metal__chains512: s: rhat=1.2154
- uncertaintea-batched-metal__chains4096: mu: mean z=9.8 (0.437 vs ref 0.495)
- uncertaintea-batched-metal__chains4096: mu: q95 z=4.1 (0.561 vs ref 0.555)
- uncertaintea-batched-metal__chains4096: mu: rhat=1.2317
- uncertaintea-batched-metal__chains4096: s: mean z=17.6 (1.123 vs ref 1.184)
- uncertaintea-batched-metal__chains4096: s: q05 z=36.6 (0.383 vs ref 1.140)
- uncertaintea-batched-metal__chains4096: s: rhat=1.1904

## logistic

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 9,804 ± 5.9e+02 | 5,290 | 0.523 | 0.64 | 1.31 | 0 |
| stan | 4 | 1000 | float64 | PASS | 56,304 ± 3,040 | 30,727 | 0.088 | 0.0817 | 3.76 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | FAIL | 50.2 ± 4.9 | 50.8 | 34.1 | 29 | 63.1 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 83 ± 6 | 73.9 | 24.5 | 24.2 | 51.5 | 0 |

Correctness failures (timings above are reported for context but MUST NOT be quoted):
- uncertaintea-batched-cpu__chains4: alpha: q95 z=4.1 (0.607 vs ref 0.580)
