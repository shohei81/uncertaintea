"""Shared diagnostics + results tables for the cross-PPL benchmark (issue #121).

Reads every run under results/raw/ (npz + json sidecar, any framework) and
computes ONE common set of diagnostics with ArviZ — rank-normalized split
R-hat and bulk/tail ESS (Vehtari et al. 2021, the Stan/posterior definition)
— instead of trusting each framework's built-in numbers.

Correctness gate (per model, all runs vs. the Stan reference):
  - every parameter's rank-normalized split R-hat < 1.01
  - posterior mean agreement: |mean_a - mean_ref| / sqrt(mcse_a^2 + mcse_ref^2) < 4
  - 5% / 95% quantile agreement within 4 combined quantile-MCSEs
Timings are only reported for runs that pass; failures are listed explicitly.

Primary metric: min-over-parameters bulk-ESS per second of sampling time
(warmup and compile/TTFX excluded, reported separately).  Tail ESS/s is
reported alongside.  Repetitions are aggregated as mean +- std.

Usage (from bench/crossppl):
    uv run --project python python/analyze.py [--out results]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent / "results" / "raw"

RHAT_MAX = 1.01
Z_MAX = 4.0
REFERENCE_FRAMEWORK = "stan"


def load_runs():
    runs = []
    for meta_path in sorted(RAW_DIR.glob("*.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        npz = np.load(meta_path.with_suffix(".npz"))
        meta["draws"] = npz["draws"]  # (chains, draws, params)
        meta["label"] = meta["framework"] if not meta.get("variant") else \
            f'{meta["framework"]}-{meta["variant"]}'
        runs.append(meta)
    return runs


def diagnostics(run):
    """Common ArviZ diagnostics for one run: per-param stats dict."""
    draws = run["draws"]
    idata = az.from_dict(
        posterior={p: draws[:, :, i] for i, p in enumerate(run["params"])})
    out = {}
    for i, p in enumerate(run["params"]):
        x = draws[:, :, i]
        da = idata.posterior[p]
        mean = float(x.mean())
        ess_bulk = float(az.ess(da, method="bulk")[p].values)
        ess_tail = float(az.ess(da, method="tail")[p].values)
        mcse_mean = float(az.mcse(da, method="mean")[p].values)
        mcse_q05 = float(az.mcse(da, method="quantile", prob=0.05)[p].values)
        mcse_q95 = float(az.mcse(da, method="quantile", prob=0.95)[p].values)
        rhat = float(az.rhat(da, method="rank")[p].values) if x.shape[0] >= 2 else np.nan
        out[p] = {
            "mean": mean, "sd": float(x.std(ddof=1)),
            "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
            "rhat": rhat, "ess_bulk": ess_bulk, "ess_tail": ess_tail,
            "mcse_mean": mcse_mean, "mcse_q05": mcse_q05, "mcse_q95": mcse_q95,
        }
    return out


def gate(run, stats, ref_stats):
    """Correctness gate; returns (passed, list of failure strings)."""
    failures = []
    for p, s in stats.items():
        if not np.isnan(s["rhat"]) and s["rhat"] >= RHAT_MAX:
            failures.append(f'{p}: rhat={s["rhat"]:.4f}')
    if ref_stats is not None:
        for p, s in stats.items():
            if p not in ref_stats:
                failures.append(f"{p}: missing in reference")
                continue
            r = ref_stats[p]
            for key, mkey in (("mean", "mcse_mean"), ("q05", "mcse_q05"), ("q95", "mcse_q95")):
                denom = np.sqrt(s[mkey] ** 2 + r[mkey] ** 2)
                if denom == 0:
                    continue
                z = abs(s[key] - r[key]) / denom
                if z >= Z_MAX:
                    failures.append(
                        f'{p}: {key} z={z:.1f} ({s[key]:.3f} vs ref {r[key]:.3f})')
    return (not failures, failures)


def aggregate(rows):
    """mean +- std over repetitions of one (model, label, chains) config."""
    def ms(key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    out = {"reps": len(rows)}
    for key in ("ess_bulk_per_s", "ess_tail_per_s", "min_ess_bulk", "min_ess_tail",
                "sampling_s", "warmup_s", "divergence_rate"):
        out[key], out[key + "_std"] = ms(key)
    out["ttfx_s"] = max((r["ttfx_s"] or 0.0) for r in rows) or None
    out["compile_s"] = max((r["compile_s"] or 0.0) for r in rows) or None
    out["all_passed"] = all(r["passed"] for r in rows)
    out["failures"] = sorted({f for r in rows for f in r["failures"]})
    return out


def fmt(x, digits=3):
    if x is None:
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    if not runs:
        raise SystemExit(f"no runs found under {RAW_DIR}")

    # Reference stats per model: the Stan run's first repetition.
    ref_stats = {}
    for run in runs:
        if run["framework"] == REFERENCE_FRAMEWORK and run["model"] not in ref_stats:
            ref_stats[run["model"]] = diagnostics(run)

    per_config = defaultdict(list)
    for run in runs:
        stats = diagnostics(run)
        is_ref = run["framework"] == REFERENCE_FRAMEWORK
        passed, failures = gate(run, stats, None if is_ref else ref_stats.get(run["model"]))
        sampling_s = run["timings"].get("sampling_s")
        min_bulk = min(s["ess_bulk"] for s in stats.values())
        min_tail = min(s["ess_tail"] for s in stats.values())
        per_config[(run["model"], run["label"], run["num_chains"])].append({
            "min_ess_bulk": min_bulk,
            "min_ess_tail": min_tail,
            "ess_bulk_per_s": min_bulk / sampling_s if sampling_s else None,
            "ess_tail_per_s": min_tail / sampling_s if sampling_s else None,
            "sampling_s": sampling_s,
            "warmup_s": run["timings"].get("warmup_s"),
            "ttfx_s": run["timings"].get("ttfx_s"),
            "compile_s": run["timings"].get("compile_s"),
            "divergence_rate": (run.get("diagnostics") or {}).get("divergence_rate"),
            "passed": passed,
            "failures": failures,
            "precision": run["sampler"].get("precision"),
            "num_samples": run["num_samples"],
            "num_warmup": run["num_warmup"],
        })

    summary = {}
    for (model, label, chains), rows in sorted(per_config.items()):
        agg = aggregate(rows)
        agg["precision"] = rows[0]["precision"]
        agg["num_samples"] = rows[0]["num_samples"]
        agg["num_warmup"] = rows[0]["num_warmup"]
        summary.setdefault(model, {})[f"{label}__chains{chains}"] = agg

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["# Cross-PPL benchmark summary", ""]
    lines.append("Common diagnostics: ArviZ rank-normalized split R-hat and "
                 "bulk/tail ESS computed identically for every framework. "
                 f"Correctness gate: R-hat < {RHAT_MAX}, mean/quantile agreement "
                 f"with the Stan reference within {Z_MAX} combined MCSEs.")
    lines.append("")
    for model, configs in summary.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| framework | chains | draws/chain | precision | correct | "
                     "min bulk ESS/s | min tail ESS/s | sampling s | warmup s | "
                     "TTFX/compile s | div rate |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for key, a in configs.items():
            label, chains = key.rsplit("__chains", 1)
            correct = "PASS" if a["all_passed"] else "FAIL"
            ttfx = a["ttfx_s"] if a["ttfx_s"] is not None else a["compile_s"]
            lines.append(
                f'| {label} | {chains} | {a["num_samples"]} | {a["precision"]} '
                f'| {correct} | {fmt(a["ess_bulk_per_s"])} ± {fmt(a["ess_bulk_per_s_std"], 2)} '
                f'| {fmt(a["ess_tail_per_s"])} | {fmt(a["sampling_s"])} '
                f'| {fmt(a["warmup_s"])} | {fmt(ttfx)} | {fmt(a["divergence_rate"], 2)} |')
        failed = {k: a for k, a in configs.items() if not a["all_passed"]}
        if failed:
            lines.append("")
            lines.append("Correctness failures (timings above are reported for "
                         "context but MUST NOT be quoted):")
            for key, a in failed.items():
                for f_ in a["failures"]:
                    lines.append(f"- {key}: {f_}")
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines))
    print(f"wrote {outdir / 'summary.json'} and {outdir / 'summary.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
