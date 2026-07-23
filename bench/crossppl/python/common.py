"""Shared I/O helpers for the cross-PPL benchmark runners (issue #121).

Output contract (one pair of files per repetition, shared with julia/run.jl):
  results/raw/<model>__<framework>[-<variant>]__chains<K>__seed<S>__rep<R>.npz
      "draws": float64 array, shape (chains, draws, params)
  ... and a .json sidecar: parameter names (canonical, 1-based brackets for
  vector components), timings, sampler settings, environment provenance.
"""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
RESULTS_DIR = HERE.parent / "results" / "raw"


def load_data(name: str) -> dict:
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def flatten_params(sites: dict[str, np.ndarray], order: list[str]):
    """Flatten NumPyro-style site draws into (chains, draws, params).

    `sites` maps site name -> array of shape (chains, draws) or
    (chains, draws, k); vector sites expand to `name[1]`..`name[k]`.
    """
    names, columns = [], []
    for site in order:
        arr = np.asarray(sites[site])
        if arr.ndim == 2:
            names.append(site)
            columns.append(arr)
        elif arr.ndim == 3:
            for j in range(arr.shape[2]):
                names.append(f"{site}[{j + 1}]")
                columns.append(arr[:, :, j])
        else:
            raise ValueError(f"site {site}: unsupported ndim {arr.ndim}")
    draws = np.stack(columns, axis=-1).astype(np.float64)
    return names, draws


def write_result(*, model: str, framework: str, variant: str | None, params: list[str],
                 draws: np.ndarray, num_chains: int, num_samples: int, num_warmup: int,
                 seed: int, rep: int, timings: dict, sampler: dict, env: dict,
                 diagnostics: dict | None = None) -> str:
    assert draws.shape == (num_chains, num_samples, len(params)), draws.shape
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fw = framework if variant is None else f"{framework}-{variant}"
    stem = f"{model}__{fw}__chains{num_chains}__seed{seed}__rep{rep}"
    np.savez(RESULTS_DIR / f"{stem}.npz", draws=draws)
    meta = {
        "model": model,
        "framework": framework,
        "variant": variant,
        "params": params,
        "num_chains": num_chains,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "seed": seed,
        "rep": rep,
        "timings": timings,
        "sampler": sampler,
        "diagnostics": diagnostics or {},
        "env": {"python": platform.python_version(),
                "machine": platform.machine(), **env},
    }
    with open(RESULTS_DIR / f"{stem}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return stem
