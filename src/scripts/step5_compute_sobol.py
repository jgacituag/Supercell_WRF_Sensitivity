#!/usr/bin/env python
"""
Step 5: Compute Sobol indices for a chosen metric (config-driven)

What it does
------------
- Reads experiment_config.yaml to get EXPERIMENT_DIR and sobol settings.
- Loads the Sobol "problem" and the Saltelli-ordered param_values.
- Builds the Y vector (in Saltelli order) from per-sample metrics.pkl.
  * If a sample is missing, or not in viable_sample_ids.npy (when present),
    it assigns a penalty value instead of dropping the sample.
- Runs SALib's sobol.analyze (S1, ST, optional S2) with bootstrap CIs.
- Performs a convergence check using progressive truncation (e.g., 1/4, 1/2, 3/4, full).
- Estimates a rough "N needed" to reach target CI widths by ~1/sqrt(N) scaling.
- Saves CSVs, a human-readable summary, and a small JSON.

Usage
-----
  python src/scripts/step5_compute_sobol.py --config configs/experiment_config.yaml

"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import pickle
import csv

from SALib.analyze import sobol

# ------------------------- Config handling -------------------------

@dataclass
class BootstrapCfg:
    n_resamples: int = 1000
    ci_level: float = 0.95

@dataclass
class SobolJobCfg:
    enabled: bool = True
    metric: str = "updraft_p99"
    fail_strategy: str = "penalize"  # "penalize" or "skip"
    penalty_value: float = -1.0e9
    calc_second_order: bool = True
    bootstrap: BootstrapCfg = field(default_factory=BootstrapCfg)
    convergence_check: bool = True
    convergence_threshold: float = 0.05
    n_blocks: int = 4

@dataclass
class PathsCfg:
    EXPERIMENT_DIR: str = "outputs/sobol_exp_default"

def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_cfg(cfg_path: Path) -> Tuple[PathsCfg, SobolJobCfg]:
    data = yaml.safe_load(Path(cfg_path).read_text())

    paths = PathsCfg(
        EXPERIMENT_DIR=_safe_get(data, ["PATHS", "EXPERIMENT_DIR"], "outputs/sobol_exp_default")
    )

    # Prefer "sobol_job", else "SOBOL"
    base = _safe_get(data, ["sobol_job"], None)
    if base is None:
        base = _safe_get(data, ["SOBOL"], {})

    # Bootstrap
    b = BootstrapCfg(
        n_resamples=int(_safe_get(base, ["bootstrap", "n_resamples"], 1000)),
        ci_level=float(_safe_get(base, ["bootstrap", "ci_level"], 0.95)),
    )

    # Allow penalty fallback from SALTELLI.penalty_value
    penalty_fallback = _safe_get(data, ["SALTELLI", "penalty_value"], -1.0e9)

    sj = SobolJobCfg(
        enabled=bool(_safe_get(base, ["enabled"], True)),
        metric=str(_safe_get(base, ["metric"], "updraft_p99")),
        fail_strategy=str(_safe_get(base, ["fail_strategy"], "penalize")).lower(),
        penalty_value=float(_safe_get(base, ["penalty_value"], penalty_fallback)),
        calc_second_order=bool(_safe_get(base, ["calc_second_order"], False)),
        bootstrap=b,
        convergence_check=bool(_safe_get(base, ["convergence_check"], True)),
        convergence_threshold=float(_safe_get(base, ["convergence_threshold"], 0.05)),
        n_blocks=int(_safe_get(base, ["n_blocks"], 4)),
    )

    return paths, sj

# ------------------------- I/O helpers -------------------------

def load_problem(exp_dir: Path):
    with open(exp_dir / "problem.pkl", "rb") as f:
        problem = pickle.load(f)
    return problem

def load_param_values(exp_dir: Path) -> np.ndarray:
    return np.load(exp_dir / "param_values.npy")

def load_viable_ids(exp_dir: Path) -> Optional[np.ndarray]:
    path = exp_dir / "soundings" / "viable_sample_ids.npy"
    if path.exists():
        return np.load(path)
    return None

def load_metric_for_sample(sample_dir: Path, metric: str) -> Optional[float]:
    pkl = sample_dir / "metrics.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            m = pickle.load(f)
        if metric in m and m[metric] is not None:
            return float(m[metric])
    except Exception:
        pass
    return None

# ------------------------- Build Y vector -------------------------

def build_Y(exp_dir: Path,
            metric: str,
            fail_strategy: str,
            penalty_value: float,
            viable_ids: Optional[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      Y: np.ndarray (length must equal N*(2D+2))
      n_missing: count of samples with no metric.pkl
      n_penalized: count penalized due to non-viable or missing/None
    """
    X = load_param_values(exp_dir)
    L = len(X)
    Y = np.empty(L, dtype=float)

    viable_mask = None
    if viable_ids is not None:
        viable_mask = np.zeros(L, dtype=bool)
        viable_mask[viable_ids] = True

    n_missing = 0
    n_penalized = 0

    for i in range(L):
        sdir = exp_dir / "wrf_results" / f"sample_{i:05d}"
        val = load_metric_for_sample(sdir, metric)
        is_viable = True if viable_mask is None else bool(viable_mask[i])

        if val is None:
            n_missing += 1
            if fail_strategy == "penalize":
                Y[i] = penalty_value
                n_penalized += 1
            else:
                # skip not allowed for Sobol (would break design)
                Y[i] = penalty_value
                n_penalized += 1
        else:
            if not is_viable:
                if fail_strategy == "penalize":
                    Y[i] = penalty_value
                    n_penalized += 1
                else:
                    Y[i] = penalty_value
                    n_penalized += 1
            else:
                Y[i] = val
    #print(f"Building Y vector for metric '{metric}' with {L} samples...")
    #print(f"  Missing samples: {n_missing}")
    #print(f"  Penalized samples: {n_penalized}")
    #print(f"  Final Y vector length: {len(Y)}")
    return Y, n_missing, n_penalized

# ------------------------- Convergence utilities -------------------------

def truncate_to_blocks_saltelli(Y: np.ndarray, D: int, calc2: bool, blocks: int) -> List[np.ndarray]:
    """
    Return progressive truncations that each keep an integer number of Saltelli base blocks.
    Each truncation has length Lb = Nb * stride, where stride = (2D+2) if calc2 else (D+2),
    and Nb = floor(N_base * frac).
    """
    stride = (2 * D + 2) if calc2 else (D + 2)
    N_base = len(Y) // stride
    outs = []
    for b in range(1, blocks + 1):
        frac = b / blocks
        Nb = max(1, int(math.floor(N_base * frac)))
        Lb = Nb * stride
        outs.append(Y[:Lb].copy())
    return outs

def rel_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative change |a-b| / max(|a|, eps)."""
    eps = 1e-12
    return np.abs(a - b) / np.maximum(np.abs(a), eps)

def suggest_extra_N(ci_width_full: np.ndarray, target_width: float, N_full: int) -> int:
    """
    Assume CI width ~ c / sqrt(N). Then N_target = N_full * (ci_width_full / target_width)^2.
    Return the max integer across parameters.
    """
    ci = np.asarray(ci_width_full)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(target_width > 0, ci / target_width, np.inf)
        N_target = (ratio ** 2) * N_full
    N_needed = int(np.ceil(np.nanmax(N_target)))
    return N_needed

# ------------------------- Main logic -------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute Sobol indices with CIs and convergence check.")
    ap.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    ap.add_argument("--metric", default=None, help="Override metric key (else read from config)")
    ap.add_argument("--outdir", default=None, help="Override output dir (else EXPERIMENT_DIR/sobol/<metric>)")
    args = ap.parse_args()

    paths, job = load_cfg(Path(args.config))
    if not job.enabled:
        print("[sobol] sobol_job.enabled is False — nothing to do.")
        return

    exp_dir = Path(paths.EXPERIMENT_DIR).resolve()
    problem = load_problem(exp_dir)
    X = load_param_values(exp_dir)
    D = int(problem.get("num_vars", len(problem.get("names", []))))
    L = len(X)

    metric = args.metric if args.metric else job.metric
    viable_ids = load_viable_ids(exp_dir)

    Y, n_missing, n_penalized = build_Y(
        exp_dir, metric, job.fail_strategy, job.penalty_value, viable_ids
    )

    # SALib sobol.analyze expects outputs in the same Saltelli order as inputs.
    # Compute base indices + CIs (SALib bootstrap)
    #print(len(Y))
    #print(problem)
    Si = sobol.analyze(
        problem, Y,
        print_to_console=False,
        calc_second_order=True,
        num_resamples=int(job.bootstrap.n_resamples),
        conf_level=float(job.bootstrap.ci_level)
    )

    # Prepare output dir
    outdir = Path(args.outdir) if args.outdir else (exp_dir / "sobol" / metric)
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(problem["names"])

    # Save primary CSVs
    with open(outdir / "sobol_indices.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["param", "S1", "ST"]
        if "S2" in Si:
            header += ["S2_json"]
        w.writerow(header)
        for i, n in enumerate(names):
            row = [n, Si["S1"][i], Si["ST"][i]]
            if "S2" in Si:
                # store S2 row for param i as JSON (pairwise vector)
                row.append(json.dumps([float(x) if np.isfinite(x) else None for x in Si["S2"][i]]))
            w.writerow(row)

    with open(outdir / "sobol_confidence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "S1_conf", "ST_conf"])
        for i, n in enumerate(names):
            w.writerow([n, Si["S1_conf"][i], Si["ST_conf"][i]])

    # Convergence check (progressive truncations)
    conv_report = {}
    if job.convergence_check and job.n_blocks >= 2:
        truncs = truncate_to_blocks_saltelli(Y, D, True, job.n_blocks)
        indices_by_block = []
        for Yb in truncs:
            Sib = sobol.analyze(
                problem, Yb,
                calc_second_order=True,
                print_to_console=False
            )
            indices_by_block.append(Sib)

        S1_full = np.array(Si["S1"])
        ST_full = np.array(Si["ST"])
        changes = []
        for k, Sib in enumerate(indices_by_block[:-1], start=1):  # compare each block vs full
            rc1 = rel_change(S1_full, np.array(Sib["S1"]))
            rct = rel_change(ST_full, np.array(Sib["ST"]))
            changes.append({"block": k, "S1_rel_change": rc1.tolist(), "ST_rel_change": rct.tolist()})

        max_rc1 = np.max([np.array(c["S1_rel_change"]) for c in changes], axis=0)
        max_rct = np.max([np.array(c["ST_rel_change"]) for c in changes], axis=0)
        conv_report = {
            "blocks": job.n_blocks,
            "threshold": job.convergence_threshold,
            "max_rel_change_S1": max_rc1.tolist(),
            "max_rel_change_ST": max_rct.tolist(),
            "per_block": changes,
        }

    # Simple N-needed suggestion based on CI width scaling
    # width ≈ 2 * conf for SALib (conf is half-width), so we'll use half-width arrays directly.
    target = float(job.convergence_threshold)  # reuse threshold as a 'target half-width' if you like
    N_full = len(Y)
    N_suggest_S1 = suggest_extra_N(np.array(Si["S1_conf"]), target, N_full)
    N_suggest_ST = suggest_extra_N(np.array(Si["ST_conf"]), target, N_full)
    N_suggest = int(max(N_suggest_S1, N_suggest_ST))

    # Save JSON bundle
    bundle = {
        "metric": metric,
        "N_total": int(L),
        "N_used": int(len(Y)),
        "D": int(D),
        "penalty_value": job.penalty_value,
        "n_missing": int(n_missing),
        "n_penalized": int(n_penalized),
        "conf_level": float(job.bootstrap.ci_level),
        "num_resamples": int(job.bootstrap.n_resamples),
        "calc_second_order": bool(job.calc_second_order),
        "S1": np.array(Si["S1"]).tolist(),
        "ST": np.array(Si["ST"]).tolist(),
        "S1_conf": np.array(Si["S1_conf"]).tolist(),
        "ST_conf": np.array(Si["ST_conf"]).tolist(),
        "names": names,
        "convergence": conv_report,
        "N_suggest_for_target_halfwidth": {
            "target_halfwidth": target,
            "N_suggest_max_over_params": N_suggest
        }
    }
    if "S2" in Si:
        bundle["S2"] = np.array(Si["S2"]).tolist()

    with open(outdir / "sobol_results.json", "w") as f:
        json.dump(bundle, f, indent=2)

    # Save a human-readable summary
    with open(outdir / "sobol_summary.txt", "w") as f:
        f.write(f"Sobol summary for metric: {metric}\n")
        f.write(f"Samples (Saltelli rows): {len(Y)}  |  Params: D={D}\n")
        f.write(f"Penalty value: {job.penalty_value}  |  Missing: {n_missing}  |  Penalized: {n_penalized}\n")
        f.write(f"Confidence: {job.bootstrap.ci_level:.2f} with {job.bootstrap.n_resamples} resamples\n")
        f.write("\nIndices (S1, ST) ± half-width:\n")
        for i, n in enumerate(names):
            s1, st = Si['S1'][i], Si['ST'][i]
            c1, ct = Si['S1_conf'][i], Si['ST_conf'][i]
            f.write(f"  {n:20s}  S1={s1:7.4f} ±{c1:6.4f}   ST={st:7.4f} ±{ct:6.4f}\n")

        if job.convergence_check and conv_report:
            f.write("\nConvergence (max relative change vs full across blocks):\n")
            thr = job.convergence_threshold
            max_rc1 = np.array(conv_report["max_rel_change_S1"])
            max_rct = np.array(conv_report["max_rel_change_ST"])
            for i, n in enumerate(names):
                flag1 = "✓" if max_rc1[i] <= thr else "✗"
                flagt = "✓" if max_rct[i] <= thr else "✗"
                f.write(f"  {n:20s}  S1 Δrel={max_rc1[i]:.3f} {flag1}   ST Δrel={max_rct[i]:.3f} {flagt}\n")

        f.write("\nN-needed estimate (target half-width ~ {:.3f}): {}\n".format(
            target, bundle["N_suggest_for_target_halfwidth"]["N_suggest_max_over_params"]
        ))

    print(f"[sobol] Saved results under: {outdir}")

if __name__ == "__main__":
    main()
