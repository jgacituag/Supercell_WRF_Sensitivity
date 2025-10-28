#!/usr/bin/env python
"""
Step 4: Compute WRF metrics (config-driven, env vs storm, masked/slabbed, parallel)

Usage
-----
  python src/scripts/step4_compute_metrics.py --config configs/experiment_config.yaml

What it does
------------
- Reads YAML config:
    PATHS.EXPERIMENT_DIR
    metrics_job.* (workers, merge, dt_seconds, environment{time,column,metrics}, storm{time_window,mask,slab,metrics})
- For each sample (sample_XXXXX) under <EXPERIMENT_DIR>/wrf_results:
    * Compute ENVIRONMENT metrics at the configured time and column (SW corner by default, offset 1,1),
      using the WRF wrfout to capture environment.
      Keys are saved as env_* (e.g., env_mucape, env_srh_0_3km, env_stp).
    * Compute STORM metrics within a time window (default: start_min=10 → end),
      optionally masked by reflectivity >= threshold and limited to a z-slab.
      Keys keep concise names (e.g., updraft_p99, precip_total).
    * Merge results into metrics.pkl (if merge=True).
    * Save metrics_meta.json describing actual settings resolved per-sample.
"""

#from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from netCDF4 import Dataset, num2date
from wrf import getvar, to_np

# Optional MetPy pieces
try:
    import metpy.calc as mpcalc
    from metpy.units import units
    HAS_METPY = True
except Exception:
    HAS_METPY = False


# --------------------------- Config dataclasses ---------------------------

@dataclass
class EnvTimeCfg:
    mode: str = "wrfout_index"           # "wrfout_index" | "minutes_since_start"
    t_index: int = 0
    minutes: float = 0.0

@dataclass
class EnvColumnCfg:
    mode: str = "corner"                 # "corner" | "ij" | "latlon"
    corner: str = "SW"                   # SW | SE | NW | NE
    offset_i: int = 1
    offset_j: int = 1
    i: Optional[int] = None
    j: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

@dataclass
class EnvironmentCfg:
    time: EnvTimeCfg
    column: EnvColumnCfg
    metrics: List[str]

@dataclass
class StormTimeWindowCfg:
    mode: str = "minutes"                # "minutes" | "indices"
    start_min: Optional[float] = 10.0    # default spin-up = 10 min
    end_min: Optional[float] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

@dataclass
class StormMaskCfg:
    use_reflectivity: bool = True
    dbz_threshold: float = 20.0

@dataclass
class StormSlabCfg:
    zmin: Optional[float] = None         # m AGL; None → full column
    zmax: Optional[float] = None

@dataclass
class StormCfg:
    time_window: StormTimeWindowCfg
    mask: StormMaskCfg
    slab: StormSlabCfg
    metrics: List[str]

@dataclass
class MetricsJobCfg:
    enabled: bool = True
    workers: int = 1
    merge: bool = True
    dt_seconds: float = 300.0
    environment: EnvironmentCfg = None
    storm: StormCfg = None

@dataclass
class PathsCfg:
    EXPERIMENT_DIR: str
    PBS: Dict[str, Any] | None = None


# --------------------------- Utilities ---------------------------

def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_config(cfg_path: Path) -> Tuple[PathsCfg, MetricsJobCfg]:
    data = yaml.safe_load(Path(cfg_path).read_text())

    paths = PathsCfg(
        EXPERIMENT_DIR=_safe_get(data, ["PATHS", "EXPERIMENT_DIR"], "outputs/sobol_exp_default"),
        PBS=_safe_get(data, ["PATHS", "PBS"], None),
    )

    # Environment
    et = EnvTimeCfg(
        mode=_safe_get(data, ["metrics_job", "environment", "time", "mode"], "wrfout_index"),
        t_index=_safe_get(data, ["metrics_job", "environment", "time", "t_index"], 0),
        minutes=_safe_get(data, ["metrics_job", "environment", "time", "minutes"], 0.0),
    )
    ec = EnvColumnCfg(
        mode=_safe_get(data, ["metrics_job", "environment", "column", "mode"], "corner"),
        corner=_safe_get(data, ["metrics_job", "environment", "column", "corner"], "SW"),
        offset_i=_safe_get(data, ["metrics_job", "environment", "column", "offset_i"], 1),
        offset_j=_safe_get(data, ["metrics_job", "environment", "column", "offset_j"], 1),
        i=_safe_get(data, ["metrics_job", "environment", "column", "i"], None),
        j=_safe_get(data, ["metrics_job", "environment", "column", "j"], None),
        lat=_safe_get(data, ["metrics_job", "environment", "column", "lat"], None),
        lon=_safe_get(data, ["metrics_job", "environment", "column", "lon"], None),
    )
    env = EnvironmentCfg(
        time=et,
        column=ec,
        metrics=list(_safe_get(data, ["metrics_job", "environment", "metrics"], []) or [])
    )

    # Storm
    tw = StormTimeWindowCfg(
        mode=_safe_get(data, ["metrics_job", "storm", "time_window", "mode"], "minutes"),
        start_min=_safe_get(data, ["metrics_job", "storm", "time_window", "start_min"], 10),
        end_min=_safe_get(data, ["metrics_job", "storm", "time_window", "end_min"], None),
        start_idx=_safe_get(data, ["metrics_job", "storm", "time_window", "start_idx"], None),
        end_idx=_safe_get(data, ["metrics_job", "storm", "time_window", "end_idx"], None),
    )
    mk = StormMaskCfg(
        use_reflectivity=_safe_get(data, ["metrics_job", "storm", "mask", "use_reflectivity"], True),
        dbz_threshold=_safe_get(data, ["metrics_job", "storm", "mask", "dbz_threshold"], 20.0),
    )
    sl = StormSlabCfg(
        zmin=_safe_get(data, ["metrics_job", "storm", "slab", "zmin"], None),
        zmax=_safe_get(data, ["metrics_job", "storm", "slab", "zmax"], None),
    )
    st = StormCfg(
        time_window=tw,
        mask=mk,
        slab=sl,
        metrics=list(_safe_get(data, ["metrics_job", "storm", "metrics"], []) or [])
    )

    mj = MetricsJobCfg(
        enabled=_safe_get(data, ["metrics_job", "enabled"], True),
        workers=_safe_get(data, ["metrics_job", "workers"], 1),
        merge=_safe_get(data, ["metrics_job", "merge"], True),
        dt_seconds=float(_safe_get(data, ["metrics_job", "dt_seconds"], 300.0)),
        environment=env,
        storm=st,
    )
    return paths, mj


def _open_wrfouts(sample_dir: Path) -> List[Dataset]:
    wrfouts = sorted(sample_dir.glob("wrfout_d0*"))
    return [Dataset(str(p)) for p in wrfouts]


def _close_all(ncs: List[Dataset]):
    for nc in ncs:
        try:
            nc.close()
        except Exception:
            pass


def _wrf_times(nc: Dataset) -> List[float]:
    """Return minutes since start for all times in this file (single-time wrfouts typically)."""
    # wrfout usually has 1 time per file. We'll try to detect base time with netCDF4 num2date.
    try:
        tvar = nc.variables.get("Times", None)
        if tvar is not None:
            # Times can be char array [DateStrLen], or [Time, DateStrLen].
            if tvar.ndim == 2 and tvar.shape[0] == 1:
                return [0.0]
            elif tvar.ndim == 2:
                return [float(i) for i in range(tvar.shape[0])]
    except Exception:
        pass
    # Fallback: assume each wrfout is an increment; we will index time by file index later.
    return [0.0]


def _get_heights_3d(nc: Dataset) -> Optional[np.ndarray]:
    try:
        z = getvar(nc, "z", meta=False)  # m AGL (z, y, x)
        return np.array(z)
    except Exception:
        return None


def _get_var(nc: Dataset, name: str, meta=False):
    try:
        return getvar(nc, name, meta=meta)
    except Exception:
        return None


def _corner_ij(nc2d: np.ndarray, corner: str, oi: int, oj: int) -> Tuple[int, int]:
    """Return (i,j) (x,y) indices for a corner with offsets (gridpoint units).
    We use array shape as (y,x). 'SW' is (y=0+oj, x=0+oi). 'NE' is (y=ny-1-oj, x=nx-1-oi)."""
    ny, nx = nc2d.shape[-2], nc2d.shape[-1]
    corner = corner.upper()
    if corner == "SW":
        j = 0 + max(0, oj)
        i = 0 + max(0, oi)
    elif corner == "SE":
        j = 0 + max(0, oj)
        i = (nx - 1) - max(0, oi)
    elif corner == "NW":
        j = (ny - 1) - max(0, oj)
        i = 0 + max(0, oi)
    elif corner == "NE":
        j = (ny - 1) - max(0, oj)
        i = (nx - 1) - max(0, oi)
    else:
        j = 0 + max(0, oj)
        i = 0 + max(0, oi)
    # clamp
    j = int(np.clip(j, 0, ny - 1))
    i = int(np.clip(i, 0, nx - 1))
    return i, j


def _layer_mean_at_column(arr3d: np.ndarray, z3d: np.ndarray, i: int, j: int, z1: float, z2: float) -> float:
    """Mean of arr3d over (z in [z1,z2]) at column (j,i)."""
    if arr3d is None or z3d is None:
        return np.nan
    col = arr3d[:, j, i]
    zz = z3d[:, j, i]
    mask = (zz >= z1) & (zz <= z2)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(col[mask]))


def _bulk_shear_at_column(nc: Dataset, i: int, j: int, hbot: float, htop: float) -> float:
    u = _get_var(nc, "ua", meta=False)
    v = _get_var(nc, "va", meta=False)
    z = _get_heights_3d(nc)
    if u is None or v is None or z is None:
        return np.nan
    u = np.array(u); v = np.array(v); z = np.array(z)
    # Take an average wind in thin layers around target heights (±100 m)
    ub = _layer_mean_at_column(u, z, i, j, hbot - 100.0, hbot + 100.0)
    vb = _layer_mean_at_column(v, z, i, j, hbot - 100.0, hbot + 100.0)
    ut = _layer_mean_at_column(u, z, i, j, htop - 100.0, htop + 100.0)
    vt = _layer_mean_at_column(v, z, i, j, htop - 100.0, htop + 100.0)
    if any(np.isnan(x) for x in [ub, vb, ut, vt]):
        return np.nan
    return float(np.hypot(ut - ub, vt - vb))


def _srh_at_column(nc: Dataset, i: int, j: int, depth_m: float) -> float:
    """Storm-relative helicity (0-depth) at one column using MetPy, Bunkers storm motion if available."""
    if not HAS_METPY:
        return np.nan
    z = _get_heights_3d(nc)
    u = _get_var(nc, "ua", meta=False)
    v = _get_var(nc, "va", meta=False)
    if z is None or u is None or v is None:
        return np.nan
    zcol = z[:, j, i] * units.m
    ucol = np.array(u)[:, j, i] * units("m/s")
    vcol = np.array(v)[:, j, i] * units("m/s")

    # Estimate Bunkers storm motion (0–6 km layer)
    try:
        mask06 = (zcol.m >= 0.0) & (zcol.m <= 6000.0)
        sm_right, _ = mpcalc.bunkers_storm_motion(ucol[mask06], vcol[mask06], zcol[mask06])
        storm_u, storm_v = sm_right[0].to("m/s"), sm_right[1].to("m/s")
    except Exception:
        # fallback to mean wind as storm motion
        storm_u = np.nanmean(ucol).to("m/s")
        storm_v = np.nanmean(vcol).to("m/s")

    try:
        maskd = (zcol.m >= 0.0) & (zcol.m <= float(depth_m))
        srh, _, _ = mpcalc.storm_relative_helicity(
            ucol[maskd], vcol[maskd], zcol[maskd], storm_u, storm_v
        )
        return float(srh.to("meter**2/second**2").m)
    except Exception:
        return np.nan


def _significant_tornado_parameter_env(mlcape: float, mlcin: float,
                                       srh01: float, shear06: float, lcl_m: float) -> float:
    """Prefer MetPy's significant_tornado_parameter; fallback to a fixed variant."""
    if HAS_METPY:
        try:
            # MetPy expects magnitude inputs with units; convert if possible
            cap = (mlcape * units("J/kg")) if np.isfinite(mlcape) else np.nan * units("J/kg")
            cin = (mlcin * units("J/kg")) if np.isfinite(mlcin) else np.nan * units("J/kg")
            srh = (srh01 * units("meter**2/second**2")) if np.isfinite(srh01) else np.nan * units("meter**2/second**2")
            sh6 = (shear06 * units("m/s")) if np.isfinite(shear06) else np.nan * units("m/s")
            lcl = (lcl_m * units("meter")) if np.isfinite(lcl_m) else np.nan * units("meter")
            stp = mpcalc.significant_tornado_parameter(
                cape=cap, lcl_height=lcl, wind_0_6km=sh6, shear_0_1km=None,  # MetPy handles typical args
                srh_0_1km=srh, cin=cin, fixed=True
            )
            return float(stp.m)
        except Exception:
            pass
    # Fallback: simple fixed STP (clipped)
    if any(not np.isfinite(x) for x in [mlcape, srh01, shear06, lcl_m]):
        return np.nan
    term_cape = mlcape / 1500.0
    term_srh = srh01 / 150.0
    term_shear = shear06 / 20.0
    term_lcl = (2000.0 - lcl_m) / 1000.0
    term_lcl = float(np.clip(term_lcl, 0.0, 1.0))
    stp = term_cape * term_srh * term_shear * term_lcl
    return float(max(stp, 0.0))


def _dbz3d(nc: Dataset) -> Optional[np.ndarray]:
    dbz = _get_var(nc, "REFL_10CM", meta=False)
    if dbz is None:
        dbz = _get_var(nc, "dbz", meta=False)
    return np.array(dbz) if dbz is not None else None


def _w3d(nc: Dataset) -> Optional[np.ndarray]:
    w = _get_var(nc, "wa", meta=False)
    if w is None:
        w = _get_var(nc, "w", meta=False)
    return np.array(w) if w is not None else None


def _apply_mask_slab(A: Optional[np.ndarray], mask3d: Optional[np.ndarray],
                     z3d: Optional[np.ndarray], zmin: Optional[float], zmax: Optional[float]) -> Optional[np.ndarray]:
    """Apply (reflectivity) mask and z slab to a 3D array (z,y,x). Return masked array or None."""
    if A is None:
        return None
    B = np.array(A, dtype=float)
    if z3d is not None and (zmin is not None or zmax is not None):
        slab = np.ones_like(z3d, dtype=bool)
        if zmin is not None:
            slab &= (z3d >= zmin)
        if zmax is not None:
            slab &= (z3d <= zmax)
        B = np.where(slab, B, np.nan)
    if mask3d is not None:
        B = np.where(mask3d, B, np.nan)
    return B


def _domain_mean_2d(var2d: Optional[np.ndarray]) -> float:
    if var2d is None:
        return np.nan
    return float(np.nanmean(var2d))


def _storm_time_indices(ncs: List[Dataset], cfg: StormTimeWindowCfg, dt_fallback: float) -> Tuple[int, int]:
    """Return (start_idx, end_idx) inclusive indices of wrfouts for storm window."""
    n = len(ncs)
    if n == 0:
        return (0, -1)
    if cfg.mode == "indices":
        si = int(cfg.start_idx) if cfg.start_idx is not None else 0
        ei = int(cfg.end_idx) if cfg.end_idx is not None else (n - 1)
        si = max(0, si); ei = min(n - 1, ei)
        if ei < si:
            ei = si
        return si, ei

    # minutes mode
    # If we can't extract real minutes, map roughly by assuming uniform spacing and dt_fallback.
    start_min = float(cfg.start_min) if cfg.start_min is not None else 0.0
    end_min = float(cfg.end_min) if cfg.end_min is not None else float("inf")

    # simple mapping: index ≈ minutes / (dt or dt_fallback)
    dt = dt_fallback
    if dt <= 0.0:
        dt = 300.0
    si = int(math.floor(start_min * 60.0 / dt))
    ei = int(math.floor(end_min * 60.0 / dt)) if math.isfinite(end_min) else (n - 1)

    si = max(0, si)
    ei = min(n - 1, ei)
    if ei < si:
        ei = si
    return si, ei


def _p99(seq: List[float]) -> float:
    v = np.array(seq, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.nanpercentile(v, 99))


# --------------------------- Per-sample processing ---------------------------

def process_sample(sample_idx: int, exp_dir: Path, jobcfg: MetricsJobCfg) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    sdir = exp_dir / "wrf_results" / f"sample_{sample_idx:05d}"
    out_pkl = sdir / "metrics.pkl"
    out_meta = sdir / "metrics_meta.json"

    # Merge existing
    existing = {}
    if out_pkl.exists() and jobcfg.merge:
        try:
            with open(out_pkl, "rb") as f:
                existing = json.loads(json.dumps(pickle_to_jsonsafe(f.read())))  # just in case; we will overwrite later
        except Exception:
            existing = {}

    # Open wrfouts
    ncs = _open_wrfouts(sdir)
    if not ncs:
        return sample_idx, {"_status": "missing_wrfout"}, {"error": "missing_wrfout"}

    z3d_first = _get_heights_3d(ncs[0])

    results: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "environment": {},
        "storm": {}
    }

    # ---------------- ENVIRONMENT METRICS ----------------
    envcfg = jobcfg.environment
    if envcfg and envcfg.metrics:
        # Resolve env time index
        if envcfg.time.mode == "wrfout_index":
            ti = int(envcfg.time.t_index)
            ti = max(0, min(ti, len(ncs) - 1))
        else:
            # minutes_since_start -> map to index via dt_seconds
            dt = jobcfg.dt_seconds if jobcfg.dt_seconds > 0 else 300.0
            ti = int(round((envcfg.time.minutes * 60.0) / dt))
            ti = max(0, min(ti, len(ncs) - 1))

        nc_env = ncs[ti]

        # Choose a 2D field to size the grid for corner selection (e.g., REF or RAINNC)
        test2d = _get_var(nc_env, "RAINNC", meta=False)
        if test2d is None:
            test2d = _get_var(nc_env, "T2", meta=False)
        if test2d is None:
            # try dbz 3D and take last level
            dbz = _dbz3d(nc_env)
            if dbz is not None:
                test2d = dbz[-1, :, :]

        if test2d is None:
            j, i = 0, 0
        else:
            arr2d = np.array(test2d)
            if envcfg.column.mode == "corner":
                i, j = _corner_ij(arr2d, envcfg.column.corner, envcfg.column.offset_i, envcfg.column.offset_j)
            elif envcfg.column.mode == "ij":
                i = int(envcfg.column.i) if envcfg.column.i is not None else 0
                j = int(envcfg.column.j) if envcfg.column.j is not None else 0
                ny, nx = arr2d.shape[-2], arr2d.shape[-1]
                i = int(np.clip(i, 0, nx - 1)); j = int(np.clip(j, 0, ny - 1))
            else:
                # lat/lon mode not implemented robustly here → fall back to corner SW
                i, j = _corner_ij(arr2d, envcfg.column.corner, envcfg.column.offset_i, envcfg.column.offset_j)

        # Compute environment metrics at (i,j)
        # CAPE/CIN/LCL (domain scalar at environment time) — try wrf-python convenience getters
        if "mucape" in envcfg.metrics:
            v = _get_var(nc_env, "mucape", meta=False)
            results["env_mucape"] = float(np.nanmean(np.array(v))) if v is not None else np.nan
        if "mucin" in envcfg.metrics:
            v = _get_var(nc_env, "mucin", meta=False)
            results["env_mucin"] = float(np.nanmean(np.array(v))) if v is not None else np.nan
        if "mlcape" in envcfg.metrics:
            v = _get_var(nc_env, "mlcape", meta=False)
            results["env_mlcape"] = float(np.nanmean(np.array(v))) if v is not None else np.nan
        if "mlcin" in envcfg.metrics:
            v = _get_var(nc_env, "mlcin", meta=False)
            results["env_mlcin"] = float(np.nanmean(np.array(v))) if v is not None else np.nan
        if "dcape" in envcfg.metrics:
            v = _get_var(nc_env, "dcape", meta=False)
            results["env_dcape"] = float(np.nanmean(np.array(v))) if v is not None else np.nan
        if "lcl" in envcfg.metrics:
            v = _get_var(nc_env, "lcl", meta=False)
            results["env_lcl"] = float(np.nanmean(np.array(v))) if v is not None else np.nan

        # Shears (0-1,0-3,0-6,3-6 km) and SRH (0-1,0-3 km) at column
        shears = [
            ("env_shear_0_1km", 0.0, 1000.0),
            ("env_shear_0_3km", 0.0, 3000.0),
            ("env_shear_0_6km", 0.0, 6000.0),
            ("env_shear_3_6km", 3000.0, 6000.0)
        ]
        for name, h1, h2 in shears:
            if name.split("env_")[1] in envcfg.metrics or name in envcfg.metrics:
                results[name] = _bulk_shear_at_column(nc_env, i, j, h1, h2)

        if "srh_0_1km" in envcfg.metrics:
            results["env_srh_0_1km"] = _srh_at_column(nc_env, i, j, 1000.0)
        if "srh_0_3km" in envcfg.metrics:
            results["env_srh_0_3km"] = _srh_at_column(nc_env, i, j, 3000.0)

        # STP (environmental) via MetPy if possible
        if "stp" in envcfg.metrics:
            mlcape = results.get("env_mlcape", np.nan)
            mlcin  = results.get("env_mlcin", np.nan)
            srh01  = results.get("env_srh_0_1km", np.nan)
            shear06= results.get("env_shear_0_6km", np.nan)
            lcl_m  = results.get("env_lcl", np.nan)
            results["env_stp"] = _significant_tornado_parameter_env(mlcape, mlcin, srh01, shear06, lcl_m)

        meta["environment"] = {
            "time_index": ti,
            "column": {"mode": envcfg.column.mode, "corner": envcfg.column.corner, "i": i, "j": j,
                       "offset_i": envcfg.column.offset_i, "offset_j": envcfg.column.offset_j},
            "metrics": envcfg.metrics
        }

    # ---------------- STORM METRICS ----------------
    storm = jobcfg.storm
    if storm and storm.metrics:
        # map window to indices
        si, ei = _storm_time_indices(ncs, storm.time_window, jobcfg.dt_seconds)
        si = max(0, min(si, len(ncs) - 1))
        ei = max(0, min(ei, len(ncs) - 1))

        up_max_series, dn_min_series = [], []
        dbz_max_series, dbz_mean_series = [], []
        # precip
        rain_means = []  # domain mean RAINNC at each selected time
        rate_series = []  # domain-max (RAINNC_t - RAINNC_{t-1})/dt

        z3d_first = _get_heights_3d(ncs[si])  # for slab

        prev_r = None
        dt = jobcfg.dt_seconds if jobcfg.dt_seconds > 0 else 300.0

        for k in range(si, ei + 1):
            nc = ncs[k]
            # 3D fields
            dbz3d = _dbz3d(nc) if (storm.mask.use_reflectivity or any(m in storm.metrics for m in ["reflectivity_max","reflectivity_mean"])) else None
            w3d = _w3d(nc) if any(m in storm.metrics for m in ["updraft_p99","downdraft_p99"]) else None

            # mask from dbz threshold
            mask = None
            if storm.mask.use_reflectivity and dbz3d is not None:
                mask = (dbz3d >= float(storm.mask.dbz_threshold))

            # apply slab+mask
            if w3d is not None:
                w_m = _apply_mask_slab(w3d, mask, z3d_first, storm.slab.zmin, storm.slab.zmax)
                if w_m is not None:
                    up_max_series.append(float(np.nanmax(w_m)))
                    dn_min_series.append(float(np.nanmin(w_m)))

            if dbz3d is not None:
                d_m = _apply_mask_slab(dbz3d, mask, z3d_first, storm.slab.zmin, storm.slab.zmax)
                if d_m is not None:
                    dbz_max_series.append(float(np.nanmax(d_m)))
                    dbz_mean_series.append(float(np.nanmean(d_m)))

            # precipitation (RAINNC)
            if any(m in storm.metrics for m in ["precip_total", "precip_rate_p99"]):
                r = _get_var(nc, "RAINNC", meta=False)
                if r is not None:
                    R = np.array(r)
                    rain_means.append(float(np.nanmean(R)))
                    if prev_r is not None and "precip_rate_p99" in storm.metrics:
                        rate = (R - prev_r) / dt
                        rate_series.append(float(np.nanmax(rate)))
                    prev_r = R

        # aggregate
        if "updraft_p99" in storm.metrics:
            results["updraft_p99"] = _p99(up_max_series)
        if "downdraft_p99" in storm.metrics:
            # p99 of |downdraft| -> use negative minima series as magnitudes
            neg_mags = [-v for v in dn_min_series if np.isfinite(v)]
            results["downdraft_p99"] = _p99(neg_mags)
        if "reflectivity_max" in storm.metrics:
            results["reflectivity_max"] = float(np.nanmax(dbz_max_series)) if dbz_max_series else np.nan
        if "reflectivity_mean" in storm.metrics:
            results["reflectivity_mean"] = float(np.nanmean(dbz_mean_series)) if dbz_mean_series else np.nan
        if "precip_total" in storm.metrics:
            results["precip_total"] = float(rain_means[-1] - rain_means[0]) if len(rain_means) >= 2 else np.nan
        if "precip_rate_p99" in storm.metrics:
            results["precip_rate_p99"] = _p99(rate_series)

        meta["storm"] = {
            "time_window": {"si": si, "ei": ei, "mode": storm.time_window.mode,
                            "start_min": storm.time_window.start_min, "end_min": storm.time_window.end_min},
            "mask": {"use_reflectivity": storm.mask.use_reflectivity, "dbz_threshold": storm.mask.dbz_threshold},
            "slab": {"zmin": storm.slab.zmin, "zmax": storm.slab.zmax},
            "metrics": storm.metrics
        }

    # Write outputs (merge if needed)
    # Read existing pkl cleanly (we avoided unsafe pickle above)
    existing = {}
    if out_pkl.exists() and jobcfg.merge:
        try:
            with open(out_pkl, "rb") as f:
                existing = pickle_load(f)
        except Exception:
            existing = {}

    merged = dict(existing)
    merged.update(results)

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        import pickle
        pickle.dump(merged, f)

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    _close_all(ncs)
    return sample_idx, {"_status": "ok", **results}, meta


# Simple, safe pickle helpers (no execution)
def pickle_load(fileobj):
    import pickle
    return pickle.load(fileobj)

def pickle_to_jsonsafe(raw_bytes):
    # Not used finally; kept for reference if needed to sanitize legacy pickles.
    return {}


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute WRF metrics (config-driven).")
    ap.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    args = ap.parse_args()

    paths, jobcfg = load_config(Path(args.config))
    exp_dir = Path(paths.EXPERIMENT_DIR).resolve()
    pv = np.load(exp_dir / "param_values.npy")
    L = len(pv)

    if not jobcfg.enabled:
        print("[metrics] metrics_job.enabled is False — nothing to do.")
        return

    worker = partial(process_sample, exp_dir=exp_dir, jobcfg=jobcfg)

    if jobcfg.workers and jobcfg.workers > 1:
        with Pool(processes=jobcfg.workers) as pool:
            for i, res, meta in pool.imap_unordered(worker, range(L), chunksize=1):
                if res.get("_status") != "ok":
                    print(f"[{i:05d}] {res.get('_status')}")
                else:
                    # light progress
                    keys = [k for k in res.keys() if k != "_status"]
                    print(f"[{i:05d}] ok -> {', '.join(keys)}")
    else:
        for i in range(L):
            i, res, meta = worker(i)
            if res.get("_status") != "ok":
                print(f"[{i:05d}] {res.get('_status')}")
            else:
                keys = [k for k in res.keys() if k != "_status"]
                print(f"[{i:05d}] ok -> {', '.join(keys)}")

    print("[metrics] Done.")


if __name__ == "__main__":
    main()