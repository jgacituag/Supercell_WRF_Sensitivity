
#!/usr/bin/env python3
# envgen_analytic.py
"""
Analytic environment generator for idealized WRF soundings.

- Inputs: 10-D u in [0,1].
- Mapping: u -> physical parameters (Tsfc, Γ, Dtrp, RHsfc, Dwv, Umax, Dshear, curvature, LLJ).
- Output: sounding dict with arrays (z [m], p [hPa], T [K], qv [g/kg], u [m/s], v [m/s]).
- Utilities: diagnostics (CAPE/CIN if MetPy available; bulk shear; PW), writer for input_sounding.

This module is dependency-light; MetPy is optional.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

try:
    import metpy.calc as mpcalc
    from metpy.units import units
    HAS_METPY = True
except Exception:
    HAS_METPY = False


# ---------------------- Parameterization -------------------------------------

@dataclass
class PhysParams:
    Tsfc_C: float
    Gamma_bdy: float
    Gamma_trp: float
    Dtrp_km: float
    RHsfc: float
    Dwv_km: float
    Umax_ms: float
    Dshear_km: float
    curved_shear_per: float
    llj_ms: float
    llj_h_km: float


def lin(a: float, b: float, u: float) -> float:
    return a + (b - a) * np.clip(u, 0.0, 1.0)


def u01_to_phys(u: List[float]) -> PhysParams:
    if len(u) != 10:
        raise ValueError("Expected len(u)==10")
    u0,u1,u2,u3,u4,u5,u6,u7,u8,u9 = list(map(float, u))
    Tsfc_C    = lin(21.0, 30.0, u0)
    Gamma_bdy = 6.5
    Gamma_trp = lin(5.5, 7.5, u1)
    Dtrp_km   = lin(12.0, 16.0, u2)
    RHsfc     = lin(0.75, 0.90, u3)
    Dwv_max   = max(4.0, Dtrp_km - 1.0)
    Dwv_km    = lin(4.0, Dwv_max, u4)
    Umax_ms   = lin(10.0, 45.0, u5)
    Dshear_km = lin(3.0, 12.0, u6)
    curved    = lin(0.0, 0.6, u7)
    llj_ms    = lin(0.0, 8.0, u8)
    llj_h_km  = lin(0.5, 1.2, u9)
    return PhysParams(Tsfc_C, Gamma_bdy, Gamma_trp, Dtrp_km, RHsfc, Dwv_km,
                      Umax_ms, Dshear_km, curved, llj_ms, llj_h_km)


# ---------------------- Thermodynamics & Winds -------------------------------

def build_temperature_profile(z: np.ndarray, p: PhysParams) -> np.ndarray:
    """Piecewise T(z) [K]: BL with Γ_bdy, then Γ_trp to tropopause, then +2 K/km above."""
    z_km = z / 1000.0
    T0 = p.Tsfc_C + 273.15
    T = np.empty_like(z, dtype=float)
    # BL 0-1 km
    T_bl = T0 - p.Gamma_bdy * np.minimum(z_km, 1.0)
    T = T_bl.copy()
    # Troposphere 1 km - Dtrp
    mask_trp = z_km > 1.0
    z_trp = np.clip(z_km - 1.0, 0.0, None)
    T[mask_trp] = (T_bl[np.searchsorted(z_km, 1.0, side='left')] -
                   p.Gamma_trp * z_trp[mask_trp])
    # Above Dtrp: +2 K/km increase
    mask_strat = z_km > p.Dtrp_km
    z_strat = z_km - p.Dtrp_km
    T[mask_strat] = T[np.searchsorted(z_km, p.Dtrp_km, side='left')] + 2.0 * z_strat[mask_strat]
    return T


def saturation_mixing_ratio(p_hpa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """ws [kg/kg] using Tetens saturation vapour pressure over water."""
    # es in hPa
    T_C = T_K - 273.15
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    epsilon = 0.622
    ws = epsilon * (es / np.maximum(p_hpa - es, 1e-3))
    return ws  # kg/kg


def build_moisture_profile(z: np.ndarray, p_hpa: np.ndarray, T_K: np.ndarray, phys: PhysParams) -> np.ndarray:
    """qv [g/kg] using RH(z): RH=RHsfc in BL then tapers linearly to 0.2 by Dwv_km."""
    z_km = z / 1000.0
    RH = np.ones_like(z, dtype=float) * phys.RHsfc
    top = phys.Dwv_km
    RH[z_km > top] = np.linspace(phys.RHsfc, 0.2, np.sum(z_km > top))
    ws = saturation_mixing_ratio(p_hpa, T_K)  # kg/kg
    qv = RH * ws  # kg/kg
    return qv * 1000.0  # g/kg


def hypsometric_pressure(z: np.ndarray, T_K: np.ndarray, qv_kgkg: np.ndarray, p0_hpa: float = 1000.0) -> np.ndarray:
    """Hydrostatic p(z) (hPa) using virtual temperature and forward integration."""
    g = 9.80665
    R_d = 287.04
    Tv = T_K * (1.0 + 0.61 * qv_kgkg)
    p = np.empty_like(z, dtype=float)
    p[0] = p0_hpa * 100.0  # Pa
    for k in range(1, len(z)):
        dz = z[k] - z[k-1]
        Tv_mid = 0.5 * (Tv[k] + Tv[k-1])
        p[k] = p[k-1] * np.exp(-g * dz / (R_d * Tv_mid))
    return p / 100.0  # hPa

def build_wind_profile(z, phys,
                       style='linear_arc',
                       turn_frac=0.4,          # fraction of shear depth where the turn begins
                       theta0_deg=0.0,         # direction of the straight segment (deg)
                       arc_deg=180.0,          # arc sweep (deg): 180 = semicircle
                       clockwise=False):       # arc direction
    """
    Return u, v with a linear segment up to z_turn, then a circular arc to the top.

    Inputs (from phys):
      phys.Umax_ms     : target speed at z = phys.Dshear_km
      phys.Dshear_km   : shear depth (km)
      phys.llj_ms, phys.llj_h_km : optional LLJ added to u-component

    tunables:
      turn_frac   : z_turn / Dshear (0.2–0.6 typical)
      theta0_deg  : heading of the linear segment (0° = +u / east)
      arc_deg     : how much the hodograph turns after z_turn (default 180°)
      clockwise   : True = clockwise rotation, False = counterclockwise
    """

    z_km = z / 1000.0
    D = max(phys.Dshear_km, 0.1)
    Uend = float(phys.Umax_ms)

    # --- Straight segment (0 -> z_turn) ---
    z_turn = np.clip(turn_frac, 0.05, 0.95) * D
    theta0 = np.deg2rad(theta0_deg)

    # Speed increases linearly to r0 at z_turn; choose r0 so that the
    # fraction of U at the join equals turn_frac (keeps total growth roughly linear)
    r0 = Uend * (z_turn / D)                     # speed at the join
    u_lin = r0 * np.cos(theta0)
    v_lin = r0 * np.sin(theta0)
    P0 = np.array([u_lin, v_lin])                # end point of straight segment

    # --- Arc segment (z_turn -> D) ---
    sgn = -1.0 if clockwise else +1.0
    theta1 = theta0 + sgn * np.deg2rad(arc_deg)  # desired tangent at the top

    # Circle geometry so the tangent is continuous at the join:
    # center C lies along the normal to the straight segment at P0
    # C = P0 - R * [sin(theta0), -cos(theta0)]
    # end point at top: P1 = C + R * [sin(theta1), -cos(theta1)]
    # enforce |P1| = Uend -> solve quadratic for R >= 0
    Dvec = np.array([np.sin(theta1) - np.sin(theta0),
                     -(np.cos(theta1) - np.cos(theta0))])
    a = float(Dvec @ Dvec)
    b = 2.0 * float(P0 @ Dvec)
    c = float(P0 @ P0) - Uend**2
    disc = max(0.0, b*b - 4*a*c)
    R = (-b + np.sqrt(disc)) / (2*a) if a > 0 else 0.0
    R = max(R, 1e-6)

    C = P0 - R * np.array([np.sin(theta0), -np.cos(theta0)])

    # Build profile
    u = np.zeros_like(z, dtype=float)
    v = np.zeros_like(z, dtype=float)

    # 1) Straight part
    m_lin = z_km <= z_turn + 1e-6
    if z_turn > 0:
        frac = (z_km[m_lin] / z_turn)
    else:
        frac = np.zeros_like(z_km[m_lin])
    u[m_lin] = frac * P0[0]
    v[m_lin] = frac * P0[1]

    # 2) Arc part
    m_arc = z_km > z_turn
    if np.any(m_arc):
        # sweep angle along depth (linear in z)
        phi0 = theta0 - np.pi/2.0
        phi1 = theta1 - np.pi/2.0
        t = (z_km[m_arc] - z_turn) / max(D - z_turn, 1e-6)
        phi = phi0 + t * (phi1 - phi0)
        u[m_arc] = C[0] + R * np.cos(phi)
        v[m_arc] = C[1] + R * np.sin(phi)

    # Optional LLJ bump (Gaussian) added to u
    if getattr(phys, "llj_ms", 0.0) > 0.0:
        sigma = 0.3  # km
        u += phys.llj_ms * np.exp(-0.5 * ((z_km - phys.llj_h_km) / sigma)**2)

    return u, v
# ---------------------- Generator --------------------------------------------

def generate_sounding(u: List[float], z_top_km: float = 20.0, dz_m: float = 50.0) -> Dict[str, np.ndarray]:
    """Return dict with z, p(hPa), T(K), qv(g/kg), u, v for a given u∈[0,1]^10."""
    phys = u01_to_phys(u)
    z = np.arange(0.0, z_top_km * 1000.0 + dz_m, dz_m, dtype=float)
    T = build_temperature_profile(z, phys)
    # First guess p with dry Tv (qv=0), then update with moist Tv iteratively
    p_hpa = np.full_like(z, 1000.0)
    qv = np.zeros_like(z)
    for _ in range(2):
        p_hpa = hypsometric_pressure(z, T, qv / 1000.0, p0_hpa=1000.0)
        qv = build_moisture_profile(z, p_hpa, T, phys)
    u, v = build_wind_profile(z, phys)
    return dict(height=z, p=p_hpa, t=T, qv=qv, u=u, v=v)


# ---------------------- Diagnostics & Writer ---------------------------------

def dewpoint_from_qv(p_hpa: np.ndarray, qv_gkg: np.ndarray) -> np.ndarray:
    """Return Td [K] from (p, qv)."""
    epsilon = 0.622
    r = qv_gkg / 1000.0  # kg/kg
    p_pa = p_hpa * 100.0
    e_pa = (r / (epsilon + r)) * p_pa
    e_hpa = e_pa / 100.0
    ln_ratio = np.log(np.maximum(e_hpa, 1e-6) / 6.112)
    Td_C = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return Td_C + 273.15


def bulk_shear(u: np.ndarray, v: np.ndarray, z: np.ndarray, depth_m: float) -> float:
    """|V(depth)-V(0)| in m/s via linear interpolation."""
    ut = np.interp(depth_m, z, u)
    vt = np.interp(depth_m, z, v)
    u0, v0 = u[0], v[0]
    return float(np.hypot(ut - u0, vt - v0))


def precipitable_water(p_hpa: np.ndarray, qv_gkg: np.ndarray) -> float:
    """Return PW [mm] via vertical integration of qv."""
    if not HAS_METPY:
        return np.nan

    dewpoint = dewpoint_from_qv(p_hpa, qv_gkg)
    pw = mpcalc.precipitable_water(p_hpa * units.hectopascal,dewpoint * units.kelvin)
    return float(pw.to('millimeter').m)


def diagnostics(sound: Dict[str, np.ndarray]) -> Dict[str, float]:
    z = sound['height']; p = sound['p']; T = sound['t']; qv = sound['qv']; u = sound['u']; v = sound['v']
    td = dewpoint_from_qv(p, qv)
    diag = {}
    # CAPE/CIN (Mixed-layer) if MetPy is available
    if HAS_METPY:
        p_q = (p * units.hectopascal)
        T_q = (T * units.kelvin)
        Td_q = (td * units.kelvin)
        cape, cin = mpcalc.mixed_layer_cape_cin(p_q, T_q, Td_q, depth=100 * units.hectopascal)
        diag['MLCAPE'] = float(cape.to('joule / kilogram').m)
        diag['MLCIN']  = float(cin.to('joule / kilogram').m)
    else:
        diag['MLCAPE'] = np.nan
        diag['MLCIN']  = np.nan
    diag['SH06'] = bulk_shear(u, v, z, 6000.0)
    diag['SH01'] = bulk_shear(u, v, z, 1000.0)
    diag['PW']   = precipitable_water(p, qv)
    return diag


def write_input_sounding(path: str, sound: Dict[str, np.ndarray]) -> None:
    """
    Write a simple WRF-like input_sounding with columns:
    p(hPa)  T(K)  qv(g/kg)  u(m/s)  v(m/s)  z(m)
    """
    data = np.column_stack([sound['p'], sound['t'], sound['qv'], sound['u'], sound['v'], sound['height']])
    header = "p_hPa  T_K  qv_gkg  u_ms  v_ms  z_m"
    np.savetxt(path, data, fmt="%.3f", header=header, comments='')
