"""
WRF Idealized Supercell Sounding Generator
===========================================

Generates realistic atmospheric soundings for WRF idealized supercell simulations.
Designed for Sobol sensitivity analysis with physics-based parameter controls.

Organization:
  1. Physical constants and imports
  2. Auxiliary/helper functions (thermodynamic calculations, interpolation)
  3. Core I/O functions (read/write WRF input_sounding files)
  4. Profile generation functions (temperature, moisture, wind)
  5. Diagnostic calculation functions (CAPE, CIN, shear, etc.)
  6. Consistency check functions
  7. Main orchestrator function

Author: [Your name]
Date: 2025
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

cp = 1004.0      # Specific heat at constant pressure (J/kg/K)
Rd = 287.0       # Gas constant for dry air (J/kg/K)
g = 9.81         # Gravitational acceleration (m/s²)
P0 = 100000.0    # Reference pressure (Pa)
epsilon = 0.622  # Ratio of molecular weights (water vapor / dry air)

# ============================================================================
# AUXILIARY / HELPER FUNCTIONS
# ============================================================================

def es_bolton(Tk):
    """
    Calculate saturation vapor pressure using Bolton's formula.
    
    Parameters
    ----------
    Tk : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    es : float or ndarray
        Saturation vapor pressure in hPa
    """
    return 6.112 * np.exp(17.67 * (Tk - 273.15) / (Tk - 29.65))


def qvs_hPa_gpkg(Tk, p_hpa):
    """
    Calculate saturation mixing ratio.
    
    Parameters
    ----------
    Tk : float or ndarray
        Temperature in Kelvin
    p_hpa : float or ndarray
        Pressure in hPa
        
    Returns
    -------
    qvs : float or ndarray
        Saturation mixing ratio in g/kg
    """
    es = es_bolton(Tk)
    return epsilon * es / (p_hpa - (1.0 - epsilon) * es) * 1000.0


def lin_interp(z, y, z0):
    """
    Linear interpolation helper function.
    
    Parameters
    ----------
    z : ndarray
        Height array (monotonically increasing)
    y : ndarray
        Value array to interpolate
    z0 : float
        Target height for interpolation
        
    Returns
    -------
    y0 : float
        Interpolated value at height z0
    """
    i = np.searchsorted(z, z0)
    if i == 0:
        return y[0]
    if i >= len(z):
        return y[-1]
    
    z1, z2 = z[i-1], z[i]
    y1, y2 = y[i-1], y[i]
    w = (z0 - z1) / (z2 - z1)
    return (1.0 - w) * y1 + w * y2


# ============================================================================
# CORE I/O FUNCTIONS
# ============================================================================

def read_input_sounding(filename):
    """
    Read WRF input_sounding file.
    
    Format:
      Line 1: surf_pressure(hPa) surf_theta(K) surf_qv(g/kg)
      Remaining lines: height(m) theta(K) qv(g/kg) u(m/s) v(m/s)
    
    Parameters
    ----------
    filename : str
        Path to input_sounding file
        
    Returns
    -------
    sounding : dict
        Dictionary containing sounding profile data
    """
    with open(filename) as f:
        lines = f.readlines()
    
    # Parse surface values (first line)
    line_split = lines[0].split()
    surf_pressure = float(line_split[0])
    surf_theta = float(line_split[1])
    surf_qv = float(line_split[2])
    
    # Parse vertical profile
    nlevs = len(lines) - 1
    height = np.zeros(nlevs)
    theta = np.zeros(nlevs)
    qv = np.zeros(nlevs)
    u = np.zeros(nlevs)
    v = np.zeros(nlevs)
    
    for i in range(nlevs):
        line_split = lines[i + 1].split()
        height[i] = float(line_split[0])
        theta[i] = float(line_split[1])
        qv[i] = float(line_split[2])
        u[i] = float(line_split[3])
        v[i] = float(line_split[4])
    
    # Calculate pressure at each level (hydrostatic balance)
    p = np.zeros(nlevs)
    p[0] = surf_pressure
    
    for i in range(1, nlevs):
        theta_mean = 0.5 * (theta[i] + theta[i-1])
        dz = height[i] - height[i-1]
        pi_prev = (p[i-1] * 100) ** (Rd / cp)
        pi = pi_prev - g * (P0 ** (Rd / cp)) * dz / (cp * theta_mean)
        p[i] = (pi ** (cp / Rd)) / 100  # Convert to hPa
    
    # Calculate temperature from theta using Exner function
    pi = ((p * 100.0) / P0) ** (Rd / cp)
    t = theta * pi
    
    return {
        'surf_pressure': surf_pressure,
        'surf_theta': surf_theta,
        'surf_qv': surf_qv,
        'nlevs': nlevs,
        'height': height,
        'theta': theta,
        'qv': qv,
        'u': u,
        'v': v,
        'p': p,
        't': t
    }


def write_input_sounding(filename, sounding):
    """
    Write WRF input_sounding file.
    
    Parameters
    ----------
    filename : str
        Output path for input_sounding file
    sounding : dict
        Sounding dictionary containing profile data
    """
    with open(filename, 'w') as f:
        # Write surface values
        f.write(f"{sounding['surf_pressure']:.4f} {sounding['surf_theta']:.4f} "
                f"{sounding['surf_qv']:.6f}\n")
        
        # Write vertical profile
        for i in range(sounding['nlevs']):
            f.write(f"{sounding['height'][i]:10.4f} {sounding['theta'][i]:10.4f} "
                   f"{sounding['qv'][i]:10.6f} {sounding['u'][i]:10.4f} "
                   f"{sounding['v'][i]:10.4f}\n")


# ============================================================================
# PROFILE GENERATION FUNCTIONS
# ============================================================================

def generate_temperature_profile(sounding, params):
    """
    Generate potential temperature (θ) profile with physics-based layers.
    
    The atmosphere is divided into three layers:
      1. Boundary layer (0 to z_ml_top): Slight θ increase due to surface heating
      2. Troposphere (z_ml_top to z_trop_top): Moderate θ increase
      3. Stratosphere (above z_trop_top): Strong θ increase (stable layer)
    
    Temperature (T) is always derived from θ via the Exner function to ensure
    thermodynamic consistency.
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary (will be modified in place)
    params : dict
        Parameter dictionary with keys:
        - surface_theta (K): Surface potential temperature [296-306]
        - z_ml_top (m): Boundary layer top height [600-1500]
        - z_trop_top (m): Tropopause height [11000-13000]
        - theta_lapse_bl (K/km): BL θ lapse rate [0.5-2.5]
        - theta_lapse (K/km): Tropospheric θ lapse rate [2.0-3.5]
        - theta_lapse_strat (K/km): Stratospheric θ lapse rate [15-25]
    
    Returns
    -------
    sounding : dict
        Modified sounding with updated 'theta' and 't' fields
    """
    z = sounding['height']              # m
    p = sounding['p'] * 100.0           # Pa
    th = np.empty_like(z, dtype=float)

    # --- Extract parameters with defaults ---
    surface_theta     = float(params.get('surface_theta', 300.0))
    z_ml_top          = float(params.get('z_ml_top', 1000.0))
    z_trop_top        = float(params.get('z_trop_top', 12000.0))
    theta_lapse_bl    = float(params.get('theta_lapse_bl', 1.9))
    theta_lapse       = float(params.get('theta_lapse', 2.5))
    theta_lapse_strat = float(params.get('theta_lapse_strat', 20.0))

    # --- 1) Boundary layer: slight lapse (surface heating effect) ---
    mask_bl = z <= z_ml_top
    th[mask_bl] = surface_theta + theta_lapse_bl * z[mask_bl] / 1000.0

    # --- 2) Troposphere: moderate lapse ---
    mask_trop = (z > z_ml_top) & (z <= z_trop_top)
    if np.any(mask_trop):
        # Start from BL top value
        th_ml_top = surface_theta + theta_lapse_bl * z_ml_top / 1000.0
        th[mask_trop] = th_ml_top + theta_lapse * (z[mask_trop] - z_ml_top) / 1000.0

    # --- 3) Stratosphere: strong lapse (very stable) ---
    mask_strat = z > z_trop_top
    if np.any(mask_strat):
        # Start from tropopause value
        th_ml_top = surface_theta + theta_lapse_bl * z_ml_top / 1000.0
        th_trop_top = th_ml_top + theta_lapse * (z_trop_top - z_ml_top) / 1000.0
        th[mask_strat] = th_trop_top + theta_lapse_strat * (z[mask_strat] - z_trop_top) / 1000.0

    # --- 4) Derive temperature from θ using Exner function ---
    pi = (p / P0) ** (Rd / cp)
    sounding['theta'] = th
    sounding['t'] = th * pi

    return sounding


def generate_moisture_profile(sounding, params):
    """
    Generate moisture (qv) profile with specified RH at different levels.
    
    Strategy:
      - Boundary layer: Well-mixed in qv (not RH!) to avoid supersaturation
      - Above BL: Linear RH transitions through mid-troposphere
      - Upper levels: Exponential decay
    
    Key constraint: qv is set from RH at the TOP of the mixed layer, then
    held constant below. This ensures RH ≤ target everywhere in the BL.
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary (will be modified in place)
    params : dict
        Parameter dictionary with keys:
        - low_level_rh (%): BL relative humidity [60-95]
        - mid_level_rh (%): Mid-troposphere RH [30-60]
        - upper_level_rh (%): Upper-troposphere RH [10-40]
        - z_ml_top (m): BL top height [600-1500]
        - z_moisture_top (m): Top of moisture transition [1000-2500]
    
    Returns
    -------
    sounding : dict
        Modified sounding with updated 'qv' field
    """
    # Extract parameters
    low_level_rh  = float(params['low_level_rh'])
    mid_level_rh  = float(params['mid_level_rh'])
    upper_level_rh = float(params['upper_level_rh'])
    z_ml_top      = float(params.get('z_ml_top', 1000.0))
    moisture_transition_depth = float(params.get('moisture_transition_depth', 1000.0))
    z_moisture_top = z_ml_top + moisture_transition_depth 
    z_dry_top     = 10000.0  # Height where exponential decay begins
    
    # Ensure proper ordering of heights
    z_moisture_top = max(z_moisture_top, z_ml_top + 100.0)
    
    z = sounding['height']
    T = sounding['t']      # K
    p = sounding['p']      # hPa
    qv = np.zeros_like(z, dtype=float)

    # --- Set BL qv from RH at the TOP of the mixed layer ---
    # This ensures RH ≤ target throughout the BL (no supersaturation)
    T_ml = lin_interp(z, T, z_ml_top)
    p_ml = lin_interp(z, p, z_ml_top)
    qvs_ml = qvs_hPa_gpkg(T_ml, p_ml)       # g/kg
    qv_ml = qvs_ml * (low_level_rh / 100.0)  # g/kg (constant in BL)

    # --- Build moisture profile ---
    for i, zi in enumerate(z):
        if zi <= z_ml_top:
            # Boundary layer: constant qv
            qv[i] = qv_ml
            
        elif zi <= z_moisture_top:
            # Transition layer: linear RH blend (low → mid)
            frac = (zi - z_ml_top) / (z_moisture_top - z_ml_top)
            target_rh = low_level_rh + (mid_level_rh - low_level_rh) * frac
            qv_target = qvs_hPa_gpkg(T[i], p[i]) * (target_rh / 100.0)
            
            # Smooth transition just above BL top to avoid kink
            if zi <= (z_ml_top + 200.0):
                s = (zi - z_ml_top) / 200.0
                s = s * s * (3.0 - 2.0 * s)  # Smoothstep function
                qv[i] = (1.0 - s) * qv_ml + s * qv_target
            else:
                qv[i] = qv_target
                
        elif zi <= z_dry_top:
            # Mid to upper troposphere: linear RH blend (mid → upper)
            frac = (zi - z_moisture_top) / (z_dry_top - z_moisture_top)
            target_rh = mid_level_rh + (upper_level_rh - mid_level_rh) * frac
            qv[i] = qvs_hPa_gpkg(T[i], p[i]) * (target_rh / 100.0)
            
        else:
            # Above dry_top: exponential decay
            target_rh = upper_level_rh * np.exp(-(zi - z_dry_top) / 3000.0)
            qv[i] = qvs_hPa_gpkg(T[i], p[i]) * (target_rh / 100.0)

        # --- Safety guards ---
        # 1. Never exceed saturation
        qv[i] = min(qv[i], qvs_hPa_gpkg(T[i], p[i]))
        
        # 2. Optional monotonic guard: no increase with height above BL
        if i > 0 and zi > z_ml_top:
            qv[i] = min(qv[i], qv[i-1] + 1e-6)

    sounding['qv'] = qv
    return sounding


def generate_wind_profile_sh(sounding, params, remove_mean_wind=True):
    """
    Generate wind profile typical for Southern Hemisphere supercells.
    
    Implements a "curved-then-linear" hodograph that turns ANTI-CLOCKWISE
    (appropriate for SH) from the origin into the U>0, V>0 quadrant.
    
    The curvature parameter (0-1) blends between:
      - curvature = 0: Straight line hodograph (unidirectional shear)
      - curvature = 1: 180° curved hodograph (strong directional shear)
    
    Optional low-level jet can be added perpendicular to mean shear direction.
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary (will be modified in place)
    params : dict
        Parameter dictionary with keys:
        - shear_rate (1/s): Shear rate [1/1200 to 1/400]
        - shear_depth (m): Depth of shear layer [3000-12000]
        - shear_curvature (0-1): Hodograph curvature [0-1]
        - low_level_jet (m/s): LLJ strength (optional, default 0)
        - llj_height (m): LLJ height (optional, default 1000)
        - llj_dir (deg): LLJ direction (optional, default 0)
        - llj_width (m): LLJ width (optional, default 400)
    remove_mean_wind : bool
        If True, remove 0-6km mean wind to center hodograph
    
    Returns
    -------
    sounding : dict
        Modified sounding with updated 'u' and 'v' fields
    """
    # --- Extract parameters ---
    curvature = float(params['shear_curvature'])
    total_shear_depth = float(params['shear_depth'])
    shear_mag = float(params['shear_rate']) * total_shear_depth
    
    llj_strength = float(params.get('low_level_jet', 0))
    llj_height = float(params.get('llj_height', 1000))
    llj_width = float(params.get('llj_width', 400))
    llj_dir_rad = np.radians(float(params.get('llj_dir', 0)))
    
    z = sounding['height']
    u = np.zeros_like(z)
    v = np.zeros_like(z)

    # --- Define hodograph geometry ---
    # Total angle of curved part (0 to π)
    total_angle = curvature * np.pi
    
    # Path lengths
    arc_length = shear_mag * curvature
    linear_length = shear_mag * (1.0 - curvature)
    
    # Height where curve stops and line begins
    curved_depth = total_shear_depth * curvature
    
    # Radius of circular arc (with small epsilon to avoid division by zero)
    radius = arc_length / (total_angle + 1e-6)
    
    # --- Calculate endpoints ---
    # End of curved part (start of linear part)
    u_ini = -radius * np.cos(total_angle)
    v_ini = -radius * np.sin(total_angle)
    
    # Tangent direction at end of curve
    tangent_dir_u = np.sin(total_angle)
    tangent_dir_v = -np.cos(total_angle)
    
    # Final endpoint
    u_end = u_ini + linear_length * tangent_dir_u
    v_end = v_ini + linear_length * tangent_dir_v

    # --- Assign wind components by layer ---
    # Create masks for efficiency
    mask_curved = (z <= curved_depth) & (curved_depth > 1e-6)
    mask_linear = (z > curved_depth) & (z <= total_shear_depth)
    mask_constant = (z > total_shear_depth)
    
    # A) Curved part
    if np.any(mask_curved):
        frac_h = z[mask_curved] / curved_depth
        theta_h = frac_h * total_angle
        u[mask_curved] = -radius * np.cos(theta_h)
        v[mask_curved] = -radius * np.sin(theta_h)
    
    # B) Linear part
    if np.any(mask_linear):
        denom = (total_shear_depth - curved_depth)
        if denom > 1e-6:
            frac_h = (z[mask_linear] - curved_depth) / denom
            u[mask_linear] = u_ini + (u_end - u_ini) * frac_h
            v[mask_linear] = v_ini + (v_end - v_ini) * frac_h
    
    # C) Handle pure-linear case (curvature = 0)
    if curved_depth < 1e-6:
        mask_linear_only = z <= total_shear_depth
        frac_h = z[mask_linear_only] / (total_shear_depth + 1e-6)
        u[mask_linear_only] = u_end * frac_h
        v[mask_linear_only] = v_end * frac_h
        
    # D) Constant part (above shear layer)
    u[mask_constant] = u_end
    v[mask_constant] = v_end
    
    # --- Add Low-Level Jet (optional) ---
    if llj_strength > 0:
        llj_profile = llj_strength * np.exp(-0.5 * (z - llj_height)**2 / (llj_width**2))
        u += -llj_profile * np.sin(llj_dir_rad)
        v += -llj_profile * np.cos(llj_dir_rad)
    
    # --- Final assignment ---
    sounding['u'] = u
    sounding['v'] = v
    
    # --- Remove mean wind (centers hodograph) ---
    if remove_mean_wind:
        mask_0_6km = z <= 6000.0
        if np.any(mask_0_6km):
            mean_u = np.mean(u[mask_0_6km])
            mean_v = np.mean(v[mask_0_6km])
            sounding['u'] = u - mean_u
            sounding['v'] = v - mean_v
    
    return sounding


# ============================================================================
# DIAGNOSTIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_cape_cin(sounding, parcel_type='most_unstable'):
    """
    Calculate CAPE and CIN using MetPy.
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary with p, t, qv, height
    parcel_type : str
        Parcel choice: 'most_unstable', 'surface', or 'mixed_layer'
        
    Returns
    -------
    cape : float
        Convective Available Potential Energy (J/kg)
    cin : float  
        Convective Inhibition (J/kg)
    """
    from metpy.units import units
    from metpy.calc import (most_unstable_cape_cin, 
                            surface_based_cape_cin, 
                            mixed_layer_cape_cin,
                            dewpoint_from_specific_humidity)
    
    # Prepare arrays with MetPy units
    p = sounding['p'] * units.hPa
    T = sounding['t'] * units.kelvin
    qv = sounding['qv'] * units('g/kg')
    
    # Convert mixing ratio to dewpoint
    q = qv / (1 + qv)  # Specific humidity (approximate)
    Td = dewpoint_from_specific_humidity(p, T, q)
    
    try:
        if parcel_type == 'most_unstable':
            cape_val, cin_val = most_unstable_cape_cin(p, T, Td)
        elif parcel_type == 'surface':
            cape_val, cin_val = surface_based_cape_cin(p, T, Td)
        elif parcel_type == 'mixed_layer':
            cape_val, cin_val = mixed_layer_cape_cin(p, T, Td, depth=100*units.hPa)
        else:
            raise ValueError(f"Unknown parcel_type: {parcel_type}")
        
        # Convert to regular floats (remove units)
        cape_j = float(cape_val.magnitude) if hasattr(cape_val, 'magnitude') else float(cape_val)
        cin_j = float(cin_val.magnitude) if hasattr(cin_val, 'magnitude') else float(cin_val)
        
        # Handle NaN or infinite values
        if not np.isfinite(cape_j):
            cape_j = 0.0
        if not np.isfinite(cin_j):
            cin_j = 0.0
            
        return cape_j, cin_j
        
    except Exception as e:
        print(f"Warning: CAPE/CIN calculation failed: {e}")
        return 0.0, 0.0


def calculate_diagnostics(sounding):
    """
    Calculate comprehensive atmospheric diagnostics using MetPy.
    
    Computes:
      - CAPE/CIN (multiple parcel types)
      - Precipitable water (PWAT)
      - Lifted index
      - Bulk wind shear (0-1, 0-3, 0-6 km)
      - Storm-relative helicity (0-3 km)
      - Surface values
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary with p, t, qv, height, u, v
    
    Returns
    -------
    diag : dict
        Dictionary containing all diagnostic variables
    """
    from metpy.units import units
    from metpy.calc import (precipitable_water, lifted_index, 
                           bulk_shear, storm_relative_helicity,
                           bunkers_storm_motion,
                           dewpoint_from_specific_humidity,
                           wind_speed)
    
    diag = {}
    
    try:
        # Setup MetPy units
        p = sounding['p'] * units.hPa
        T = sounding['t'] * units.kelvin
        qv = sounding['qv'] * units('g/kg')
        z = sounding['height'] * units.meter
        u = sounding['u'] * units('m/s')
        v = sounding['v'] * units('m/s')
        
        # Convert qv to dewpoint
        q = qv / (1 + qv)
        Td = dewpoint_from_specific_humidity(p, T, q)
        
        # --- CAPE and CIN (most unstable) ---
        cape, cin = calculate_cape_cin(sounding, parcel_type='most_unstable')
        diag['mucape'] = cape
        diag['mucin'] = cin
        
        # --- Surface-based CAPE/CIN ---
        cape_sfc, cin_sfc = calculate_cape_cin(sounding, parcel_type='surface')
        diag['sbcape'] = cape_sfc
        diag['sbcin'] = cin_sfc
        
        # --- Precipitable water ---
        pwat = precipitable_water(p, Td)
        diag['pwat'] = float(pwat.magnitude)
        
        # --- Lifted Index (500 hPa) ---
        try:
            li = lifted_index(p, T, Td, p[0], T[0], Td[0])
            diag['lifted_index'] = float(li.magnitude) if hasattr(li, 'magnitude') else float(li)
        except:
            diag['lifted_index'] = np.nan
        
        # --- Bulk wind shear (various layers) ---
        try:
            # 0-1 km shear
            shear_0_1 = bulk_shear(p, u, v, height=z, depth=1000*units.meter)
            diag['shear_0_1km'] = float(wind_speed(*shear_0_1).magnitude)
            
            # 0-3 km shear  
            shear_0_3 = bulk_shear(p, u, v, height=z, depth=3000*units.meter)
            diag['shear_0_3km'] = float(wind_speed(*shear_0_3).magnitude)
            
            # 0-6 km shear
            shear_0_6 = bulk_shear(p, u, v, height=z, depth=6000*units.meter)
            diag['shear_0_6km'] = float(wind_speed(*shear_0_6).magnitude)
        except Exception as e:
            print(f"Warning: Shear calculation failed: {e}")
            diag['shear_0_1km'] = np.nan
            diag['shear_0_3km'] = np.nan
            diag['shear_0_6km'] = np.nan
        
        # --- Storm Relative Helicity (0-3 km) ---
        try:
            # Estimate storm motion using Bunkers right-mover method
            storm_u, storm_v = bunkers_storm_motion(p, u, v, z)
            
            srh_0_3, _, _ = storm_relative_helicity(z, u, v, 
                                                    depth=3000*units.meter,
                                                    storm_u=storm_u, 
                                                    storm_v=storm_v)
            diag['srh_0_3km'] = float(srh_0_3.magnitude)
        except:
            diag['srh_0_3km'] = np.nan
        
        # --- Surface-based variables ---
        diag['surface_theta'] = float(sounding['surf_theta'])
        diag['surface_qv'] = float(sounding['surf_qv'])
        diag['surface_p'] = float(sounding['surf_pressure'])
        diag['surface_t'] = float(sounding['t'][0])
        
    except Exception as e:
        print(f"Warning: Diagnostic calculation failed: {e}")
        # Return dict with NaN values
        for key in ['mucape', 'mucin', 'sbcape', 'sbcin', 'pwat', 
                   'lifted_index', 'shear_0_1km', 'shear_0_3km', 'shear_0_6km',
                   'srh_0_3km', 'surface_theta', 'surface_qv', 'surface_p', 'surface_t']:
            if key not in diag:
                diag[key] = np.nan
    
    return diag


# ============================================================================
# CONSISTENCY CHECK FUNCTIONS
# ============================================================================

def ensure_thermodynamic_consistency(sounding):
    """
    Final checks to ensure sounding is thermodynamically consistent.
    
    Performs two main corrections:
      1. Prevents theta from decreasing with height (except in inversions)
      2. Ensures moisture never exceeds saturation
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary (will be modified in place)
    
    Returns
    -------
    sounding : dict
        Corrected sounding
    corrections : dict
        Dictionary tracking what corrections were made
    """
    corrections = {
        'theta_adjusted': False,
        'theta_n_levels': 0,
        'theta_max_adjustment': 0.0,
        'qv_adjusted': False,
        'qv_n_levels': 0,
        'qv_max_reduction': 0.0
    }
    
    # --- 1. Ensure theta doesn't decrease significantly with height ---
    for i in range(1, len(sounding['theta'])):
        if sounding['theta'][i] < sounding['theta'][i-1] - 0.5:
            old_theta = sounding['theta'][i]
            sounding['theta'][i] = sounding['theta'][i-1] - 0.3
            corrections['theta_adjusted'] = True
            corrections['theta_n_levels'] += 1
            corrections['theta_max_adjustment'] = max(
                corrections['theta_max_adjustment'],
                abs(sounding['theta'][i] - old_theta)
            )
    
    # --- 2. Recompute temperature from corrected theta ---
    pi = ((sounding['p'] * 100.0) / P0) ** (Rd / cp)
    sounding['t'] = sounding['theta'] * pi
    
    # --- 3. Ensure moisture doesn't exceed saturation ---
    for i in range(len(sounding['qv'])):
        es = es_bolton(sounding['t'][i])
        qvs = epsilon * es / (sounding['p'][i] - (1 - epsilon) * es) * 1000.0
        
        if sounding['qv'][i] > qvs:
            old_qv = sounding['qv'][i]
            sounding['qv'][i] = 0.99 * qvs  # 99% of saturation
            corrections['qv_adjusted'] = True
            corrections['qv_n_levels'] += 1
            reduction = (old_qv - sounding['qv'][i]) / old_qv * 100
            corrections['qv_max_reduction'] = max(
                corrections['qv_max_reduction'],
                reduction
            )
    
    return sounding, corrections


# ============================================================================
# MAIN ORCHESTRATOR FUNCTION
# ============================================================================

def generate_sounding(params, base_sounding_file='input_sounding'):
    """
    Generate realistic atmospheric sounding for WRF idealized experiments.
    
    Main orchestrator function that coordinates all profile generation steps:
      1. Read base sounding template
      2. Generate temperature profile (θ → T)
      3. Generate moisture profile (qv from RH targets)
      4. Generate wind profile (hodograph with optional LLJ)
      5. Apply thermodynamic consistency checks
      6. Update surface values
    
    Designed for Sobol sensitivity analysis with Southern Hemisphere 
    convection. All parameters are controlled via the input dictionary.
    
    Parameters
    ----------
    params : dict
        Dictionary with parameter values. See individual profile generation
        functions for details. Key parameters:
        
        MOISTURE:
        - low_level_rh (%): BL relative humidity [60-95]
        - mid_level_rh (%): Mid-level RH [30-60]
        - upper_level_rh (%): Upper-level RH [10-40]
        - z_moisture_top (m): Moisture transition height [1000-2500]
        
        TEMPERATURE:
        - surface_theta (K): Surface potential temperature [296-306]
        - z_ml_top (m): Boundary layer top [600-1500]
        - z_trop_top (m): Tropopause height [11000-13000]
        - theta_lapse_bl (K/km): BL lapse rate [0.5-2.5]
        - theta_lapse (K/km): Tropospheric lapse rate [2.0-3.5]
        - theta_lapse_strat (K/km): Stratospheric lapse [15-25]
        
        WIND SHEAR:
        - shear_rate (1/s): Shear rate [1/1200-1/400]
        - shear_depth (m): Shear layer depth [3000-12000]
        - shear_curvature (0-1): Hodograph curvature [0-1]
        - low_level_jet (m/s): Optional LLJ strength [0-15]
        
    base_sounding_file : str
        Path to base input_sounding template file
        
    Returns
    -------
    sounding : dict
        Dictionary containing the complete modified sounding profile
        
    Raises
    ------
    Exception
        If any step in the sounding generation process fails
    """
    # Set defaults for any missing parameters
    defaults = {
        # Moisture parameters
        'low_level_rh': 85,
        'mid_level_rh': 50,
        'upper_level_rh': 20,
        'z_moisture_top': 2000,
        
        # Temperature parameters
        'surface_theta': 300,
        'z_ml_top': 1000,
        'z_trop_top': 12000,
        'theta_lapse_bl': 1.9,
        'theta_lapse': 2.5,
        'theta_lapse_strat': 20.0,
        
        # Wind shear parameters
        'shear_depth': 6000,
        'shear_rate': 0.00166,
        'shear_curvature': 0.0,
        
        # Low-level jet parameters (optional)
        'low_level_jet': 0,
        'llj_height': 1000,
        'llj_dir': 0,
        'llj_width': 400
    }

    for key, value in defaults.items():
        if key not in params:
            params[key] = value
    
    try:
        # --- Read base sounding template ---
        sounding = read_input_sounding(base_sounding_file)
        
        # --- Update surface theta ---
        sounding['surf_theta'] = params['surface_theta']

        # --- Update surface moisture ---
        t_sfc = params['surface_theta']  # At surface, theta ≈ T for p≈1000 hPa
        p_sfc = sounding['surf_pressure']
        es_sfc = es_bolton(t_sfc)
        qvs_sfc = epsilon * es_sfc / (p_sfc - (1 - epsilon) * es_sfc) * 1000.0
        sounding['surf_qv'] = qvs_sfc * (params['low_level_rh'] / 100.0)

        # --- 1. Generate temperature profile ---
        sounding = generate_temperature_profile(sounding, params)
        
        # --- 2. Generate moisture profile ---
        sounding = generate_moisture_profile(sounding, params)
        
        # --- 3. Generate wind profile ---
        sounding = generate_wind_profile_sh(sounding, params)
        
        # --- 4. Apply thermodynamic consistency checks ---
        sounding, corrections = ensure_thermodynamic_consistency(sounding)
        
        # --- Warn if significant corrections were made ---
        if corrections['theta_adjusted']:
            print(f"  WARNING: Theta adjusted at {corrections['theta_n_levels']} levels "
                  f"(max: {corrections['theta_max_adjustment']:.2f} K)")
        
        if corrections['qv_adjusted']:
            print(f"  WARNING: Moisture reduced at {corrections['qv_n_levels']} levels "
                  f"(max reduction: {corrections['qv_max_reduction']:.1f}%)")
        
        # Store corrections in sounding for later analysis
        sounding['corrections'] = corrections
        
        return sounding
        
    except Exception as e:
        print(f"ERROR in generate_sounding: {e}")
        raise