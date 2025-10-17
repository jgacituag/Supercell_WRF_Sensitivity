from metpy.units import units
from metpy.calc import mixed_layer_cape_cin,most_unstable_cape_cin,surface_based_cape_cin
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle

# Physical constants
cp = 1004.0  # Specific heat at constant pressure (J/kg/K)
Rd = 287.0   # Gas constant for dry air (J/kg/K)
g = 9.81     # Gravity (m/s²)
P0 = 100000.0  # Reference pressure (Pa)
epsilon = 0.622  # Ratio of molecular weights (water vapor / dry air)

def generate_sounding(params, base_sounding_file='input_sounding'):
    """
    Generate realistic atmospheric sounding for WRF idealized experiments.
    Designed for Sobol sensitivity analysis with Southern Hemisphere convection.
    
    Parameters
    ----------
    params : dict
        Dictionary with the following keys:
        
        Moisture control:  # UPDATED DESCRIPTION
        - 'low_level_rh': Low-level relative humidity (%), range: [75, 95]
        - 'mid_level_rh': Mid-level relative humidity (%), range: [30, 70]
        - 'upper_level_rh': Upper-level relative humidity (%), range: [10, 40]
        
        Temperature profile:
        - 'surface_theta': Surface potential temperature (K), range: [297, 305]
        - 'low_level_lapse': Low-level lapse rate (K/km), range: [6.0, 8.5]
        - 'mid_level_lapse': Mid-level lapse rate (K/km), range: [5.0, 7.0]
        
        Wind shear (Southern Hemisphere typical):
        - 'shear_magnitude': 0-6km bulk shear (m/s), range: [10, 35]
        - 'shear_curvature': 0=linear, 1=quarter circle, range: [0, 1]
        - 'shear_direction': Clockwise from north (degrees), range: [0, 360]
        - 'shear_depth': Height of shear layer (m), range: [4000, 8000]
        - 'low_level_jet': LLJ strength (m/s), range: [0, 15]
        - 'llj_height': LLJ height (m), range: [500, 1500]
        
    base_sounding_file : str
        Path to base input_sounding file
        
    Returns
    -------
    sounding : dict
        Dictionary containing the modified sounding profile
    """
    
    # Set defaults for any missing parameters
    defaults = {
        'low_level_rh': 85,
        'mid_level_rh': 50,
        'upper_level_rh': 20,
        'surface_theta': 300,
        'low_level_lapse': 7.0,
        'mid_level_lapse': 6.0,
        'shear_magnitude': 25,
        'shear_curvature': 0.0,
        'shear_direction': 270,
        'shear_depth': 6000,
        'low_level_jet': 0,
        'llj_height': 1000
    }

    for key, value in defaults.items():
        if key not in params:
            params[key] = value
    
    try:
        # Read base sounding
        sounding = read_input_sounding(base_sounding_file)
        sounding['surf_theta'] = params['surface_theta']

        t_sfc = params['surface_theta']  # At surface, theta ≈ T for p=1000 hPa
        p_sfc = sounding['surf_pressure']

        es_sfc = 6.112 * np.exp(17.67 * (t_sfc - 273.16) / (t_sfc - 29.65))
        qvs_sfc = epsilon * es_sfc / (p_sfc - (1 - epsilon) * es_sfc) * 1000.0
        sounding['surf_qv'] = qvs_sfc * (params['low_level_rh'] / 100.0)

        # 1. Generate temperature profile
        sounding = generate_temperature_profile(sounding, params)
        
        # 2. Generate moisture profile
        sounding = generate_moisture_profile(sounding, params)
        
        # 3. Generate wind profile (Southern Hemisphere typical)
        sounding = generate_wind_profile_sh(sounding, params)
        
        # 4. Final consistency checks
        sounding, corrections = ensure_thermodynamic_consistency(sounding)
        
        # Warn if significant corrections were made
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

def calculate_diagnostics(sounding):
    """
    Calculate comprehensive diagnostics using MetPy.
    
    Returns
    -------
    diag : dict
        Dictionary with all diagnostic variables
    """
    from metpy.units import units
    from metpy.calc import (precipitable_water, lifted_index, 
                           bulk_shear, storm_relative_helicity)
    import warnings
    warnings.filterwarnings('ignore')
    
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
        from metpy.calc import dewpoint_from_specific_humidity
        q = qv / (1 + qv)
        Td = dewpoint_from_specific_humidity(p, T, q)
        
        # CAPE and CIN (most unstable)
        cape, cin = calculate_cape_cin(sounding, parcel_type='most_unstable')
        diag['mucape'] = cape
        diag['mucin'] = cin
        
        # Surface-based CAPE/CIN
        cape_sfc, cin_sfc = calculate_cape_cin(sounding, parcel_type='surface')
        diag['sbcape'] = cape_sfc
        diag['sbcin'] = cin_sfc
        
        # Precipitable water
        pwat = precipitable_water(p, Td)
        diag['pwat'] = float(pwat.magnitude)
        
        # Lifted Index (500 hPa)
        try:
            li = lifted_index(p, T, Td, p[0], T[0], Td[0])
            diag['lifted_index'] = float(li.magnitude) if hasattr(li, 'magnitude') else float(li)
        except:
            diag['lifted_index'] = np.nan
        
        # Bulk wind shear (various layers)
        try:
            # 0-1 km shear
            shear_0_1 = bulk_shear(p, u, v, height=z, depth=1000*units.meter)
            diag['shear_0_1km'] = float(shear_0_1[0].magnitude)
            
            # 0-3 km shear  
            shear_0_3 = bulk_shear(p, u, v, height=z, depth=3000*units.meter)
            diag['shear_0_3km'] = float(shear_0_3[0].magnitude)
            
            # 0-6 km shear
            shear_0_6 = bulk_shear(p, u, v, height=z, depth=6000*units.meter)
            diag['shear_0_6km'] = float(shear_0_6[0].magnitude)
        except:
            diag['shear_0_1km'] = np.nan
            diag['shear_0_3km'] = np.nan
            diag['shear_0_6km'] = np.nan
        
        # Storm Relative Helicity (0-3 km)
        try:
            # Need to estimate storm motion (use Bunkers right-mover as default)
            from metpy.calc import bunkers_storm_motion
            storm_u, storm_v = bunkers_storm_motion(p, u, v, z)
            
            srh_0_3, _, _ = storm_relative_helicity(z, u, v, 
                                                    depth=3000*units.meter,
                                                    storm_u=storm_u, 
                                                    storm_v=storm_v)
            diag['srh_0_3km'] = float(srh_0_3.magnitude)
        except:
            diag['srh_0_3km'] = np.nan
        
        # Surface-based variables
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


def generate_temperature_profile(sounding, params):
    """Generate realistic temperature profile with controlled lapse rates."""
    
    surface_theta = params['surface_theta']
    low_level_lapse = params['low_level_lapse']  # K/km
    mid_level_lapse = params['mid_level_lapse']  # K/km
    
    # Define layer boundaries
    low_level_top = 3000  # m
    mid_level_top = 9000  # m
    tropopause = 13000   # m 
    
    theta = np.zeros_like(sounding['height'])
    
    for i, z in enumerate(sounding['height']):
        if z <= low_level_top:
            # Boundary layer: steeper lapse rate
            dtheta = low_level_lapse * (z / 1000.0)
            theta[i] = surface_theta + dtheta
        elif z <= mid_level_top:
            # Mid-levels: more stable
            theta_at_top_low = surface_theta + low_level_lapse * (low_level_top / 1000.0)
            dz = z - low_level_top
            dtheta = mid_level_lapse * (dz / 1000.0)
            theta[i] = theta_at_top_low + dtheta
        elif z <= tropopause:
            # Upper troposphere: transition to stratosphere
            theta_at_mid_top = (surface_theta + 
                              low_level_lapse * (low_level_top / 1000.0) +
                              mid_level_lapse * ((mid_level_top - low_level_top) / 1000.0))
            dz = z - mid_level_top
            # Decreasing lapse rate toward tropopause
            lapse = mid_level_lapse * (1 - 0.3 * (dz / (tropopause - mid_level_top)))
            dtheta = lapse * (dz / 1000.0)
            theta[i] = theta_at_mid_top + dtheta
        else:
            # Stratosphere: stable layer
            theta_at_trop = (surface_theta + 
                           low_level_lapse * (low_level_top / 1000.0) +
                           mid_level_lapse * ((mid_level_top - low_level_top) / 1000.0) +
                           mid_level_lapse * 0.85 * ((tropopause - mid_level_top) / 1000.0))
            dz = z - tropopause
            theta[i] = theta_at_trop + 1.5 * (dz / 1000.0)  # Stratospheric inversion
    
    sounding['theta'] = theta
    
    # Convert theta to temperature
    pi = ((sounding['p'] * 100.0) / P0) ** (Rd / cp)
    sounding['t'] = theta * pi
    
    return sounding


def generate_moisture_profile(sounding, params):
    """Generate moisture profile with specified RH at different levels."""
    
    low_level_rh = params['low_level_rh']
    mid_level_rh = params['mid_level_rh']
    upper_level_rh = params['upper_level_rh']
    
    # Define moisture layer boundaries
    low_level_top = 2000  # m
    mid_level_top = params['moisture_depth']  # m
    dry_level_top = 10000  # m
    
    qv = np.zeros_like(sounding['height'])
    
    for i, (z, t, p) in enumerate(zip(sounding['height'], sounding['t'], sounding['p'])):
        # Calculate saturation mixing ratio
        es = 6.112 * np.exp(17.67 * (t - 273.16) / (t - 29.65))  # hPa
        qvs = epsilon * es / (p - (1 - epsilon) * es) * 1000.0  # g/kg
        
        # Determine target RH based on height
        if z <= low_level_top:
            # Boundary layer: high moisture
            target_rh = low_level_rh
        elif z <= mid_level_top:
            # Mid-levels: transition
            frac = (z - low_level_top) / (mid_level_top - low_level_top)
            target_rh = low_level_rh + (mid_level_rh - low_level_rh) * frac
        elif z <= dry_level_top:
            # Upper mid-levels
            frac = (z - mid_level_top) / (dry_level_top - mid_level_top)
            target_rh = mid_level_rh + (upper_level_rh - mid_level_rh) * frac
        else:
            # Upper troposphere: very dry
            target_rh = upper_level_rh * np.exp(-(z - dry_level_top) / 3000.0)
        
        qv[i] = qvs * (target_rh / 100.0)
    
    sounding['qv'] = qv
    
    return sounding


def generate_wind_profile_sh(sounding, params,remove_mean_wind=True):
    """
    Generate wind profile typical for Southern Hemisphere supercells.
    
    Uses shear_rate * shear_depth to define bulk shear magnitude (U_max).
    """
    
    shear_depth = params['shear_depth']  # 
    shear_rate = params['shear_rate']     # s^-1 (New input parameter)
    
    # Calculate the total bulk shear magnitude (U_max) from the new inputs
    shear_mag = shear_rate * shear_depth # <-- UPDATED: Derived value (equivalent to U_max/D_shear * D_shear)
    
    curvature = params['shear_curvature']  # 0=linear, 1=half circle
    direction = params['shear_direction']  # degrees
    llj_strength = params['low_level_jet']
    llj_height = params['llj_height']
    
    u = np.zeros_like(sounding['height'])
    v = np.zeros_like(sounding['height'])
    
    for i, z in enumerate(sounding['height']):
        if z <= shear_depth:
            # Fraction through shear layer
            frac = z / shear_depth
            
            if curvature == 0:
                # Linear shear
                speed = shear_mag * frac
                angle = direction
            else:
                # Interpolate between linear and circular
                linear_speed = shear_mag * frac
                linear_angle = direction
                
                # Circular component (SH: clockwise rotation)
                theta_rad = curvature * (np.pi / 2) * frac
                circular_u = shear_mag * np.sin(theta_rad) * np.sin(np.radians(direction))
                circular_v = -shear_mag * np.sin(theta_rad) * np.cos(np.radians(direction))
                
                # Blend
                u_linear = linear_speed * np.sin(np.radians(linear_angle))
                v_linear = -linear_speed * np.cos(np.radians(linear_angle))
                
                u[i] = (1 - curvature) * u_linear + curvature * circular_u
                v[i] = (1 - curvature) * v_linear + curvature * circular_v
                continue
            
            u[i] =  speed * np.sin(np.radians(angle))
            v[i] = -speed * np.cos(np.radians(angle))  # Negative for SH
        else:
            # Above shear layer: constant wind
            idx = int(shear_depth / (sounding['height'][1] - sounding['height'][0]))
            if idx < len(u):
                u[i] = u[idx]
                v[i] = v[idx]
    
    # Add low-level jet if specified
    if llj_strength > 0:
        llj_profile = llj_strength * np.exp(-0.5 * ((sounding['height'] - llj_height) / 400) ** 2)
        # Add jet perpendicular to shear direction (typical configuration)
        jet_dir = (direction + 90) % 360
        u +=  llj_profile * np.sin(np.radians(jet_dir))
        v += -llj_profile * np.cos(np.radians(jet_dir))
    
    sounding['u'] = u
    sounding['v'] = v
    if remove_mean_wind:
        # Remove 0–6 km mean wind (keep convection near domain center longer)
        mask = sounding['height'] < 6000.0
        mean_u = np.mean(sounding['u'][mask])
        mean_v = np.mean(sounding['v'][mask])
        sounding['u'] = sounding['u'] - mean_u
        sounding['v'] = sounding['v'] - mean_v
    
    return sounding

def calculate_cape_cin(sounding, parcel_type='most_unstable'):
    """
    Calculate CAPE and CIN using MetPy (much more accurate!).
    
    Parameters
    ----------
    sounding : dict
        Sounding dictionary with p, t, qv, height
    parcel_type : str
        'most_unstable', 'surface', or 'mixed_layer'
        
    Returns
    -------
    cape : float
        CAPE in J/kg
    cin : float  
        CIN in J/kg
    """
    
    # Prepare arrays with MetPy units
    p = sounding['p'] * units.hPa
    T = sounding['t'] * units.kelvin
    qv = sounding['qv'] * units('g/kg')
    
    # Convert mixing ratio to dewpoint
    from metpy.calc import dewpoint_from_specific_humidity
    # First convert qv (g/kg) to specific humidity (dimensionless)
    q = qv / (1 + qv)  # Approximate conversion
    Td = dewpoint_from_specific_humidity(p, T, q)
    
    try:
        if parcel_type == 'most_unstable':
            # Find most unstable parcel in lowest 300 hPa
            #parcel_p, parcel_t, parcel_td, _ = most_unstable_parcel(p, T, Td, depth=300*units.hPa)
            #cape_val, cin_val = cape_cin(p, T, Td, parcel_p, parcel_t, parcel_td)
            cape_val, cin_val = most_unstable_cape_cin(p, T, Td)
        elif parcel_type == 'surface':
            # Surface-based parcel
            cape_val, cin_val = surface_based_cape_cin(p, T, Td)
            
        elif parcel_type == 'mixed_layer':
            # Mixed layer (lowest 100 hPa)
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
        print(f"Warning: MetPy CAPE/CIN calculation failed: {e}")
        # Fallback to zero
        return 0.0, 0.0


def ensure_thermodynamic_consistency(sounding):
    """
    Final checks to ensure sounding is thermodynamically consistent.
    
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
    
    # 1. Ensure temperature decreases with height (except in inversions)
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
    
    # 2. Recompute temperature from theta
    pi = ((sounding['p'] * 100.0) / P0) ** (Rd / cp)
    sounding['t'] = sounding['theta'] * pi
    
    # 3. Ensure moisture doesn't exceed saturation
    for i in range(len(sounding['qv'])):
        es = 6.112 * np.exp(17.67 * (sounding['t'][i] - 273.16) / 
                           (sounding['t'][i] - 29.65))
        qvs = epsilon * es / (sounding['p'][i] - (1 - epsilon) * es) * 1000.0
        
        if sounding['qv'][i] > qvs:
            old_qv = sounding['qv'][i]
            sounding['qv'][i] = 0.99 * qvs
            corrections['qv_adjusted'] = True
            corrections['qv_n_levels'] += 1
            reduction = (old_qv - sounding['qv'][i]) / old_qv * 100
            corrections['qv_max_reduction'] = max(
                corrections['qv_max_reduction'],
                reduction
            )
    
    return sounding, corrections

def read_input_sounding(filename):
    """Read WRF input_sounding file."""
    
    with open(filename) as f:
        lines = f.readlines()
    # First line: surface values
    line_split = lines[0].split()
    surf_pressure = float(line_split[0])
    surf_theta = float(line_split[1])
    surf_qv = float(line_split[2])
    
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
    
    # Calculate pressure at each level (hydrostatic)
    p = np.zeros(nlevs)
    p[0] = surf_pressure
    
    for i in range(1, nlevs):
        theta_mean = 0.5 * (theta[i] + theta[i-1])
        dz = height[i] - height[i-1]
        pi_prev = (p[i-1] * 100) ** (Rd / cp)
        pi = pi_prev - g * (P0 ** (Rd / cp)) * dz / (cp * theta_mean)
        p[i] = (pi ** (cp / Rd)) / 100  # hPa
    
    # Calculate temperature
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
    """Write WRF input_sounding file."""
    
    with open(filename, 'w') as f:
        # Write surface values
        f.write(f"{sounding['surf_pressure']:.4f} {sounding['surf_theta']:.4f} "
                f"{sounding['surf_qv']:.6f}\n")
        
        # Write profile
        for i in range(sounding['nlevs']):
            f.write(f"{sounding['height'][i]:10.4f} {sounding['theta'][i]:10.4f} "
                   f"{sounding['qv'][i]:10.6f} {sounding['u'][i]:10.4f} "
                   f"{sounding['v'][i]:10.4f}\n")