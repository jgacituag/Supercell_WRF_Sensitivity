
# mapper_u01_to_conf.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    return a + (b - a) * min(max(u, 0.0), 1.0)

def u01_to_phys(u: List[float]) -> PhysParams:
    if len(u) != 10:
        raise ValueError("Expected len(u)==10")
    u0,u1,u2,u3,u4,u5,u6,u7,u8,u9 = u
    Tsfc_C    = lin(21.0, 30.0, u0)
    Gamma_bdy = 6.5
    Gamma_trp = lin(5.5, 7.5, u1)
    Dtrp_km   = lin(12.0, 16.0, u2)
    RHsfc     = lin(0.75, 0.90, u3)
    Dwv_max   = max(4.0, Dtrp_km - 1.0)
    Dwv_km    = lin(4.0, Dwv_max, u4)
    Umax_ms   = lin(10.0, 45.0, u5)
    Dshear_km = lin(3.0, 12.0, u6)
    curved_shear_per = lin(0.0, 0.6, u7)
    llj_ms    = lin(0.0, 8.0, u8)
    llj_h_km  = lin(0.5, 1.2, u9)
    return PhysParams(Tsfc_C, Gamma_bdy, Gamma_trp, Dtrp_km, RHsfc, Dwv_km,
                      Umax_ms, Dshear_km, curved_shear_per, llj_ms, llj_h_km)

def shear_intensity(Umax_ms: float, Dshear_km: float) -> float:
    S_ms_per_km = Umax_ms / max(Dshear_km, 0.1)
    return max(0.0, S_ms_per_km / 1000.0)

def phys_to_conf(p: PhysParams) -> Dict:
    conf = {}
    conf['modify_wind_profile'] = True
    conf['remove_mean_wind']    = True
    conf['shear_type']          = 'Curved'
    conf['total_shear_depth']   = p.Dshear_km * 1000.0
    conf['int_total_shear']     = max(1.0e-5, shear_intensity(p.Umax_ms, p.Dshear_km))
    conf['curved_shear_per']    = p.curved_shear_per
    conf['shear_depth_u']    = 0.0
    conf['shear_strength_u'] = 5.0e-3
    conf['shear_depth_v']    = 8000.0
    conf['shear_strength_v'] = 0.0
    conf['llj_amp']   = p.llj_ms
    conf['llj_h']     = p.llj_h_km * 1000.0
    conf['llj_width'] = 500.0
    conf['llj_dir']   = 360.0
    conf['surf_u'] = 0.0
    conf['surf_v'] = 0.0
    conf['modify_stability'] = True
    conf['stability_factor_height'] = min( (p.Dtrp_km * 1000.0), 5000.0 )
    conf['stability_factor'] = (p.Gamma_trp - 6.5) / 1.0
    conf['surface_temperature_offset'] = p.Tsfc_C - 24.0
    conf['modify_moisture_profile'] = True
    conf['dry_run'] = False
    conf['low_level_moisture_height'] = min(p.Dwv_km * 1000.0, p.Dtrp_km * 1000.0 - 1000.0)
    mult = -10.0 + (p.RHsfc - 0.75) / (0.90 - 0.75) * (5.0 + 10.0)
    conf['low_level_moisture_mult_factor'] = mult
    conf['mid_level_moisture_height'] = 2000.0
    conf['mid_level_moisture_mult_factor'] = 0.0
    return conf

def map_u01_to_conf(u: List[float]):
    phys = u01_to_phys(u)
    conf = phys_to_conf(phys)
    return phys, conf
