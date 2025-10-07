#!/usr/bin/env python3
"""
envgen_analytic.py
------------------
Analytic environment generator for idealized WRF soundings with a **linear→arc** hodograph.
(… full docstring omitted here for brevity in this code cell …)
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

@dataclass
class PhysParams:
    Tsfc_C: float; Gamma_bdy: float; Gamma_trp: float; Dtrp_km: float; RHsfc: float; Dwv_km: float
    Umax_ms: float; Dshear_km: float
    turn_frac: float; join_speed_frac: float; theta0_deg: float; arc_signed_deg: float
    llj_ms: float; llj_h_km: float
    curved_shear_per: float = 0.0

def _lin(a,b,u): return a + (b-a)*float(np.clip(u,0.0,1.0))

def u01_to_phys(u: List[float]) -> PhysParams:
    uu = list(map(float,u))
    if len(uu)<10: raise ValueError("Expected len(u) >= 10")
    Tsfc_C=_lin(21,30,uu[0]); Gamma_bdy=6.5; Gamma_trp=_lin(5.5,7.5,uu[1])
    Dtrp_km=_lin(12,16,uu[2]); RHsfc=_lin(0.75,0.90,uu[3]); Dwv_max=max(4.0,Dtrp_km-1.0)
    Dwv_km=_lin(4.0,Dwv_max,uu[4]); Umax_ms=_lin(10,45,uu[5]); Dshear_km=_lin(3,12,uu[6])
    curved_shear_per=_lin(0,0.6,uu[7])
    turn_frac=0.35; join_speed_frac=0.5; theta0_deg=-30.0; arc_signed_deg=180.0
    llj_ms=_lin(0,8,uu[8]) if len(uu)>=9 else 0.0; llj_h_km=_lin(0.5,1.2,uu[9]) if len(uu)>=10 else 0.9
    if len(uu)>=13:
        turn_frac=_lin(0.25,0.50,uu[7]); join_speed_frac=_lin(0.30,0.70,uu[8])
        theta0_deg=_lin(-60,15,uu[9]); arc_signed_deg=_lin(-150,150,uu[10])
        llj_ms=_lin(0,8,uu[11]); llj_h_km=_lin(0.5,1.2,uu[12])
    return PhysParams(Tsfc_C,Gamma_bdy,Gamma_trp,Dtrp_km,RHsfc,Dwv_km,Umax_ms,Dshear_km,
                      turn_frac,join_speed_frac,theta0_deg,arc_signed_deg,llj_ms,llj_h_km,
                      curved_shear_per=curved_shear_per)

def build_temperature_profile(z,p):
    z_km=z/1000.0; T0=p.Tsfc_C+273.15; T=np.empty_like(z,float)
    T_bl=T0-p.Gamma_bdy*np.minimum(z_km,1.0); T[:]=T_bl
    m=z_km>1.0; z_trp=np.clip(z_km-1.0,0.0,None)
    T[m]=(T_bl[np.searchsorted(z_km,1.0,side='left')] - p.Gamma_trp*z_trp[m])
    m2=z_km>p.Dtrp_km; z_str=z_km-p.Dtrp_km
    T[m2]=T[np.searchsorted(z_km,p.Dtrp_km,side='left')] + 2.0*z_str[m2]
    return T

def _sat_mr(p_hpa,T_K):
    T_C=T_K-273.15; es=6.112*np.exp((17.67*T_C)/(T_C+243.5)); eps=0.622
    return eps*(es/np.maximum(p_hpa-es,1e-3))

def build_moisture_profile(z,p_hpa,T_K,phys):
    z_km=z/1000.0; RH=np.ones_like(z)*phys.RHsfc; top=phys.Dwv_km
    mask=z_km>top
    if np.any(mask):
        idx=np.where(mask)[0]; RH[idx]=np.linspace(phys.RHsfc,0.2,len(idx))
    qv=RH*_sat_mr(p_hpa,T_K); return qv*1000.0

def _hypsometric(z,T_K,qv_kgkg,p0_hpa=1000.0):
    g=9.80665; Rd=287.04; Tv=T_K*(1.0+0.61*np.maximum(qv_kgkg,0.0))
    p=np.empty_like(z,float); p[0]=p0_hpa*100.0
    for k in range(1,len(z)):
        dz=z[k]-z[k-1]; Tv_mid=0.5*(Tv[k]+Tv[k-1]); p[k]=p[k-1]*np.exp(-g*dz/(Rd*Tv_mid))
    return p/100.0

def build_wind_profile_linear_arc(z,phys):
    z_km=z/1000.0; D=max(phys.Dshear_km,0.1); Uend=float(phys.Umax_ms)
    z_turn=float(np.clip(phys.turn_frac,0.05,0.95)*D)
    th0=np.deg2rad(phys.theta0_deg)
    alpha=np.deg2rad(np.clip(phys.arc_signed_deg,-179.9,179.9))
    r0=np.clip(phys.join_speed_frac,0.05,0.95)*Uend
    P0=np.array([r0*np.cos(th0),r0*np.sin(th0)])
    th1=th0+alpha
    Dv=np.array([np.sin(th1)-np.sin(th0), -(np.cos(th1)-np.cos(th0))])
    a=float(Dv@Dv); b=2.0*float(P0@Dv)
    c=float(P0@P0)-Uend**2
    disc=max(0.0,b*b-4*a*c)
    R=(-b+np.sqrt(disc))/(2*a) if a>0 else 0.0
    R=max(R,1e-6)
    C=P0 - R*np.array([np.sin(th0),-np.cos(th0)])
    u=np.zeros_like(z,float)
    v=np.zeros_like(z,float)
    
    m=z_km<=z_turn+1e-6; frac=(z_km[m]/z_turn) if z_turn>0 else 0.0; u[m]=frac*P0[0]; v[m]=frac*P0[1]
    m2=z_km>z_turn
    if np.any(m2):
        phi0=th0-np.pi/2.0; phi1=th1-np.pi/2.0; t=(z_km[m2]-z_turn)/max(D-z_turn,1e-6)
        phi=phi0 + t*(phi1-phi0); u[m2]=C[0]+R*np.cos(phi); v[m2]=C[1]+R*np.sin(phi)
    if getattr(phys,"llj_ms",0.0)>0.0:
        sigma=0.3; u+=phys.llj_ms*np.exp(-0.5*((z_km-phys.llj_h_km)/sigma)**2)
    return u,v

def generate_sounding(u: List[float], z_top_km: float = 30.0, dz_m: float = 100.0) -> Dict[str, np.ndarray]:
    phys=u01_to_phys(u); z=np.arange(0.0,z_top_km*1000.0+dz_m,dz_m,float)
    T=build_temperature_profile(z,phys); p=np.full_like(z,1000.0); qv=np.zeros_like(z)
    for _ in range(2):
        p=_hypsometric(z,T,qv/1000.0,1000.0); qv=build_moisture_profile(z,p,T,phys)
    u_arr,v_arr=build_wind_profile_linear_arc(z,phys)
    return dict(height=z,p=p,t=T,qv=qv,u=u_arr,v=v_arr)

def dewpoint_from_qv(p_hpa,qv_gkg):
    eps=0.622; r=qv_gkg/1000.0; p_pa=p_hpa*100.0; e_pa=(r/(eps+r))*p_pa; e_hpa=e_pa/100.0
    ln=np.log(np.maximum(e_hpa,1e-6)/6.112); Td_C=(243.5*ln)/(17.67-ln); return Td_C+273.15

def bulk_shear(u,v,z,depth_m):
    ut=np.interp(depth_m,z,u); vt=np.interp(depth_m,z,v); return float(np.hypot(ut-u[0],vt-v[0]))

def precipitable_water(p_hpa,qv_gkg):
    g=9.80665; q=np.maximum(qv_gkg,0.0)/1000.0; p_pa=p_hpa*100.0; return float(np.trapz(q,p_pa)/g)

def diagnostics(sound):
    z=sound['height']; p=sound['p']; T=sound['t']; qv=sound['qv']; u=sound['u']; v=sound['v']
    td=dewpoint_from_qv(p,qv); d={}
    if HAS_METPY:
        p_q=p*units.hectopascal; T_q=T*units.kelvin; Td_q=td*units.kelvin
        cape,cin=mpcalc.mixed_layer_cape_cin(p_q,T_q,Td_q,depth=100*units.hectopascal)
        d['MLCAPE']=float(cape.to('joule / kilogram').m); d['MLCIN']=float(cin.to('joule / kilogram').m)
        try:
            sm_u,sm_v,*_=mpcalc.bunkers_storm_motion(u*units('m/s'),v*units('m/s'),z*units.m)
            srh,_,_=mpcalc.storm_relative_helicity(u*units('m/s'),v*units('m/s'),z*units.m,
                                                   depth=3000*units.m,storm_u=sm_u,storm_v=sm_v)
            d['SRH03']=float(srh.m)
        except Exception:
            d['SRH03']=np.nan
    else:
        d['MLCAPE']=np.nan; d['MLCIN']=np.nan; d['SRH03']=np.nan
    d['SH06']=bulk_shear(u,v,z,6000.0); d['SH01']=bulk_shear(u,v,z,1000.0); d['PW']=precipitable_water(p,qv)
    return d

def write_input_sounding(path, sound, psfc_hpa=1000.0):
    p=sound['p']; T=sound['t']; qv=sound['qv']; u=sound['u']; v=sound['v']; z=sound['height']
    T_sfc=float(T[0]); qv_sfc=float(qv[0])
    with open(path,'w') as f:
        f.write(f"{psfc_hpa:12.4f}{T_sfc:12.4f}{qv_sfc:12.6f}\n")
        for zi,Ti,qvi,ui,vi in zip(z,T,qv,u,v):
            f.write(f"{zi:12.4f}{Ti:12.4f}{qvi:12.6f}{ui:12.4f}{vi:12.4f}\n")

if __name__=="__main__":
    s=generate_sounding([0.5]*13); write_input_sounding("input_sounding_test",s,psfc_hpa=1000.0); print("Wrote input_sounding_test")
