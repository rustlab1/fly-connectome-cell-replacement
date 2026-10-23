"""Compiled integration of the unchanged reduced yaw-controller equations."""
import numpy as np
from numba import njit
import flight as reference

@njit(cache=True)
def _integrate(noise,gain_R,gain_L,bias,drive,p):
    tau_yaw,G,w_sat,B0,tau_gust,sigma,speed,dt=p
    d=min(max(drive,.05),1.);speed*=d;sigma/=d
    a=np.exp(-dt/tau_gust);s=sigma*np.sqrt(1-a*a)
    omega=0.;gust=0.;psi=0.;x=0.;y=0.
    out=np.zeros((len(noise),5),np.float32)
    for i in range(len(noise)):
        gust=a*gust+s*noise[i]
        g=gain_L if omega>0 else gain_R
        cmd=-G*g*w_sat*np.tanh(omega/w_sat)+B0*bias
        omega+=dt/tau_yaw*(-omega+cmd+gust)
        psi+=omega*dt;x+=speed*np.cos(psi)*dt;y+=speed*np.sin(psi)*dt
        out[i,0]=i*dt;out[i,1]=psi;out[i,2]=omega;out[i,3]=x;out[i,4]=y
    return out

def simulate(gain_R=1.,gain_L=1.,bias=0.,T=10.,seed=0,params=None,drive=1.):
    if not np.isfinite([gain_R,gain_L,bias,drive,T]).all():raise ValueError('Nonfinite controller input; trajectory is undefined')
    p=dict(reference.P,**(params or {}));noise=np.random.default_rng(seed).standard_normal(int(T/p['dt']))
    pars=np.array([p[k] for k in ['tau_yaw','G','w_sat','B0','tau_gust','sigma_gust','speed','dt']],float)
    return _integrate(noise,float(gain_R),float(gain_L),float(bias),float(drive),pars)

def evaluate(gain_R,gain_L,bias,n_seeds=20,T=10.,drive=1.,params=None):
    perf=[reference.performance(simulate(gain_R,gain_L,bias,T=T,seed=s,drive=drive,params=params)) for s in range(n_seeds)]
    return {k:float(np.mean([q[k] for q in perf])) for k in perf[0]}|{k+'_sd':float(np.std([q[k] for q in perf])) for k in perf[0]}
