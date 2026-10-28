"""Controller sensitivity analysis (Methods).

Replays the stored gains, bias and motor outputs of each realization through the same
movement controller and 20 gust sequences, with the motor drive defined as in the paper
(M0_current), by wing-motor output alone (M1), held constant (M2), or scaling speed but not
gusts (M3). Values are distances normalized to matched intact, averaged over realizations.

Usage:
    python code/v8/controller_sensitivity.py results/v8/integrated_20261012/per_realization.csv \
        results/v8/tables/controller_sensitivity.csv
"""
import numpy as np, pandas as pd, sys
d=pd.read_csv(sys.argv[1])
print(np.allclose(d.drive,(d.motor_output+d.takeoff_output)/2))
DEG=np.pi/180; tau_yaw=.02;G=3.;w_sat=300*DEG;B0=150*DEG;tau_g=.15;sig0=100*DEG;sp0=.3;dt=.001;T=10.
nsteps=int(T/dt); nseeds=20
noise=np.stack([np.random.default_rng(s).standard_normal(nsteps) for s in range(nseeds)])  # 20 x n
def run(gR,gL,b,drive,scale_speed=True,scale_gust=True):
    gR=np.asarray(gR)[:,None];gL=np.asarray(gL)[:,None];b=np.asarray(b)[:,None]
    dd=np.clip(np.asarray(drive),.05,1.)[:,None]
    sp=sp0*(dd if scale_speed else 1.); sig=sig0/(dd if scale_gust else 1.)
    a=np.exp(-dt/tau_g); s=sig*np.sqrt(1-a*a)
    N=gR.shape[0]; om=np.zeros((N,nseeds)); gu=np.zeros_like(om); psi=np.zeros_like(om); x=np.zeros_like(om); y=np.zeros_like(om); path=np.zeros_like(om)
    for i in range(nsteps):
        gu=a*gu+s*noise[None,:,i]
        g=np.where(om>0,gL,gR)
        cmd=-G*g*w_sat*np.tanh(om/w_sat)+B0*b
        om=om+dt/tau_yaw*(-om+cmd+gu)
        psi=psi+om*dt
        x=x+sp*np.cos(psi)*dt; y=y+sp*np.sin(psi)*dt
    dist=np.hypot(x,y)
    return dist.mean(1)
conds=['intact','lesion20_500','lesion20_1000','lesion20_2000','selective_descending','selective_other_37','selection_random_100','selection_interface_100','selection_ranked_100','connections_random_partners_100','connections_correct_types_100','non_target_types_100','typed_dose_correct_types_2000','non_target_types_2000','review_full_random_partners_2000']
sub=d[d.name.isin(conds)].copy()
mo_int=None
res={}
maps={'M0_current':dict(drv='drive',ss=True,sg=True),
      'M1_wingpower_only':dict(drv='motor_output',ss=True,sg=True),
      'M2_no_drive':dict(drv=None,ss=False,sg=False),
      'M3_speed_only_no_gust_scaling':dict(drv='drive',ss=True,sg=False)}
for m,cfg in maps.items():
    drv=sub[cfg['drv']].values if cfg['drv'] else np.ones(len(sub))
    sub[m]=run(sub.gain_R.values,sub.gain_L.values,sub.bias.values,drv,cfg['ss'],cfg['sg'])
# check reproduction
print('M0 vs stored distance: max abs diff', np.abs(sub['M0_current']-sub['distance_m']).max(), 'corr', np.corrcoef(sub['M0_current'],sub['distance_m'])[0,1])
intact=sub[sub.name=='intact'].set_index('seed')
out=[]
for c in conds:
    s=sub[sub.name==c].set_index('seed')
    row={'cond':c}
    for m in maps:
        row[m]=(s[m]/intact.loc[s.index,m]).mean()
    row['stored_norm']=s['distance_m_norm'].mean()
    out.append(row)
o=pd.DataFrame(out); pd.set_option('display.width',200); print(o.round(3).to_string(index=False))
o.to_csv(sys.argv[2] if len(sys.argv)>2 else 'controller_sensitivity.csv',index=False)
