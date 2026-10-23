"""Build payloads for journal-style flight movies from frozen V8 example paths (no re-simulation)."""
import json, base64, hashlib, os, numpy as np, pandas as pd
from pathlib import Path
# FLY project root (defaults to three levels above code/v8/journal_movies)
U=Path(os.environ.get('FLY_ROOT', Path(__file__).resolve().parents[3]))
TR=U/'results/v8/trajectories'; TR2=U/'results/v8/figure2_full_replacement_20261012/trajectories'
AN=U/'videos/v8/publication_movies_20261012'
per=pd.read_csv(U/'results/v8/integrated_20261012/per_realization.csv')
means=per.groupby('name')[['distance_m_norm','motor_output_norm','takeoff_output_norm']].mean()
FPS=60; NF=600
ORANGE='#ce682b'; TEAL='#138479'; BLUE='#2465a6'; GREY='#50545b'; PINK='#c16888'; PURPLE='#7250ae'; GREEN='#008b78'
hashes={}
def traj(name):
    p=TR/f'{name}.csv'
    if not p.exists(): p=TR2/f'{name}.csv'
    hashes[str(p.relative_to(U))]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.read_csv(p); return d
def panel(name,title,sub,color,anat):
    d=traj(name); t=d.time_s.to_numpy(); x=d.x_m.to_numpy(); y=d.y_m.to_numpy()
    psi=np.unwrap(d.heading_rad.to_numpy()); om=d.yaw_rad_s.to_numpy()
    # display-only roll: low-pass filtered yaw rate (tau 80 ms), does not affect positions
    a=np.exp(-0.001/0.08); f=np.zeros_like(om); acc=0.
    for i,v in enumerate(om): acc=a*acc+(1-a)*v; f[i]=acc
    tf=np.arange(NF+1)/FPS; tf[-1]=min(tf[-1],t[-1])
    # positions start at origin (time 0 state before first step)
    t0=np.concatenate([[0.],t+0.0]) if t[0]>0 else t
    X=np.interp(tf,np.concatenate([[0.],t]),np.concatenate([[0.],x])); Y=np.interp(tf,np.concatenate([[0.],t]),np.concatenate([[0.],y]))
    P=np.interp(tf,np.concatenate([[0.],t]),np.concatenate([[psi[0]],psi])); R=np.interp(tf,t,f)
    # cumulative path length at frame times (display of straightness not used)
    seg=np.hypot(np.diff(np.concatenate([[0.],x])),np.diff(np.concatenate([[0.],y])))
    L=np.interp(tf,t,np.cumsum(seg))
    path=np.stack([np.concatenate([[0.],x[9::10]]),np.concatenate([[0.],y[9::10]])],1)  # every 10 ms
    pt=np.concatenate([[0.],t[9::10]])
    dots=[[float(np.interp(s,t,x)),float(np.interp(s,t,y))] for s in range(1,11)]
    an=AN/anat
    return dict(name=name,title=title,sub=sub,color=color,
        frames=np.round(np.stack([X,Y,P,R,L],1),6).tolist(),path=np.round(path,5).tolist(),path_t=np.round(pt,4).tolist(),dots=dots,
        anatomy='data:image/png;base64,'+base64.b64encode(an.read_bytes()).decode(),
        mean_pct=float(100*means.loc[name,'distance_m_norm']),end_dist=float(np.hypot(x[-1],y[-1])))
intact=traj('intact'); ip=np.stack([np.concatenate([[0.],intact.x_m.to_numpy()[9::10]]),np.concatenate([[0.],intact.y_m.to_numpy()[9::10]])],1)
def table(names):
    return [[round(100*means.loc[n,c]) for n in names] for c in ['distance_m_norm','motor_output_norm','takeoff_output_norm']]
M={}
M[1]=dict(title='Supplementary Movie 1  |  Lesion size and location',chapters=[dict(sub='Matched example flights after 1,000, 4,000 or 16,000 neurons are lost at two sites',grid=[2,3],panels=[
  panel('lesion20_1000','1,000 neurons lost','Site 20, central brain',ORANGE,'movie1_chapter1_lesion20_1000_anatomy.png'),
  panel('lesion20_4000','4,000 neurons lost','Site 20, central brain',ORANGE,'movie1_chapter1_lesion20_4000_anatomy.png'),
  panel('lesion20_16000','16,000 neurons lost','Site 20, central brain',ORANGE,'movie1_chapter1_lesion20_16000_anatomy.png'),
  panel('lesion25_1000','1,000 neurons lost','Site 25, optic lobe',BLUE,'movie1_chapter1_lesion25_1000_anatomy.png'),
  panel('lesion25_4000','4,000 neurons lost','Site 25, optic lobe',BLUE,'movie1_chapter1_lesion25_4000_anatomy.png'),
  panel('lesion25_16000','16,000 neurons lost','Site 25, optic lobe',BLUE,'movie1_chapter1_lesion25_16000_anatomy.png')])])
M[3]=dict(title='Supplementary Movie 3  |  Selecting cells to restore',chapters=[dict(sub='2,000 neurons lost at site 20; 100 lost cells restored with their original identities and partners',grid=[1,4],panels=[
  panel('lesion20_2000','No graft','No cells restored',ORANGE,'movie3_chapter1_lesion20_2000_anatomy.png'),
  panel('selection_random_100','Random selection','100 lost cells chosen at random',GREY,'movie3_chapter1_selection_random_100_anatomy.png'),
  panel('selection_interface_100','Projection priority','Descending and optic-lobe projection cells first',PURPLE,'movie3_chapter1_selection_interface_100_anatomy.png'),
  panel('selection_ranked_100','Motor-control priority','Descending and tangential cells first',GREEN,'movie3_chapter1_selection_ranked_100_anatomy.png')])])
n100=['lesion20_2000','selection_ranked_100','connections_random_partners_100','connections_correct_types_100','non_target_types_100']
n2k=['lesion20_2000','complete_original_2000','review_full_random_partners_2000','typed_dose_correct_types_2000','non_target_types_2000']
subs=['No replacement cells','Original identities and partners','Original identities, random nearby partners','Matched cell types, partners copied by type','Types absent from the lesion, partners copied by type']
titles=['No graft','Original repair','Random wiring','Correct-type graft','Non-target graft']; cols=[ORANGE,TEAL,GREY,BLUE,PINK]
def ch4(names,k,dose):
    return dict(sub=f'2,000 neurons lost at site 20; {dose} replacement cells',grid=[2,3],panels=[panel(n,titles[i],subs[i] if not (i==0) else 'No replacement cells',cols[i],f'movie4_chapter{k}_{n}_anatomy.png') for i,n in enumerate(names)],
        table=dict(head=titles,colors=cols,rows=['Distance','Wing motor','Looming motor'],values=table(names),dose=dose))
M[4]=dict(title='Supplementary Movie 4  |  Replacement identity and wiring',chapters=[ch4(n100,1,'100'),ch4(n2k,2,'2,000')])
common=dict(intact=np.round(ip,5).tolist(),arena=dict(fwd=[-1.0,3.0],lat=[-1.0,1.5]),fps=FPS,frames=NF)
for k,v in M.items():
    v.update(common); json.dump(v,open(f'movie{k}.json','w'))
json.dump(hashes,open('source_hashes.json','w'),indent=1)
print({k:[len(c['panels']) for c in v['chapters']] for k,v in M.items()})
print(means.loc[n100+n2k+['lesion20_1000','lesion25_1000']].round(3))
