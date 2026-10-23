"""Paired outcomes and uncertainty for the prespecified V8 realizations."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import t
from batch_inputs import read_inputs, STATE_FILES

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/v5'))
import flight_fast as flight
import experiments as ex
OUT=ROOT/'results/v8'; TAB=OUT/'tables'


def beam_crossings(x, planes=(.5,1.,1.5,2.), hysteresis=.005):
    """Count both directions after a 5-mm excursion beyond each virtual plane."""
    count=0
    for plane in planes:
        sides=np.where(x<plane-hysteresis,-1,np.where(x>plane+hysteresis,1,0))
        sides=sides[sides!=0]
        count+=int(np.count_nonzero(sides[1:]!=sides[:-1]))
    return count


def interval(values):
    a=np.asarray(values,float)
    assert np.isfinite(a).all()
    n=len(a); mean=float(a.mean())
    sd=float(a.std(ddof=1)) if n>1 else 0.
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
    return dict(n=n,mean=mean,sd=sd,sem=sd/np.sqrt(n),ci_low=mean-half,ci_high=mean+half,p025=float(np.quantile(a,.025)),p975=float(np.quantile(a,.975)))


def outcomes(row,ref):
    row=row.copy()
    row['motor_output']=(row['powerMN_R_F']+row['powerMN_L_F'])/2/ref['power']
    row['takeoff_output']=(row['TTMn_loom']+row['DLMn_loom'])/ref['takeoff']
    row['drive']=(row['motor_output']+row['takeoff_output'])/2
    row['evoked_activity']=np.mean([row[f'activeCB_{k}']/ref[f'act_{k}'] for k in ['R','L','F','feed','loom']])
    trials=[]
    for gust in range(20):
        tr=flight.simulate(row['gain_R'],row['gain_L'],row['bias'],drive=row['drive'],seed=gust)
        delta=tr[-1,3:5]-tr[0,3:5]
        path=np.linalg.norm(np.diff(tr[:,3:5],axis=0),axis=1).sum()
        distance=np.linalg.norm(delta)
        crossings=beam_crossings(tr[:,3])
        trials.append(dict(name=row['name'],rep=row['rep'],seed=row['seed'],gust_seed=gust,
            distance_m=float(distance),path_m=float(path),forward_m=float(delta[0]),
            straightness=float(distance/path),gates=int(np.sum(np.max(tr[:,3])>=np.array([.5,1.,1.5,2.]))),
            beam_crossings=crossings,beam_crossings_per_s=crossings/10.,mean_speed_m_s=float(path/10.)))
        if row['rep']==0 and gust==3:
            (OUT/'trajectories').mkdir(exist_ok=True)
            pd.DataFrame(tr,columns=['time_s','heading_rad','yaw_rad_s','x_m','y_m']).to_csv(OUT/'trajectories'/f'{row["name"]}.csv',index=False)
    for k in ['distance_m','path_m','forward_m','straightness','gates','beam_crossings','beam_crossings_per_s','mean_speed_m_s']:
        row[k]=float(np.mean([q[k] for q in trials]))
    row['takeoff_rate_hz']=(row['TTMn_loom']+row['DLMn_loom'])/2
    spec=row.pop('spec')
    row.update({k:spec.get(k) for k in ['category','site','dose','rule','wiring','subset','intervention_seed']})
    return row,trials


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    TAB.mkdir(exist_ok=True)
    protocol,amendment,specs,rows=read_inputs()
    keys=[(r['name'],r['seed']) for r in rows]
    assert len(keys)==len(set(keys))
    expected={(r['name'],r['seed']) for r in specs}
    if not a.partial:assert set(keys)==expected,(len(keys),len(expected))
    for rel,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    ref=json.loads((ROOT/'results/v5/reference.json').read_text())
    cache=OUT/'controller_cache_v2.jsonl'
    existing={}
    if cache.exists():
        for l in cache.read_text().splitlines():
            q=json.loads(l);existing[(q['row']['name'],q['row']['seed'])]=q
    with cache.open('a') as f:
        for i,row in enumerate(rows):
            key=(row['name'],row['seed'])
            if key not in existing:
                result,trials=outcomes(row,ref)
                q={'row':result,'trials':trials};existing[key]=q
                f.write(json.dumps(q)+'\n');f.flush()
            if i%500==0:print(f'Controller outcomes {i+1}/{len(rows)}',flush=True)
    df=pd.DataFrame([existing[k]['row'] for k in keys])
    intact=df[df.name=='intact'].set_index('rep')
    normalized=['distance_m','path_m','straightness','motor_output','takeoff_output','evoked_activity','LPTC_L_L','drive','beam_crossings_per_s']
    for metric in normalized:
        denominator=df.rep.map(intact[metric])
        assert (denominator.dropna()>0).all(),metric
        df[metric+'_norm']=df[metric]/denominator
    df=df[df.rep.isin(intact.index)].copy()
    assert np.isfinite(df[[k+'_norm' for k in normalized]]).all().all()
    df.to_csv(TAB/'per_realization.csv',index=False)
    movie_update=OUT/'movie_source_rows_non_target.csv'
    new_example=df[(df.name=='non_target_types_100')&(df.rep==0)]
    if len(new_example) and not movie_update.exists():
        previous=pd.read_csv(OUT/'movie_source_rows.csv')
        updated=pd.concat([previous,new_example.reindex(columns=previous.columns)],ignore_index=True)
        assert len(updated)==len(previous)+1 and not updated.name.duplicated().any()
        updated.to_csv(movie_update,index=False)
    pd.DataFrame([q for k in keys for q in existing[k]['trials']]).to_csv(TAB/'gust_trials.csv',index=False)
    metrics=['gain_R','gain_L','bias','distance_m','path_m','forward_m','straightness','gates','motor_output','takeoff_output','evoked_activity','LPTC_L_L','drive','beam_crossings','beam_crossings_per_s','mean_speed_m_s','takeoff_rate_hz']+[k+'_norm' for k in normalized]
    agg=[]
    for name,g in df.groupby('name',sort=False):
        for metric in metrics:
            agg.append(dict(name=name,metric=metric,**interval(g[metric])))
    summary=pd.DataFrame(agg);summary.to_csv(TAB/'summary.csv',index=False)
    comparisons=[('selection_ranked_100','selection_random_100'),('selection_ranked_100','selection_interface_100'),('selection_ranked_100','lesion20_2000'),*protocol['connection_tests'],('selective_descending','selective_other_37'),('selective_visual','selective_other_4')]
    comparisons.append(('connections_correct_types_100','non_target_types_100'))
    contrasts=[]
    for left,right in comparisons:
        aa=df[df.name==left].set_index('rep');bb=df[df.name==right].set_index('rep');reps=aa.index.intersection(bb.index)
        for metric in ['distance_m','distance_m_norm','straightness','motor_output_norm','takeoff_output_norm','LPTC_L_L_norm','gates']:
            if len(reps):contrasts.append(dict(left=left,right=right,metric=metric,**interval(aa.loc[reps,metric]-bb.loc[reps,metric])))
    pd.DataFrame(contrasts).to_csv(TAB/'paired_contrasts.csv',index=False)
    complete=df[df.name=='complete_original_2000'].set_index('rep')
    reconstruction={}
    for metric in ['y_R','y_L','y_F','motor_output','takeoff_output','distance_m','LPTC_L_L']:
        reconstruction[metric]=float(np.abs(complete[metric]-intact.loc[complete.index,metric]).max()) if len(complete) else None
    qa=dict(status='COMPLETE' if set(keys)==expected else 'PARTIAL',realizations=len(intact),states=len(rows),expected_states=len(expected),
        unique_state_seeds=True,finite_readouts=True,source_model_hashes_unchanged=True,
        exact_reconstruction_max_difference=reconstruction,
        replication_unit='one independent neural/intervention realization, with 20 nested gusts; one anatomical connectome',
        source_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in STATE_FILES})
    (TAB/'analysis_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
