"""Recover matched looming profiles without changing the frozen V8 batch."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from run_confirmatory import ROOT,initialize,prepare,ex
import graft_identity

OUT=ROOT/'results/v8/figure2_activity_profiles_20261012'
NAMES=['intact','lesion20_2000','selection_ranked_100','connections_random_partners_100',
       'connections_correct_types_100','non_target_types_100']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def trailing_rate(times,cells,members):
    counts,_=np.histogram(times[np.isin(cells,members)],bins=np.arange(301))
    cumulative=np.r_[0,np.cumsum(counts)];i=np.arange(301);prior=np.maximum(i-25,0)
    return (cumulative[i]-cumulative[prior])/np.maximum(np.minimum(i,25),1)/.001/len(members)

def main():
    OUT.mkdir(exist_ok=True);initialize();net=ex.get_net();neu=net['neu'];n=len(neu)
    rows={q['name']:q for filename in ['states.jsonl','non_target_states.jsonl']
          for line in (ROOT/'results/v8'/filename).read_text().splitlines()
          if (q:=json.loads(line)).get('rep')==0 and q['name'] in NAMES}
    with np.load(ROOT/'videos/v8/activity_current_model_recordings.npz') as z:old={k:z[k] for k in z.files}
    arrays={};checks=[];exports=[]
    ttm=np.flatnonzero(neu.type.eq('TTMn'));dlm=np.flatnonzero(neu.type.astype(str).str.startswith('DLMn'))
    assert len(ttm)==2 and len(dlm)==10
    for name in NAMES:
        ref=rows[name];spec=ref['spec'];source='Existing verified Movie 1 recording'
        if name in NAMES[:3]:
            times=old[name+'_loom_t'];cells=old[name+'_loom_i'];rates=old[name+'_loom_rates']
            lost=np.array([],dtype=int) if name=='intact' else old['lesion_idx']
        else:
            source='Exact seed replay of an existing V8 realization, recording spikes'
            lost,radius,graft,stats=prepare(dict(spec,rep=-1,wiring='original' if name=='non_target_types_100' else spec['wiring']))
            if name=='non_target_types_100':
                pool=pd.read_csv(ROOT/'results/v8/non_target_identity_pool.csv')['type'].to_numpy()
                center=pd.read_csv(ROOT/'results/v5/lesion_sites.csv').set_index('site').loc[20,['x','y','z']].tolist()
                E,sign,templates,stats=graft_identity.make_graft_typed(lost,100,graft[2],fate_accuracy=0,
                    seed=spec['intervention_seed'],center_um=center,local_radius_um=2*radius,identity_pool=pool)
                assert not np.isin(E.assigned_types,neu.iloc[lost]['type'].dropna()).any()
                graft=(E,sign,templates)
            assert graft[0].nnz==ref['graft_nnz']
            np.testing.assert_array_equal(graft[2],np.load(ROOT/f'results/v8/condition_indices/{name}_templates.npy'))
            print('Recovering matched profile:',name,flush=True)
            output,saved,spikes=ex.run_state(name,lesion_idx=lost,graft=graft,seed=1000,
                save_rates=False,record_spikes=True,with_feed=True,stimuli_subset=['loom'])
            for key,value in output.items():
                if key.endswith('_loom') and not key.startswith('time_'):
                    np.testing.assert_allclose(value,ref[key],rtol=0,atol=1e-6,err_msg=name+' '+key)
            times,cells=spikes['loom'];rates=saved['loom']
        keep=(times>=50)&(times<300)
        measured=np.bincount(cells[keep],minlength=len(rates))/.25
        np.testing.assert_array_equal(measured.astype(np.float32),rates)
        with np.load(ROOT/f'results/v8/rates/{name}_s1000.npz') as z:
            np.testing.assert_array_equal(measured[:n].astype(np.float32),z['loom'])
            if len(rates)>n:np.testing.assert_array_equal(measured[n:].astype(np.float32),z['graft_loom'])
        assert not np.isin(cells[cells<n],lost).any()
        mean=.5*(measured[ttm].mean()+measured[dlm].mean())
        np.testing.assert_allclose(mean,.5*(ref['TTMn_loom']+ref['DLMn_loom']),rtol=0,atol=2e-6)
        curve=.5*(trailing_rate(times,cells,ttm)+trailing_rate(times,cells,dlm))
        arrays[name+'_t']=times;arrays[name+'_i']=cells
        exports.extend(dict(condition=name,t_ms=t,motor_rate_hz=float(rate)) for t,rate in enumerate(curve))
        checks.append(dict(condition=name,source=source,neural_seed=1000,intervention_seed=2000,
            poisson_seed=7005,spike_count=len(times),all_host_and_graft_rates_match_exactly=True,
            mean_motor_hz=float(mean),removed_cell_spikes=0))
    np.savez_compressed(OUT/'recordings.npz',**arrays);pd.DataFrame(exports).to_csv(OUT/'motor_traces.csv',index=False)
    dependencies=['videos/v8/activity_current_model_recordings.npz','results/v8/states.jsonl','results/v8/non_target_states.jsonl',
        'results/v8/non_target_identity_pool.csv','code/v8/record_fig2a_profiles.py','code/v8/run_confirmatory.py',
        'code/v8/graft_identity.py','code/v5/experiments.py','code/v5/flysim.py','code/v5/flysim_fast.py',
        'data/v5/W_csc.npz','data/v5/neurons.parquet','data/v5/cell_groups.json','data/v5/decoder.npz']
    dependencies += [f'results/v8/rates/{name}_s1000.npz' for name in NAMES]
    manifest=dict(status='PASS',conditions=NAMES,checks=checks,recording_duration_ms=300,
        rate_measurement_window_ms=[50,300],trace_window_ms=25,
        trace_definition='Arithmetic mean of TTMn and DLMn population rates over the preceding 25 ms; early bins use elapsed duration.',
        neural_realizations_per_condition=1,additional_independent_neural_realizations=0,
        existing_realizations_replayed_for_recording=3,frozen_batch_files_unchanged=True,
        source_sha256={p:sha(ROOT/p) for p in dependencies},
        outputs={p.name:sha(p) for p in [OUT/'recordings.npz',OUT/'motor_traces.csv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('PASS: six matched profiles, all rates exactly reproduce saved states.',flush=True)

if __name__=='__main__':main()
