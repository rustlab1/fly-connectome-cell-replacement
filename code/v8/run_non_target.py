"""Matched graft sizes with non-target central-brain identities absent from the lesion."""
from pathlib import Path
import argparse,hashlib,json,time
from multiprocessing import get_context
import numpy as np
import pandas as pd
from run_confirmatory import ROOT,OUT,initialize,ex
import graft_identity as graft

DOSES=[50,100,200,500,1000,1250,1500,1750,2000]
POOL=None


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    neu=pd.read_parquet(ROOT/'data/v5/neurons.parquet')
    lesion=np.load(OUT/'condition_indices/lesion20_2000_lesion.npy')
    lost_types=set(neu.iloc[lesion]['type'].dropna())
    typed=neu[neu['type'].notna()]
    pure_cb=typed.groupby('type')['superclass'].agg(lambda s:s.eq('cb_intrinsic').all())
    eligible_types=set(pure_cb[pure_cb].index)-lost_types
    eligible=neu[neu['type'].isin(eligible_types)].copy()
    assert len(eligible)>0 and not np.isin(eligible.index,lesion).any()
    pool=eligible[['bodyId','type','superclass','somaSide']].copy()
    pool.insert(0,'neuron_index',eligible.index)
    path=OUT/'non_target_identity_pool.csv'
    content=pool.to_csv(index=False)
    if path.exists():assert path.read_text()==content
    else:path.write_text(content)
    specs=[dict(name=f'non_target_types_{dose}',category='non_target_dose',rep=rep,seed=1000+rep,
        intervention_seed=2000+rep,site=20,n=2000,dose=dose,rule='ranked',wiring='non_target_types')
        for rep in range(100) for dose in DOSES]
    protocol=dict(date='2026-10-09',author_choice='Non-target brain cell types absent from the lesion, using the same integration rule.',
        specs=specs,n_realizations=100,pool_cells=len(pool),pool_types=pool['type'].nunique(),pool_sha256=digest(path),
        pool_definition='Annotated types represented exclusively by cb_intrinsic neurons in this connectome, excluding every annotated type present anywhere in the 2000-cell lesion. Sample cells uniformly with replacement, therefore weighting types by their abundance in the eligible surviving central-brain population.',
        integration='The unchanged V5 donor-based wiring operations with an explicit identity pool. Match intended ranked positions, graft count, neuronal/intervention seeds and 20 gust sequences. Identity changes transmitter sign, population membership, donor connections and synaptic load.',
        comparator='Required identities under V5 typed wiring at matching doses. The previous lost-type-shuffle comparison is retained separately. No numerical ceiling, fitted plateau or outcome-based pool selection.',
        interpretation='Non-target denotes absence from this lesion and exclusion of sensory, projection, descending and motor superclasses. It does not prove that these cells have no influence on movement. These are central-brain interneuron grafts, not experimentally validated products.',
        timing='Author-approved extension after preliminary lost-type-shuffle results showed why shuffled lost types were not an unrelated graft. The non-target pool is fixed from anatomical annotations before examining its outcomes.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'code/v5/graft_typed.py',ROOT/'code/v8/graft_identity.py',ROOT/'data/v5/neurons.parquet']})
    path=OUT/'non_target_amendment.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def work(s):
    global POOL
    initialize();net=ex.get_net();neu=net['neu']
    if POOL is None:POOL=pd.read_csv(OUT/'non_target_identity_pool.csv')['type'].to_numpy()
    site=pd.read_csv(ROOT/'results/v5/lesion_sites.csv').set_index('site').loc[20]
    center=[float(site[k]) for k in ['x','y','z']]
    lesion,radius=ex.make_lesion(center,2000)
    sc=neu.superclass.to_numpy()[lesion]
    first=np.isin(lesion,net['G']['LPTC_R']+net['G']['LPTC_L'])|(sc=='descending_neuron')
    second=~first&np.isin(sc,['visual_projection','visual_centrifugal'])
    rng=np.random.default_rng(s['intervention_seed'])
    positions=np.concatenate([rng.permutation(lesion[first]),rng.permutation(lesion[second]),rng.permutation(lesion[~first&~second])])[:s['dose']]
    E,sign,positions,stats=graft.make_graft_typed(lesion,s['dose'],positions,fate_accuracy=0,
        seed=s['intervention_seed'],center_um=center,local_radius_um=2*radius,identity_pool=POOL)
    assert not np.isin(E.assigned_types,neu.iloc[lesion]['type'].dropna()).any()
    assert stats['n_correct_fate']==0 and stats['n_with_donor']==s['dose']
    stats.update(n_non_target_cells=s['dose'],n_matching_any_lost_type=0,
        assigned_type_sha256=hashlib.sha256('\n'.join(E.assigned_types).encode()).hexdigest())
    if s['rep']==0:
        np.save(OUT/'condition_indices'/f'{s["name"]}_lesion.npy',lesion)
        np.save(OUT/'condition_indices'/f'{s["name"]}_templates.npy',positions)
        pd.DataFrame(dict(position=positions,intended_type=neu.iloc[positions]['type'].to_numpy(),assigned_type=E.assigned_types)).to_csv(OUT/'condition_indices'/f'{s["name"]}_assignments.csv',index=False)
    result=ex.run_state(s['name'],lesion_idx=lesion,graft=(E,sign,positions),seed=s['seed'],with_feed=True,save_rates=s['rep']==0)
    result.update(rep=s['rep'],spec=s,lesion_radius_um=radius,graft_nnz=int(E.nnz),graft_frac_E=float((sign>0).mean()),graft_stats=stats)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    protocol=freeze()
    if a.freeze_only:
        print(json.dumps({k:protocol[k] for k in ['pool_cells','pool_types','pool_sha256']}));return
    path=OUT/'non_target_states.jsonl';done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(row['name'],row['seed']);assert key not in done;done.add(key)
    pending=[s for s in protocol['specs'] if (s['name'],s['seed']) not in done]
    start=time.monotonic();print(f'{len(pending)} pending, {len(done)} complete',flush=True)
    with get_context('spawn').Pool(a.workers,initializer=initialize) as pool,path.open('a') as f:
        for i,row in enumerate(pool.imap_unordered(work,pending,chunksize=1)):
            f.write(json.dumps(row)+'\n');f.flush()
            if i%10==0 or i+1==len(pending):print(f'{i+1}/{len(pending)} {row["name"]} rep={row["rep"]} elapsed_min={(time.monotonic()-start)/60:.1f}',flush=True)


if __name__=='__main__':main()
