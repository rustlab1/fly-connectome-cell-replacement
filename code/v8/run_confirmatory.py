"""Prespecified V8 stochastic replications, using the unchanged V5 neural model."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
from multiprocessing import get_context
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/v5'))
import experiments as ex
import graft_typed as gt
OUT = ROOT / 'results/v8'
ex.RES = str(OUT) + os.sep
SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
DOSES = [50, 100, 200, 500, 1000, 1250, 1500, 1750]
RULES = ['random', 'interface', 'ranked']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def specs():
    rows = []
    for rep in range(100):
        common = dict(rep=rep, seed=1000 + rep, intervention_seed=2000 + rep)
        rows.append(common | dict(name='intact', category='intact'))
        for n in SIZES:
            rows.append(common | dict(name=f'lesion20_{n}', category='lesion_size', site=20, n=n))
        for n in [1000, 4000, 16000]:
            rows.append(common | dict(name=f'lesion25_{n}', category='location_control', site=25, n=n))
        for rule in RULES:
            for dose in DOSES:
                rows.append(common | dict(name=f'selection_{rule}_{dose}', category='selection', site=20, n=2000, dose=dose, rule=rule, wiring='original'))
        rows.append(common | dict(name='complete_original_2000', category='complete', site=20, n=2000, dose=2000, rule='ranked', wiring='original'))
        for wiring in ['random_partners', 'correct_types', 'incorrect_types']:
            rows.append(common | dict(name=f'connections_{wiring}_100', category='connections', site=20, n=2000, dose=100, rule='ranked', wiring=wiring))
        for subset in ['descending', 'visual', 'descending_visual', 'other_37', 'other_4']:
            rows.append(common | dict(name=f'selective_{subset}', category='selective_ablation', site=20, n=1000, subset=subset))
    return rows


def freeze_protocol():
    OUT.mkdir(exist_ok=True)
    source_paths = [ROOT/'code/v5'/n for n in ['experiments.py', 'flysim.py', 'flysim_fast.py', 'graft_typed.py', 'flight.py', 'flight_fast.py']]
    source_paths += [ROOT/'data/v5/decoder.npz', ROOT/'results/v5/reference.json', ROOT/'results/v5/lesion_sites.csv']
    protocol = dict(version=8, date='2026-10-09', n_realizations=100,
        neural_seeds=list(range(1000,1100)), intervention_seeds=list(range(2000,2100)),
        gust_seeds=list(range(20)), sensory_stimuli=['R','L','F','feed','loom'],
        primary_specs=specs(), source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        selection='One seeded permutation within each priority tier; nested prefixes at all doses. The same ranked 100 templates are used for every connection comparison in a realization.',
        reference='Decoder and population activity calibration fixed from the original five intact calibration seeds. Outcomes are additionally normalized to the new matched intact realization.',
        uncertainty='Mean and two-sided 95% Student t confidence interval across 100 independent outer realizations. Average 20 matched gust trials within each realization first. Also report 2.5th and 97.5th outcome percentiles. One anatomical connectome, not 100 animals.',
        connection_tests=[['selection_ranked_100','connections_random_partners_100'],['connections_correct_types_100','connections_incorrect_types_100']],
        spared_readout='LPTC_L_L rate divided by matched intact, selected from old exploratory seeds 0-2 before this fresh batch. Report observed change and CI; no post-hoc equivalence claim.',
        virtual_gate='Flight-plane gates perpendicular to initial heading at x=0.5,1.0,1.5,2.0 m. Count each gate once if any saved trajectory point reaches or exceeds it. This is a simulated flight measure, not walking beam counts.',
        exclusions='No outcome-dependent exclusions or forced monotonic fits. Zero denominators are reported as undefined and inspected.',
        frozen_exploratory='The original 56-site screen and legacy robustness controls remain exploratory with their original sample counts, separately labeled.',
        selective_controls='Within the original 1000-cell site-20 mask: all descending neurons, all LPTCs, their union, and randomly chosen other cells matched to 37 or 4 cells. These are circuit ablations, not focal lesions of equal volume.',
        physical_limits='No wing posture, muscle mechanics, recovery over time, or animal-to-animal biological variability is predicted.')
    path=OUT/'prespecified_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old==protocol, 'Refusing to change a protocol after starting'
    else:
        path.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def initialize():
    ex.RES=str(OUT)+os.sep
    ex.get_net(); ex.load_decoder()


def prepare(s):
    initialize()
    net=ex.get_net(); neu=net['neu']
    if s['category']=='intact':
        return None,None,None,None
    sites=pd.read_csv(ROOT/'results/v5/lesion_sites.csv').set_index('site')
    row=sites.loc[s['site']]
    center=[float(row[k]) for k in ['x','y','z']]
    lesion,radius=ex.make_lesion(center,s['n'])
    lptc=np.isin(lesion,net['G']['LPTC_R']+net['G']['LPTC_L'])
    desc=neu.superclass.values[lesion]=='descending_neuron'
    rng=np.random.default_rng(s['intervention_seed'])
    if 'subset' in s:
        pools={'descending':lesion[desc],'visual':lesion[lptc], 'descending_visual':lesion[desc|lptc]}
        other=rng.permutation(lesion[~(desc|lptc)])
        pools.update(other_37=other[:37],other_4=other[:4])
        lesion=pools[s['subset']]
    graft=None; stats=None
    if 'dose' in s:
        sc=neu.superclass.values[lesion]
        if s['rule']=='random':
            order=rng.permutation(lesion)
        elif s['rule']=='interface':
            priority=np.isin(sc,['descending_neuron','visual_projection','visual_centrifugal'])
            order=np.concatenate([rng.permutation(lesion[priority]),rng.permutation(lesion[~priority])])
        else:
            tier1=lptc|desc
            tier2=~tier1 & np.isin(sc,['visual_projection','visual_centrifugal'])
            order=np.concatenate([rng.permutation(lesion[tier1]),rng.permutation(lesion[tier2]),rng.permutation(lesion[~tier1&~tier2])])
        templ=order[:s['dose']]
        assert len(np.unique(templ))==s['dose'] and np.isin(templ,lesion).all()
        if s['wiring'] in ['correct_types','incorrect_types']:
            E,sign,templ,stats=gt.make_graft_typed(lesion,s['dose'],templ,fate_accuracy=float(s['wiring']=='correct_types'),seed=s['intervention_seed'],center_um=center,local_radius_um=2*radius)
            graft=(E,sign,templ)
        else:
            graft=ex.make_graft(lesion,s['dose'],specificity=float(s['wiring']=='original'),templates=templ,seed=s['intervention_seed'],center_um=center,local_radius_um=2*radius)
    if s['rep']==0:
        folder=OUT/'condition_indices';folder.mkdir(exist_ok=True)
        np.save(folder/(s['name']+'_lesion.npy'),lesion)
        if graft is not None:np.save(folder/(s['name']+'_templates.npy'),graft[2])
    return lesion,radius,graft,stats


def work(s):
    lesion,radius,graft,stats=prepare(s)
    out=ex.run_state(s['name'],lesion_idx=lesion,graft=graft,seed=s['seed'],with_feed=True,save_rates=s['rep']==0)
    out.update(rep=s['rep'],spec=s,lesion_radius_um=radius)
    if graft is not None:
        out.update(graft_nnz=int(graft[0].nnz),graft_frac_E=float((graft[1]>0).mean()),graft_stats=stats)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);a=p.parse_args()
    protocol=freeze_protocol()
    path=OUT/'states.jsonl'
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            q=json.loads(line);done.add((q['name'],q['seed']))
    todo=[s for s in protocol['primary_specs'] if (s['name'],s['seed']) not in done]
    if a.limit:todo=todo[:a.limit]
    print(f'{len(todo)} pending, {len(done)} complete',flush=True)
    started=time.time()
    with get_context('spawn').Pool(a.workers,initializer=initialize) as pool, path.open('a') as f:
        for i,out in enumerate(pool.imap_unordered(work,todo,chunksize=1)):
            f.write(json.dumps(out)+'\n');f.flush()
            if i%10==0 or i+1==len(todo):print(f'{i+1}/{len(todo)} {out["name"]} rep={out["rep"]} elapsed_min={(time.time()-started)/60:.1f}',flush=True)


if __name__=='__main__':main()
