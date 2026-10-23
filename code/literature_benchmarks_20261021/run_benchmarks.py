"""Exploratory literature benchmarks using the unmodified V5 NumPy reference.

The old SciPy/PyArrow/Numba environment is absent. Load the exact reference
class and readout functions from the archived source through AST selection;
only data loading is adapted. No scientific model equations are rewritten.
The CSC arrays are loaded with NumPy, and flat Parquet metadata with the
adjacent read-only decoder. Validation against saved V8 arrays is mandatory.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from parquet_read import read_columns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/literature_benchmarks_20261021'
NAMES = ['bodyId','type','instance','somaSide','superclass','class','subclass']
SEEDS = list(range(1000, 1010))
CACHE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()


def source_definitions(path, function_names, assignment_names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body
             if (isinstance(n,(ast.FunctionDef, ast.ClassDef)) and n.name in function_names)
             or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignment_names for t in n.targets))]
    found = {n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert found == set(function_names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


def initialize():
    if CACHE: return CACHE
    z = np.load(ROOT/'data/v5/W_csc.npz')
    assert z['format'].item() == b'csc'
    W = SimpleNamespace(shape=tuple(z['shape']),indptr=z['indptr'],indices=z['indices'],data=z['data'])
    annotations = OUT/'annotations.json'
    table = json.loads(annotations.read_text()) if annotations.exists() else read_columns(ROOT/'data/v5/neurons.parquet',NAMES)
    T = {k:np.asarray(v,dtype=object) for k,v in table.items()}
    N = len(T['type']); assert N == 165122 == W.shape[0]
    G = json.loads((ROOT/'data/v5/cell_groups.json').read_text())
    pick = lambda m: np.flatnonzero(m).tolist()
    G['GRN_labellar'] = pick((T['class']=='gustatory') & np.isin(T['subclass'],['labellar bristle','taste peg','pharyngeal sensillum']))
    G['probMN'] = pick((T['superclass']=='cb_motor') & np.array([isinstance(t,str) and t.startswith('MN') for t in T['type']]))
    G['powerMN'] = G['powerMN_R']+G['powerMN_L']
    G['steerMN'] = G['steerMN_R']+G['steerMN_L']
    G['LC_loom'] = pick(np.isin(T['type'],['LC4','LPLC2']))
    G['GF'] = pick(T['type']=='DNp01')
    G['TTMn'] = pick(T['type']=='TTMn')
    G['DLMn'] = pick(np.isin(T['type'],['DLMn a, b','DLMn c-f']))
    extra_groups = {'HS_L':pick(np.isin(T['type'],['HSN','HSE','HSS']) & (T['somaSide']=='L')),
                    'HS_R':pick(np.isin(T['type'],['HSN','HSE','HSS']) & (T['somaSide']=='R')),
                    'LC4':pick(T['type']=='LC4'), 'LPLC2':pick(T['type']=='LPLC2')}
    G.update(extra_groups)
    G['HS_bilateral'] = sorted(G['HS_L']+G['HS_R'])
    G['T4T5'] = pick(np.array([isinstance(t,str) and t[:2] in ['T4','T5'] and len(t)==3 and t[2] in 'abcd' for t in T['type']]))
    assert len(G['HS_L']) == len(G['HS_R']) == 3
    inventory = json.loads((ROOT/'results/v5/model_inventory.json').read_text())
    for key,count in inventory['group_counts'].items():
        assert len(G[key]) == count, (key,len(G[key]),count)
    assert len(set(T['type'])-{None}) == inventory['n_cell_type_labels']
    for group,ids in G.items():
        if group.startswith(('T4','T5')) and group!='T4T5':
            typ,side=group.split('_')
            assert ids == pick((T['type']==typ)&(T['somaSide']==side)),group
    for row in csv.DictReader((ROOT/'results/v8/site20_descending_neuron_annotations.csv').open()):
        i=int(row['model_index'])
        assert int(T['bodyId'][i])==int(row['bodyId'])
        assert T['type'][i] == (row['type'] or None)
    state_cache = {}
    context = {'G':G,'neu':range(N),'nonOL':~np.isin(T['superclass'],['ol_intrinsic','ol_sensory'])}
    scope = {'np':np,'_cache':state_cache,'get_net':lambda:context,'PROC':str(ROOT/'data/v5')+os.sep}
    source_definitions(ROOT/'code/v5/experiments.py', ['stimuli','readouts','load_decoder','_validate_directional_refs'], ['W_SYN','STIM_RATE','T_SIM'], scope)
    scope['load_decoder']()
    engine = {'np':np}
    source_definitions(ROOT/'code/v5/flysim.py',['LIFNetReference','_gather'],['PARAMS'],engine)
    CACHE.update(W=W,T=T,G=G,N=N,scope=scope,engine=engine,decoder=state_cache)
    return CACHE


def source_hashes():
    paths=['data/v5/W_csc.npz','data/v5/neurons.parquet','data/v5/cell_groups.json','data/v5/decoder.npz',
           'code/v5/flysim.py','code/v5/experiments.py','results/v8/states.jsonl','results/v8/rates/intact_s1000.npz']
    return {p:sha(ROOT/p) for p in paths}


def specifications():
    # Fixed before outcome inspection. Ten paired neural-event seeds.
    return [dict(name=name,group=group,stimuli=stims,seed=seed)
            for seed in SEEDS for name,group,stims in [
                ('HS_left_loss','HS_L',['R','L','F']),
                ('HS_right_loss','HS_R',['R','L','F']),
                ('HS_bilateral_loss','HS_bilateral',['R','L','F']),
                ('T4T5_block','T4T5',['R','L','F','loom']),
                ('LPLC2_block','LPLC2',['R','L','F','loom']),
                ('LC4_block','LC4',['loom']),
                ('LC4_LPLC2_block','LC_loom',['loom'])]]


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    annotation_path=OUT/'annotations.json'
    if not annotation_path.exists():
        annotation_path.write_text(json.dumps(read_columns(ROOT/'data/v5/neurons.parquet',NAMES)))
    c=initialize()
    hashes=source_hashes()
    archived=json.loads((ROOT/'results/v8/model_input_hashes.json').read_text())['files']
    for p in ['data/v5/W_csc.npz','data/v5/neurons.parquet','data/v5/cell_groups.json','data/v5/decoder.npz']:
        assert hashes[p]==archived[p]['sha256'],p
    protocol=dict(
        label='Exploratory pilot fixed before benchmark outcome inspection',
        seeds=SEEDS,unit='paired sensory-event realizations of one MaleCNS connectome; not biological replicates',
        model='Unmodified V5 NumPy LIF reference source; V8 input matrix and frozen steering decoder',
        trial_ms=300,measurement_ms=[50,300],input_Hz=100,
        intervention='Silence specified cells using the original dead mask; surviving inputs retain original random-event slots. This approximates output blockade/axon lesion, not its full physiology.',
        benchmark_studies=[
            dict(id='HS',citation='Hausen and Wehrhahn 1983',doi='10.1098/rspb.1983.0068',species='Calliphora erythrocephala',
                 expected='Unilateral loss of canonical HS cells alters/reduces yaw steering asymmetrically; bilateral loss is an added diagnostic.',
                 primary=['gain_R','gain_L','y_R','y_L','y_F','gain','bias'],
                 limits='Cross-species qualitative comparison. Published yaw torque versus model wing-motor decoder. No quantitative experimental curve fitted. Model omits graded HS physiology and electrical coupling. Inspect absolute decoder responses as well as forward-referenced gains.'),
            dict(id='T4T5',citation='Bahl et al. 2013',doi='10.1038/nn.3386',species='Drosophila melanogaster',
                 expected='Blocking T4/T5 abolishes directional optomotor responses.',
                 primary=['gain_R','gain_L','gain'],added_specificity='looming motor response to direct LC4/LPLC2 input',
                 limits='Published optomotor experiment used walking; model decoder uses wing motor neurons. T4/T5 are directly driven model inputs, making response loss a basic input-pathway sanity check. Object fixation and actual free-flight ability are not represented; directly driven looming bypasses upstream T4/T5 processing.'),
            dict(id='LPLC2',citation='Ache et al. 2019',doi='10.1016/j.cub.2019.01.079',species='Drosophila melanogaster',
                 expected='LPLC2 silencing reduces GF recruitment during combined LC4/LPLC2 looming drive; motor output may also decline. LC4-only and combined blockade are diagnostic controls.',
                 primary=['GF_loom'],secondary=['TTMn_loom','DLMn_loom','gain_R','gain_L'],
                 limits='Only a pathway-level consistency check: constant Poisson input does not encode looming size or velocity, GF mean spike rate is not the measured graded voltage response, and escape probability/timing are not modeled.')],
        inference='Report all three tests and all predefined controls. No tuning, outcome-based study replacement, or quantitative biological validation claim. Paired descriptive effects and seed variability; matching expected sign is not sufficient to validate the graft model.',
        sources=hashes,specifications=specifications(),
        driver_sha256=sha(Path(__file__)),parquet_reader_sha256=sha(Path(__file__).with_name('parquet_read.py')))
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text());old.pop('created_utc',None)
        assert old==protocol,'Protocol changed; preserve old run and use a new directory.'
    else:
        protocol_path.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),**protocol),indent=2)+'\n')
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),backend='unmodified_numpy_reference_AST_import',reason='Archived conda environment absent; SciPy/PyArrow/Numba not installed and dependency download unavailable. Only file loading adapted; validation gate compares all saved rates.'),indent=2)+'\n')
    with (OUT/'target_cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['group','model_index']+NAMES);writer.writeheader()
        for group in ['HS_L','HS_R','T4T5','LPLC2','LC4','GF','TTMn','DLMn']:
            for i in c['G'][group]:writer.writerow(dict(group=group,model_index=i,**{k:c['T'][k][i] for k in NAMES}))
    print('Prepared frozen protocol', {k:len(c['G'][k]) for k in ['HS_L','HS_R','T4T5','LPLC2','LC4','GF']},flush=True)


def run_trial(stimulus,seed,lesion):
    c=initialize(); net=c['engine']['LIFNetReference'](c['W'],params={'w_syn':c['scope']['W_SYN']})
    dead=np.zeros(c['N'],bool);dead[lesion]=True
    ix=c['scope']['stimuli'](with_feed=True)[stimulus]
    return net.run(300.,{100.:ix},dead=dead,seed=seed*7+{'R':1,'L':2,'F':3,'feed':4,'loom':5}[stimulus],t_measure=(50.,300.))['rates']


def validate():
    c=initialize(); saved=np.load(ROOT/'results/v8/rates/intact_s1000.npz');checks=[]
    for stim in ['R','L','F','loom']:
        start=time.monotonic();r=run_trial(stim,1000,[])
        row=dict(stimulus=stim,exact_equal=bool(np.array_equal(r,saved[stim])),max_abs_diff_Hz=float(np.max(abs(r-saved[stim]))),different_cells=int(np.sum(r!=saved[stim])),seconds=time.monotonic()-start)
        checks.append(row);print(row,flush=True)
        assert row['exact_equal'], 'Reference replay differs from V8; do not run benchmarks.'
    reread=c['scope']['readouts']({s:saved[s] for s in ['R','L','F','loom']})
    archived=next(json.loads(line) for line in (ROOT/'results/v8/states.jsonl').open() if '"intact"' in line and json.loads(line)['seed']==1000)
    readout_checks=[]
    for k,v in reread.items():
        if k in archived and isinstance(v,(float,int)):
            # Float32 dot reductions can vary between NumPy/BLAS versions.
            # The underlying neuron-by-neuron firing rates must remain exact.
            ok=bool(np.isclose(v,archived[k],rtol=1e-7,atol=1e-5))
            readout_checks.append(dict(metric=k,current=v,archived=archived[k],absolute_difference=abs(v-archived[k]),within_tolerance=ok))
            assert ok,(k,v,archived[k])
    (OUT/'validation.json').write_text(json.dumps(dict(passed=True,checks=checks,readouts_exact=all(x['absolute_difference']==0 for x in readout_checks),readout_tolerance=dict(rtol=1e-7,atol=1e-5),readout_checks=readout_checks,source_hashes=source_hashes()),indent=2)+'\n')


def work(spec):
    c=initialize();start=time.monotonic();lesion=np.asarray(c['G'][spec['group']],dtype=int)
    rates={s:run_trial(s,spec['seed'],lesion) for s in spec['stimuli']}
    out=dict(spec, n_silenced=len(lesion),seconds=time.monotonic()-start,**c['scope']['readouts'](rates))
    for s,r in rates.items():
        for group in ['HS_L','HS_R','LC4','LPLC2']:
            out[f'{group}_{s}']=float(np.mean(r[c['G'][group]]))
        assert np.all(r[lesion]==0)
    rates_dir=OUT/'rates';rates_dir.mkdir(exist_ok=True)
    np.savez_compressed(rates_dir/f'{spec["name"]}_s{spec["seed"]}.npz',**rates)
    return out


def run(workers):
    assert json.loads((OUT/'validation.json').read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['driver_sha256']==sha(Path(__file__))
    assert protocol['sources']==source_hashes()
    output=OUT/'states.jsonl';done=set()
    if output.exists():
        for line in output.open():
            r=json.loads(line);done.add((r['name'],r['seed']))
    todo=[s for s in protocol['specifications'] if (s['name'],s['seed']) not in done]
    print('pending states',len(todo),'workers',workers,flush=True)
    with get_context('spawn').Pool(workers,initializer=initialize) as pool, output.open('a') as f:
        for i,out in enumerate(pool.imap_unordered(work,todo,chunksize=1)):
            f.write(json.dumps(out)+'\n');f.flush()
            print(f'{i+1}/{len(todo)} {out["name"]} seed={out["seed"]} {out["seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','run']);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='validate':validate()
    else:run(args.workers)
