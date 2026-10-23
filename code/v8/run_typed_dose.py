"""Author-requested dose extension using the frozen V8 cell-type rules."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from multiprocessing import get_context
from run_confirmatory import ROOT, OUT, initialize, work

DOSES = [50, 200, 500, 1000, 1250, 1500, 1750, 2000]


def freeze():
    specs = []
    for rep in range(100):
        for dose in DOSES:
            for wiring in ['correct_types', 'incorrect_types']:
                specs.append(dict(name=f'typed_dose_{wiring}_{dose}', category='typed_dose',
                    rep=rep, seed=1000+rep, intervention_seed=2000+rep, site=20,
                    n=2000, dose=dose, rule='ranked', wiring=wiring))
    amendment = dict(date='2026-10-09', reason='Author requested incorrect-type dose response in Fig. 2c, with a matched required-type comparison.',
        extension_specs=specs, primary_protocol_sha256=hashlib.sha256((OUT/'prespecified_protocol.json').read_bytes()).hexdigest(),
        n_realizations=100, reused_100_cell_conditions=['connections_correct_types_100', 'connections_incorrect_types_100'],
        design='The same intended positions at each dose for required and alternative identities. Positions are nested ranked prefixes. Alternative identities exclude the intended type. Identity and donor draws are dose-specific under the existing typed-graft algorithm, not nested graft connections.',
        limits='This tests deliberately mismatched identity, not an unconstrained random mixture. Full typed replacement is not exact reconstruction. No forced plateaus or outcome ordering.',
        display='Means and two-sided 95% Student t confidence intervals; 20 matched gust trials averaged within each realization. Supersedes the intermediate SEM display request. Original exploratory conditions keep their actual n.',
        timing='Frozen after partial primary-batch results were available and before any extension simulation. This is an author-requested extension, not a retrospectively prespecified primary analysis.')
    path=OUT/'typed_dose_amendment.json'
    if path.exists():
        assert json.loads(path.read_text()) == amendment
    else:
        path.write_text(json.dumps(amendment, indent=2)+'\n')
    return amendment


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    amendment=freeze(); path=OUT/'typed_dose_states.jsonl'
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line); key=(row['name'],row['seed']); assert key not in done; done.add(key)
    pending=[s for s in amendment['extension_specs'] if (s['name'],s['seed']) not in done]
    start=time.monotonic(); print(f'{len(pending)} pending, {len(done)} complete',flush=True)
    with get_context('spawn').Pool(a.workers,initializer=initialize) as pool,path.open('a') as f:
        for i,row in enumerate(pool.imap_unordered(work,pending,chunksize=1)):
            f.write(json.dumps(row)+'\n'); f.flush()
            if i%10==0 or i+1==len(pending):
                print(f'{i+1}/{len(pending)} {row["name"]} rep={row["rep"]} elapsed_min={(time.monotonic()-start)/60:.1f}',flush=True)


if __name__=='__main__': main()
