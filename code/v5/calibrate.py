"""Calibrate the V5 decoder from corrected intact-network trials."""
from pathlib import Path
from multiprocessing import Pool
import argparse,json,time
import numpy as np
import experiments as ex

def one(seed):
    ex.get_net()
    out=ex.run_state('intact',seed=seed,with_feed=True,save_rates=True)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);a=ap.parse_args()
    res=Path(ex.RES);res.mkdir(exist_ok=True);raw=res/'intact_calibration_raw.json'
    if raw.exists():rows=json.loads(raw.read_text())
    else:
        start=time.time()
        with Pool(a.workers) as pool:rows=list(pool.imap_unordered(one,a.seeds))
        raw.write_text(json.dumps(rows,indent=2));print('Calibration simulation seconds',round(time.time()-start,2),flush=True)
    net=ex.get_net();N=len(net['neu']);rates={k:[] for k in ['R','L','F']}
    for seed in a.seeds:
        with np.load(res/f'rates/intact_s{seed}.npz') as z:
            for k in rates:rates[k].append(z[k].astype(np.float32))
    ex.set_decoder(*[np.mean(rates[k],axis=0) for k in ['R','L','F']])
    scored=[]
    for seed in a.seeds:
        with np.load(res/f'rates/intact_s{seed}.npz') as z:
            rd=ex.readouts({k:z[k].astype(np.float32) for k in ['R','L','F','feed','loom']})
        original=next(r for r in rows if r['seed']==seed)
        scored.append(original|rd|{'name':'intact','seed':seed,'spec':{'name':'intact','seed':seed,'feed':True}})
    (res/'intact.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in sorted(scored,key=lambda r:r['seed'])))
    refs=dict(power=np.mean([np.mean([r['powerMN_R_F'],r['powerMN_L_F']]) for r in scored]),takeoff=np.mean([r['TTMn_loom']+r['DLMn_loom'] for r in scored]),**{f'act_{k}':np.mean([r[f'activeCB_{k}'] for r in scored]) for k in ['R','L','F','feed','loom']})
    (res/'reference.json').write_text(json.dumps(refs,indent=2))
    print('Intact calibrated',[(r['seed'],round(r['gain_R'],3),round(r['gain_L'],3),round(r['bias'],3)) for r in scored],flush=True)
if __name__=='__main__':main()
