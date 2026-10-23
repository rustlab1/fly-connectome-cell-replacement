"""Exact continuation tests for algorithmic state, independent of the connectome."""
import sys
import tempfile
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/v5'))
import adaptation as ad
from adaptation import update,checkpoint,restore,DEFAULTS


def fake_trajectory(folder, spec, *, reuse=True, split=None, changing=False):
    """Exercise the real round/checkpoint loop without a connectome simulation."""
    root=Path(folder); (root/'rates').mkdir(parents=True)
    pd.DataFrame([dict(site=20,x=0.,y=0.,z=0.)]).to_csv(root/'lesion_sites.csv',index=False)
    target=np.array([8.,2.,4.,6.],np.float32)
    np.savez(root/'rates/intact_s0.npz',R=target,L=target,F=target)
    calls=[]
    def run(name,**kwargs):
        gain=kwargs['gain_in']; calls.append(gain.copy())
        current=target*(gain*.5 if changing else 1.)
        rates={key:current.astype(np.float32).copy() for key in ['R','L','F','feed','loom']}
        return dict(name=name,gain_R=float(current[2]),gain_L=float(current[3]),bias=float(current[2]-current[3]),
                    time_R=1.,nested={'array_sum':float(current.sum())}),rates
    with patch.multiple(ad.ex,RES=str(root),get_net=lambda:{'neu':pd.DataFrame(index=range(4))},
                        load_decoder=lambda:None,stimuli=lambda _: {'R':np.array([0])},
                        make_lesion=lambda center,n:(np.arange(1,n+1),2.),make_graft=lambda *a,**k:('fixed_graph',),
                        run_state=run),redirect_stdout(io.StringIO()):
        if split is not None:
            ad.experiment(spec|{'rounds':split},reuse_evaluations=reuse)
        path=ad.experiment(spec,reuse_evaluations=reuse)
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    states=[restore(root/'checkpoints'/f'{spec["name"]}_s0_r{r}.npz') for r in range(spec['rounds']+1)]
    final=restore(root/'checkpoints'/f'{spec["name"]}_s0_resume.npz')
    return rows,calls,states,final


def assert_trajectory_equal(a,b):
    # Only runtime/provenance differs when an evaluation is reused.
    clean=lambda rows:[{k:v for k,v in r.items() if k!='evaluation_reused' and not k.startswith('time_')} for r in rows]
    assert clean(a[0])==clean(b[0])
    for left,right in zip(a[2]+[a[3]],b[2]+[b[3]]):
        np.testing.assert_array_equal(left[0],right[0])
        if left[1] is None or right[1] is None: assert left[1] is right[1]
        else: np.testing.assert_array_equal(left[1],right[1])
        assert left[2:]==right[2:]


def check_evaluation_reuse():
    base=dict(name='trajectory',seed=0,n_lesion=0,site=20,rounds=5)
    scenarios={'intact':{},'frozen':{'n_lesion':1,'frozen':True},
               'acute_reconstruction':{'n_lesion':1,'graft_n':1,'rule':'random'}}
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for label,changes in scenarios.items():
            spec=base|changes
            uncached=fake_trajectory(root/(label+'_uncached'),spec,reuse=False)
            cached=fake_trajectory(root/(label+'_cached'),spec)
            resumed=fake_trajectory(root/(label+'_resumed'),spec,split=2)
            assert_trajectory_equal(uncached,cached);assert_trajectory_equal(uncached,resumed)
            assert len(uncached[1])==6 and len(cached[1])==1 and len(resumed[1])==2
            assert [r['evaluation_reused'] for r in cached[0]]==[False,True,True,True,True,True]
            assert [r['evaluation_reused'] for r in resumed[0]]==[False,True,True,False,True,True]
            assert [r['name'] for r in cached[0]]==[f'trajectory_r{r}' for r in range(6)]
            assert all(r['time_R']==0 for r in cached[0][1:])
        changing=base|{'n_lesion':1}
        uncached=fake_trajectory(root/'changing_uncached',changing,reuse=False,changing=True)
        cached=fake_trajectory(root/'changing_cached',changing,changing=True)
        assert_trajectory_equal(uncached,cached)
        assert len(cached[1])==6 and not any(r['evaluation_reused'] for r in cached[0])
        assert all(not np.array_equal(a,b) for a,b in zip(cached[1],cached[1][1:]))
    print('Evaluation reuse: six rounds required one evaluation; resumed runs required two; changed gains required six')

def main():
    gain=np.ones(4,np.float32);ema=None;target=np.array([8.,2.,0.,6.],np.float32);mask=np.array([True,True,True,False])
    sequence=[np.array([2+i*.4,4,0,1],np.float32) for i in range(12)]
    for cur in sequence[:5]: gain,ema=update(gain,ema,cur,target,mask,DEFAULTS)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'state.npz';checkpoint(path,gain,ema,5,{'test':True});g2,e2,r,m=restore(path)
        assert r==5 and m=={'test':True}
        for cur in sequence[5:]:
            gain,ema=update(gain,ema,cur,target,mask,DEFAULTS);g2,e2=update(g2,e2,cur,target,mask,DEFAULTS)
        np.testing.assert_array_equal(gain,g2);np.testing.assert_array_equal(ema,e2)
        assert gain[3]==1
        assert np.all(gain>=.5) and np.all(gain<=2)
        # No-lesion, identical training activity is a fixed point of the rule.
        g,e=update(np.ones(4,np.float32),None,target,target,np.ones(4,bool),DEFAULTS)
        np.testing.assert_array_equal(g,np.ones(4))
    check_evaluation_reuse()
    print('Adaptation exact-resume, fixed-point, bounds and evaluation-reuse checks passed')
if __name__=='__main__':main()
