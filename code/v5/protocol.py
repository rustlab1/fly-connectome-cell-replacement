"""Prespecified V5 experiment specifications; no legacy readouts reused."""
from pathlib import Path
import json
import pandas as pd
import experiments as ex
ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'results/v5'
SEEDS=[0,1,2]

def main():
    sites=pd.read_csv(RES/'lesion_sites.csv').set_index('site')
    specs=[]
    def add(name,site,n,seed,graft=None,**extra):
        row=sites.loc[site]
        s=dict(name=name,seed=seed,lesion={'center':[float(row.x),float(row.y),float(row.z)],'n':n},feed=True,save_rates=False,
               site=site,n_lesion=n,**extra)
        if graft is not None:s['graft']=graft|{'seed':seed}
        specs.append(s)
    for site in sites.index:
        for n in [1000,4000,16000]:
            for seed in SEEDS:add(f'map_site{site}_n{n}',int(site),n,seed,category='screen')
    for n in [250,500,2000,8000,32000]:
        for seed in SEEDS:add(f'size_site20_n{n}',20,n,seed,category='size')
    for rule,targeted in [('random',None),('interface',True),('ranked','ranked')]:
        for pct in [2.5,5,10,25,50]:
            for seed in SEEDS:
                g={'n':int(2000*pct/100),'specificity':1.0,'composition':'matched','strength':1.0}
                if targeted:g['targeted']=targeted
                add(f'partial_{rule}_p{pct:g}',20,2000,seed,g,category='partial',rule=rule,pct=pct)
    for seed in SEEDS:add('partial_full',20,2000,seed,{'n':2000,'specificity':1.0},category='partial',rule='complete',pct=100.)
    for dose in [.5,1.,2.]:
        for p in [0.,.25,.5,.75,1.]:
            for seed in SEEDS:add(f'grid_d{dose:g}_p{p:g}',20,1000,seed,{'n':int(1000*dose),'specificity':p},category='grid',dose=dose,specificity=p)
    for p in [0.,.5]:
        for seed in SEEDS:add(f'wiring_core100_p{p:g}',20,2000,seed,{'n':100,'specificity':p,'targeted':'ranked'},category='wiring',rule='ranked',wiring='copy',specificity=p)
    for fate in [0.,.25,.5,.75,1.]:
        for seed in SEEDS:add(f'typed_core100_fate{fate:g}',20,2000,seed,{'n':100,'wiring':'typed','fate_accuracy':fate,'targeted':'ranked'},category='typed',rule='ranked',fate_accuracy=fate)
    for seed in SEEDS:add('typed_full2000',20,2000,seed,{'n':2000,'wiring':'typed','fate_accuracy':1.,'targeted':'ranked'},category='typed',rule='complete',fate_accuracy=1.)
    for count in [100,200]:
        for survival in [.5,.75]:
            for seed in SEEDS:add(f'survival_n{count}_s{survival:g}',20,2000,seed,{'n':count,'specificity':1.,'targeted':'ranked','survival':survival},category='survival',rule='ranked',survival=survival)
    for count in [100,2000]:
        for sd in [.5,1.]:
            for seed in SEEDS:add(f'heterogeneity_n{count}_sd{sd:g}',20,2000,seed,{'n':count,'specificity':1.,'targeted':'ranked','strength_sd':sd,'strength_mean_norm':True},category='heterogeneity',rule='ranked',strength_sd=sd)
    # Sensitivity conditions are supplementary and use a single fixed neural
    # seed. They assess parameters, not biological replication.
    for site in [20,25]:
        for n in [1000,4000]:
            for factor in [.85,1.15]:add(f'weight_site{site}_n{n}_f{factor:g}',site,n,0,category='sensitivity',extra_params={'w_syn':ex.W_SYN*factor},weight_factor=factor)
            for rate in [75.,125.]:add(f'rate_site{site}_n{n}_r{rate:g}',site,n,0,category='sensitivity',stim_rate=rate)
    for factor in [.85,1.15]:
        specs.append(dict(name=f'weight_intact_f{factor:g}',seed=0,feed=True,save_rates=False,category='sensitivity_intact',weight_factor=factor,extra_params={'w_syn':ex.W_SYN*factor}))
    for rate in [75.,125.]:
        specs.append(dict(name=f'rate_intact_r{rate:g}',seed=0,feed=True,save_rates=False,category='sensitivity_intact',stim_rate=rate))
    (RES/'specs_states.json').write_text(json.dumps(specs,indent=2))
    # The first pass is a gate before the complete batch. Each record also
    # appears in the full protocol and is skipped there when already complete.
    key_names=['map_site20_n1000','map_site20_n4000','map_site20_n16000','map_site25_n1000','map_site25_n4000','map_site25_n16000','size_site20_n2000','partial_full','partial_ranked_p5','partial_random_p5']
    gate=[s for s in specs if s['name'] in key_names and s['seed']==0]
    (RES/'specs_gate.json').write_text(json.dumps(gate,indent=2))
    design={'neural_seeds':SEEDS,'screen_sizes':[1000,4000,16000],'screen_sites':len(sites),'state_count':len(specs),
            'gust_seeds':list(range(20)),'movie_gust_seed':3,'flight_duration_s':10,
            'adaptation_rounds':20,'adaptation_sizes':[1000,2000,4000,8000],
            'late_summary_rounds':[16,17,18,19,20],
            'adaptation_note':'Algorithmic rounds. No plateau or biological time claim. Display individual seeds and temporal variation.',
            'adaptation_params':dict(eta=.25,eps=2.,step=1.15,ema=.5,gmin=.5,gmax=2.),
            'delayed_branch_round':10,'delayed_primary_lesion':2000,'delayed_graft_counts':[100,2000],
            'data_policy':'Only corrected V5 results; no V4 result mixing.'}
    (RES/'protocol.json').write_text(json.dumps(design,indent=2))
    print('states',len(specs),'gate',len(gate))
if __name__=='__main__':main()
