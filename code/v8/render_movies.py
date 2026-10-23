"""V8 movies from the first prespecified realization, with shared fly scale."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import comparison_base as base

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'videos/v8'
TABLE=ROOT/'results/v8/movie_source_rows.csv'
TABLE3=ROOT/'results/v8/movie_source_rows_non_target.csv'
COLORS={'none':[224,94,65],'random':[80,85,93],'interface':[122,81,180],'ranked':[0,148,125],
        'random_partners':[80,84,91],'correct_types':[48,117,184],'incorrect_types':[193,104,136]}
STEMS={1:'Supplementary_Movie_1_lesion_location_and_size',2:'Supplementary_Movie_2_replacement_selection',3:'Supplementary_Movie_3_replacement_connections'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(number):
    table=TABLE3 if number==3 else TABLE
    data=pd.read_csv(table).set_index('name');ref=data.loc['intact']
    if number==1:
        names=[f'lesion{site}_{n}' for n in [1000,4000,16000] for site in [20,25]]
        titles=[f'{n:,} lost neurons, site {site}' for n in [1000,4000,16000] for site in [20,25]]
        cols=[COLORS['none'] if site==20 else COLORS['interface'] for n in [1000,4000,16000] for site in [20,25]]
        title='Lesion location and size';subtitle='Matched anatomy and steering trajectories';rows=3
    elif number==2:
        names=['lesion20_2000']+[f'selection_{r}_100' for r in ['random','interface','ranked']]
        titles=['No replacement','Random selection','Projection first','Visual + descending first']
        cols=[COLORS[r] for r in ['none','random','interface','ranked']]
        title='Cell selection';subtitle='2,000 cells lost; 100 correctly connected replacements';rows=2
    else:
        names=['selection_ranked_100','connections_random_partners_100','connections_correct_types_100','non_target_types_100']
        titles=['Original partners','Random partners','Required types','Non-target types']
        cols=[COLORS[r] for r in ['ranked','random_partners','correct_types','incorrect_types']]
        title='Replacement connections';subtitle='The same 100 replacement positions; paired connection and identity tests';rows=2
    scenes=[]
    for i,(name,label,col) in enumerate(zip(names,titles,cols)):
        r=data.loc[name]
        scene=dict(id=name,row=i//2,column=i%2,title=label,site=int(r.site),n_lesion=int(r.n_lesion),neural_seed=1000,
            source_table=str(table.relative_to(ROOT)),source_condition=name,lesion_idx_file=f'results/v8/condition_indices/{name}_lesion.npy',
            color=col,region_label='Posterior central brain' if r.site==20 else 'Optic lobe',
            **{k:float(r[k]) for k in ['gain_R','gain_L','bias','drive']})
        if r.n_graft:
            scene.update(graft_template_idx_file=f'results/v8/condition_indices/{name}_templates.npy',replacement_label='100 replacement neurons')
        scenes.append(scene)
    return dict(duration_s=10,fps=30,gust_seed=3,neural_seed=1000,
        selection_rule='First prespecified realization, without selection by outcome or appearance.',
        reference_note='Matched intact neural seed 1000 and gust seed 3, with the same glyph size.',
        intact_parameters={k:float(ref[k]) for k in ['gain_R','gain_L','bias','drive']},
        reference_dashed=number==3,
        title=title,subtitle=subtitle,rows=rows,columns=2,width=2160,height=1920 if rows==3 else 1620,scenes=scenes)


def comparison(number):
    m=manifest(number);OUT.mkdir(exist_ok=True)
    original=base.fly_sprite
    def equal_sprite(heading,phase,scale=1.):return original(heading,phase,scale=1.1)
    base.fly_sprite=equal_sprite
    movie=base.Movie(m)
    for name,tr in [(s['id'],tr) for s,tr in zip(movie.scenes,movie.traces)]+[('intact',movie.intact)]:
        prior=pd.read_csv(ROOT/'results/v8/trajectories'/f'{name}.csv').to_numpy()
        np.testing.assert_allclose(tr,prior,rtol=0,atol=1e-6)
    movie.write(STEMS[number]);base.fly_sprite=original
    (OUT/f'Movie{number}_v8_conditions.json').write_text(json.dumps(m,indent=2)+'\n')
    p=OUT/f'{STEMS[number]}_provenance.json';q=json.loads(p.read_text());q.update(fly_glyph_scale=1.1,neural_seed=1000,reference_is_matched=True)
    p.write_text(json.dumps(q,indent=2)+'\n')


def payloads_for(number):
    import render_navigation3d as three
    data=pd.read_csv(TABLE).set_index('name');neu=pd.read_parquet(ROOT/'data/v5/neurons.parquet')
    intact=pd.read_csv(ROOT/'results/v8/trajectories/intact.csv').to_numpy()
    if number==5:
        names=['lesion20_2000','selection_ranked_100'];labels=['No replacement','100 selected neurons'];cols=['#e05e41','#00947d']
    else:
        m=manifest(1);names=[s['id'] for s in m['scenes']];labels=[s['title'] for s in m['scenes']];cols=['#e05e41' if s['site']==20 else '#7a51b4' for s in m['scenes']]
    payloads=[];records=[];sources=[TABLE,ROOT/'data/v5/neurons.parquet',ROOT/'results/v8/trajectories/intact.csv']
    for name,title,color in zip(names,labels,cols):
        row=data.loc[name];lp=ROOT/f'results/v8/condition_indices/{name}_lesion.npy';tp=ROOT/f'results/v8/condition_indices/{name}_templates.npy'
        trajectory_path=ROOT/f'results/v8/trajectories/{name}.csv';tr=pd.read_csv(trajectory_path).to_numpy()
        lost=np.load(lp);templ=np.load(tp) if int(row.n_graft) else None;sources.extend([lp,trajectory_path])
        if templ is not None:sources.append(tp)
        payload=dict(traj=tr.tolist(),title=title,subtitle=f'{int(row.n_lesion):,} neurons lost · site {int(row.site)}',color=color,
            tier_label='',tier_color=color,stats_html='',inset_label='Lesion + replacement' if templ is not None else 'Lesion',
            cns=three.cns_payload(neu,lost,templ),fall=None,graft_size=4.)
        if number==8:payload['reference']=intact.tolist()
        payloads.append(payload)
        distance=float(np.linalg.norm(tr[-1,3:5]-tr[0,3:5]));length=float(np.linalg.norm(np.diff(tr[:,3:5],axis=0),axis=1).sum())
        records.append(dict(source_condition=name,site=int(row.site),n_lesion=int(row.n_lesion),n_replacement=int(row.n_graft),neural_seed=1000,gust_seed=3,
            trajectory_file=str(trajectory_path.relative_to(ROOT)),trajectory_points=len(tr),final_net_displacement_m=distance,final_path_length_m=length,final_straightness=distance/length,
            **{k:float(row[k]) for k in ['gain_R','gain_L','bias','drive']}))
    return payloads,records,{str(p.relative_to(ROOT)):sha(p) for p in sources}


def navigation(number,preview=False):
    import navigation3d_smooth as smooth
    smooth.base.current_payloads=lambda:payloads_for(5)
    smooth.movie8_payloads=lambda:payloads_for(8)
    smooth.render(number,preview_only=preview)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--movie',type=int,required=True,choices=[1,2,3,5,8]);p.add_argument('--preview',action='store_true');a=p.parse_args()
    comparison(a.movie) if a.movie<=3 else navigation(a.movie,a.preview)
