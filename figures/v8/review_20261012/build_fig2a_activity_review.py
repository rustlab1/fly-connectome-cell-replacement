"""Figure-only review: damage baselines, complete dose series and restrained visual cues."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import runpy
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--plain',action='store_true')
args=parser.parse_args()
EXT=ROOT/'results/v8/figure2_full_replacement_20261012'
DOSE_EXT=ROOT/'results/v8/figure2_random_dose_20261012'
b=SimpleNamespace(**runpy.run_path(str(ROOT/'code/v8/build_figures.py'),run_name='full_replacement_review_helpers'))
b.style()
centers=np.linspace(.13,.925,5)
b.replacement_schematic.__globals__['CONN_CENTERS']=centers
qa=json.loads((EXT/'analysis_qa.json').read_text());assert qa['status']=='PASS' and qa['states']==100
dose_qa=json.loads((DOSE_EXT/'analysis_qa.json').read_text());assert dose_qa['status']=='PASS' and dose_qa['states']==700
data=pd.read_csv(ROOT/'results/v8/tables/per_realization.csv')
new=pd.read_csv(EXT/'per_realization.csv')
missing=pd.read_csv(DOSE_EXT/'per_realization.csv')
d=b.old.Data()
height=7.3;vertical_scale=7.1/height
fig=b.plt.figure(figsize=(7.2,height))
a=b.container(fig,[.035,.735*vertical_scale,.945,(.249*7.1+.18)/height]);b.letter(fig,.010,.990,'a')


def schematic(ax):
    from matplotlib.patches import Circle,Polygon,FancyArrowPatch
    rows=[('n/a','None'),('+','Original'),('+','Nearby cells\nat random'),
          ('+','By cell type'),('−','By cell type')]
    for j,(x,header,(identity,partners)) in enumerate(zip(centers,b.CONN_HEADERS,rows)):
        ax.text(x,.985,header,ha='center',va='top',fontsize=8.6,color=b.INK,linespacing=1.02)
        net=ax.inset_axes([x-.074,.705,.148,.15])
        net.set_xlim(-1.05,1.05);net.set_ylim(-.65,.65);net.set_aspect('equal');net.axis('off')
        pts=np.array([[-.80,.36],[-.80,-.36],[.80,.36],[.80,-.36]])
        for p in pts:net.add_patch(Circle(p,.09,fc='#cbd3da',ec='none',zorder=3))
        if j==0:
            net.scatter([0],[0],marker='x',s=22,color=b.CONN_COLORS[j],linewidths=1.,zorder=4)
        else:
            edges=[[],[0,1,2],[1,2,3],[0,2,3],[1,2,3]][j]
            color=b.CONN_COLORS[j];style=['-','-',':','--','--'][j]
            for i in edges:
                p=pts[i];unit=p/np.linalg.norm(p);inner=unit*.18;outer=p-unit*.09
                start,end=(outer,inner) if i<2 else (inner,outer)
                net.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6,
                                             lw=1.0,linestyle=style,color=color))
            if j==4:net.add_patch(Polygon([[-.20,-.16],[.20,-.16],[0,.20]],fc=color,ec='none',zorder=4))
            else:net.add_patch(Circle((0,0),.18,fc=b.TEAL,ec='none',zorder=4))
        ax.text(x,.145,identity,ha='center',va='center',fontsize=10.5,color=b.INK)
        ax.text(x,.045,partners,ha='center',va='center',fontsize=7.8,color=b.INK,linespacing=1.02)
    ax.text(.012,.145,'Correct type',va='center',fontsize=8.4,color=b.INK)
    ax.text(.012,.045,'Partners',va='center',fontsize=8.4,color=b.INK)



def anatomy_with_activity(fig,ax,d):
    lesion=np.load(ROOT/'results/v8/condition_indices/selection_ranked_100_lesion.npy')
    template=np.load(ROOT/'results/v8/condition_indices/selection_ranked_100_templates.npy')
    known=np.isfinite(d.xyz).all(axis=1);panels=[];scales=[];files={};activity_rows=[];motor_rows=[];box_pairs=[]
    profile_dir=ROOT/'results/v8/figure2_activity_profiles_20261012'
    profile_manifest=json.loads((profile_dir/'manifest.json').read_text());assert profile_manifest['status']=='PASS'
    profiles=pd.read_csv(profile_dir/'motor_traces.csv')
    profile_ymax=float(25*np.ceil(profiles.motor_rate_hz.max()/25))
    for name,digest in profile_manifest['outputs'].items():assert hashlib.sha256((profile_dir/name).read_bytes()).hexdigest()==digest
    for path in [profile_dir/'manifest.json',profile_dir/'motor_traces.csv',profile_dir/'recordings.npz']:
        files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    motor=np.flatnonzero(d.neurons.type.eq('TTMn')|d.neurons.type.astype(str).str.startswith('DLMn'))
    intact_row=data[(data.name=='intact')&(data.rep==0)].iloc[0]
    intact_motor_hz=float((intact_row.TTMn_loom+intact_row.DLMn_loom)/2)
    intact_ratefile=ROOT/'results/v8/rates/intact_s1000.npz'
    with np.load(intact_ratefile) as saved:intact_rates=saved['loom'].copy()
    motor_groups=[(np.flatnonzero(d.neurons.type.eq('TTMn')),'TTMn_loom'),
                  (np.flatnonzero(d.neurons.type.astype(str).str.startswith('DLMn')),'DLMn_loom')]
    for group,key in motor_groups:np.testing.assert_allclose(intact_rates[group].mean(),intact_row[key],rtol=0,atol=2e-6)
    files[str(intact_ratefile.relative_to(ROOT))]=hashlib.sha256(intact_ratefile.read_bytes()).hexdigest()
    for j,(name,color) in enumerate(zip(b.CONDITIONS,b.CONN_COLORS)):
        ix=np.array([],dtype=int) if j==0 else np.load(ROOT/f'results/v8/condition_indices/{name}_templates.npy')
        if j>0:assert np.array_equal(ix,template)
        ratefile=ROOT/f'results/v8/rates/{name}_s1000.npz'
        with np.load(ratefile) as saved:rates=saved['loom'].copy()
        assert rates.shape==(len(d.xyz),) and np.isfinite(rates).all() and (rates>=0).all()
        assert rates[lesion].sum()==0
        active=np.flatnonzero(known&(rates>0));x=centers[j]
        anatomy=ax.inset_axes([x-.113,.225,.140,.470]);b.cloud(anatomy,d,brain=False,alpha=.38)
        anatomy.scatter(b.old.display_x(d.xyz[active,0]),d.xyz[active,2],s=.38+.014*rates[active],c='#3b7dab',alpha=.80,lw=0,zorder=2)
        motor_active=motor[known[motor]&(rates[motor]>0)]
        anatomy.scatter(b.old.display_x(d.xyz[motor_active,0]),d.xyz[motor_active,2],s=1.0+.025*rates[motor_active],c='#d7652a',alpha=.96,lw=0,zorder=5)
        anatomy.scatter(b.old.display_x(d.xyz[lesion,0]),d.xyz[lesion,2],s=.62,c='#edb8c5',alpha=.83,lw=0,zorder=3)
        if len(ix):anatomy.scatter(b.old.display_x(d.xyz[ix,0]),d.xyz[ix,2],s=1.0,c=color,lw=0,zorder=4)
        zoom=ax.inset_axes([x+.005,.525,.076,.145])
        zx,zz=b.old.display_x(d.xyz[lesion,0]),d.xyz[lesion,2];pad=max(np.ptp(zx),np.ptp(zz))*.08
        zoom.scatter(zx,zz,s=.65,c='#edb8c5',alpha=.72,lw=0)
        if len(ix):zoom.scatter(b.old.display_x(d.xyz[ix,0]),d.xyz[ix,2],s=1.6,c=color,lw=0)
        zoom.set_xlim(zx.min()-pad,zx.max()+pad);zoom.set_ylim(zz.max()+pad,zz.min()-pad);zoom.set_aspect('equal');zoom.set_xticks([]);zoom.set_yticks([])
        for spine in zoom.spines.values():spine.set_visible(True);spine.set_color('#d4c9cf');spine.set_linewidth(.5)
        anatomy.apply_aspect();zoom.apply_aspect()
        origin=ax.transAxes.inverted().transform(anatomy.transData.transform((zx.mean(),zz.mean())))
        target=ax.transAxes.inverted().transform(zoom.transAxes.transform((0,.45)))
        ax.annotate('',xy=target,xytext=origin,xycoords=ax.transAxes,textcoords=ax.transAxes,zorder=20,
                    arrowprops={'arrowstyle':'->','color':b.GRAY,'lw':.6,'mutation_scale':5})
        for group,key in motor_groups:
            expected=data[(data.name==name)&(data.rep==0)].iloc[0]
            np.testing.assert_allclose(rates[group].mean(),expected[key],rtol=0,atol=2e-6)
        motor_hz=float((expected.TTMn_loom+expected.DLMn_loom)/2)
        pct_intact=100*motor_hz/intact_motor_hz
        np.testing.assert_allclose(pct_intact,100*expected.takeoff_output_norm,rtol=0,atol=1e-9)
        ax.text(x+.043,.484,f'{motor_hz:.1f} Hz',fontsize=6.4,ha='center',va='center',color=color,transform=ax.transAxes)
        ax.text(x+.043,.214,f'{pct_intact:.0f}% intact',fontsize=6.4,ha='center',va='center',color=b.INK,transform=ax.transAxes)
        bounds=zoom.get_position().bounds
        shift=ax.get_position().height*.205
        profile=fig.add_axes([bounds[0],bounds[1]-shift,bounds[2],bounds[3]])
        series=profiles[profiles.condition==name]
        np.testing.assert_array_equal(series.t_ms.to_numpy(),np.arange(301))
        profile.plot(series.t_ms,series.motor_rate_hz,color=color,lw=.65)
        profile.set_xlim(0,300);profile.set_ylim(0,profile_ymax)
        profile.set_xticks([0,300]);profile.set_yticks([0,100])
        profile.tick_params(axis='both',labelsize=5.0,length=1.5,width=.45,pad=1.0)
        profile.set_xlabel('ms',fontsize=5.3,labelpad=0)
        profile.set_ylabel('Hz',fontsize=5.3,labelpad=0)
        for spine in profile.spines.values():spine.set_visible(True);spine.set_color('#9ba9b5');spine.set_linewidth(.45)
        box_pairs.append((zoom,profile))
        motor_rows.append(dict(condition=name,neural_seed=1000,intervention_seed=2000,n=1,stimulus='looming',
                               window_start_ms=50,window_end_ms=300,TTMn_hz=expected.TTMn_loom,DLMn_hz=expected.DLMn_loom,
                               motor_hz=motor_hz,intact_motor_hz=intact_motor_hz,percent_intact=pct_intact))
        source=d.neurons.iloc[active][['x_um','y_um','z_um','type']].copy()
        source['neuron_index']=active;source['condition']=name;source['mean_rate_hz']=rates[active];source['motor_highlight']=np.isin(active,motor)
        activity_rows.append(source)
        panels.append({'condition':name,'lost':len(lesion),'replacements':len(ix),'same_intended_positions':j==0 or np.array_equal(ix,template),
                       'rate_source':str(ratefile.relative_to(ROOT)),'stimulus':'loom','measurement_window_ms':[50,300],'active_host_cells_with_known_soma':len(active),
                       'active_host_vnc_cells':int(np.sum(d.xyz[active,2]>=460)),
                       'activity_marker_size_rule':'.38 + .014 * mean firing rate in Hz',
                       'host_rate_max_hz':float(rates.max()),'illustrative_activity_map_neural_seed':1000,'intervention_seed':2000,
                       'motor_mean_hz':motor_hz,'motor_percent_intact':pct_intact,'intact_motor_mean_hz':intact_motor_hz,
                       'motor_rate_definition':'Arithmetic mean of TTMn and DLMn population mean rates, including zero-rate cells.',
                       'profile_window_ms':25,'profile_xlim_ms':[0,300],'profile_ylim_hz':[0,profile_ymax],
                       'zoom_arrow_start_axes':origin.tolist(),'zoom_arrow_end_axes':target.tolist()})
        scales.append(anatomy);files[str(ratefile.relative_to(ROOT))]=hashlib.sha256(ratefile.read_bytes()).hexdigest()
    ax.scatter([.006,.006],[.420,.290],s=6,c=['#3b7dab','#d7652a'],lw=0,transform=ax.transAxes)
    ax.text(.015,.420,'Host',fontsize=7.0,ha='left',va='center',transform=ax.transAxes)
    ax.text(.015,.290,'Motor',fontsize=7.0,ha='left',va='center',transform=ax.transAxes)
    fig.canvas.draw();display=[(q.transData.transform((1,1))-q.transData.transform((0,0))).tolist() for q in scales]
    for scale in display[1:]:np.testing.assert_allclose(scale,display[0],rtol=0,atol=1e-9)
    boxes=[]
    for zoom,profile in box_pairs:
        z=zoom.get_window_extent();p=profile.get_window_extent()
        np.testing.assert_allclose([z.width,z.height],[p.width,p.height],rtol=0,atol=1e-9)
        boxes.append(dict(zoom=[z.width,z.height],profile=[p.width,p.height]))
    source_table=OUT/'Fig2a_activity_source_values.csv';pd.concat(activity_rows).to_csv(source_table,index=False)
    motor_table=OUT/'Fig2a_motor_readouts.csv';pd.DataFrame(motor_rows).to_csv(motor_table,index=False)
    return {'conditions':panels,'matched_brain_scale':True,'matched_full_cns_scale':True,'full_ventral_nerve_cord_visible':True,
            'activity_is_single_matched_realization':True,'activity_n':1,'activity_colour':'#3b7dab','motor_activity_colour':'#d7652a','rate_mean_comparison_to_frozen_states':'PASS','display_scales':display,'activity_source_hashes':files,'activity_source_table':str(source_table.relative_to(ROOT)),'activity_source_table_sha256':hashlib.sha256(source_table.read_bytes()).hexdigest(),
            'motor_readout_table':str(motor_table.relative_to(ROOT)),'motor_readout_table_sha256':hashlib.sha256(motor_table.read_bytes()).hexdigest(),
            'activity_profile_table':str((profile_dir/'motor_traces.csv').relative_to(ROOT)),
            'activity_profile_table_sha256':hashlib.sha256((profile_dir/'motor_traces.csv').read_bytes()).hexdigest(),
            'profile_window_ms':25,'profile_boxes_same_size_as_zooms':True,'profile_box_sizes_pixels':boxes,
            'profile_common_xlim_ms':[0,300],'profile_common_ylim_hz':[0,profile_ymax]}

schematic(a);anatomy=anatomy_with_activity(fig,a,d)


hundred=b.CONDITIONS
full=['lesion20_2000','complete_original_2000','review_full_random_partners_2000',
      'typed_dose_correct_types_2000','non_target_types_2000']
intact=pd.read_csv(ROOT/'results/v8/trajectories/intact.csv').to_numpy()
intact=intact.copy();intact[:,3:5]-=intact[0,3:5].copy()
arrays=[]
for names in [hundred,full]:
    group=[]
    for name in names:
        base=EXT if name.startswith('review_full_') else ROOT/'results/v8'
        arr=pd.read_csv(base/'trajectories'/f'{name}.csv').to_numpy()
        arr[:,3:5]-=arr[0,3:5].copy()
        group.append(arr)
    arrays.append(group)
xy=np.concatenate([p[:,3:5] for group in arrays for p in group]+[intact[:,3:5]])
pad_x,pad_y=(.10,.10) if args.plain else (.18,.25)
limits=((min(-.25,xy[:,1].min())-pad_x,max(.25,xy[:,1].max())+pad_x),
        (min(0,xy[:,0].min())-pad_y,max(.5,xy[:,0].max())+pad_y))
limits=((min(limits[0][0],-.1),max(limits[0][1],1.1)),limits[1])
paths=b.container(fig,[.035,.326*vertical_scale,.945,.378*vertical_scale]);b.letter(fig,.010,.717*vertical_scale,'b')
path_axes=[]
path_records=[]
for row,(group,label) in enumerate(zip(arrays,['100 cells','2,000 cells'])):
    bottom=.61-row*.52
    paths.text(-.022,bottom+.185,label,rotation=90,va='center',ha='center',fontsize=8.2,transform=paths.transAxes)
    for j,(x,arr,color) in enumerate(zip(centers,group,b.CONN_COLORS)):
        ax=paths.inset_axes([x-.061,bottom,.14,.37])
        b.path(ax,arr,intact,color,limits,left=True,bottom=True)
        ax.set_aspect('auto')
        ax.set_xlim(limits[0]);ax.set_ylim(limits[1])
        ax.tick_params(labelsize=7,pad=1)
        ax.set_xlabel('Lateral\ndisplacement (m)',fontsize=7.3,labelpad=1,linespacing=1.0)
        ax.set_ylabel('Forward\ndisplacement (m)',fontsize=7.3,labelpad=1,linespacing=1.0)
        path_axes.append(ax)
        path_records.append((ax,arr,color))
paths.plot([.86,.90],[1.04,1.04],transform=paths.transAxes,color=b.GRAY,ls='--',lw=1,clip_on=False)
paths.text(.912,1.04,'Intact',transform=paths.transAxes,va='center',fontsize=7.5,color=b.INK)


def fly_symbol(ax,arr,color):
    """Fixed-size schematic glyph, oriented along the final displayed path segment."""
    from matplotlib.offsetbox import AnnotationBbox,DrawingArea
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import Affine2D
    from matplotlib.lines import Line2D
    screen=ax.transData.transform(arr[-2:, [4,3]])
    delta=screen[1]-screen[0]
    angle=np.degrees(np.arctan2(delta[1],delta[0]))-90
    da=DrawingArea(12,12,0,0)
    rotation=Affine2D().rotate_deg_around(6,6,angle)
    for xs,ys in [([4.7,1.3],[7.5,9.8]),([4.8,1.0],[6.2,6.0]),([5.0,1.8],[4.9,2.4]),
                  ([7.3,10.7],[7.5,9.8]),([7.2,11.0],[6.2,6.0]),([7.0,10.2],[4.9,2.4])]:
        leg=Line2D(xs,ys,color=color,lw=.45);leg.set_transform(rotation+da.get_transform());da.add_artist(leg)
    parts=[(3.9,6.1,3.6,7.0,30,'white',color,.65),
           (8.1,6.1,3.6,7.0,-30,'white',color,.65),
           (6,4.0,2.6,5.0,0,color,color,1),
           (6,7.2,3.1,3.2,0,color,color,1),
           (6,9.7,3.0,2.3,0,color,color,1)]
    for x,y,w,h,ang,face,edge,alpha in parts:
        part=Ellipse((x,y),w,h,angle=ang,fc=face,ec=edge,lw=.45,alpha=alpha)
        part.set_transform(rotation+da.get_transform());da.add_artist(part)
    marker=AnnotationBbox(da,(arr[-1,4],arr[-1,3]),frameon=False,pad=0,
                         box_alignment=(.5,.5),annotation_clip=False,zorder=8)
    ax.add_artist(marker)


if not args.plain:
    fig.canvas.draw()
    for ax,arr,color in path_records:
        for artist in list(ax.collections)[:2]:artist.remove()
        for example,col in [(intact,b.GRAY),(arr,color)]:
            target_times=np.arange(1,10,dtype=float)
            indices=np.array([np.argmin(np.abs(example[:,0]-time)) for time in target_times])
            ax.scatter(example[indices,4],example[indices,3],s=5,c=col,alpha=.58,lw=0,zorder=4)
            fly_symbol(ax,example,col)

metrics=[('distance_m_norm','Distance',b.ORANGE,'o'),
         ('beam_crossings_per_s_norm','Virtual crossings/s',b.PURPLE,'s'),
         ('motor_output_norm','Wing motor',b.TEAL,'^'),
         ('takeoff_output_norm','Looming motor',b.BLUE,'D'),
         ('LPTC_L_L_norm','Spared visual',b.INK,'v')]
baseline=data[data.name=='lesion20_2000'].copy();baseline['dose']=0
random=pd.concat([baseline,data[data.name=='connections_random_partners_100'],missing,new],ignore_index=True)
frames=[baseline,b.dose_frame(data,'ranked'),random,
        b.typed_dose_frame(data,'correct_types'),b.typed_dose_frame(data,'non_target_types')]
zero=baseline.set_index('rep')
assert len(zero)==100 and zero.index.is_unique
expected_doses=[0,50,100,200,500,1000,1250,1500,1750,2000]
exports=[]
for name,q in zip(b.CONN_LABELS,frames):
    assert not q.duplicated(['dose','rep']).any()
    if name!='No graft':assert sorted(q.dose.astype(int).unique().tolist())==expected_doses
    for metric,label,color,marker in metrics:
        q[metric+'_change']=q[metric]-q.rep.map(zero[metric])
        assert q.loc[q.dose==0,metric+'_change'].eq(0).all()
        for dose,group in q.groupby('dose'):
            vals=group[metric].to_numpy();assert len(vals)==100
            m,h=b.interval(vals)
            change,ch=b.interval(group[metric+'_change'])
            exports.append(dict(condition=name,metric=metric,readout=label,dose=int(dose),n=100,
                                mean_intact_normalized=m,ci95_low=m-h,ci95_high=m+h,
                                mean_change_from_no_graft=change,change_ci95_low=change-ch,change_ci95_high=change+ch))
values=pd.DataFrame(exports)
upper=max(1.12,float(values.ci95_high.max())*1.06)
lower=min(-.025,float(values.ci95_low.min())*1.15)
box=b.container(fig,[.035,.095*vertical_scale,.945,.185*vertical_scale]);b.letter(fig,.010,.285*vertical_scale,'c')
damage_baselines={}
for j,(x,(metric,label,_,_)) in enumerate(zip(centers,metrics)):
    ax=box.inset_axes([x-.081,0,.162,1])
    ax.set_title(label,fontsize=8.5,color=b.INK,pad=4)
    damage,damage_half=b.interval(zero[metric])
    damage_baselines[metric]={'mean':damage,'ci95_low':damage-damage_half,'ci95_high':damage+damage_half}
    ax.axhline(damage,color=b.CONN_COLORS[0],ls='--',lw=1.,zorder=2)
    ax.fill_between([-60,2060],damage-damage_half,damage+damage_half,
                    color=b.CONN_COLORS[0],alpha=.07,lw=0,zorder=1)
    ax.axhline(1,color=b.GRAY,ls=':',lw=.9,zorder=2)
    for q,condition,color in zip(frames[1:],b.CONN_LABELS[1:],b.CONN_COLORS[1:]):
        b.summary(ax,q,metric,color,condition,x='dose',marker='o')
    ax.scatter([0],[damage],s=15,facecolor=b.CONN_COLORS[0],edgecolor='white',lw=.45,zorder=6)
    ax.set_ylim(lower,upper)
    ax.set_yticks([0,.5,1])
    ax.set_xlim(-60,2060);ax.set_xticks([0,1000,2000],['0','1k','2k'])
    if j==0:
        ax.set_ylabel('Readout (intact = 1)',fontsize=8.1,labelpad=2)
        ax.text(.02,.96,'n = 100; 95% CI',transform=ax.transAxes,ha='left',va='top',fontsize=6.8,color=b.INK)
    else:
        ax.tick_params(labelleft=False)
    b.old.clean(ax);ax.tick_params(labelsize=7,pad=1)
box.text(.50,-.205,'Replacement cells',transform=box.transAxes,ha='center',fontsize=8.6)
handles=[b.Line2D([0],[0],color=b.CONN_COLORS[0],ls='--',lw=1.,label='No graft')]
handles.extend(b.Line2D([0],[0],color=color,marker='o',ms=3.5,lw=1.1,label=label)
               for label,color in zip(b.CONN_LABELS[1:],b.CONN_COLORS[1:]))
handles.append(b.Line2D([0],[0],color=b.GRAY,ls=':',lw=.9,label='Intact reference'))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.008*vertical_scale),ncol=3,
           fontsize=8,frameon=False,handlelength=1.35,columnspacing=1.2,
           handletextpad=.4,labelspacing=.3)
display_limits=[{'x':list(ax.get_xlim()),'y':list(ax.get_ylim())} for ax in path_axes]
assert all(q==display_limits[0] for q in display_limits)

stem='Fig2_v8_activity_anatomy_review'
outputs=[]
for ext in ['png','pdf','svg']:
    path=OUT/f'{stem}.{ext}';fig.savefig(path,dpi=300,facecolor='white');outputs.append(path)
b.plt.close(fig)
values.to_csv(OUT/f'{stem}_source_values.csv',index=False)
meta={'status':'FIGURE_ONLY_REVIEW','panels':{'a':'Five aligned repair conditions and anatomy; no graft shown as a plain X',
      'b':'Matched 100-cell and 2,000-cell examples below each condition','c':'Five labelled readouts starting from residual function after damage, with intact = 1'},
      'conditions':b.CONN_LABELS,'path_conditions':{'100_cells':hundred,'2000_cells':full},
      'example_seeds':{'neural':1000,'intervention':2000,'gust':3},
      'path_axes_common_for_all_ten':True,'path_axis_limits':limits,
      'actual_displayed_path_limits':display_limits[0],'figure_inches':[7.2,height],
      'panel_a_extra_height_inches':.18,
      'schematic_host_nodes_per_condition':4,
      'no_graft_symbol':'Small vector X marker surrounded by the same gray host-neuron dots as the other conditions.',
      'partner_label':'By cell type',
      'partner_label_definition':'Partners are copied from a surviving neuron of the graft cell\'s assigned type. This rule applies to both correct-type and non-target grafts.',
      'path_axis_labels_each_panel':['Lateral displacement (m)','Forward displacement (m)'],
      'path_origin':'Each displayed path starts at its first saved position. Raw trajectories unchanged.',
      'path_plot_shape':'Identical rectangular axes and identical numerical ranges in all ten panels; lateral and forward metres have different display scales.',
      'visual_style':'Plain paths' if args.plain else 'Identical-size 12-point schematic flies and trajectory dots at 1-second intervals',
      'fly_glyph_note':'Glyphs use the same pose and size for every condition. Orientation follows the displayed path tangent. They do not model wing posture.',
      'fly_glyph_size_points':None if args.plain else 12,
      'trajectory_dot_times_s':[] if args.plain else list(range(1,10)),
      'anatomy':anatomy,'column_centers':centers.tolist(),'readout_y_limits':[float(lower),float(upper)],
      'readouts':[m[0] for m in metrics],
      'sample_size':100,'uncertainty':'Pointwise 95% Student t confidence interval across neural/intervention realizations. Average 20 gusts first.',
      'all_repair_group_doses':expected_doses,'random_wiring_doses':expected_doses,
      'zero_dose_definition':'Shared untreated 2,000-cell lesion. Every group starts at the measured remaining function for that readout, not at zero by construction.',
      'quantity':'Graft readout divided by matched intact readout, calculated within each realization before summarizing.',
      'no_graft_curve':'Dashed orange reference at remaining function after damage, with its 95% CI. It is a shared untreated baseline, not a graft dose series.',
      'spared_response':'Remaining visual response starts near intact = 1 and remains close to that level. No forced invariance or clipping at 1.',
      'intact_reference':'Dotted gray line at 1 in all five readout plots.',
      'damage_baselines_intact_normalized':damage_baselines,
      'background_band':None,
      'lines':'Join tested doses only. No smoothing, forced plateau or clipping at intact.',
      'dose_interpretation':'Intended positions are nested dose prefixes. Correct-type and non-target donor assignments and wiring are resampled at each dose, so these are separate grafts rather than additions to an unchanged smaller graft.',
      'primary_states_unchanged':7100,'new_review_states':800,'new_random_dose_states':700,'new_extension_separate_from_primary':True,
      'manuscript_and_package_updated_for_this_review':False,
      'annotation_limit':'All 100-cell positions are annotated. At full dose, 42 positions are untyped and the matched-type algorithm samples annotated lost-cell identities there.',
      'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
      'source_values_sha256':hashlib.sha256((OUT/f'{stem}_source_values.csv').read_bytes()).hexdigest(),
      'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'results/v8/tables/per_realization.csv',EXT/'per_realization.csv',EXT/'analysis_qa.json',
                 DOSE_EXT/'per_realization.csv',DOSE_EXT/'analysis_qa.json',DOSE_EXT/'extension_protocol.json',Path(__file__)]}}
meta['sources'].update(anatomy['activity_source_hashes'])
meta['activity_source_table']=anatomy['activity_source_table']
meta['activity_source_table_sha256']=anatomy['activity_source_table_sha256']
meta['motor_readout_table']=anatomy['motor_readout_table']
meta['motor_readout_table_sha256']=anatomy['motor_readout_table_sha256']
meta['activity_profile_table']=anatomy['activity_profile_table']
meta['activity_profile_table_sha256']=anatomy['activity_profile_table_sha256']
meta['panels']['a']='Five repair conditions, full CNS, matched looming-evoked host activity and graft close-ups'
(OUT/f'{stem}_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'status':meta['status'],'outputs':[str(p) for p in outputs]},indent=2))
