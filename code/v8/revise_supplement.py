"""Build three focused supplementary figures, then align their manuscript callouts."""
from pathlib import Path
import hashlib
import json
import re
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'report/v8'
OUT = ROOT / 'figures/v8'
DATA = ROOT / 'results/v8/supplement_revision_20261012'
DATA.mkdir(exist_ok=True)
b = SimpleNamespace(**runpy.run_path(str(ROOT / 'code/v8/build_figures.py'), run_name='supplement_helpers'))
b.style()
sys.path.insert(0, str(ROOT / 'code/v5'))
import flight_fast as flight

STEMS = {'S1': 'SuppFig1_v8_model_checks', 'S2': 'SuppFig2_v8_lesion_specificity', 'S3': 'SuppFig3_v8_replacement_controls'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(values):
    a = np.asarray(values, float)
    assert np.isfinite(a).all()
    mean = a.mean(); half = t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(mean), float(half)


def clean(ax, label):
    b.old.clean(ax)
    ax.text(-.17, 1.10, label, transform=ax.transAxes, fontsize=12, fontweight='bold')


def gust_checks(data):
    source = DATA / 'gust_strength_checks.csv'
    if source.exists():
        return pd.read_csv(source)
    protocol = {'status': 'EXPLORATORY_CONTROLLER_CHECK', 'recorded_before_simulation': True,
                'neural_states': 'Reuse the 100 existing intact realizations', 'gust_seeds': list(range(20)),
                'gust_amplitude_factors': [0, .5, 1, 1.5, 2], 'baseline_amplitude_deg_per_s': 100,
                'additional_neural_states': 0, 'purpose': 'Internal sensitivity, not experimental validation',
                'original_batch_results_available_before_this_check': True,
                'independent_uncertainty_unit': 'Intact neural realization, average 20 gusts first',
                'input_sha256': {'results/v8/tables/per_realization.csv': sha(ROOT/'results/v8/tables/per_realization.csv'),
                                  'code/v5/flight.py': sha(ROOT/'code/v5/flight.py'),
                                  'code/v5/flight_fast.py': sha(ROOT/'code/v5/flight_fast.py')}}
    (DATA / 'controller_check_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    for _, q in data[data.name=='intact'].iterrows():
        for factor in protocol['gust_amplitude_factors']:
            for gust in range(20):
                tr = flight.simulate(q.gain_R, q.gain_L, q.bias, drive=q.drive, seed=gust,
                                     params={'sigma_gust': 100*np.pi/180*factor})
                path = np.linalg.norm(np.diff(tr[:,3:5], axis=0), axis=1).sum()
                distance = np.linalg.norm(tr[-1,3:5]-tr[0,3:5])
                records.append(dict(rep=int(q.rep), seed=int(q.seed), gust_seed=gust, gust_factor=factor,
                                    distance_m=float(distance), straightness=float(distance/path)))
    checks = pd.DataFrame(records)
    assert len(checks)==10000 and not checks.duplicated(['rep','gust_seed','gust_factor']).any()
    baseline = checks[checks.gust_factor==1].groupby('rep')[['distance_m','straightness']].mean()
    reference = data[data.name=='intact'].set_index('rep')[['distance_m','straightness']]
    np.testing.assert_allclose(baseline.sort_index(), reference.sort_index(), rtol=1e-10, atol=1e-10)
    checks.to_csv(source,index=False)
    return checks


def model_checks(data):
    checks = gust_checks(data)
    fig, axes = b.plt.subplots(2,2,figsize=(7.2,5.5))
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.90,wspace=.47,hspace=.62)
    a, path, c, d = axes.flat
    intact = data[data.name=='intact'].set_index('rep')
    a.hist(intact.distance_m,bins=15,color=b.BLUE,alpha=.8,edgecolor='white')
    a.set(xlabel='Distance from start (m)',ylabel='Realizations',title='Intact movement')
    a.text(.98,.96,'n = 100',ha='right',va='top',transform=a.transAxes,fontsize=8)
    q = intact.loc[0]
    example = []
    for gust in range(20):
        tr = flight.simulate(q.gain_R,q.gain_L,q.bias,drive=q.drive,seed=gust)
        xy = tr[:,3:5]-tr[0,3:5]
        path.plot(xy[:,1],xy[:,0],color=b.BLUE,alpha=.18,lw=.8)
        if gust==3:
            chosen = xy
        example.extend(dict(gust_seed=gust,time_s=float(tr[j,0]),x_m=float(xy[j,0]),y_m=float(xy[j,1])) for j in range(0,len(tr),100))
    saved = pd.read_csv(ROOT/'results/v8/trajectories/intact.csv')
    expected = saved[['x_m','y_m']].to_numpy(np.float32)
    expected -= expected[0]
    np.testing.assert_allclose(chosen,expected,rtol=1e-6,atol=1e-6)
    path.plot(chosen[:,1],chosen[:,0],color=b.BLUE,lw=1.4,label='Gust seed 3')
    no_gust = flight.simulate(q.gain_R,q.gain_L,q.bias,drive=q.drive,seed=3,params={'sigma_gust':0})
    xy = no_gust[:,3:5]-no_gust[0,3:5]
    path.plot(xy[:,1],xy[:,0],color=b.INK,lw=1.1,ls='--',label='No gusts')
    path.set(xlabel='Lateral displacement (m)',ylabel='Forward displacement (m)',title='Wind disturbances')
    path.legend(fontsize=7.4,frameon=False,loc='lower right')
    path.text(.02,.96,'20 gust sequences',transform=path.transAxes,fontsize=8,va='top')
    pd.DataFrame(example).to_csv(DATA/'intact_gust_example_paths.csv',index=False)
    per_seed = checks.groupby(['gust_factor','rep']).straightness.mean().reset_index()
    b.summary(c,per_seed,'straightness',b.BLUE,'Intact',x='gust_factor')
    c.set(xlabel='Gust amplitude / standard',ylabel='Path straightness',title='Gust sensitivity',ylim=(0,1.05),xticks=[0,.5,1,1.5,2])
    c.text(.98,.96,'n = 100; 95% CI',transform=c.transAxes,fontsize=8,ha='right',va='top')
    d.axis('off');d.set_title('Original reconstruction',fontsize=9,pad=8)
    full = data[data.name=='complete_original_2000'].set_index('rep')
    metrics = [('Steering commands',['y_R','y_L','y_F']),('Wing motor',['motor_output']),
               ('Looming motor',['takeoff_output']),('Distance',['distance_m'])]
    values=[]
    for label, fields in metrics:
        delta = float(max((full[f]-intact[f]).abs().max() for f in fields))
        assert delta==0
        values.append([label,f'{delta:g}'])
    table = d.table(cellText=values,colLabels=['Readout','Maximum difference'],cellLoc='left',colLoc='left',bbox=[0,.12,1,.78],colWidths=[.48,.52])
    table.auto_set_font_size(False);table.set_fontsize(8.2)
    for (r,col), cell in table.get_celld().items():
        cell.set_edgecolor('#dce3e7');cell.set_linewidth(.5)
        if r==0:cell.set_text_props(weight='bold')
    d.text(.5,-.02,'Compared with matched intact; n = 100',ha='center',transform=d.transAxes,fontsize=8)
    for i, ax in enumerate(axes.flat):
        if ax is not d:clean(ax,chr(97+i))
        else:ax.text(-.17,1.10,'d',transform=ax.transAxes,fontsize=12,fontweight='bold')
    return fig, {'controller_checks':10000,'additional_neural_states':0,'independent_realizations':100,
                 'nested_gusts':20,'baseline_replays_match_original':True,'maximum_reconstruction_differences':values,
                 'purpose':'Internal controller and reconstruction checks; no external biological validation'}


def lesion_specificity(data):
    fig = b.plt.figure(figsize=(7.2,7.0))
    axes = [fig.add_axes([x,y,w,.195]) for y in [.73,.43,.13] for x,w in [(.11,.33),(.68,.30)]]
    q = data[data.category=='lesion_size']
    specs = [('distance_m','Distance from start (m)'),('beam_crossings_per_s','Virtual crossings/s'),
             ('takeoff_rate_hz','Looming motor rate (Hz)'),('LPTC_L_L','Spared visual rate (Hz)')]
    for ax, (metric,label) in zip(axes,specs):
        b.summary(ax,q,metric,b.ORANGE,'Lesion',x='n_lesion')
        m,h = interval(data.loc[data.name=='intact',metric])
        ax.axhline(m,color=b.GRAY,ls=':',lw=1,label='Intact')
        ax.set_xscale('log',base=2);ax.set_xticks([500,2000,8000,64000],['500','2k','8k','64k'])
        ax.set_xlabel('Neurons lost');ax.set_ylabel(label,fontsize=8.5)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False,fontsize=7.4)
    names=['lesion20_1000','selective_descending','selective_visual','selective_descending_visual','selective_other_37','selective_other_4']
    labels=['Focal (1,000)','Descending (37)','Tangential (4)','Both (41)','Other (37)','Other (4)']
    cols=[b.ORANGE,b.PURPLE,b.BLUE,b.TEAL,b.GRAY,b.GRAY]
    for ax,metric,label in [(axes[4],'distance_m_norm','Distance / intact'),(axes[5],'motor_output_norm','Wing motor / intact')]:
        b.dots(ax,data,names,labels,cols,metric,1,1)
        ax.set_xlabel(label,fontsize=8.5);ax.tick_params(axis='y',labelsize=8)
        ax.set_xlim(-.05,1.62);ax.set_xticks([0,.5,1,1.5])
    axes[4].set_position([.22,.13,.25,.195])
    axes[5].set_position([.725,.13,.25,.195])
    for i,ax in enumerate(axes):
        b.old.clean(ax)
        fig.text(.015 if i%2==0 else .535, [.946,.946,.646,.646,.346,.346][i],chr(97+i),weight='bold',fontsize=12)
    fig.text(.98,.985,'n = 100; 95% CI',ha='right',va='top',fontsize=8)
    return fig, {'lesion_readouts_absolute_units':['distance_m','beam_crossings_per_s','takeoff_rate_hz','LPTC_L_L'],
                 'selective_lesion_readouts':['distance_m_norm','motor_output_norm'],'n':100,
                 'removed_redundant_panels':'Selective leftward gain and repeated spared-response ablation plot'}


def replacement_controls(data):
    source = pd.read_csv(ROOT/'results/v5/revision_20261002_clarity/all_direct_readouts.csv')
    fig = b.plt.figure(figsize=(7.2,7.2))
    a,bx,c,d,e,f = [fig.add_axes([x,y,w,.19]) for y in [.745,.44,.13] for x,w in [(.11,.33),(.68,.285)]]
    for rule in ['random','interface','ranked']:
        q = b.dose_frame(data,rule)
        b.summary(a,q,'distance_m',b.RULE_COLORS[rule],b.RULE_LABELS[rule],x='dose')
    a.axhline(data.loc[data.name=='intact','distance_m'].mean(),color=b.GRAY,ls=':',lw=.85)
    a.set(xlim=(-35,2035),xticks=[0,1000,2000],xlabel='Replacement cells',ylabel='Distance from start (m)',title='Cell selection; n = 100')
    a.legend(frameon=False,fontsize=7.3,loc='lower right',handlelength=1.2,labelspacing=.2)
    intact = data[data.name=='intact'].set_index('rep')
    direction_rows=[]
    for j,(name,col) in enumerate(zip(b.CONDITIONS,b.CONN_COLORS)):
        for shift,metric,marker in [(-.11,'gain_R','o'),(.11,'gain_L','^')]:
            q=data[data.name==name].set_index('rep');vals=q[metric]/intact[metric]
            mean,half=interval(vals)
            bx.errorbar(mean,j+shift,xerr=half,fmt=marker,color=col,ms=4,capsize=2,lw=.9)
            direction_rows.append(dict(name=name,metric=metric+'_matched_intact',n=len(vals),mean=mean,ci_low=mean-half,ci_high=mean+half))
    bx.set_yticks(range(5),b.CONN_LABELS,fontsize=7.4);bx.set_ylim(4.5,-.5);bx.axvline(1,color=b.GRAY,ls=':',lw=.8)
    bx.set_xlabel('Response / intact',fontsize=8.5);bx.set_title('Directional responses; n = 100',fontsize=8.7)
    bx.legend(handles=[b.Line2D([],[],marker='o',color=b.INK,ls='',label='Right'),b.Line2D([],[],marker='^',color=b.INK,ls='',label='Left')],frameon=False,ncol=2,fontsize=7.1,loc='upper left',bbox_to_anchor=(0,1.02),borderaxespad=0)
    pd.DataFrame(direction_rows).to_csv(DATA/'directional_responses.csv',index=False)
    for wiring,col,label,ls in [('correct_types',b.BLUE,'Correct-type graft','-'),('incorrect_types','#b98a3c','Shuffled lost types','-'),('non_target_types','#c16888','Non-target graft','-')]:
        q=b.typed_dose_frame(data,wiring)
        b.summary(c,q,'distance_m_norm',col,label,x='dose',ls=ls)
    c.axhline(1,color=b.GRAY,ls=':',lw=.85)
    c.set(xlim=(-35,2035),xticks=[0,1000,2000],xlabel='Replacement cells',ylabel='Distance / intact',title='Shuffled identities; n = 100')
    c.legend(frameon=False,fontsize=6.9,loc='upper left',handlelength=1.2,labelspacing=.2)
    d.set_position([.68,.44,.235,.19])
    grid = b.exploratory_partner_grid(fig,d,source)
    d.set_title('Partner retention; n = 3',fontsize=9)
    baseline=float(source[source.name=='size_site20_n2000'].net_displacement_m.mean())
    intact_m=float(source[(source.audit_group=='intact')&source.seed.isin([0,1,2])].net_displacement_m.mean())
    raw = []
    for count,base,marker in [(100,'partial_ranked_p5','o'),(200,'partial_ranked_p10','s')]:
        q=source[(source.category=='survival')&(source.n_transplanted==count)].copy()
        full=source[source.name==base].copy();full['survival']=1
        q=pd.concat([q,full]);q['survival_pct']=q.survival*100
        col=b.TEAL if count==100 else b.BLUE
        for _, seed in q.groupby('seed'):
            seed=seed.sort_values('survival_pct')
            e.plot(seed.survival_pct,seed.net_displacement_m,color=col,alpha=.23,lw=.7)
            e.scatter(seed.survival_pct,seed.net_displacement_m,color=col,alpha=.5,s=11,marker=marker)
        means=q.groupby('survival_pct').net_displacement_m.mean()
        e.plot(means.index,means,color=col,marker=marker,ms=5,label=f'{count} cells added')
        raw.extend(dict(test='survival',name=r['name'],seed=r['seed'],initial_cells=count,x=r['survival_pct'],surviving_cells=r['n_graft'],distance_m=r['net_displacement_m']) for r in q.to_dict('records'))
    for count,base,marker in [(100,'partial_ranked_p5','o'),(2000,'partial_full','s')]:
        q=source[(source.category=='heterogeneity')&(source.n_graft==count)].copy()
        full=source[source.name==base].copy();full['strength_sd']=0
        q=pd.concat([q,full]);col=b.TEAL if count==100 else b.BLUE
        for _, seed in q.groupby('seed'):
            seed=seed.sort_values('strength_sd')
            f.plot(seed.strength_sd,seed.net_displacement_m,color=col,alpha=.23,lw=.7)
            f.scatter(seed.strength_sd,seed.net_displacement_m,color=col,alpha=.5,s=11,marker=marker)
        means=q.groupby('strength_sd').net_displacement_m.mean()
        f.plot(means.index,means,color=col,marker=marker,ms=5,label=f'{count:,} cells')
        raw.extend(dict(test='strength',name=r['name'],seed=r['seed'],initial_cells=count,x=r['strength_sd'],surviving_cells=r['n_graft'],distance_m=r['net_displacement_m']) for r in q.to_dict('records'))
    for ax in [e,f]:
        ax.axhline(intact_m,color=b.GRAY,ls=':',lw=.85);ax.axhline(baseline,color=b.ORANGE,ls='--',lw=.85)
        ax.set_ylim(0,3);ax.set_yticks([0,1,2,3]);ax.set_ylabel('Distance from start (m)',fontsize=8.5)
        ax.legend(frameon=False,fontsize=7.1,loc='lower left')
    e.set(xlim=(45,105),xticks=[50,75,100],xlabel='Cell survival probability (%)',title='Graft survival; n = 3')
    f.set(xlim=(-.05,1.05),xticks=[0,.5,1],xticklabels=['Uniform','Moderate','High'],xlabel='Synaptic-strength variation',title='Graft strength; n = 3')
    pd.DataFrame(raw).to_csv(DATA/'survival_and_strength_individuals.csv',index=False)
    for i,ax in enumerate([a,bx,c,d,e,f]):
        b.old.clean(ax);ax.text(-.26,1.11,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=12)
    return fig, {'n_by_panel':{'a':100,'b':100,'c':100,'d':3,'e':3,'f':3},
                 'exploratory_partner_grid':grid,'survival_strength_display':'All three outcomes and their mean, no confidence intervals',
                 'shuffled_lost_types_distinct_from_non_target':True,
                 'removed_redundant_panels':'Repeated full-dose identity bars, looming bars, unique-gate bars and identity-mixture straightness'}


LEGENDS = {
 'S1': '**Supplementary Figure 1 | Controller and reconstruction checks.** **a**, Intact distance after ten seconds across 100 neural realizations, each averaged over 20 gusts. **b**, Twenty gust sequences applied to the same intact neural realization (seed 1000), with gust seed 3 highlighted and a no-gust reference. **c**, Path straightness as gust amplitude changes, reusing the 100 intact neural states. Points and ribbons show means and pointwise 95% confidence intervals after averaging 20 gusts within each realization. Standard amplitude reproduces the original controller outcomes. This separate diagnostic contributes 10,000 controller runs, including 2,000 repeated standard-amplitude runs, and no new neural states. **d**, Maximum absolute differences between intact and complete original reconstruction across the 100 paired realizations. All differences are zero. These are internal computational checks, not validation against observed fly behaviour.',
 'S2': '**Supplementary Figure 2 | Absolute lesion readouts and cell-class contributions.** **a–d**, Distance after ten seconds, repeated virtual crossings per second, looming-evoked motor rate and spared visual rate across the nested site-20 lesion series. Dotted grey lines show intact means. **e,f**, Distance and wing-motor activity relative to matched intact after the 1,000-cell focal lesion or selective removal of its 37 descending neurons, four tangential cells, their 41-cell union, or 37 or four other cells from the same lesion mask. Dotted lines show intact. Points and uncertainty show means and pointwise 95% confidence intervals across 100 realizations; movement averages 20 matched gusts first. Cell-class ablations do not match focal-lesion geometry. Crossing rate counts repeated passages through virtual flight planes, not walking beam counts; looming rates are neural activity, not takeoff probability.',
 'S3': '**Supplementary Figure 3 | Additional replacement controls.** **a**, Distance under random selection, projection priority or motor-control priority, all with original identities and available partners. Random selection differs from random wiring. **b**, Rightward (circles) and leftward (triangles) responses for the five main repair conditions with 100 replacements, divided by each matched intact response. **c**, Shuffled lost types, compared with correct-type and non-target grafts under the same donor-based wiring rule. Shuffling draws other types present in the lesion; non-target types are absent. Panels a–c use means and pointwise 95% confidence intervals across 100 realizations. **d**, Earlier partner-retention grid: absolute distance after a 1,000-cell lesion, n = 3. The 2,000-cell graft duplicates templates and changes synaptic load. **e**, Cell survival probability after adding 100 or 200 selected cells. **f**, Uniform, moderately variable or highly variable synaptic strengths among 100 or 2,000 replacements, with all cells surviving. Lognormal multipliers have log standard deviations 0, 0.5 or 1 and expected multiplier one. Panels e,f show all three exploratory realizations and their means; faint lines join matched seeds. Grey dotted and orange dashed lines show intact and untreated means. All movement averages 20 gusts per realization.'
}


def update_text():
    p=REPORT/'manuscript_v8.md';main=p.read_text()
    original_citations=re.findall(r'\^(\d+(?:[–,]\d+)*)\^',main)
    if 'Supplementary Fig. 3a' not in main:
        mapping={'6':'2a–d','6d':'2d','4':'2e,f','2g':'3a','2':'3a,c','5e':'3d','5b':'3b','5':'3b–d','3':'3e,f'}
        pattern=r'(Supplementary (?:Figures?|Figs?\.?) )(\d+[a-g]?(?:\s*(?:and|,|–)\s*\d+[a-g]?)*)'
        main=re.sub(pattern,lambda m:m[1]+re.sub(r'\d+[a-g]?',lambda q:mapping.get(q[0],q[0]),m[2]),main)
        main=main.replace('(Supplementary Figs. 3e,f and 3b–d)', '(Supplementary Fig. 3d–f)')
        main=main.replace('retained in Supplementary Figures 3a,c and 3b–d.', 'retained in Supplementary Fig. 3c.')
        main=main.replace('(Fig. 2c and Supplementary Fig. 3b)', '(Fig. 2c)')
        main=main.replace('Directional responses, neural activity and virtual flight-gate crossings provide complementary readouts (Supplementary Fig. 3a,c). Earlier exploratory tests with three realizations also examined replacement survival, variable synaptic strength and complete-dose connection rules (Supplementary Fig. 3d–f).',
           'Directional responses and the shuffled-lost-type control provide additional tests of integration (Supplementary Fig. 3b,c). Earlier exploratory tests with three realizations examined original-partner retention, replacement survival and variable synaptic strength (Supplementary Fig. 3d–f).')
        main=main.replace('(Supplementary Fig. 1c,d)', '(archived sensitivity source tables)')
        main=main.replace('The survival and strength analyses in Supplementary Figure 3e,f retain three original seeds, individual outcomes and 95% confidence intervals.',
                          'The survival and strength analyses in Supplementary Figure 3e,f retain three original seeds and show every outcome with its mean, without confidence intervals.')
        main=main.replace('Except for Figure 1f, uncertainty displays', 'Except for Figure 1f and the individual-outcome displays in Supplementary Fig. 3e,f, uncertainty displays')
        anchor='These trajectories describe steering without full flight biomechanics.'
        main=main.replace(anchor, 'Changing the gust sequence altered the path of the same intact circuit, while complete original reconstruction reproduced its intact responses exactly (Supplementary Fig. 1b–d). '+anchor)
        diagnostic='For the separate controller checks in Supplementary Fig. 1b,c, we reused the 100 intact neural states while setting gust amplitude to 0, 0.5, 1, 1.5 or 2 times its standard value. Each state used the same 20 gust seeds, giving 10,000 diagnostic trajectories, including 2,000 repeated standard-amplitude trajectories. This exploratory analysis added no neural states and is separate from the 158,000 trajectories in the replacement and lesion batch. Standard amplitude reproduced the existing intact outcomes. The no-gust example retained the circuit\'s existing gains, bias and drive.'
        main=main.replace('\n\nDistance from the starting position was', '\n\n'+diagnostic+'\n\nDistance from the starting position was')
    main=main.replace('implementation check (Supplementary Fig. 1b)', 'implementation check (Supplementary Fig. 1d)')
    assert re.findall(r'\^(\d+(?:[–,]\d+)*)\^',main)==original_citations
    p.write_text(main)
    old=(REPORT/'supplementary_information_v8.md').read_text()
    before=old.split('[[FIGURE:S1]]')[0]
    before=re.sub(r'Supplementary Figure 1 covers[^\n]*', 'Three supplementary figures support the main results. Figure 1 checks the controller and exact reconstruction. Figure 2 provides absolute lesion readouts and selective cell-class ablations. Figure 3 combines the cell-selection, wiring, graft-survival and synaptic-strength controls. Redundant main-figure readouts are omitted; their raw data remain in the source package.', before)
    before=before.replace('The new supplement contains six figures.', 'The supplement contains three figures.')
    supplement=before+'\n\n'+'\n\n'.join('[[FIGURE:'+key+']]\n\n'+legend for key,legend in LEGENDS.items())+'\n'
    (REPORT/'supplementary_information_v8.md').write_text(supplement)
    from integrate_approved_fig2 import refresh_captions
    # The reduced supplement intentionally has three figures, not six.
    legends={}
    for text in [main,supplement]:
        for block in text.split('\n\n'):
            match=re.match(r'\*\*(Supplementary )?Figure (\d+) \|',block)
            if match:legends[('S' if match[1] else '')+match[2]]=block.strip()
    assert set(legends)=={'1','2','S1','S2','S3'}
    (REPORT/'figure_legends.json').write_text(json.dumps(legends,indent=2)+'\n')
    captions=dict(re.findall(r'\*\*Supplementary Movie (\d+)\.\*\* ([^\n]+)',supplement))
    (REPORT/'movie_captions.json').write_text(json.dumps(captions,indent=2)+'\n')


def main():
    data=pd.read_csv(ROOT/'results/v8/tables/per_realization.csv')
    qpath=OUT/'figure_qa.json';qa=json.loads(qpath.read_text())
    qa['figures']={k:v for k,v in qa['figures'].items() if not k.startswith('SuppFig')}
    qa['figures']['Fig2_v8_cell_replacement']['old_panel_b_moved_to']='Supplementary Figure 3d'
    qa['figures']['Fig2_v8_cell_replacement']['earlier_cell_selection_distance_plot']='Supplementary Figure 3a'
    qa['figures']['Fig1_v8_focal_loss']['spared_visual_response']='Black line in Figure 1f and retained in Supplementary Figure 2d'
    qa['output_sha256']={k:v for k,v in qa['output_sha256'].items() if not Path(k).name.startswith('SuppFig')}
    for key,builder in [('S1',model_checks),('S2',lesion_specificity),('S3',replacement_controls)]:
        fig,meta=builder(data);stem=STEMS[key];qa['figures'][stem]=meta
        for ext in ['png','pdf','svg']:
            p=OUT/(stem+'.'+ext);fig.savefig(p,dpi=300,facecolor='white');qa['output_sha256'][str(p.relative_to(ROOT))]=sha(p)
        b.plt.close(fig)
        print(stem,flush=True)
    qa['input_sha256']['code/v8/revise_supplement.py']=sha(Path(__file__))
    qa['input_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in DATA.iterdir() if p.suffix in ['.json','.csv']})
    qa['supplement_revision_20261012']={'figures':3,'order':'Model checks, lesion specificity, replacement controls','status':'PENDING_VISUAL_AND_SOURCE_QA'}
    qpath.write_text(json.dumps(qa,indent=2)+'\n')
    update_text()
    print('Three supplementary figures and synchronized callouts prepared.')


if __name__=='__main__':
    main()
