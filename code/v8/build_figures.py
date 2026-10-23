"""V8 main figures: frozen exploratory screen and new paired realizations."""
from pathlib import Path
import os
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/fly-v7-mainfigs-matplotlib')
sys.path.insert(0, str(ROOT / 'code/v5'))
import build_figures as old
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import t
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT = ROOT / 'figures/v8'
BLUE, ORANGE, TEAL, PURPLE, RED = old.BLUE, old.ORANGE, old.TEAL, old.PURPLE, old.RED
INK, MUTED, GRAY = old.INK, old.MUTED, old.GRAY
RULE_COLORS = old.RULE_COLORS
RULE_LABELS = {'random': 'Random selection', 'interface': 'Projection priority', 'ranked': 'Motor-control priority'}



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    old.style()
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
                         'ytick.labelsize': 8.5, 'legend.fontsize': 8,
                         'axes.linewidth': .7, 'lines.linewidth': 1.35,
                         'xtick.major.size': 3, 'ytick.major.size': 3})


def letter(fig, x, y, value):
    fig.text(x, y, value, fontsize=12, fontweight='bold', ha='left', va='top', color=INK)


def container(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis('off')
    return ax


def cloud(ax, d, brain=False, alpha=.48):
    old.cloud(ax, d, brain=brain, alpha=alpha)


def summary(ax, q, metric, color, label, x='n_lesion', marker='o', ls='-'):
    g = q.groupby(x)[metric].agg(['mean', 'std', 'count']).sort_index()
    half=t.ppf(.975,g['count']-1)*g['std']/np.sqrt(g['count'])
    ax.plot(g.index,g['mean'],color=color,marker=marker,ms=3.4,label=label,zorder=3,ls=ls)
    ax.fill_between(g.index,g['mean']-half,g['mean']+half,color=color,alpha=.16,lw=0)


def simulation_range(ax, q, metric, color, label, x='n_lesion', marker='o'):
    g=q.groupby(x)[metric]
    mean=g.mean().sort_index()
    low=g.quantile(.025).reindex(mean.index)
    high=g.quantile(.975).reindex(mean.index)
    ax.fill_between(mean.index,low,high,color=color,alpha=.25,lw=0,zorder=1)
    ax.plot(mean.index,mean,color=color,marker=marker,ms=3.4,label=label,lw=1.5,zorder=3)
    return dict(metric=metric,lesion_sizes=mean.index.astype(int).tolist(),
                mean=mean.tolist(),p025=low.tolist(),p975=high.tolist(),
                n=g.size().reindex(mean.index).astype(int).tolist())


def trajectory_arrays(d, names):
    arrays=[pd.read_csv(ROOT/'results/v8/trajectories'/f'{name}.csv').to_numpy() for name in names+['intact']]
    xy=np.concatenate([p[:,3:5] for p in arrays])
    limits=((min(-.25,xy[:,1].min())-.1,max(.25,xy[:,1].max())+.1),(min(0,xy[:,0].min())-.1,max(.5,xy[:,0].max())+.1))
    return arrays[:-1],arrays[-1],limits


def path(ax, arr, intact, color, limits, left=True, bottom=True):
    ax.plot(intact[:, 4], intact[:, 3], color=GRAY, ls='--', lw=1.15)
    ax.plot(arr[:, 4], arr[:, 3], color=color, lw=1.7)
    ax.scatter([intact[-1, 4]], [intact[-1, 3]], s=16, c=GRAY, zorder=5)
    ax.scatter([arr[-1, 4]], [arr[-1, 3]], s=18, c=color, zorder=5)
    ax.scatter([0], [0], s=12, c=INK, zorder=6)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1, 2])
    old.clean(ax, False)
    ax.tick_params(labelsize=8)
    if not left:
        ax.tick_params(labelleft=False)
    if not bottom:
        ax.tick_params(labelbottom=False)


def figure1(d):
    fig = plt.figure(figsize=(7.2, 6.8))
    a = container(fig, [.035, .65, .36, .32]); letter(fig, .015, .988, 'a')
    anatomy = a.inset_axes([0, .08, .55*.305/.36, .30*.91/.32]); cloud(anatomy, d, alpha=.6)
    for key, col, size in [('LPTC', BLUE, 3.2), ('DN', PURPLE, 1.1), ('steerMN', ORANGE, 4.7)]:
        ix = np.array(d.groups[key + '_L'] + d.groups[key + '_R'])
        ix = ix[np.isfinite(d.xyz[ix, 0])]
        anatomy.scatter(old.display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=size, c=col, linewidths=0)
    anatomy.text(395, 50, 'Brain +\noptic lobes', ha='center', fontsize=7.7, color=MUTED)
    anatomy.text(395, 1140, 'Ventral\nnerve cord', ha='center', fontsize=7.7, color=MUTED)
    steps = [('Visual\nmotion', INK), ('Sensory\nprocessing', BLUE), ('Sensorimotor\nintegration', BLUE),
             ('Descending motor\ncontrol', PURPLE), ('Wing motor\noutput', ORANGE), ('Steering', INK)]
    for i, (label, color) in enumerate(steps):
        y = .95-i*.166
        a.text(.76, y, label, ha='center', va='center', fontsize=8.1, color=color, linespacing=1.03)
        if i < 5:
            a.annotate('', xy=(.76, y-.115), xytext=(.76, y-.057),
                       arrowprops={'arrowstyle': '-|>', 'lw': .65, 'color': GRAY, 'mutation_scale': 6})

    b = container(fig, [.43, .67, .55, .30]); letter(fig, .405, .988, 'b')
    norm = colors.Normalize(0, 1)
    b_maps=[]
    for j, size in enumerate([1000, 4000, 16000]):
        ax = b.inset_axes([j/3, .045, .305, .91]); old.cmap_plot(ax, d, 'straightness', size, norm)
        b_maps.append(ax)
        ax.set_title(f'{size//1000}k', fontsize=9.3, pad=4)
        for text in ax.texts:
            text.set_visible(False)
    cax = b.inset_axes([.10, -.07, .80, .032])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face'); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, pad=1); cb.set_label('Path straightness', fontsize=8.5, labelpad=1)

    c = fig.add_axes([.10, .485, .78, .09]); letter(fig, .015, .60, 'c')
    screen = pd.read_csv(ROOT/'results/v5/revision_20261002/all_sites_mean_readouts.csv')
    map_values=d.sites.set_index(['site','n_lesion']).straightness.sort_index()
    screen_values=screen.set_index(['site','n_lesion']).straightness.sort_index()
    np.testing.assert_allclose(map_values,screen_values,rtol=0,atol=1e-12)
    wide = screen.pivot(index='site', columns='n_lesion', values='net_displacement_m')[[1000, 4000, 16000]]
    assert wide.shape == (56, 3) and int((wide[16000] < wide[1000]).sum()) == 55
    wide=wide.sort_index()
    im=c.pcolormesh(np.arange(57)-.5,np.arange(4)-.5,wide.to_numpy().T,
                   cmap='viridis',vmin=0,vmax=3,shading='flat',rasterized=False)
    c.set_xlim(-.5,55.5);c.set_ylim(2.5,-.5)
    site_ids=wide.index.astype(int).to_numpy()
    ticks=[i for i,site in enumerate(site_ids) if site%5==0 or site==55]
    c.set_xticks(ticks,[str(site_ids[i]) for i in ticks],fontsize=7.2)
    c.set_yticks([0,1,2],['1k','4k','16k'],fontsize=8)
    c.set_xlabel('Lesion site',fontsize=8.5,labelpad=2)
    c.set_ylabel('Neurons lost',fontsize=8.5,labelpad=5)
    for site,color in [(20,ORANGE),(25,BLUE)]:
        col=int(np.flatnonzero(site_ids==site)[0])
        c.add_patch(plt.Rectangle((col-.5,-.5),1,3,fill=False,edgecolor=color,lw=1.2,clip_on=False))
        for tick in c.get_xticklabels():
            if tick.get_text()==str(site):tick.set_color(color);tick.set_fontweight('bold')
    c.tick_params(axis='both',length=2,pad=2)
    for spine in c.spines.values():spine.set_visible(False)
    c.text(0,1.18,'56 sites; n = 3 per site and lesion size',transform=c.transAxes,ha='left',fontsize=7.2,color=MUTED)
    cax=fig.add_axes([.90,.485,.013,.09])
    cb=fig.colorbar(im,cax=cax,ticks=[0,1,2,3]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7,length=2,pad=2);cb.set_label('Distance (m)',fontsize=8,labelpad=3)

    local = container(fig, [.06, .04, .33, .345]); letter(fig, .015, .415, 'd')
    arrays, intact, limits = trajectory_arrays(d, ['lesion20_1000', 'lesion25_1000'])
    counts = {}
    for j, site in enumerate([20, 25]):
        ix = d.lesion(site, 1000)
        dn = ix[d.neurons.superclass.to_numpy()[ix] == 'descending_neuron']
        vi = ix[np.isin(ix, d.groups['LPTC_L'] + d.groups['LPTC_R'])]
        counts[str(site)] = {'descending': len(dn), 'visual_integration': len(vi), 'total': len(ix)}
        ax = local.inset_axes([j*.52, .775, .47, .22]); cloud(ax, d, brain=True)
        ax.scatter(old.display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=.9, c='#efa260', lw=0)
        ax.scatter(old.display_x(d.xyz[dn, 0]), d.xyz[dn, 2], s=4.4, c=PURPLE, edgecolors='white', lw=.15, zorder=5)
        ax.scatter(old.display_x(d.xyz[vi, 0]), d.xyz[vi, 2], s=9.5, c=BLUE, edgecolors='white', lw=.2, zorder=6)
        ax.set_title(f'Site {site}', fontsize=9, pad=1)
        local.text(j*.52+.235, .68, f'{len(dn)} descending', ha='center', color=PURPLE, fontsize=8, fontweight='bold')
        local.text(j*.52+.235, .61, f'{len(vi)} visual', ha='center', color=BLUE, fontsize=8, fontweight='bold')
        ax = local.inset_axes([j*.52+.07, .075, .36, .40])
        path(ax, arrays[j], intact, [ORANGE, BLUE][j], limits, left=(j == 0))
        ax.set_xlabel('Lateral (m)', fontsize=8)
        if j == 0:
            ax.set_ylabel('Forward (m)', fontsize=8, labelpad=2)
    local.text(.5,.52,'Brain to motor circuits',ha='center',color=PURPLE,fontsize=7.8)
    assert counts == {'20': {'descending': 37, 'visual_integration': 4, 'total': 1000},
                      '25': {'descending': 0, 'visual_integration': 0, 'total': 1000}}

    nested = container(fig, [.445, .06, .18, .33]); letter(fig, .42, .415, 'e')
    ax = nested.inset_axes([0, .24, 1, .72]); cloud(ax, d)
    sizes = [(64000, '#733047'), (16000, '#c75c68'), (4000, '#efa260'), (500, '#259c84')]
    for n, color in sizes:
        ix = d.lesion(20, n)
        ax.scatter(old.display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=.5, c=color, lw=0, zorder=3)
    ax.set_title('Site 20', fontsize=9, pad=2)
    for j, (n, color) in enumerate(reversed(sizes)):
        nested.scatter(.01, .17-j*.085, s=14, color=color, transform=nested.transAxes, clip_on=False)
        label = f'{n/1000:g}k' if n >= 1000 else '500'
        nested.text(.10, .17-j*.085, f'{label} ({100*n/len(d.neurons):.1f}%)', va='center', fontsize=7.8, transform=nested.transAxes)

    f = fig.add_axes([.745, .085, .24, .29]); letter(fig, .69, .407, 'f')
    q = pd.read_csv(ROOT/'results/v8/tables/per_realization.csv'); q=q[q.category=='lesion_size']
    bands=[]
    for metric, color, label, marker in [
        ('distance_m_norm', ORANGE, 'Distance', 'o'),
        ('beam_crossings_per_s_norm', PURPLE, 'Virtual crossings/s', 's'),
        ('motor_output_norm', TEAL, 'Wing motor', '^'),
        ('takeoff_output_norm', BLUE, 'Looming motor', 'D'),
        ('LPTC_L_L_norm', INK, 'Spared visual', 'v')]:
        bands.append(simulation_range(f,q,metric,color,label,marker=marker))
    f.set_xscale('log', base=2); f.set_xticks([500, 2000, 8000, 64000], ['500', '2k', '8k', '64k'])
    f.set_ylim(-.035, 1.63); f.set_yticks([0, .5, 1, 1.5]); f.axhline(1, color=GRAY, ls=':', lw=.85)
    f.set_xlabel('Neurons lost'); f.set_ylabel('Readout / intact', fontsize=8.5)
    handles,labels=f.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.1,loc='upper right',bbox_to_anchor=(.988,.465),ncol=2,handlelength=1.3,columnspacing=.7,labelspacing=.25)
    f.text(.01,1.035,f'n = {q.groupby("name").size().min()}; 95% simulation range',transform=f.transAxes,fontsize=6.7,color=MUTED)
    old.clean(f)
    fig.canvas.draw()
    probes=np.array([[0,0],[100,0],[0,100]])
    a_scale=np.linalg.norm(np.diff(anatomy.transData.transform(probes),axis=0),axis=1)
    b_scales=[np.linalg.norm(np.diff(ax.transData.transform(probes),axis=0),axis=1) for ax in b_maps]
    for scale in b_scales:np.testing.assert_allclose(a_scale,scale,rtol=0,atol=1e-9)
    return fig, {'lost_cells_verified': counts, 'screen_locations': 56, 'screen_sizes': [1000, 4000, 16000],
                 'locations_decreased_1000_to_16000': 55, 'panel_mapping': {'a': 'circuit model', 'b': 'fixed-location maps',
                 'c': 'site-by-size distance heatmap', 'd': 'local examples and actual lost cell counts', 'e': 'nested lesion', 'f': 'size series'},
                 'panel_c_site_order':site_ids.tolist(),'panel_c_distance_matrix_m':wide.to_numpy().T.tolist(),
                 'panel_f_uncertainty':'2.5th to 97.5th empirical percentiles across 100 realization means',
                 'panel_f_bands':bands,'spared_visual_response':'Black line in Figure 1f and retained in Supplementary Figure 6d',
                 'matched_cns_scale_a_b':True,'cns_display_scale_pixels':{'a':a_scale.tolist(),'b':[scale.tolist() for scale in b_scales]}}


CONDITIONS=['lesion20_2000','selection_ranked_100','connections_random_partners_100','connections_correct_types_100','non_target_types_100']
CONN_LABELS=['No graft','Original repair','Random wiring','Correct-type graft','Non-target graft']
CONN_COLORS=[ORANGE,TEAL,'#50545b',BLUE,'#c16888']
CONN_HEADERS=['No\ngraft','Original\nrepair','Random\nwiring','Correct-type\ngraft','Non-target\ngraft']
CONN_CENTERS=np.linspace(.24,.935,5)


def replacement_schematic(ax):
    from matplotlib.patches import Circle, Polygon, FancyArrowPatch
    centers=CONN_CENTERS
    rows=[('n/a','None'),('+','Original'),('+','Nearby cells\nat random'),
          ('+','Copied from\nsame-type cell'),('−','Copied from\nsame-type cell')]
    for j,(x,header,(identity,partners)) in enumerate(zip(centers,CONN_HEADERS,rows)):
        ax.text(x,.985,header,ha='center',va='top',fontsize=8.6,color=INK,linespacing=1.02)
        net=ax.inset_axes([x-.074,.735,.148,.17])
        net.set_xlim(-1.05,1.05);net.set_ylim(-.65,.65);net.set_aspect('equal');net.axis('off')
        pts=np.array([[-.80,.36],[-.80,-.36],[.80,.36],[.80,-.36]])
        edges=[[],[0,1,2],[1,2,3],[0,2,3],[1,2,3]][j]
        color=CONN_COLORS[j];style=['-','-',':','--','--'][j]
        for i in edges:
            p=pts[i];unit=p/np.linalg.norm(p)
            inner=unit*.18;outer=p-unit*.09
            start,end=(outer,inner) if i<2 else (inner,outer)
            net.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6,
                                         lw=1.0,linestyle=style,color=color))
        for p in pts:net.add_patch(Circle(p,.09,fc='#cbd3da',ec='none',zorder=3))
        if j==0:
            net.add_patch(Circle((0,0),.18,fc='white',ec=ORANGE,lw=.9,linestyle='--'))
            net.plot([-.10,.10],[-.10,.10],color=ORANGE,lw=.9)
            net.plot([-.10,.10],[.10,-.10],color=ORANGE,lw=.9)
        elif j==4:net.add_patch(Polygon([[-.20,-.16],[.20,-.16],[0,.20]],fc=color,ec='none',zorder=4))
        else:net.add_patch(Circle((0,0),.18,fc=TEAL,ec='none',zorder=4))
        ax.text(x,.145,identity,ha='center',va='center',fontsize=10.5,color=INK)
        ax.text(x,.045,partners,ha='center',va='center',fontsize=7.8,color=INK,linespacing=1.02)
    ax.text(.012,.145,'Correct type',va='center',fontsize=8.4,color=INK)
    ax.text(.012,.045,'Partners',va='center',fontsize=8.4,color=INK)


def replacement_anatomy(fig,ax,d):
    lesion=np.load(ROOT/'results/v8/condition_indices/selection_ranked_100_lesion.npy')
    template=np.load(ROOT/'results/v8/condition_indices/selection_ranked_100_templates.npy')
    panels=[];scales=[]
    for j,(name,header,color) in enumerate(zip(CONDITIONS,CONN_HEADERS,CONN_COLORS)):
        ix=np.array([],dtype=int) if j==0 else np.load(ROOT/f'results/v8/condition_indices/{name}_templates.npy')
        if j>0:assert np.array_equal(ix,template) and len(ix)==100 and np.isin(ix,lesion).all()
        x=CONN_CENTERS[j]
        brain=ax.inset_axes([x-.078,.43,.156,.27]);cloud(brain,d,brain=True,alpha=.43)
        brain.scatter(old.display_x(d.xyz[lesion,0]),d.xyz[lesion,2],s=.65,c='#edb8c5',alpha=.85,lw=0,zorder=3)
        if len(ix):brain.scatter(old.display_x(d.xyz[ix,0]),d.xyz[ix,2],s=1.0,c=color,lw=0,zorder=4)
        zoom=ax.inset_axes([x-.065,.235,.13,.15])
        zx,zz=old.display_x(d.xyz[lesion,0]),d.xyz[lesion,2]
        pad=max(np.ptp(zx),np.ptp(zz))*.08
        zoom.scatter(zx,zz,s=.65,c='#edb8c5',alpha=.72,lw=0)
        if len(ix):zoom.scatter(old.display_x(d.xyz[ix,0]),d.xyz[ix,2],s=1.6,c=color,lw=0)
        zoom.set_xlim(zx.min()-pad,zx.max()+pad);zoom.set_ylim(zz.max()+pad,zz.min()-pad)
        zoom.set_aspect('equal');zoom.set_xticks([]);zoom.set_yticks([])
        for spine in zoom.spines.values():spine.set_visible(True);spine.set_color('#d4c9cf');spine.set_linewidth(.5)
        ax.annotate('',xy=(x,.40),xytext=(x,.44),
                    arrowprops={'arrowstyle':'->','color':GRAY,'lw':.6,'mutation_scale':5})
        panels.append({'condition':name,'lost':len(lesion),'replacements':len(ix),
                       'same_intended_positions':j==0 or np.array_equal(ix,template)})
        scales.append(brain)
    fig.canvas.draw()
    display=[(brain.transData.transform((1,1))-brain.transData.transform((0,0))).tolist() for brain in scales]
    for scale in display[1:]:np.testing.assert_allclose(scale,display[0],rtol=0,atol=1e-9)
    return {'conditions':panels,'matched_brain_scale':True,'display_scales':display}


def exploratory_partner_grid(fig,b,direct):
    intact_m=float(direct[(direct.audit_group=='intact')&direct.seed.isin([0,1,2])].net_displacement_m.mean())
    grid=direct[direct.category=='grid']
    pivot=grid.pivot_table(index='n_graft',columns='specificity',values='net_displacement_m',aggfunc='mean').reindex(index=[500,1000,2000],columns=[0,.25,.5,.75,1])
    cmap=colors.LinearSegmentedColormap.from_list('absolute_movement',['#c53b79','#faf8f5','#07855d'])
    im=b.pcolormesh(np.arange(6)-.5,np.arange(4)-.5,pivot.to_numpy(),cmap=cmap,
                   norm=colors.Normalize(0,max(intact_m,float(pivot.to_numpy().max()))),shading='flat',rasterized=False)
    b.set_xlim(-.5,4.5);b.set_ylim(2.5,-.5)
    for i in range(3):
        for j in range(5):
            value=pivot.iloc[i,j]
            b.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9,
                   color='white' if value/intact_m<.17 or value/intact_m>.82 else INK)
    b.set_xticks(range(5),['0','25','50','75','100']);b.set_yticks(range(3),['500','1,000','2,000'])
    b.set_xlabel('Original partners retained (%)',fontsize=8.5);b.set_ylabel('Replacement cells')
    b.set_title('1,000 neurons lost; n = 3',fontsize=9,pad=4)
    cax=b.inset_axes([1.04,0,.035,1]);cb=fig.colorbar(im,cax=cax,ticks=[0,1,2])
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8,pad=1);cb.set_label('Distance (m)',fontsize=8.5,labelpad=3)
    return {'lesion_size':1000,'realizations_per_point':3,'replacement_counts':pivot.index.astype(int).tolist(),
            'original_partner_probability':pivot.columns.tolist(),'distance_matrix_m':pivot.to_numpy().tolist()}


def interval(values):
    a=np.asarray(values,float);m=a.mean()
    half=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
    return m,half


def dots(ax,data,names,labels,cols,metric,scale=1.,reference=None):
    for y,(name,col) in enumerate(zip(names,cols)):
        values=data.loc[data.name==name,metric].to_numpy()*scale
        m,h=interval(values)
        ax.errorbar(m,y,xerr=h,fmt='o',color=col,ms=5,capsize=3,lw=1.3,zorder=4)
        if len(values)>1:
            offset=np.random.default_rng(0).uniform(-.10,.10,len(values))
            ax.scatter(values,y+offset,s=4,color=col,alpha=.14,lw=0,zorder=2)
    ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.set_ylim(len(names)-.5,-.5)
    if reference is not None:ax.axvline(reference,color=GRAY,ls=':',lw=.9)
    old.clean(ax);ax.grid(axis='x',alpha=.13,lw=.5)


def dose_frame(data,rule):
    q=data[(data.category=='selection')&(data.rule==rule)].copy()
    zero=data[data.name=='lesion20_2000'].copy();zero['dose']=0
    full=data[data.name=='complete_original_2000'].copy();full['dose']=2000
    return pd.concat([zero,q,full])


def typed_dose_frame(data,wiring):
    if wiring=='non_target_types':
        zero=data[data.name=='lesion20_2000'].copy();zero['dose']=0
        return pd.concat([zero,data[data.category=='non_target_dose'].copy()])
    q=data[(data.category=='typed_dose')&(data.wiring==wiring)].copy()
    zero=data[data.name=='lesion20_2000'].copy();zero['dose']=0
    hundred=data[data.name==f'connections_{wiring}_100'].copy()
    return pd.concat([zero,q,hundred])


def dose_curves(ax,data,metric,scale=1,shuffled=False):
    for rule in ['random','interface','ranked']:
        q=dose_frame(data,rule).copy();q[metric]*=scale
        summary(ax,q,metric,RULE_COLORS[rule],RULE_LABELS[rule],x='dose')
    for wiring,col,label in [('correct_types',BLUE,'Correct-type graft'),('non_target_types','#c16888','Non-target graft')]:
        q=typed_dose_frame(data,wiring).copy();q[metric]*=scale
        summary(ax,q,metric,col,label,x='dose',marker='s',ls='--')
    if shuffled:
        q=typed_dose_frame(data,'incorrect_types').copy();q[metric]*=scale
        summary(ax,q,metric,'#b98a3c','Shuffled lost types',x='dose',marker='x',ls=':')


def figure2(d):
    fig = plt.figure(figsize=(7.2, 7.35))
    data = pd.read_csv(ROOT/'results/v8/tables/per_realization.csv')
    a=container(fig,[.04,.705,.94,.28]);letter(fig,.015,.988,'a')
    replacement_schematic(a)
    anatomy_qa=replacement_anatomy(fig,a,d)

    c = fig.add_axes([.10, .43, .345, .215]); letter(fig, .015, .68, 'b')
    dose_curves(c,data,'distance_m')
    intact_m=float(data.loc[data.name=='intact','distance_m'].mean())
    c.axhline(intact_m, color=GRAY, ls=':', lw=.95)
    c.text(.98, intact_m+.05, 'Intact', transform=c.get_yaxis_transform(), ha='right', fontsize=8, color=MUTED)
    upper=3.12
    for wiring in ['correct_types','non_target_types']:
        q=typed_dose_frame(data,wiring)
        for _,group in q.groupby('dose'):
            mean,half=interval(group.distance_m)
            if np.isfinite(half):upper=max(upper,(mean+half)*1.04)
    c.set_xlim(-35, 2035); c.set_xticks([0,500,1000,1500,2000]); c.set_ylim(0,upper)
    if upper<4:c.set_yticks([0,1,2,3])
    c.set_xlabel('Replacement cells'); c.set_ylabel('Distance from start (m)', fontsize=8.5)
    n_min=int(data.groupby('name').size().min())
    c.set_title(f'2,000 neurons lost; n = {n_min}', fontsize=9, pad=4)
    c.legend(frameon=True, edgecolor='none', facecolor='white', framealpha=.94, fontsize=7.0, loc='upper left',
             handlelength=1.25, labelspacing=.1, borderpad=.2)
    old.clean(c)

    dbox = container(fig, [.575, .415, .40, .255]); letter(fig, .535, .68, 'c')
    cases=list(zip(CONDITIONS[1:],CONN_LABELS[1:],CONN_COLORS[1:]))
    arrays, intact, limits = trajectory_arrays(d, [x[0] for x in cases])
    for i, (arr, (_, label, color)) in enumerate(zip(arrays, cases)):
        row, col = divmod(i, 2)
        ax = dbox.inset_axes([.10+col*.51, .55-row*.56, .36, .40])
        path(ax, arr, intact, color, limits, left=(col == 0), bottom=(row == 1))
        untreated=pd.read_csv(ROOT/'results/v8/trajectories/lesion20_2000.csv').to_numpy()
        ax.plot(untreated[:,4],untreated[:,3],color=ORANGE,lw=.75,alpha=.6)
        ax.set_title(label, fontsize=8.2, color=color, pad=1)
        if col == 0:
            ax.set_ylabel('Forward (m)', fontsize=7.8, labelpad=1)
        if row == 1:
            ax.set_xlabel('Lateral (m)', fontsize=7.8, labelpad=1)

    e = fig.add_axes([.10,.115,.345,.205]);letter(fig,.015,.36,'d')
    outcomes={}
    ticklabels=['No\ngraft','Original\nrepair','Random\nwiring','Correct\ntype graft','Non-target\ngraft']
    for j,(name,color) in enumerate(zip(CONDITIONS,CONN_COLORS)):
        vals=100*data.loc[data.name==name,'distance_m_norm'];m,sem=interval(vals)
        outcomes[name]={'mean_percent_intact':float(m),'ci95_half_width':float(sem),'n':len(vals)}
        e.bar(j,m,width=.63,color=color,alpha=.85,zorder=2)
        e.errorbar(j,m,yerr=sem,fmt='none',color=INK,lw=.85,capsize=2,zorder=4)
        e.scatter(j+np.random.default_rng(0).uniform(-.14,.14,len(vals)),vals,s=4,color=INK,alpha=.15,lw=0,zorder=3)
    e.axhline(100,color=GRAY,ls=':',lw=.95);e.set_ylim(0,max(112,float(data[data.name.isin(CONDITIONS)].distance_m_norm.max())*100+3))
    e.set_yticks([0,25,50,75,100]);e.set_xticks(range(5),ticklabels,fontsize=7.3)
    e.set_ylabel('Distance (% intact)');e.set_title('100 selected replacements',fontsize=9,pad=4);old.clean(e)
    f=fig.add_axes([.605,.115,.365,.205]);letter(fig,.535,.36,'e')
    for j,(name,color) in enumerate(zip(CONDITIONS,CONN_COLORS)):
        vals=100*data.loc[data.name==name,'motor_output_norm'];m,sem=interval(vals)
        f.bar(j,m,width=.63,color=color,alpha=.85,zorder=2)
        f.errorbar(j,m,yerr=sem,fmt='none',color=INK,lw=.85,capsize=2,zorder=4)
        visual=100*data.loc[data.name==name,'LPTC_L_L_norm'];vm,vs=interval(visual)
        f.errorbar(j,vm,yerr=vs,fmt='o',mfc='white',mec=INK,color=INK,ms=4,capsize=2,lw=.85,zorder=5)
    f.axhline(100,color=GRAY,ls=':',lw=.95);f.set_ylim(0,123);f.set_yticks([0,25,50,75,100])
    f.set_xticks(range(5),ticklabels,fontsize=7.3);f.set_ylabel('Activity (% intact)')
    f.set_title('100 selected replacements',fontsize=9,pad=4);old.clean(f)
    from matplotlib.patches import Patch
    f.legend(handles=[Patch(facecolor=TEAL,label='Wing motor'),Line2D([0],[0],marker='o',mfc='white',mec=INK,color='none',label='Spared visual')],loc='upper left',ncol=2,fontsize=7.2,frameon=False,handlelength=.9,columnspacing=.8,borderpad=.1)
    e.text(.98,.91,f'n = {len(vals)}',transform=e.transAxes,ha='right',fontsize=7.2,color=MUTED)
    f.text(.98,.91,f'n = {len(vals)}',transform=f.transAxes,ha='right',fontsize=7.2,color=MUTED)
    return fig,{'design':'One integrated panel a aligns five circuit sketches, brain maps and lesion views; original outcome chart types retained in b-e','primary_n':data.groupby('name').size().to_dict(),'panel_a_conditions':CONDITIONS,'panel_a_anatomy':anatomy_qa,'schematic_anatomy_column_centers':CONN_CENTERS.tolist(),'old_panel_b_moved_to':'Supplementary Figure 5e','source_values':outcomes,'dose_maximum':2000,'display':'mean and 95% confidence interval','same_100_cell_groups_in_a_c_d_e':True}


def supplementary2(data):
    fig = plt.figure(figsize=(7.2, 7.0))
    axes = np.array([[fig.add_axes([.10 + j*.305, .64-i*.29, .235, .205]) for j in range(3)] for i in range(2)])
    metrics=[('gain_R','Rightward gain',1),('gain_L','Leftward gain',1),('motor_output_norm','Wing motor activity (% intact)',100),('takeoff_output_norm','Looming motor activity (% intact)',100),('LPTC_L_L_norm','Spared visual activity (% intact)',100),('gates','Virtual gates crossed (0–4)',1)]
    for i,(ax,(metric,label,scale)) in enumerate(zip(axes.flat,metrics)):
        dose_curves(ax,data,metric,scale,shuffled=True)
        ax.set_ylabel(label,fontsize=8.5);ax.set_xlabel('Replacement neurons',fontsize=8.5);ax.set_xticks([0,1000,2000]);old.clean(ax)
        ax.text(-.24,1.06,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=12)
        if metric!='gates':ax.axhline(scale,color=GRAY,ls=':',lw=.8)
        if metric=='LPTC_L_L_norm':ax.set_ylim(0,max(110,ax.get_ylim()[1]))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.6,loc='upper center',bbox_to_anchor=(.53,.995),
               ncol=3,handlelength=1.4,columnspacing=1.1,labelspacing=.3)
    g = fig.add_axes([.10, .075, .82, .19])
    dose_curves(g, data, 'distance_m')
    g.axhline(data.loc[data.name=='intact', 'distance_m'].mean(), color=GRAY, ls=':', lw=.85)
    g.set_xlim(-35,2035);g.set_xticks([0,500,1000,1500,2000])
    g.set_ylabel('Distance from start (m)', fontsize=8.5);g.set_xlabel('Replacement neurons', fontsize=8.5)
    g.set_title('2,000 neurons lost; n = 100', fontsize=9, pad=4)
    g.text(-.069,1.08,'g',transform=g.transAxes,weight='bold',fontsize=12)
    old.clean(g)
    return fig, {'panel_g': 'Retained cell-selection and type-based dose comparison in absolute metres',
                 'source': 'results/v8/tables/per_realization.csv', 'n': 100,
                 'uncertainty': 'Pointwise 95% Student t confidence interval',
                 'random_selection_uses_original_partners': True}


def supplementary4(data):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4));fig.subplots_adjust(left=.22,right=.98,bottom=.1,top=.91,wspace=.7,hspace=.4)
    names=['lesion20_1000','selective_descending','selective_visual','selective_descending_visual','selective_other_37','selective_other_4']
    labels=['Focal lesion (1,000)','Descending (37)','Visual (4)','Both groups (41)','Other cells (37)','Other cells (4)']
    n=int(data[data.name.isin(names)].groupby('name').size().min())
    fig.text(.98,.975,f'n = {n}; 95% CI',ha='right',va='top',fontsize=8,color=MUTED)
    cols=[ORANGE,PURPLE,BLUE,TEAL,GRAY,GRAY]
    metrics=[('distance_m_norm','Distance (% intact)',100),('motor_output_norm','Wing motor activity (% intact)',100),('gain_L','Leftward gain',1),('LPTC_L_L_norm','Spared visual activity (% intact)',100)]
    for i,(ax,(metric,label,scale)) in enumerate(zip(axes.flat,metrics)):
        dots(ax,data,names,labels,cols,metric,scale,scale);ax.set_xlabel(label,fontsize=8.5);ax.tick_params(axis='y',labelsize=8)
        if metric=='LPTC_L_L_norm':ax.set_xlim(0,125);ax.set_xticks([0,50,100])
        ax.text(-.62,1.1,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=12)
    return fig,{}


def supplementary3(data):
    import importlib.util
    spec=importlib.util.spec_from_file_location('legacy_supplement',ROOT/'code/v7/build_supplement_figures.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    def mean_sem(ax,data,x,y,color,marker='o',label=None,individual=False):
        groups=data.groupby(x)[y].agg(['mean','std','count']).sort_index()
        ax.errorbar(groups.index,groups['mean'],yerr=t.ppf(.975,groups['count']-1)*groups['std']/np.sqrt(groups['count']),color=color,marker=marker,ms=5,capsize=3,elinewidth=1,label=label,zorder=5)
        if individual:
            for _,row in data.groupby('seed'):
                row=row.sort_values(x);ax.plot(row[x],row[y],color=color,alpha=.24,lw=.85,zorder=2);ax.scatter(row[x],row[y],s=12,color=color,alpha=.45,zorder=3)
    legacy.mean_error=mean_sem;legacy.save=lambda fig,stem:(fig,{'sample_size':3,'display':'95% confidence intervals; individual seeds retained'})
    fig,qa=legacy.supplement3(pd.read_csv(ROOT/'results/v5/revision_20261002_clarity/all_direct_readouts.csv'))
    for ax in fig.axes:
        low,high=ax.dataLim.intervaly
        if np.isfinite([low,high]).all():
            margin=max(.03,(high-low)*.06)
            ax.set_ylim(min(0,low-margin),max(ax.get_ylim()[1],high+margin))
        ax.text(.98,1.02,'n = 3',transform=ax.transAxes,ha='right',fontsize=8,color=MUTED)
    return fig,qa


def supplementary1(data):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.0));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.90,wspace=.4,hspace=.60)
    a,b,c,d=axes.flat
    q=data[data.name=='intact']
    a.hist(q.distance_m,bins=15,color=BLUE,alpha=.75,edgecolor='white');a.set_xlabel('Intact distance from start (m)');a.set_ylabel('Simulation realizations')
    intact=q.set_index('rep');full=data[data.name=='complete_original_2000'].set_index('rep');common=full.index.intersection(intact.index)
    for k,col in [('y_R',ORANGE),('y_L',BLUE),('y_F',TEAL)]:b.scatter(intact.loc[common,k],full.loc[common,k],s=8,c=col,alpha=.6,label=k[-1])
    lo,hi=b.get_xlim();b.plot([lo,hi],[lo,hi],color=GRAY,ls=':',lw=.8);b.set_xlabel('Intact steering command');b.set_ylabel('Complete original replacement');b.legend(frameon=False,ncol=3,fontsize=8)
    olddata=pd.read_csv(ROOT/'results/v5/tables/sensitivity_matched_intact.csv')
    for ax,param,metric,yl in [(c,'weight_factor','steering_similarity','Directional-response similarity'),(d,'stim_rate','straightness','Path straightness')]:
        for (site,n),g in olddata[olddata.parameter==param].groupby(['site','n_lesion']):
            g=g.sort_values('value');ax.plot(g.value,g[metric],marker='o',ms=3,color=ORANGE if site==20 else BLUE,ls='-' if n==1000 else '--',label=f'Site {int(site)}, {int(n)//1000}k')
        ax.set_xlabel('Synaptic-weight factor' if param=='weight_factor' else 'Sensory-input rate (Hz)');ax.set_ylabel(yl)
    handles,labels=c.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.0,ncol=4,loc='lower center',bbox_to_anchor=(.54,.035),
               handlelength=1.5,columnspacing=1.1)
    for i,ax in enumerate(axes.flat):old.clean(ax);ax.text(-.23,1.1,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=12)
    return fig,{}


def supplementary5(data):
    source=pd.read_csv(ROOT/'results/v5/revision_20261002_clarity/all_direct_readouts.csv')
    fig=plt.figure(figsize=(7.2,6.5))
    axes=[fig.add_axes([x,y,.255,.19]) for y in [.73,.42] for x in [.23,.725]]
    a,b,c,d=axes
    names=['size_site20_n2000','clarity_full2000_random_partners','clarity_full2000_wrong_types','typed_full2000','partial_full']
    labels=['No graft','Random wiring','Shuffled lost types','Correct-type graft','Original repair']
    dots(a,source,names,labels,[ORANGE,'#50545b','#b98a3c',BLUE,TEAL],'net_displacement_m_paired_norm',100,100)
    a.set_xlabel('Distance (% intact)',fontsize=8.5);a.set_title('2,000-cell replacement; n = 3',fontsize=9)
    dots(b,data,CONDITIONS,CONN_LABELS,CONN_COLORS,'takeoff_output_norm',100,100)
    b.set_xlabel('Looming motor activity (% intact)',fontsize=8.5);b.set_title('100 selected cells; n = 100',fontsize=9)
    dots(c,data,CONDITIONS,CONN_LABELS,CONN_COLORS,'gates')
    c.set_xlabel('Virtual gates crossed (0–4)',fontsize=8.5);c.set_xlim(-.1,4.1)
    for j,name in enumerate(CONDITIONS):
        for shift,metric,col,marker in [(-.10,'gain_R',BLUE,'o'),(.10,'gain_L',ORANGE,'s')]:
            m,se=interval(data.loc[data.name==name,metric]);d.errorbar(m,j+shift,xerr=se,fmt=marker,color=col,ms=4,capsize=2,lw=.9)
    d.set_yticks(range(5),CONN_LABELS);d.set_ylim(4.5,-.5);d.axvline(1,color=GRAY,ls=':',lw=.8)
    d.set_xlabel('Directional gain',fontsize=8.5);old.clean(d)
    d.legend(handles=[Line2D([0],[0],marker='o',color=BLUE,label='Right'),Line2D([0],[0],marker='s',color=ORANGE,label='Left')],frameon=False,fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.23))
    for ax,letter_ in zip(axes,'abcd'):
        ax.tick_params(axis='y',labelsize=8);pos=ax.get_position();letter(fig,pos.x0-.215,pos.y1+.075,letter_)
    heatmap=fig.add_axes([.23,.075,.61,.205]);letter(fig,.015,.34,'e')
    grid_qa=exploratory_partner_grid(fig,heatmap,source)
    return fig,{'panel_a':'Original complete-dose controls, n=3','panels_b_d':'New matched 100-cell connection comparisons, n=100','panel_e_exploratory_grid':grid_qa,'display':'95% confidence intervals'}


def supplementary6(data):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
    fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.90,wspace=.43,hspace=.48)
    metrics=[('distance_m','Distance from start (m)',ORANGE),('beam_crossings_per_s','Virtual crossings per second',PURPLE),
             ('takeoff_rate_hz','Looming-evoked motor rate (Hz)',BLUE),('LPTC_L_L','Spared visual response (Hz)',INK)]
    q=data[data.category=='lesion_size']
    for i,(ax,(metric,label,color)) in enumerate(zip(axes.flat,metrics)):
        summary(ax,q,metric,color,'Lesion')
        intact=data.loc[data.name=='intact',metric];m,h=interval(intact)
        ax.axhline(m,color=GRAY,ls=':',lw=1,label='Intact mean')
        ax.axhspan(m-h,m+h,color=GRAY,alpha=.12,lw=0)
        ax.set_xscale('log',base=2);ax.set_xticks([500,2000,8000,64000],['500','2k','8k','64k'])
        ax.set_xlabel('Neurons lost');ax.set_ylabel(label,fontsize=8.5);old.clean(ax)
        ax.text(-.25,1.12,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax.text(.97,1.035,f'n = {q.groupby("name").size().min()}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
        ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
        low,high=ax.get_ylim();ax.set_ylim(low,high*1.06)
    axes[0,0].legend(frameon=False,fontsize=8)
    return fig,{'display':'Means and 95% confidence intervals in absolute units','motor_rate':'Mean of the TTMn and DLMn population rates; no locomotion onset is simulated'}


def main():
    import shutil
    OUT.mkdir(exist_ok=True);style();d=old.Data();data=pd.read_csv(ROOT/'results/v8/tables/per_realization.csv')
    report={'status':'PENDING_VISUAL_QA','analysis_status':json.loads((ROOT/'results/v8/tables/analysis_qa.json').read_text())['status'],'figures':{},'output_sha256':{}}
    builders=[('Fig1_v8_focal_loss',lambda:figure1(d)),('Fig2_v8_cell_replacement',lambda:figure2(d)),('SuppFig1_v8_model_validation',lambda:supplementary1(data)),('SuppFig2_v8_replacement_readouts',lambda:supplementary2(data)),('SuppFig3_v8_replacement_robustness',lambda:supplementary3(data)),('SuppFig4_v8_selective_ablation',lambda:supplementary4(data)),('SuppFig5_v8_exploratory_controls',lambda:supplementary5(data))]
    builders.append(('SuppFig6_v8_behavioral_readouts',lambda:supplementary6(data)))
    for stem,build in builders:
        fig,qa=build();report['figures'][stem]=qa
        if report['analysis_status'] != 'COMPLETE':
            fig.text(.5,.007,'Layout preview: simulation batch incomplete',ha='center',fontsize=7,color='#a64262')
        for ext in ['png','pdf','svg']:
            path=OUT/f'{stem}.{ext}';fig.savefig(path,dpi=300,facecolor='white');report['output_sha256'][str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig);print(stem,flush=True)
    inputs=[Path(__file__),ROOT/'code/v5/build_figures.py',ROOT/'code/v7/build_supplement_figures.py',
            ROOT/'data/v5/neurons.parquet',ROOT/'data/v5/cell_groups.json',ROOT/'results/v5/lesion_sites.csv',
            ROOT/'results/v5/tables/sites_summary.csv',ROOT/'results/v5/tables/sensitivity_matched_intact.csv',
            ROOT/'results/v8/tables/per_realization.csv',ROOT/'results/v8/tables/summary.csv',
            ROOT/'results/v5/revision_20261002/all_sites_mean_readouts.csv',ROOT/'results/v5/revision_20261002_clarity/all_direct_readouts.csv']
    inputs+=sorted((ROOT/'results/v8/condition_indices').glob('*.npy'))
    inputs+=sorted((ROOT/'results/v8/trajectories').glob('*.csv'))
    report['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    (OUT/'figure_qa.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
