"""Build V5 main and supplementary figures from corrected source tables.

Use --preview while results are incomplete. Preview files live in a separate
directory and include an explicit label. --final requires the complete state
protocol, checked adaptation coverage, and current table provenance.
All plotting primitives remain vector objects in PDF and SVG outputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import flight_fast as flight

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/v5'
TAB = RES / 'tables'
OUT = ROOT / 'figures/v5'
BLUE = '#2465a6'
ORANGE = '#ce682b'
TEAL = '#138479'
PURPLE = '#8060a6'
RED = '#b34755'
INK = '#242d35'
MUTED = '#67727b'
GRAY = '#acb7bf'
PALE = '#e3e8ec'
SIZE_COLORS = {1000: BLUE, 2000: TEAL, 4000: ORANGE, 8000: PURPLE}
RULE_COLORS = {'random': '#50545b', 'interface': '#7250ae', 'ranked': '#008b78'}
LABELS = {'random': 'Random selection', 'interface': 'Projection cells first',
          'ranked': 'Visual + descending first'}


def display_x(x):
    """Match the movies: the animal's right appears on the page right."""
    return 779.6-np.asarray(x)


def style():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7.0, 'axes.titlesize': 7.6,
        'axes.labelsize': 7.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.2, 'axes.linewidth': .65, 'lines.linewidth': 1.2,
        'axes.edgecolor': '#78858e', 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_table(name):
    path = TAB / (name + '.csv')
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean(ax, grid=True):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(pad=2)
    if grid:
        ax.grid(axis='y', color='#e7ecef', linewidth=.55, zorder=0)
        ax.set_axisbelow(True)


def panel(ax, letter, title):
    bounds = ax.get_position()
    ax.figure.text(bounds.x0-.027, bounds.y1+9/72/ax.figure.get_figheight(), letter,
                   fontsize=11, fontweight='bold', color=INK, va='baseline')
    ax.set_title(title, loc='left', pad=9, fontweight='normal')


def container(fig, rect, letter, title):
    ax = fig.add_axes(rect)
    ax.axis('off')
    panel(ax, letter, title)
    return ax


def missing(ax, label='Awaiting corrected results'):
    ax.text(.5, .48, label, ha='center', va='center', transform=ax.transAxes,
            color=MUTED, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['left', 'bottom']].set_visible(False)


def dots(ax, xs, values, color, width=.11, alpha=.48, size=11):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        offsets = np.linspace(-width, width, len(vals)) if len(vals) > 1 else np.zeros(1)
        ax.scatter(np.repeat(xs, len(vals)) + offsets, vals, s=size, color=color,
                   alpha=alpha, edgecolors='none', zorder=4)


class Data:
    def __init__(self):
        self.states = load_table('states')
        self.intact = load_table('intact')
        self.sites = load_table('sites_summary')
        self.grafts = load_table('grafts_summary')
        self.adaptation = load_table('adaptation')
        self.contrasts = load_table('adaptation_treatment_contrasts')
        self.frozen_contrasts = load_table('adaptation_frozen_contrasts')
        self.matched = load_table('adaptation_matched_followup')
        self.sensitivity = load_table('sensitivity_matched_intact')
        for df in [self.states, self.intact]:
            if len(df) and 'response_R' not in df:
                df['response_R'] = df.y_R - df.y_F
                df['response_L'] = df.y_F - df.y_L
        self.neurons = pd.read_parquet(ROOT / 'data/v5/neurons.parquet')
        self.groups = json.loads((ROOT / 'data/v5/cell_groups.json').read_text())
        self.centres = pd.read_csv(RES / 'lesion_sites.csv').set_index('site')
        self.xyz = self.neurons[['x_um', 'y_um', 'z_um']].to_numpy()
        has = np.isfinite(self.xyz).all(axis=1)
        eligible = has & ~self.neurons.superclass.isin(['vnc_motor', 'cb_motor']).to_numpy()
        self.pool = np.flatnonzero(eligible)
        self.tree = cKDTree(self.xyz[self.pool])
        valid = np.flatnonzero(has)
        rng = np.random.default_rng(194)
        self.background = np.sort(rng.choice(valid, min(12000, len(valid)), replace=False))
        self.brain_background = self.background[self.xyz[self.background, 2] < 460]
        self.movie_seed = json.loads((RES / 'protocol.json').read_text()).get('movie_gust_seed', 3)

    def lesion(self, site, n):
        c = self.centres.loc[site]
        _, ix = self.tree.query([c.x, c.y, c.z], k=int(n))
        return self.pool[np.atleast_1d(ix)]

    def select(self, name, seed=None):
        if self.states.empty:
            return pd.DataFrame()
        q = self.states[self.states.name == name]
        return q[q.seed == seed] if seed is not None else q

    def untreated(self):
        if self.adaptation.empty:
            return pd.DataFrame()
        q = self.adaptation[self.adaptation.condition == 'untreated'].copy()
        if 'frozen' in q:
            q = q[~q.frozen.fillna(False).astype(bool)]
        return q


def cloud(ax, d, brain=False, alpha=.6):
    ix = d.brain_background if brain else d.background
    ax.scatter(display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=.45, color='#c9d2d9', alpha=alpha,
               linewidths=0, rasterized=False)
    ax.set_xlim(-15, 790)
    ax.set_ylim(470 if brain else 1100, 70)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def cmap_plot(ax, d, metric, size, norm):
    cloud(ax, d)
    q = d.sites[d.sites.n_lesion == size] if not d.sites.empty else pd.DataFrame()
    if len(q):
        ax.scatter(display_x(q.x), q.z, c=q[metric], norm=norm, cmap='viridis', s=26,
                   edgecolors='white', linewidths=.55, zorder=4)
        for site, marker in [(20, 'o'), (25, 's')]:
            r = d.centres.loc[site]
            ax.scatter(display_x([r.x]), [r.z], facecolors='none', edgecolors=INK,
                       s=53, linewidths=.85, marker=marker, zorder=5)
        count_label = '56 fixed locations' if q.site.nunique()==56 else f'{q.site.nunique()}/56 locations (incomplete)'
        ax.text(.5, -.045, count_label, transform=ax.transAxes,
                ha='center', fontsize=5.7, color=MUTED)
    else:
        ax.text(.5, .42, 'Results pending', transform=ax.transAxes,
                ha='center', fontsize=6.5, color=MUTED)
    ax.set_title(f'{size:,} neurons lost', fontsize=7.3, pad=3)


def trajectory(ax, row, d, color, intact=True, title=None, label_axes=False, limits=None):
    paths = []
    if intact and len(d.intact):
        r = d.intact[d.intact.seed == 0]
        r = r.iloc[0] if len(r) else d.intact.iloc[0]
        paths.append((flight.simulate(r.gain_R, r.gain_L, r.bias, seed=d.movie_seed,
                                     drive=r.drive), GRAY, .95, '--'))
    if row is not None:
        paths.append((flight.simulate(row.gain_R, row.gain_L, row.bias,
                                     seed=d.movie_seed, drive=row.drive), color, 1.35, '-'))
    for arr, c, lw, ls in paths:
        ax.plot(arr[:, 4], arr[:, 3], color=c, lw=lw, ls=ls)
        ax.scatter(arr[-1, 4], arr[-1, 3], s=12, color=c, zorder=5)
    ax.scatter([0], [0], marker='o', s=8, color=INK, zorder=5)
    ax.set_aspect('equal', adjustable='box')
    clean(ax, False)
    if limits is not None:
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    if label_axes:
        ax.set_xlabel('Lateral displacement (m)')
        ax.set_ylabel('Forward displacement (m)')
    else:
        ax.tick_params(labelsize=5.7)
    if title:
        ax.set_title(title, fontsize=6.8, pad=3)


def line_summary(ax, q, metric, color, label, x='n_lesion', linestyle='-', marker='o'):
    if q.empty:
        return
    g = q.groupby(x)[metric].agg(['mean', 'std']).sort_index()
    xx = g.index.to_numpy(float)
    ax.plot(xx, g['mean'], color=color, marker=marker, ms=3,
            ls=linestyle, label=label)
    ax.errorbar(xx, g['mean'], yerr=g['std'].fillna(0), fmt='none',
                color=color, lw=.75, capsize=1.7)


def site_size_rows(d):
    if d.states.empty:
        return pd.DataFrame()
    return d.states[(d.states.site == 20) & (d.states.n_graft == 0) &
                    d.states.category.isin(['screen', 'size'])]


def neural_size_axis(ax, d, which):
    q = site_size_rows(d)
    if q.empty:
        missing(ax); return
    if which == 'steering':
        line_summary(ax, q, 'response_R', BLUE, 'Rightward response')
        line_summary(ax, q, 'response_L', ORANGE, 'Leftward response', linestyle='--', marker='s')
        if len(d.intact):
            ax.axhline(d.intact.response_R.mean(), color=BLUE, ls=':', alpha=.55, lw=.8)
            ax.axhline(d.intact.response_L.mean(), color=ORANGE, ls=':', alpha=.55, lw=.8)
        ax.set_ylabel('Decoded steering response (Hz)')
        ax.axhline(0, color='#a9b3ba', lw=.65)
        ax.text(.98, .03, 'Dotted: intact reference', transform=ax.transAxes,
                ha='right', fontsize=5.6, color=MUTED)
    else:
        line_summary(ax, q, 'motor_output', TEAL, 'Wing-power output')
        line_summary(ax, q, 'evoked_activity', PURPLE, 'Evoked activity', linestyle='--', marker='s')
        ax.set_ylabel('Response / intact')
        ax.set_ylim(0, max(1.08, float(q[['motor_output','evoked_activity']].max().max())*1.08))
    if which != 'steering':
        ax.axhline(1, color=GRAY, ls=':', lw=.9, zorder=0)
    ax.set_xscale('log', base=2)
    ax.set_xticks([500, 2000, 8000, 32000], ['500', '2k', '8k', '32k'])
    ax.set_xlabel('Neurons lost at site 20')
    ax.legend(frameon=False, loc='best', handlelength=1.4)
    clean(ax)


def add_preview(fig, preview, d):
    if preview:
        fig.text(.5, .992, 'INCOMPLETE PREVIEW | corrected results available so far',
                 ha='center', va='top', fontsize=8, color=RED, fontweight='bold')
    n = len(d.states)
    footer = (f'Model outputs | points/lines show simulation seeds | neural states available: {n}' if preview
              else 'Points and thin lines show stochastic simulation repeats; normalized neural reference = 1')
    fig.text(.5, .007, footer,
             ha='center', va='bottom', fontsize=5.8, color=MUTED)


def figure1(d, preview):
    fig = plt.figure(figsize=(8.1, 7.55))
    fig.text(.05, .964, 'Focal neuronal loss and movement impairment', fontsize=11.5, fontweight='bold')
    # a: anatomical context and the task, without unsupported brain artwork.
    a = container(fig, [.065, .686, .265, .235], 'a', 'From visual motion to steering')
    anatomy = a.inset_axes([0, .03, .52, .9])
    cloud(anatomy, d)
    for key, col, size in [('LPTC', BLUE, 2.7), ('DN', PURPLE, 1.0), ('steerMN', ORANGE, 4.3)]:
        ix = np.array(d.groups[key + '_L'] + d.groups[key + '_R'])
        ix = ix[np.isfinite(d.xyz[ix, 0])]
        anatomy.scatter(display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=size, c=col, linewidths=0)
    anatomy.text(395, 73, 'Brain + optic lobes', ha='center', fontsize=5.6, color=MUTED)
    anatomy.text(390, 1120, 'Ventral nerve cord', ha='center', fontsize=5.6, color=MUTED)
    anatomy.text(15, 485, 'L', fontsize=5.2, color=MUTED)
    anatomy.text(725, 485, 'R', fontsize=5.2, color=MUTED)
    steps = [('Visual motion', INK), ('Motion detectors', BLUE), ('Visual integration', BLUE),
             ('Descending neurons', PURPLE), ('Wing motor output', ORANGE), ('Steering controller', INK)]
    for i, (text, col) in enumerate(steps):
        y = .92 - i * .158
        a.text(.77, y, text, transform=a.transAxes, ha='center', va='center',
               fontsize=5.9, color=col)
        if i < len(steps) - 1:
            a.annotate('', xy=(.77, y - .116), xytext=(.77, y - .038), xycoords='axes fraction',
                       arrowprops={'arrowstyle': '-|>', 'lw': .6, 'color': GRAY, 'mutation_scale': 6})
    a.text(.75, -.035, 'Connectivity to activity\nto simulated movement', ha='center',
           transform=a.transAxes, fontsize=5.7, color=MUTED)
    # b: same fixed colour scale for all sizes and all sites.
    b = container(fig, [.395, .686, .57, .235], 'b', 'Location and size of neuronal loss')
    norm = colors.Normalize(0, 1)
    for j, size in enumerate([1000, 4000, 16000]):
        ax = b.inset_axes([j / 3, .015, .29, .90])
        cmap_plot(ax, d, 'straightness', size, norm)
    cax = b.inset_axes([.2, -.08, .54, .035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax,
                      orientation='horizontal', ticks=[0, .5, 1])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.7, pad=1)
    cb.set_label('Simulated path straightness', fontsize=6, labelpad=1)
    # c: anatomical intervention and its paired trajectory.
    c = container(fig, [.07, .399, .255, .19], 'c', 'A local lesion and its movement output')
    c_paths=[]
    for name in ['map_site20_n1000','map_site25_n1000']:
        rows=d.select(name,0)
        if len(rows):
            r=rows.iloc[0];c_paths.append(flight.simulate(r.gain_R,r.gain_L,r.bias,seed=d.movie_seed,drive=r.drive))
    if len(d.intact):
        rows=d.intact[d.intact.seed==0];r=rows.iloc[0] if len(rows) else d.intact.iloc[0]
        c_paths.append(flight.simulate(r.gain_R,r.gain_L,r.bias,seed=d.movie_seed,drive=r.drive))
    c_limits=None
    if c_paths:
        xy=np.concatenate([p[:,3:5] for p in c_paths])
        c_limits=((min(-.25,xy[:,1].min())-.1,max(.25,xy[:,1].max())+.1),
                  (min(0,xy[:,0].min())-.1,max(.5,xy[:,0].max())+.1))
    for j, (site, col) in enumerate([(20, ORANGE), (25, BLUE)]):
        ax = c.inset_axes([j * .52, .63, .47, .36])
        cloud(ax, d, brain=True)
        ix = d.lesion(site, 1000)
        ax.scatter(display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=.6, c=col, linewidths=0)
        ax.set_title(f'Site {site}', fontsize=6.5, pad=1)
        path = c.inset_axes([j * .52 + .035, -.01, .41, .60])
        q = d.select(f'map_site{site}_n1000', seed=0)
        if len(q):
            trajectory(path, q.iloc[0], d, col, title=None, limits=c_limits)
        else:
            missing(path)
    c.text(.5, -.26, '1,000 neurons lost; identical gusts\nDashed: intact reference; seed 0',
           ha='center', transform=c.transAxes, fontsize=5.8, color=MUTED)
    # d/e: actual nested anatomical loss paired with interpretable readouts.
    revised = RES/'revision_20261002'
    qsize = pd.read_csv(revised/'site20_size_readout_by_seed.csv')
    nested = container(fig, [.395, .367, .20, .224], 'd', 'An expanding focal lesion')
    anat = nested.inset_axes([0, .15, 1, .91]); cloud(anat, d)
    nested_sizes = [(64000,'#733047'), (16000,'#c75c68'), (4000,'#efa260'), (500,'#259c84')]
    for n,col in nested_sizes:
        ix = d.lesion(20,n)
        anat.scatter(display_x(d.xyz[ix,0]), d.xyz[ix,2], s=.5, c=col, linewidths=0, zorder=3)
    for j,(n,col) in enumerate(reversed(nested_sizes)):
        nested.scatter(.035,.105-j*.078,s=9,color=col,transform=nested.transAxes,clip_on=False)
        nested.text(.10,.105-j*.078,f'{n:,} cells ({100*n/len(d.neurons):.1f}% CNS)',
                    transform=nested.transAxes,va='center',fontsize=5.7,color=INK)
    ax = fig.add_axes([.695, .399, .27, .19]); panel(ax, 'e', 'Loss of movement and activity')
    for metric,col,label,mk in [
        ('net_displacement_m_paired_norm',ORANGE,'Distance from start','o'),
        ('straightness_paired_norm',BLUE,'Path straightness','s'),
        ('drive_paired_norm',TEAL,'Motor drive','^'),
        ('evoked_activity_paired_norm',PURPLE,'Evoked activity','D')]:
        line_summary(ax,qsize,metric,col,label,marker=mk)
    ax.set_xscale('log',base=2);ax.set_xticks([500,2000,8000,32000,64000],['500','2k','8k','32k','64k'])
    ax.set_ylim(-.035,1.3);ax.axhline(1,color=GRAY,ls=':',lw=.8)
    ax.set_xlabel('Neurons lost at site 20');ax.set_ylabel('Readout / matched intact')
    ax.legend(frameon=False,fontsize=5.5,handlelength=1.2,loc='upper right',labelspacing=.25)
    clean(ax)
    # f: paired fixed locations show the lesion-size effect across the screen.
    ax=fig.add_axes([.12,.10,.80,.165]);panel(ax,'f','Larger lesions reduce movement across locations')
    screen=pd.read_csv(revised/'all_sites_mean_readouts.csv')
    wide=screen.pivot(index='site',columns='n_lesion',values='net_displacement_m')[[1000,4000,16000]]
    for site,row in wide.iterrows():
        ax.plot([0,1,2],row.values,color='#bcc6ce',alpha=.45,lw=.55,zorder=1)
    for j,n,col in [(0,1000,BLUE),(1,4000,ORANGE),(2,16000,PURPLE)]:
        yy=wide[n].to_numpy();xx=j+np.linspace(-.055,.055,len(yy))
        ax.scatter(xx,yy,s=7,color=col,alpha=.50,linewidths=0,zorder=2)
    mean=wide.mean();ax.plot([0,1,2],mean.values,color=INK,lw=1.5,marker='o',ms=3.5,label='Mean across locations',zorder=4)
    ax.set_xticks([0,1,2],['1,000','4,000','16,000']);ax.set_xlim(-.15,2.15);ax.set_ylim(0,3.0)
    ax.set_xlabel('Neurons lost');ax.set_ylabel('Distance from start (m / 10 s)')
    ax.text(.99,.95,'56 fixed locations; each pale line is one location\nEach location averages three neural seeds and 20 gusts',
            transform=ax.transAxes,ha='right',va='top',fontsize=5.8,color=MUTED)
    clean(ax)
    add_preview(fig, preview, d)
    if not preview:
        fig.texts[-1].set_text('Fixed-location screen: 56 sites, three neural seeds per condition. Panel e uses matched intact normalization.')
    return fig


def templates(d, rule, n, seed=0):
    lesion = d.lesion(20, 2000)
    if rule == 'random':
        return np.random.default_rng(seed).choice(lesion, n, replace=False)
    rng = np.random.default_rng(1000 + seed)
    sc = d.neurons.superclass.to_numpy()[lesion]
    lptc = np.isin(lesion, d.groups['LPTC_L'] + d.groups['LPTC_R'])
    first = lptc | (sc == 'descending_neuron')
    second = ~first & np.isin(sc, ['visual_projection', 'visual_centrifugal'])
    return np.concatenate([rng.permutation(lesion[first]), rng.permutation(lesion[second]),
                           rng.permutation(lesion[~first & ~second])])[:n]


def treatment_rows(d):
    q = (d.matched if len(d.matched) else d.contrasts).copy()
    if q.empty:
        return q
    # Frozen-host branches are sensitivity controls, never additional repeats
    # of the adapting-host treatment arm.
    q = q[q.condition.isin(['acute_100','acute_2000','delayed_100','delayed_2000'])].copy()
    q = q[pd.to_numeric(q.graft_n, errors='coerce').fillna(0) > 0].copy()
    if q.empty:
        return q
    # A branch file begins at its intervention round. Acute graft files begin at 0.
    onset = q.groupby('experiment')['round'].transform('min')
    explicit = pd.to_numeric(q.get('intervention_round',q.get('branch_round', pd.Series(np.nan, index=q.index))), errors='coerce')
    q['onset'] = explicit.fillna(onset).astype(int)
    q['since_graft'] = q['round'] - q.onset
    q['timing'] = np.where(q.onset == 0, 'Acute', 'After adaptation')
    return q


def figure2(d, preview):
    fig = plt.figure(figsize=(8.1, 8.15))
    fig.text(.05, .964, 'Which cells and connections restore movement?', fontsize=11.5, fontweight='bold')
    a = container(fig, [.075, .708, .37, .19], 'a', 'The same lesion, different replacement cells')
    lesion = np.load(RES/'condition_indices/partial_random_p5_s0_lesion.npy')
    for j, (rule, title) in enumerate([('random', 'Random'),
                                      ('interface', 'Projection first'),
                                      ('ranked', 'Visual + descending')]):
        ax = a.inset_axes([j*.345, .60, .31, .35]); cloud(ax, d, brain=True, alpha=.42)
        actual_lesion = np.load(RES/f'condition_indices/partial_{rule}_p5_s0_lesion.npy')
        ix = np.load(RES/f'condition_indices/partial_{rule}_p5_s0_templates.npy')
        if not np.array_equal(lesion, actual_lesion) or len(ix) != 100 or not np.isin(ix, lesion).all():
            raise ValueError('Figure 2a requires the same lesion and 100 actual replacement templates')
        ax.scatter(display_x(d.xyz[lesion, 0]), d.xyz[lesion, 2], s=1.05,
                   c='#edb8c5', alpha=.8, linewidths=0, zorder=3)
        ax.scatter(display_x(d.xyz[ix, 0]), d.xyz[ix, 2], s=.9,
                   c=RULE_COLORS[rule], edgecolors='white', linewidths=.06, zorder=4)
        ax.set_title(title, fontsize=6.0, pad=2, color=RULE_COLORS[rule])
        zoom = a.inset_axes([j*.345+.025, .255, .26, .30])
        zx=display_x(d.xyz[lesion,0]);zz=d.xyz[lesion,2]
        zx0,zx1=np.min(zx),np.max(zx);zz0,zz1=np.min(zz),np.max(zz)
        pad=max(zx1-zx0,zz1-zz0)*.08
        zoom.scatter(zx,zz,s=.58,c='#edb8c5',alpha=.68,linewidths=0)
        zoom.scatter(display_x(d.xyz[ix,0]),d.xyz[ix,2],s=1.35,c=RULE_COLORS[rule],linewidths=0)
        zoom.set_xlim(zx0-pad,zx1+pad);zoom.set_ylim(zz1+pad,zz0-pad)
        zoom.set_aspect('equal');zoom.set_xticks([]);zoom.set_yticks([])
        for spine in zoom.spines.values():spine.set_visible(True);spine.set_color('#d4c9cf');spine.set_linewidth(.5)
        a.annotate('',xy=(j*.345+.155,.57),xytext=(j*.345+.155,.64),xycoords='axes fraction',
                   arrowprops={'arrowstyle':'->','color':GRAY,'lw':.55,'mutation_scale':5})
    a.text(.5,.205,'Magnified lesion and selected cells',transform=a.transAxes,ha='center',fontsize=5.5,color=MUTED)
    a.text(.5, .05, 'Same 2,000-neuron lesion; 100 cells replaced (5%)\nPink: full lesion footprint; color: replacement templates',
           transform=a.transAxes, ha='center', va='center', fontsize=6.4, color=MUTED)
    direct=pd.read_csv(RES/'revision_20261002_clarity/all_direct_readouts.csv')
    # b: measured displacement in metres, not an arbitrary composite benefit.
    ax=fig.add_axes([.575,.708,.345,.19]);panel(ax,'b','Cell number and accuracy of connections')
    grid=direct[direct.category=='grid']
    p=grid.pivot_table(index='n_graft',columns='specificity',values='net_displacement_m',aggfunc='mean').reindex(index=[500,1000,2000],columns=[0,.25,.5,.75,1])
    intact=direct[(direct.audit_group=='intact')&direct.seed.isin([0,1,2])]
    intact_m=float(intact.net_displacement_m.mean())
    cmap=colors.LinearSegmentedColormap.from_list('absolute_movement',['#c53b79','#faf8f5','#07855d'])
    im=ax.pcolormesh(np.arange(6)-.5,np.arange(4)-.5,p.to_numpy(),cmap=cmap,
                     norm=colors.Normalize(0,max(intact_m,float(p.to_numpy().max()))),shading='flat',rasterized=False)
    ax.set_xlim(-.5,4.5);ax.set_ylim(2.5,-.5)
    for i in range(3):
        for j in range(5):
            v=p.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.5,
                color='white' if v/intact_m<.17 or v/intact_m>.82 else INK)
    ax.set_xticks(range(5),['0','25','50','75','100']);ax.set_yticks(range(3),['500','1,000','2,000'])
    ax.set_xlabel('Original partners retained (%)');ax.set_ylabel('Replacement cells')
    cax=ax.inset_axes([1.04,.12,.035,.77]);cb=fig.colorbar(im,cax=cax);cb.outline.set_visible(False)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.set_label('Distance from start (m)',fontsize=6)
    cb.ax.tick_params(labelsize=5.6)
    untreated=direct[direct.name=='map_site20_n1000'].net_displacement_m.mean()
    ax.text(.5,-.28,f'1,000 cells lost; no replacement: {untreated:.2f} m; intact: {intact_m:.2f} m',
            transform=ax.transAxes,ha='center',fontsize=5.6,color=MUTED)
    # c: random selection of lost templates is separate from random connectivity.
    ax=fig.add_axes([.075,.405,.37,.19]);panel(ax,'c','Which lost cells are restored?')
    q=direct[direct.category=='partial'];no=direct[direct.name=='size_site20_n2000']
    for rule in ['random','interface','ranked']:
        z=q[q.rule==rule].copy();z=pd.concat([z,no.assign(n_graft=0)],ignore_index=True)
        line_summary(ax,z,'net_displacement_m',RULE_COLORS[rule],LABELS[rule],x='n_graft')
    ax.axhline(intact_m,color=GRAY,ls=':',lw=.8,label='Matched intact')
    ax.set_xlim(-35,1035);ax.set_xticks([0,100,500,1000]);ax.set_ylim(0,2.85)
    ax.set_xlabel('Replacement cells (of 2,000 lost)');ax.set_ylabel('Distance from start (m / 10 s)')
    ax.legend(frameon=False,fontsize=5.3,loc='lower left',bbox_to_anchor=(0,1.20),ncol=2,handlelength=1.2)
    ax.text(.02,-.32,'Every selected cell retains its correct identity and original partners.\nFull replacement is compared separately in d.',
            transform=ax.transAxes,ha='left',va='top',fontsize=5.5,color=MUTED)
    clean(ax)
    # d: full cell-count replacement is compared with exact reconstruction.
    ax=fig.add_axes([.575,.405,.37,.19]);panel(ax,'d','Replacing every lost cell is not sufficient')
    fullcases=[('size_site20_n2000','No\nreplacement',ORANGE),
               ('clarity_full2000_random_partners','Random\npartners',RULE_COLORS['random']),
               ('clarity_full2000_wrong_types','Incorrect\ncell types',RED),
               ('typed_full2000','Correct\ncell types',TEAL),
               ('partial_full','Exact\nreconstruction',BLUE)]
    for j,(name,label,col) in enumerate(fullcases):
        z=direct[direct.name==name];vals=100*z.net_displacement_m_paired_norm
        ax.bar(j,vals.mean(),width=.61,color=col,alpha=.80,zorder=2)
        ax.errorbar(j,vals.mean(),yerr=vals.std(),fmt='none',color=INK,lw=.7,capsize=2,zorder=3)
        dots(ax,j,vals,INK,width=.085,alpha=.85,size=10)
    ax.axhline(100,color=GRAY,ls=':',lw=.8);ax.set_ylim(0,112)
    ax.set_xticks(range(5),[z[1] for z in fullcases],fontsize=5.3)
    ax.set_ylabel('Distance from start (% matched intact)')
    ax.text(.01,.98,'2,000 cells replaced in every graft arm',transform=ax.transAxes,va='top',fontsize=5.5,color=MUTED)
    clean(ax)
    # e: all paths use a single prespecified seed and identical perturbations.
    e = container(fig, [.075, .115, .565, .16], 'e', 'Simulated paths under matched gusts')
    cases = [('size_site20_n2000', 'No graft', ORANGE), ('partial_random_p5', 'Random 100', RULE_COLORS['random']),
             ('partial_ranked_p5', 'Selected 100', RULE_COLORS['ranked']), ('partial_full', 'Full 2,000', BLUE)]
    paths = []
    for name, _, _ in cases:
        z = d.select(name, 0)
        if len(z):
            r = z.iloc[0]; paths.append(flight.simulate(r.gain_R, r.gain_L, r.bias, seed=d.movie_seed, drive=r.drive))
    if len(d.intact):
        r = d.intact[d.intact.seed == 0].iloc[0]
        paths.append(flight.simulate(r.gain_R, r.gain_L, r.bias, seed=d.movie_seed, drive=r.drive))
    limits = None
    if paths:
        coords = np.concatenate([p[:, 3:5] for p in paths])
        limits = ((min(-.25, coords[:, 1].min())-.1, max(.25, coords[:, 1].max())+.1),
                  (min(0, coords[:, 0].min())-.1, max(.5, coords[:, 0].max())+.1))
    for j, (name, lab, col) in enumerate(cases):
        ax = e.inset_axes([j*.257, 0, .214, 1.0])
        z = d.select(name, 0)
        if len(z):
            trajectory(ax, z.iloc[0], d, col, title=lab, limits=limits)
        else:
            missing(ax)
            ax.set_title(lab, fontsize=6.7, pad=3)
        if j:
            ax.tick_params(labelleft=False)
    e.text(.5, -.3, 'Forward versus lateral displacement (m); dashed: intact; seed 0',
           transform=e.transAxes, ha='center', fontsize=5.8, color=MUTED)
    # f: the small graft improves measurable movement, without a composite score.
    ax=fig.add_axes([.735,.115,.22,.16]);panel(ax,'f','Movement restored by 100 cells')
    for j,(name,label,col) in enumerate(cases):
        z=direct[direct.name==name];vals=100*z.net_displacement_m_paired_norm
        ax.bar(j,vals.mean(),width=.63,color=col,alpha=.8,zorder=2)
        ax.errorbar(j,vals.mean(),yerr=vals.std(),fmt='none',color=INK,lw=.7,capsize=1.5,zorder=3)
        dots(ax,j,vals,INK,width=.07,size=9,alpha=.85)
    ax.axhline(100,color=GRAY,ls=':',lw=.8);ax.set_ylim(0,112)
    ax.set_xticks(range(4),['None','Random\n100','Selected\n100','Exact\n2,000'],rotation=35,ha='right',fontsize=5.2)
    ax.set_ylabel('Distance from start (% intact)',fontsize=6.3);clean(ax)
    add_preview(fig,preview,d)
    if not preview:
        fig.texts[-1].set_text('Means and error bars: three neural seeds; individual points: seeds. Movement summarizes 20 matched gusts per seed.')
    return fig


def supplement4(d,preview):
    fig=plt.figure(figsize=(8.1,7.3))
    fig.text(.055,.954,'Supplementary Figure 4 | Exploratory network adaptation',fontsize=11.2,fontweight='bold')
    fig.text(.055,.915,'One round = neural response measurement, then an excitatory-input update. No biological time conversion.',fontsize=6.6,color=MUTED)
    ad=pd.read_csv(RES/'revision_20261002/adaptation_readout_by_round.csv')
    metrics=[('net_displacement_m_paired_norm','Distance from start'),('straightness_paired_norm','Path straightness'),
             ('drive_paired_norm','Motor drive'),('evoked_activity_paired_norm','Evoked activity')]
    for i,(metric,title) in enumerate(metrics):
        ax=fig.add_axes([.09+(i%2)*.49,.565-(i//2)*.375,.36,.24]);panel(ax,chr(97+i),title)
        for n,col in SIZE_COLORS.items():
            q=ad[ad.n_lesion==n]
            for seed,z in q.groupby('seed'):
                z=z.sort_values('round');ax.plot(np.r_[-2,z['round']],np.r_[1,z[metric]],color=col,alpha=.23,lw=.6)
            mean=q.groupby('round')[metric].mean();ax.plot(np.r_[-2,mean.index],np.r_[1,mean.values],color=col,lw=1.3,label=f'{n:,} lost')
        ax.axhline(1,color=GRAY,ls=':',lw=.8);ax.axvline(0,color=GRAY,ls='--',lw=.7)
        ax.set_xticks([-2,0,5,10,15,20],['Pre','0','5','10','15','20']);ax.set_xlim(-3,20);ax.set_ylim(0,1.8)
        ax.set_xlabel('Model update round');ax.set_ylabel('Readout / matched intact');clean(ax)
        if i==0:ax.legend(frameon=False,ncol=2,fontsize=5.6,loc='upper right')
    fig.text(.055,.06,'Thin: individual seeds; thick: mean of three seeds. Pre: intact; round 0: acute injury. Overshoots and fluctuations retained.',fontsize=6.1,color=MUTED)
    fig.text(.055,.035,'The current feedback rule does not establish stable functional recovery or a calibrated recovery timescale.',fontsize=6.1,color=MUTED)
    return fig


def supplement5(d,preview):
    fig,ax=support_canvas('Supplementary Figure 5 | Replacement identity and host adaptation',[
        'Correct cell identity in a 100-cell graft','Absolute movement after acute replacement',
        'Absolute movement after delayed replacement','Added movement at matched follow-up'])
    direct=pd.read_csv(RES/'revision_20261002_clarity/all_direct_readouts.csv')
    q=direct[(direct.category=='typed')&(direct.n_graft==100)].copy();q['correct_pct']=100*q.fate_accuracy
    line_summary(ax[0],q,'straightness',TEAL,'100 prioritized cells',x='correct_pct')
    for pct,z in q.groupby('correct_pct'):dots(ax[0],pct,z.straightness,TEAL,width=1.2,size=12)
    ax[0].set_xticks([0,25,50,75,100]);ax[0].set_ylim(0,1.04);ax[0].set_xlabel('Correctly assigned cell types (%)')
    ax[0].set_ylabel('Path straightness (0-1)');clean(ax[0])
    ad=direct[direct.audit_group=='adaptation'].copy()
    records=[]
    for timing,start,arms in [('Acute',0,[('untreated',0),('acute_100',100),('acute_2000',2000)]),
                              ('Delayed',10,[('untreated',0),('delayed_100',100),('delayed_2000',2000)])]:
        dest=ax[1] if timing=='Acute' else ax[2]
        for condition,count in arms:
            q=ad[(ad.condition==condition)&(ad.n_lesion==2000)&ad['round'].between(start,start+10)].copy()
            q['post_round']=q['round']-start;col=ORANGE if count==0 else TEAL if count==100 else BLUE
            label='No replacement' if count==0 else f'{count:,} cells'
            line_summary(dest,q,'net_displacement_m',col,label,x='post_round')
            for seed,z in q.groupby('seed'):dest.plot(z.post_round,z.net_displacement_m,color=col,alpha=.20,lw=.6)
            tail=q[q.post_round.between(6,10)].groupby('seed').net_displacement_m.mean()
            records.extend(dict(timing=timing,count=count,seed=int(seed),distance=value) for seed,value in tail.items())
        dest.set_xlabel('Model updates after intervention');dest.set_ylabel('Distance from start (m / 10 s)')
        dest.set_ylim(0,3.0);dest.legend(frameon=False,fontsize=5.3,loc='lower left',bbox_to_anchor=(0,1.15),ncol=3);clean(dest)
    tail=pd.DataFrame(records);controls=tail[tail['count']==0][['timing','seed','distance']].rename(columns={'distance':'control'})
    paired=tail[tail['count']>0].merge(controls,on=['timing','seed'],validate='many_to_one');paired['added_distance']=paired.distance-paired.control
    for j,count in enumerate([100,2000]):
        for offset,timing,col in [(-.17,'Acute',TEAL),(.17,'Delayed',PURPLE)]:
            z=paired[(paired['count']==count)&(paired.timing==timing)]
            ax[3].bar(j+offset,z.added_distance.mean(),width=.28,color=col,alpha=.80,zorder=2,label=timing if j==0 else None)
            ax[3].errorbar(j+offset,z.added_distance.mean(),yerr=z.added_distance.std(),fmt='none',color=INK,capsize=2,lw=.7)
            dots(ax[3],j+offset,z.added_distance,INK,width=.035,size=11,alpha=.8)
    ax[3].axhline(0,color=GRAY,lw=.8);ax[3].set_xticks([0,1],['100 cells','2,000 cells'])
    ax[3].set_ylabel('Added distance versus untreated host (m)');ax[3].legend(frameon=False,fontsize=6,loc='best');clean(ax[3])
    fig.text(.055,.048,'Delayed grafts begin after 10 host updates. Each arm uses its continuing untreated host at the same injury round.',fontsize=6,color=MUTED)
    fig.text(.055,.025,'Panel d first averages post-intervention rounds 6-10 within each seed. This exploratory host rule has no biological time calibration.',fontsize=6,color=MUTED)
    return fig


def support_canvas(title, titles):
    four=len(titles)==4
    fig=plt.figure(figsize=(8.1,6.65 if four else 8.1))
    fig.text(.05,.956,title,fontsize=11.2,fontweight='bold')
    axes=[]
    for i,t in enumerate(titles):
        col=i%2;row=i//2
        ax=fig.add_axes([.085+col*.49,(.605-row*.365) if four else (.697-row*.267),.36,.24 if four else .175])
        panel(ax,chr(97+i),t);axes.append(ax)
    return fig,axes


def supplement1(d, preview):
    fig,ax=support_canvas('Supplementary Figure 1 | Model reference and sensitivity',[
        'Intact responses to the three visual stimuli','Full reconstruction versus the intact network',
        'Variation in the intact directional read-out','Lesion effect across synaptic weight settings',
        'Lesion effect across sensory input rates','Variation across stochastic input seeds'])
    if len(d.intact):
        for j,(key,label,col) in enumerate([('y_R','Rightward',BLUE),('y_F','Forward',GRAY),('y_L','Leftward',ORANGE)]):
            dots(ax[0],j,d.intact[key],col,width=.09,size=18,alpha=.8)
            ax[0].plot([j-.16,j+.16],[d.intact[key].mean()]*2,color=col,lw=1.7)
        ax[0].set_xticks(range(3),['Rightward','Forward','Leftward'])
        ax[0].set_ylabel('Decoded command (Hz)');ax[0].axhline(0,color=GRAY,lw=.6);clean(ax[0])
        for j,(key,col) in enumerate([('gain_R',BLUE),('gain_L',ORANGE),('bias',PURPLE)]):
            dots(ax[2],j,d.intact[key],col,width=.1,size=18,alpha=.8)
        ax[2].set_xticks(range(3),['Right gain','Left gain','Bias'])
        ax[2].set_ylabel('Normalized neural read-out')
        ax[2].axhline(1,color=GRAY,ls=':',lw=.8);ax[2].axhline(0,color=GRAY,lw=.6)
        clean(ax[2])
    else:
        missing(ax[0]);missing(ax[2])
    full=d.select('partial_full')
    if len(full) and len(d.intact):
        paired=full.merge(d.intact[['seed','y_R','y_L','y_F']],on='seed',suffixes=('','_intact'),validate='one_to_one')
        all_delta=[]
        for j,(key,col) in enumerate([('y_R',BLUE),('y_F',GRAY),('y_L',ORANGE)]):
            delta=paired[key]-paired[key+'_intact'];all_delta.extend(delta)
            dots(ax[1],j,delta,col,width=.1,size=17,alpha=.85)
        ax[1].axhline(0,color=GRAY,lw=.7)
        ax[1].set_xticks(range(3),['Rightward','Forward','Leftward'])
        ax[1].set_ylabel('Reconstruction minus intact (Hz)')
        if max(np.abs(all_delta),default=0)<1e-5:
            ax[1].set_ylim(-1,1)
            ax[1].text(.5,.86,'All differences < 0.00001 Hz',ha='center',transform=ax[1].transAxes,
                       fontsize=6.3,color=MUTED)
        clean(ax[1])
    else:missing(ax[1])
    sens=d.sensitivity
    for dest,param,xlabel in [(ax[3],'weight_factor','Synaptic weight multiplier'),
                              (ax[4],'stim_rate','Sensory input rate (Hz)')]:
        q=sens[sens.parameter==param] if len(sens) else pd.DataFrame()
        if len(q):
            metric='steering_similarity' if param=='weight_factor' else 'straightness'
            for (site,n),z in q.groupby(['site','n_lesion']):
                z=z.sort_values('value');col=ORANGE if site==20 else BLUE;ls='-' if n==1000 else '--'
                dest.plot(z.value,z[metric],color=col,ls=ls,marker='o' if n==1000 else 's',ms=3,
                          label=f'Site {int(site)}, {int(n):,} cells')
            dest.set_ylim(0,1.05);dest.set_xlabel(xlabel)
            dest.set_ylabel('Directional steering similarity' if param=='weight_factor' else 'Simulated path straightness')
            dest.legend(frameon=False,fontsize=5.8,loc='best');clean(dest)
            note='0.85×: intact wing-power = 0; trajectory undefined' if param=='weight_factor' else 'Each condition uses its matched intact reference'
            dest.text(.01,.02,note,transform=dest.transAxes,
                      fontsize=5.4,color=MUTED)
        else:missing(dest)
    q=d.states[d.states.category=='screen'] if len(d.states) else pd.DataFrame()
    if len(q):
        z=q.groupby(['n_lesion','site']).straightness.std().reset_index()
        for j,n in enumerate([1000,4000,16000]):
            vals=z[z.n_lesion==n].straightness.dropna().sort_values().to_numpy()
            dots(ax[5],j,vals,BLUE,width=.17,size=10,alpha=.6)
            if len(vals):ax[5].plot([j-.22,j+.22],[np.median(vals)]*2,color=INK,lw=1.4)
        ax[5].set_xticks(range(3),['1,000','4,000','16,000'])
        ax[5].set_xlabel('Neurons lost');ax[5].set_ylabel('Path straightness SD across seeds')
        ax[5].text(.02,.98,'Each point is one fixed lesion location',transform=ax[5].transAxes,
                   va='top',fontsize=5.6,color=MUTED);clean(ax[5])
    else:missing(ax[5])
    add_preview(fig,preview,d);return fig


def supplement2(d, preview):
    fig,ax=support_canvas('Supplementary Figure 2 | Replacement components and robustness',[
        'Cell selection and rightward response','Cell selection and leftward response',
        'Cell selection and evoked activity','Integrated cell number after stochastic survival',
        'Variation in replacement-cell strength','Exact and type-based replacement'])
    q=d.states[d.states.category=='partial'] if len(d.states) else pd.DataFrame()
    for dest,metric,label in [(ax[0],'gain_R','Rightward response / intact'),
                               (ax[1],'gain_L','Leftward response / intact'),
                               (ax[2],'evoked_activity','Evoked activity / intact')]:
        if len(q):
            for rule in ['random','interface','ranked']:
                line_summary(dest,q[q.rule==rule],metric,RULE_COLORS[rule],LABELS[rule],x='pct')
            dest.axhline(1,color=GRAY,ls=':',lw=.8);dest.axhline(0,color=GRAY,lw=.55)
            dest.set_xlabel('Lost neurons replaced (%)');dest.set_ylabel(label)
            dest.set_xticks([0,10,25,50]);dest.legend(frameon=False,fontsize=5.8,loc='best');clean(dest)
        else:missing(dest)
    q=d.states[d.states.category=='survival'] if len(d.states) else pd.DataFrame()
    if len(q):
        for count,z in q.groupby('n_transplanted'):
            col=TEAL if count==100 else BLUE
            ax[3].scatter(z.n_graft,z.score,color=col,s=22,alpha=.75,edgecolors='white',linewidths=.35,
                          label=f'{int(count)} cells transplanted')
        ax[3].set_xlabel('Integrated replacement cells');ax[3].set_ylabel('Function summary (0-1)')
        ax[3].set_ylim(0,1.04);ax[3].legend(frameon=False,fontsize=5.8,loc='best');clean(ax[3])
    else:missing(ax[3])
    q=d.states[d.states.category=='heterogeneity'] if len(d.states) else pd.DataFrame()
    if len(q):
        for n,name,col in [(100,'partial_ranked_p5',TEAL),(2000,'partial_full',BLUE)]:
            z=q[q.n_graft==n].copy();baseline=d.select(name).copy();baseline['strength_sd']=0.
            z=pd.concat([z,baseline],ignore_index=True)
            line_summary(ax[4],z,'score',col,f'{n:,} replacement cells',x='strength_sd')
            for sd,zz in z.groupby('strength_sd'):dots(ax[4],sd,zz.score,col,width=.015,size=10)
        ax[4].set_xticks([0,.5,1]);ax[4].set_xlabel('SD of log per-cell strength factor')
        ax[4].set_ylabel('Function summary (0-1)');ax[4].set_ylim(0,1.04)
        ax[4].legend(frameon=False,fontsize=5.8,loc='best');clean(ax[4])
        ax[4].text(.02,.02,'Per-cell factors have expected mean 1',transform=ax[4].transAxes,
                   fontsize=5.5,color=MUTED)
    else:missing(ax[4])
    any_data=False
    for j,n,copy_name,type_name,col in [(0,100,'partial_ranked_p5','typed_core100_fate1',TEAL),
                                        (1,2000,'partial_full','typed_full2000',BLUE)]:
        c=d.select(copy_name);t=d.select(type_name)
        if c.empty or t.empty:continue
        c=c.set_index('seed');t=t.set_index('seed')
        for seed in sorted(set(c.index)&set(t.index)):
            any_data=True;vals=[c.loc[seed,'score'],t.loc[seed,'score']]
            ax[5].plot([j-.16,j+.16],vals,color=col,alpha=.65,lw=.85)
            ax[5].scatter(j-.16,vals[0],s=20,facecolors='white',edgecolors=col,zorder=4)
            ax[5].scatter(j+.16,vals[1],s=20,color=col,zorder=4)
    if any_data:
        ax[5].set_xticks([0,1],['100 cells','2,000 cells']);ax[5].set_ylim(0,1.05)
        ax[5].set_ylabel('Function summary (0-1)');clean(ax[5])
        ax[5].text(.03,.97,'Open: exact wiring; filled: type-based wiring\nCorrect assigned identities; paired seeds',
                   transform=ax[5].transAxes,va='top',fontsize=5.8,color=MUTED)
    else:missing(ax[5])
    add_preview(fig,preview,d);return fig


def supplement3(d, preview):
    fig,ax=support_canvas('Supplementary Figure 3 | Adaptation dynamics and rule sensitivity',[
        'Rightward steering during adaptation','Leftward steering during adaptation',
        'Turning bias during adaptation','Host neurons at the upper scaling bound',
        'Sensitivity to adaptation rate and gain bound','Replacement benefit in adapting or fixed hosts'])
    ad=d.untreated()
    for dest,key,label in [(ax[0],'gain_R','Rightward response / intact'),
                           (ax[1],'gain_L','Leftward response / intact'),
                           (ax[2],'bias','Turning bias (intact = 0)'),
                           (ax[3],'fraction_gain_capped','Fraction of host neurons at gain cap')]:
        if len(ad):
            for n,z in ad.groupby('n_lesion'):
                if n not in SIZE_COLORS:continue
                for seed,zz in z.groupby('seed'):
                    zz=zz.sort_values('round');dest.plot(zz['round'],zz[key],color=SIZE_COLORS[n],lw=.5,alpha=.24)
                z=z.groupby('round')[key].mean();dest.plot(z.index,z.values,color=SIZE_COLORS[n],label=f'{n:,} lost')
            dest.axhline(1 if key in ['gain_R','gain_L'] else 0,color=GRAY,ls=':',lw=.8)
            dest.set_xlabel('Adaptation round');dest.set_ylabel(label);dest.set_xlim(0,20)
            dest.legend(frameon=False,fontsize=5.7,loc='best',ncol=2);clean(dest)
        else:missing(dest)
    if len(d.adaptation):
        controls=[('untreated','Default',INK),('eta_0.125','Rate exponent 0.125',BLUE),
                  ('eta_0.5','Rate exponent 0.5',ORANGE),('cap_1.5','Maximum gain 1.5',TEAL),
                  ('cap_3','Maximum gain 3',PURPLE)]
        found=False
        for cond,label,col in controls:
            q=d.adaptation[(d.adaptation.condition==cond)&(d.adaptation.n_lesion==2000)&(d.adaptation.seed==0)].sort_values('round')
            if len(q):found=True;ax[4].plot(q['round'],q.score,color=col,label=label,lw=1.0)
        if found:
            ax[4].set_xlabel('Adaptation round');ax[4].set_ylabel('Function summary (0-1)')
            ax[4].set_ylim(0,1.04);ax[4].legend(frameon=False,fontsize=5.6,loc='best');clean(ax[4])
            ax[4].text(.02,.98,'2,000-neuron lesion; seed 0',transform=ax[4].transAxes,
                       ha='left',va='top',fontsize=5.5,color=MUTED)
        else:missing(ax[4])
        found=False
        for count,col in [(100,TEAL),(2000,BLUE)]:
            for frozen in [False,True]:
                source=d.frozen_contrasts if frozen else d.contrasts
                if source.empty:continue
                cond=f'delayed_{count}'+('_frozen' if frozen else '')
                key='delta_score_vs_frozen' if frozen else 'delta_score'
                q=source[(source.condition==cond)&(source.n_lesion==2000)&(source['round'].between(10,20))]
                if len(q):
                    found=True;g=q.groupby('round')[key].mean();ax[5].plot(g.index-10,g.values,color=col,
                        ls='--' if frozen else '-',label=f'{count:,} cells'+(' (fixed)' if frozen else ' (adapting)'),lw=1.1)
        if found:
            ax[5].axhline(0,color=GRAY,ls=':',lw=.8)
            ax[5].set_xlabel('Rounds after replacement');ax[5].set_ylabel('Change versus matched untreated host')
            ax[5].set_xlim(0,10)
            ax[5].legend(frameon=False,fontsize=5.5,loc='best',ncol=2,handlelength=1.5,columnspacing=.7);clean(ax[5])
        else:missing(ax[5])
    else:missing(ax[4]);missing(ax[5])
    add_preview(fig,preview,d);return fig


def validate_final(d):
    errors=[]
    specs=json.loads((RES/'specs_states.json').read_text())
    expected={(s['name'],int(s.get('seed',0))) for s in specs}
    observed=set(zip(d.states.get('name',[]),d.states.get('seed',[])))
    if expected-observed:
        errors.append(f'{len(expected-observed)} neural states are missing')
    if observed-expected:
        errors.append(f'{len(observed-expected)} neural states are outside the frozen protocol')
    if d.states.duplicated(['name','seed']).any() if len(d.states) else False:
        errors.append('Duplicate neural state rows')
    if len(d.intact)<5:
        errors.append('Five intact reference seeds are required')
    sens=d.sensitivity
    required_sens={'parameter','steering_similarity','straightness','trajectory_valid','undefined_reason'}
    if len(sens)!=16 or not required_sens.issubset(sens.columns):
        errors.append('Matched sensitivity table incomplete or missing validity metadata')
    else:
        weight=sens[sens.parameter=='weight_factor'];rate=sens[sens.parameter=='stim_rate']
        if not np.isfinite(weight.steering_similarity).all() or not np.isfinite(rate.straightness).all():
            errors.append('Plotted matched-sensitivity readout is nonfinite')
        invalid=sens[~sens.trajectory_valid.astype(bool)]
        if len(invalid) and (invalid.undefined_reason.isna().any() or invalid.straightness.notna().any()):
            errors.append('Undefined sensitivity trajectory lacks a reason or was assigned a numeric value')
    ad=d.untreated()
    for n in [1000,2000,4000,8000]:
        for seed in [0,1,2]:
            q=ad[(ad.n_lesion==n)&(ad.seed==seed)] if len(ad) else pd.DataFrame()
            if q.empty or not set(range(21)).issubset(q['round']):
                errors.append(f'Untreated adaptation incomplete: n={n}, seed={seed}')
    tr=treatment_rows(d)
    for n in [100,2000]:
        for timing in ['Acute','After adaptation']:
            for seed in [0,1,2]:
                q=tr[(tr.graft_n==n)&(tr.timing==timing)&(tr.seed==seed)] if len(tr) else pd.DataFrame()
                if q.empty or not set(range(11)).issubset(q.since_graft):
                    errors.append(f'Matched replacement follow-up incomplete: n={n}, {timing}, seed={seed}')
    for n in [100,2000]:
        for seed in [0,1,2]:
            q=d.frozen_contrasts
            q=q[(q.condition==f'delayed_{n}_frozen')&(q.seed==seed)] if len(q) else q
            if q.empty or not set(range(10,21)).issubset(q['round']):
                errors.append(f'Frozen-host contrast incomplete: n={n}, seed={seed}')
            elif not np.isfinite(q.delta_score_vs_frozen).all():
                errors.append(f'Frozen-host contrast nonfinite: n={n}, seed={seed}')
    adaptation_specs=RES/'specs_adaptation.json'
    if adaptation_specs.exists():
        plan=json.loads(adaptation_specs.read_text())
        for spec in plan.get('baseline',[])+plan.get('branches',[]):
            q=d.adaptation[(d.adaptation.experiment==spec['name'])&(d.adaptation.seed==spec['seed'])] if len(d.adaptation) else pd.DataFrame()
            start=int(spec.get('branch_round',0));end=int(spec.get('rounds',20))
            if q.empty or not set(range(start,end+1)).issubset(q['round']):
                errors.append(f'Adaptation protocol incomplete: {spec["name"]}, seed={spec["seed"]}')
    for name,df in [('states',d.states),('intact',d.intact),('adaptation',d.adaptation)]:
        metrics=[k for k in ['gain_R','gain_L','bias','drive','evoked_activity','straightness','score'] if k in df]
        if metrics and not np.isfinite(df[metrics].to_numpy(float)).all():
            errors.append(f'Nonfinite metrics in {name}')
    prov=TAB/'table_provenance.json'
    if not prov.exists():
        errors.append('Missing table provenance')
    else:
        for filename,digest in json.loads(prov.read_text()).get('generated_from',{}).items():
            path=ROOT/filename
            if not path.exists() or sha(path)!=digest:
                errors.append(f'Stale table input: {filename}')
    if errors:
        raise RuntimeError('Final figures blocked:\n'+'\n'.join(errors))


def main():
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview',action='store_true');mode.add_argument('--final',action='store_true')
    args=ap.parse_args();style();d=Data()
    if args.final:
        validate_final(d)
    dest=OUT/'preview' if args.preview else OUT
    dest.mkdir(parents=True,exist_ok=True)
    suffix='_PREVIEW' if args.preview else ''
    produced=[]
    for name,builder in [('Fig1_v5_focal_loss_adaptation',figure1),('Fig2_v5_cell_replacement',figure2),
                         ('SuppFig1_v5_model_validation',supplement1),('SuppFig2_v5_replacement_robustness',supplement2),
                         ('SuppFig3_v5_adaptation_robustness',supplement3),
                         ('SuppFig4_v5_recovery_dynamics',supplement4),('SuppFig5_v5_identity_host_state',supplement5)]:
        fig=builder(d,args.preview)
        for ext in ['pdf','svg','png']:
            path=dest/(name+suffix+'.'+ext)
            fig.savefig(path,dpi=300,facecolor='white',metadata={'Creator':'FLY V5 corrected-data figure builder'} if ext=='pdf' else None)
            produced.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    inputs=[p for p in [TAB/(n+'.csv') for n in ['states','intact','sites_summary','grafts_summary','adaptation','adaptation_treatment_contrasts','adaptation_frozen_contrasts','adaptation_matched_followup','sensitivity_matched_intact']]
            if p.exists()]
    inputs.extend([ROOT/'data/v5/neurons.parquet',ROOT/'data/v5/cell_groups.json',RES/'lesion_sites.csv',
                   RES/'protocol.json',RES/'specs_states.json',RES/'specs_adaptation.json',TAB/'table_provenance.json',Path(__file__),
                   RES/'revision_20261002/site20_size_readout_by_seed.csv',RES/'revision_20261002/adaptation_readout_by_round.csv',
                   RES/'revision_20261002/readout_audit_manifest.json',
                   RES/'revision_20261002_clarity/all_direct_readouts.csv',RES/'revision_20261002_clarity/readout_validation.json'])
    manifest={'status':'incomplete_preview' if args.preview else 'complete_verified_tables',
              'outputs':produced,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'output_sha256':{name:sha(ROOT/name) for name in produced},
              'neural_states_available':len(d.states),'anatomical_view':'mirrored x-z projection; fly right on image right, matching movies',
              'background_soma_sample':'12000 real somata, seed194, no rasterization',
              'trajectory_example':'neural seed0, prespecified gust seed'+str(d.movie_seed),
              'figure1_panels':{'a':'connectome soma positions and pathway schematic','b':'sites_summary straightness, three sizes',
                                'c':'site20/site25, 1000 neurons, actual lesion somata and trajectories',
                                'd':'actual nested site20 lesions of500,4000,16000,64000 neurons','e':'four matched-intact normalized readouts across250 to64000 lost neurons',
                                'f':'absolute distance versus lesion size across56 fixed locations; each location mean across3neural and20gust seeds'},
              'figure2_panels':{'a':'actual random,interface,ranked100 templates over the same full2000-neuron lesion','b':'absolute displacement inmetres by replacement count and original-partner probability;1k lesion',
                                'c':'absolute distance across0 to1000 selected templates with original wiring; random template selection is not random identity',
                                'd':'full2000-cell replacement with random partners, wrong types, correct types or exact reconstruction',
                                'e':'matched trajectories','f':'measured movement with no graft, random100, selected100 and exact2000, seed-paired intact percentage'},
              'supplement1_panels':{'a':'intact decoded commands, five seeds','b':'complete reconstruction minus same-seed intact',
                                    'c':'intact directional gains and bias','d':'matched-reference steering similarity across synaptic weights; 0.85x trajectory undefined',
                                    'e':'input-rate sensitivity with matched intact reference','f':'site-wise SD across neural seeds'},
              'supplement2_panels':{'a':'replacement identity and rightward response','b':'replacement identity and leftward response',
                                    'c':'replacement identity and evoked activity','d':'actual integrated cell count after survival',
                                    'e':'strength heterogeneity, per-cell factors have expected mean1; total synaptic strength not fixed','f':'exact versus typed wiring, paired neural seeds'},
              'supplement3_panels':{'a':'rightward response during adaptation','b':'leftward response during adaptation',
                                    'c':'turning bias during adaptation','d':'host fraction at upper input-gain bound',
                                    'e':'rate-exponent and gain-cap sensitivity, neural seed0',
                                    'f':'delayed replacement benefit relative to matched untreated adapting or frozen hosts'},
              'supplement4_panels':{'a':'distance','b':'straightness','c':'motor drive','d':'evoked activity; all matched-seed normalized, original round0 to20 plus intact; overshoots retained'},
              'supplement5_panels':{'a':'100-cell typed graft identity accuracy versus path straightness','b':'absolute distance after acute graft','c':'absolute distance after delayed graft','d':'paired additional metres across post-intervention rounds6 to10'}}
    (dest/('figure_manifest'+suffix+'.json')).write_text(json.dumps(manifest,indent=2))
    print('\n'.join(produced))


if __name__=='__main__':
    main()
