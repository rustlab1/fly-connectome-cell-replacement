"""
experiments.py
--------------
Shared machinery for the lesion / graft experiments on the male CNS connectome model.

Protocol (per network state):
  three 300-ms trials with Poisson drive (100 Hz) of direction-selective T4/T5 neurons
    'R' : rightward pattern rotation  (right eye front-to-back a-subtypes, left eye back-to-front b-subtypes)
    'L' : leftward pattern rotation   (mirror image)
    'F' : forward translation         (front-to-back a-subtypes in both eyes)
  firing rates are measured 50-300 ms after stimulus onset.

Read-outs:
  steering command  y = d . r_steerMN   with d the intact-network differential pattern (R minus L)
  steering gain     (y_R - y_L) / (y_R - y_L)_intact
  gain_R            (y_R - y_F) / (y_R,intact - y_F,intact)
  gain_L            (y_F - y_L) / (y_F,intact - y_L,intact)
  steering bias     2 * (y_F - y_F,intact) / (y_R - y_L)_intact
  The overall gain is not generally the arithmetic mean of directional gains.
  off-target drive  mean rate of leg motor neurons
  network activity  number of active neurons outside the optic lobes
"""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from flysim import LIFNet, load_net

ROOT = Path(__file__).resolve().parents[2]
PROC = str(ROOT / 'data' / 'v5') + os.sep
RES = str(ROOT / 'results' / 'v5') + os.sep
W_SYN = 0.275 * (54492922 / 138639) / (124025046 / 165122)   # exact integer synapse total; density-matched to FlyWire
STIM_RATE = 100.0
T_SIM, T_MEAS = 300.0, (50.0, 300.0)

_cache = {}


def get_net():
    if 'W' not in _cache:
        W, neu, G = load_net(PROC)
        _cache.update(W=W, neu=neu, G=G)
        xyz = neu[['x_um', 'y_um', 'z_um']].values
        has = ~np.isnan(xyz[:, 0])
        _cache['soma_idx'] = np.flatnonzero(has)
        _cache['tree'] = cKDTree(xyz[has])
        # neurons eligible for lesioning: all neurons with a soma in the CNS except the motor
        # neurons, which are the read-out layer of the model (kept intact, like a muscle recording)
        elig = has & ~neu.superclass.isin(['vnc_motor', 'cb_motor']).values
        _cache['lesion_idx_pool'] = np.flatnonzero(elig)
        _cache['lesion_tree'] = cKDTree(xyz[elig])
        _cache['nonOL'] = ~neu.superclass.isin(['ol_intrinsic', 'ol_sensory']).values
        # vital read-outs: labellar / pharyngeal taste neurons -> proboscis motor neurons (feeding),
        # wing power motor neurons (flight motor drive)
        t = neu.type.fillna('')
        G['GRN_labellar'] = neu.index[(neu['class'] == 'gustatory') & neu.subclass.isin(['labellar bristle', 'taste peg', 'pharyngeal sensillum'])].tolist()
        G['probMN'] = neu.index[(neu.superclass == 'cb_motor') & t.str.match(r'^MN')].tolist()
        G['powerMN'] = G['powerMN_R'] + G['powerMN_L']
        G['steerMN'] = G['steerMN_R'] + G['steerMN_L']
        # escape take-off pathway: looming-sensitive visual projection neurons -> giant fibre -> TTMn / DLMn
        G['LC_loom'] = neu.index[t.isin(['LC4', 'LPLC2'])].tolist()
        G['GF'] = neu.index[t == 'DNp01'].tolist()
        G['TTMn'] = neu.index[t == 'TTMn'].tolist()
        G['DLMn'] = neu.index[t.isin(['DLMn a, b', 'DLMn c-f'])].tolist()
    return _cache


def stimuli(with_feed=False):
    G = get_net()['G']
    d = {'R': np.array(G['T4a_R'] + G['T5a_R'] + G['T4b_L'] + G['T5b_L']),
         'L': np.array(G['T4a_L'] + G['T5a_L'] + G['T4b_R'] + G['T5b_R']),
         'F': np.array(G['T4a_L'] + G['T5a_L'] + G['T4a_R'] + G['T5a_R'])}
    if with_feed:
        d['feed'] = np.array(G['GRN_labellar'])
        d['loom'] = np.array(G['LC_loom'])
    return d


def graft_stimulus_indices(indices, lesion_idx=None, graft=None):
    """Transfer a lost input's stimulus to eligible replacements, in input order.

    Unreplaced inputs retain a dead placeholder so their random-event slot is
    consumed without stimulation. One-to-one replacement retains the intact
    random-event ordering. Repeated templates receive separate event streams.
    Typed grafts exclude misfated cells through sensory eligibility metadata.
    """
    indices = np.asarray(indices, dtype=int)
    if graft is None or lesion_idx is None:
        return indices
    N = len(get_net()['neu'])
    dead = set(np.asarray(lesion_idx, dtype=int).tolist())
    eligible = getattr(graft[0], 'stimulus_eligible', np.ones(len(graft[2]), dtype=bool))
    replacements = {}
    for gi, t in enumerate(graft[2]):
        if eligible[gi]:
            replacements.setdefault(int(t), []).append(N + gi)
    result = []
    for ix in indices:
        result.extend(replacements.get(int(ix), [int(ix)]) if ix in dead else [int(ix)])
    return np.asarray(result, dtype=int)


# ------------------------------------------------------------------ lesions
def make_lesion(center_um, n_neurons):
    """Indices of the n_neurons neurons whose somata are closest to center_um (focal, contiguous loss)."""
    c = get_net()
    d, ii = c['lesion_tree'].query(np.asarray(center_um, float), k=n_neurons)
    return c['lesion_idx_pool'][ii], float(d.max())


# ------------------------------------------------------------------- grafts
def make_graft(lesion_idx, n_graft, specificity=1.0, composition='matched', strength=1.0,
               local_radius_um=40.0, seed=0, center_um=None, templates=None, survival=1.0, strength_sd=0.0, strength_mean_norm=False):
    """Wire n_graft new neurons into the lesioned network.

    Each graft neuron is assigned a template among the lost neurons.  A fraction
    `specificity` of the template's input and output connections is copied
    (partner-specific integration); the remainder is re-assigned to random
    surviving neurons within local_radius_um of the lesion centre (unspecific,
    local integration).  Synapse counts are kept, so the graft carries the same
    synaptic load on retained connections; `strength` scales all graft weights.
    Host-to-graft edges are owned by the postsynaptic template. All outputs,
    including graft-to-graft edges, are owned by the presynaptic template and
    added once. Unreplaced lost partners are omitted. Repeated templates form
    the full Cartesian set of replacement pairs, each with the original weight;
    their extra synaptic dose is intentional and is not normalized away.
    composition: 'matched' (sign of the template), 'E' (all excitatory), 'I' (all inhibitory).
    templates: optional explicit array of lost neurons to replace (first n_graft are used), e.g. a
    targeted subset (descending and visual projection neurons first).
    survival: probability that a transplanted cell survives and integrates (Bernoulli per cell); the
    cells that do not survive are dropped, so the wired graft has about n_graft * survival neurons.
    strength_sd: s.d. of a log-normal per-cell synaptic strength factor (median = strength), i.e.
    heterogeneous maturation instead of one scale factor for every graft cell.
    Returns extra CSC matrix (N_tot x N_tot), graft signs, graft template indices (survivors only).
    """
    c = get_net(); W = c['W']; N = W.shape[0]; rng = np.random.default_rng(seed)
    lesion_idx = np.atleast_1d(np.asarray(lesion_idx, dtype=int))
    if n_graft < 0 or not 0 <= specificity <= 1 or not 0 <= survival <= 1:
        raise ValueError('Invalid graft count, specificity or survival')
    dead = np.zeros(N, bool); dead[lesion_idx] = True
    # templates: each graft neuron 'stands in' for one lost neuron (random subset unless given explicitly)
    if templates is not None:
        templ = np.asarray(templates, dtype=int)[:n_graft]
        if len(templ) != n_graft or not np.isin(templ, lesion_idx).all():
            raise ValueError('Templates must provide the requested number of lost neurons')
    elif n_graft <= len(lesion_idx):
        templ = rng.choice(lesion_idx, n_graft, replace=False)
    else:
        templ = np.concatenate([rng.permutation(lesion_idx), rng.choice(lesion_idx, n_graft - len(lesion_idx))])
    if survival < 1.0:                                   # Bernoulli survival per transplanted cell
        templ = templ[rng.random(len(templ)) < survival]
    n_graft = len(templ)
    N_tot = N + n_graft
    if n_graft == 0:
        return sp.csc_matrix((N, N), dtype=np.float32), np.zeros(0, np.int8), templ
    # per-cell synaptic strength (heterogeneous maturation); strength_sd = 0 gives the single factor
    if strength_sd > 0:
        z = strength_sd * rng.standard_normal(n_graft)
        s_cell = strength * np.exp(z - (strength_sd ** 2 / 2 if strength_mean_norm else 0.0))    # mean 1 (norm) or median 1
    else:
        s_cell = np.full(n_graft, float(strength))
    sign_t = c['neu'].sign.values
    g_sign = np.array([{'matched': sign_t[t], 'E': 1, 'I': -1}[composition] for t in templ], np.int8)
    # lost neuron -> graft neurons replacing it (lets copied lost->lost connections become graft->graft)
    replaced_by = {}
    for gi, t in enumerate(templ):
        replaced_by.setdefault(int(t), []).append(N + gi)
    multi = any(len(v) > 1 for v in replaced_by.values())
    repl_first = np.full(N, -1, np.int64)
    for t, v in replaced_by.items():
        repl_first[t] = v[0]
    # local pool of surviving partners (peri-lesional tissue)
    xyz = c['neu'][['x_um', 'y_um', 'z_um']].values
    if center_um is None:
        center_um = np.nanmean(xyz[lesion_idx], axis=0)
    pool = c['soma_idx'][c['tree'].query_ball_point(center_um, local_radius_um)]
    pool = pool[~dead[pool]]
    Wcsr = _cache.setdefault('Wcsr', W.tocsr())
    R, C, V = [], [], []

    def add(r, cc, v):
        R.append(np.asarray(r, np.int64)); C.append(np.asarray(cc, np.int64)); V.append(np.asarray(v, np.float32))

    for gi, t in enumerate(templ):
        g = N + gi
        # ---- inputs of the template (row t of W): pre -> t becomes pre -> g
        pre = Wcsr.indices[Wcsr.indptr[t]:Wcsr.indptr[t + 1]]
        win = np.abs(Wcsr.data[Wcsr.indptr[t]:Wcsr.indptr[t + 1]])
        # Incoming edges from graft cells are created by those source cells'
        # output pass. Lost sources without a replacement cannot send input.
        host_input = ~dead[pre]
        pre, win = pre[host_input], win[host_input]
        keep = rng.random(len(pre)) < specificity
        pk, wk = pre[keep], win[keep]
        al = ~dead[pk]
        add(np.full(al.sum(), g), pk[al], wk[al] * sign_t[pk[al]])
        n_rand = int((~keep).sum())
        if n_rand and len(pool):
            rp = rng.choice(pool, n_rand)
            add(np.full(n_rand, g), rp, win[~keep] * sign_t[rp])
        # ---- outputs of the template (column t of W): t -> post becomes g -> post
        post = W.indices[W.indptr[t]:W.indptr[t + 1]]
        wout = np.abs(W.data[W.indptr[t]:W.indptr[t + 1]])
        retained = ~dead[post] | (repl_first[post] >= 0)
        post, wout = post[retained], wout[retained]
        keep = rng.random(len(post)) < specificity
        qk, wq = post[keep], wout[keep]
        al = ~dead[qk]
        add(qk[al], np.full(al.sum(), g), wq[al] * g_sign[gi])
        if (~al).any():
            if multi:
                for q_, w_ in zip(qk[~al], wq[~al]):
                    for gq in replaced_by.get(int(q_), []):
                        add([gq], [g], [w_ * g_sign[gi]])
            else:
                gq = repl_first[qk[~al]]; ok = gq >= 0
                add(gq[ok], np.full(ok.sum(), g), wq[~al][ok] * g_sign[gi])
        n_rand = int((~keep).sum())
        if n_rand and len(pool):
            rq = rng.choice(pool, n_rand)
            add(rq, np.full(n_rand, g), wout[~keep] * g_sign[gi])
    if R:
        Rr, Cc, Vv = np.concatenate(R), np.concatenate(C), np.concatenate(V)
        # strength factor of the graft cell(s) involved: post cell for host->graft, pre cell for
        # graft->host, geometric mean for graft->graft
        strengths = np.concatenate([np.ones(N), s_cell])
        fp, fq = strengths[Rr], strengths[Cc]
        both = (Rr >= N) & (Cc >= N)
        fac = np.where(both, np.sqrt(fp * fq), fp * fq).astype(np.float32)
        E = sp.csc_matrix((Vv * fac, (Rr, Cc)), shape=(N_tot, N_tot))
        E.sum_duplicates()
    else:
        E = sp.csc_matrix((N_tot, N_tot), dtype=np.float32)
    return E, g_sign, templ


# ------------------------------------------------------------ run a state
def run_state(name, lesion_idx=None, graft=None, seed=0, stim_rate=STIM_RATE, save_rates=True,
              record_spikes=False, extra_params=None, with_feed=False, gain_in=None, stimuli_subset=None, return_rates=False):
    """Simulate the three stimuli for one network state; return read-outs (dict).
    gain_in: optional per-neuron excitatory input gain (host neurons, length N; graft cells get 1).
    stimuli_subset: optional list of stimulus keys to run (default: all)."""
    c = get_net(); W, neu, G = c['W'], c['neu'], c['G']; N = W.shape[0]
    n_graft = 0 if graft is None else graft[0].shape[0] - N
    gi = None; ps = None
    if gain_in is not None:
        gi = np.concatenate([np.asarray(gain_in, np.float32), np.ones(n_graft, np.float32)])
        ps = np.concatenate([neu.sign.values.astype(np.int8), (graft[1] if graft is not None else np.zeros(0, np.int8))])
    net = LIFNet(W, n_extra=n_graft, extra_csc=None if graft is None else graft[0],
                 params=dict(dict(w_syn=W_SYN), **(extra_params or {})), gain_in=gi, pre_sign=ps)
    dead = np.zeros(N + n_graft, bool)
    if lesion_idx is not None:
        dead[np.asarray(lesion_idx)] = True
    out = dict(name=name, seed=seed, n_lesion=0 if lesion_idx is None else int(len(lesion_idx)),
               n_graft=int(n_graft), simulation_backend=net.backend)
    rates = {}
    spikes = {}
    for k, ix in stimuli(with_feed).items():
        if stimuli_subset is not None and k not in stimuli_subset:
            continue
        t0 = time.time()
        ix = graft_stimulus_indices(ix, lesion_idx=lesion_idx, graft=graft)
        r = net.run(T_SIM, {stim_rate: ix}, dead=dead, seed=seed * 7 + {'R': 1, 'L': 2, 'F': 3, 'feed': 4, 'loom': 5}[k],
                    t_measure=T_MEAS, record_spikes=record_spikes)
        rates[k] = r['rates']
        if record_spikes:
            spikes[k] = (r['spike_t'], r['spike_i'])
        out[f'time_{k}'] = round(time.time() - t0, 1)
    out.update(readouts(rates, graft=graft))
    if save_rates:
        os.makedirs(RES + 'rates', exist_ok=True)
        np.savez_compressed(RES + f'rates/{name}_s{seed}.npz', **{k: v[:N].astype(np.float32) for k, v in rates.items()},
                            **{f'graft_{k}': v[N:] for k, v in rates.items()} if n_graft else {})
    if record_spikes:
        return out, rates, spikes
    if return_rates:
        return out, rates
    return out


def readouts(rates, graft=None):
    """Scalar read-outs, counting graft cells in their assigned populations.

    Population output is divided by the original group size. This preserves the
    intact scale for ablation, replacement and excess cell dose.
    """
    c = get_net(); G = c['G']; nonOL = c['nonOL']; N = len(c['neu'])
    steer = np.array(G['steerMN_R'] + G['steerMN_L'])
    d = _cache.get('decoder')
    annotation_idx = (np.asarray(getattr(graft[0], 'annotation_indices', graft[2]), dtype=int)
                      if graft is not None else np.zeros(0, dtype=int))
    o = {}
    for k, r in rates.items():
        all_rates = r
        extra_rates = r[N:]
        r = r[:N]
        nonOL_extra = nonOL[annotation_idx]
        o[f'active_{k}'] = int((all_rates > 0).sum())
        o[f'activeCB_{k}'] = int(((r > 0) & nonOL).sum() + ((extra_rates > 0) & nonOL_extra).sum())
        o[f'spikes_{k}'] = float(r[nonOL].sum() + extra_rates[nonOL_extra].sum())
        for grp in ['steerMN_R', 'steerMN_L', 'powerMN_R', 'powerMN_L', 'haltereMN_R', 'haltereMN_L',
                    'legMN_R', 'legMN_L', 'DN_R', 'DN_L', 'LPTC_R', 'LPTC_L', 'probMN', 'GF', 'TTMn', 'DLMn']:
            members = np.asarray(G[grp], dtype=int)
            added = extra_rates[np.isin(annotation_idx, members)].sum()
            o[f'{grp}_{k}'] = float((r[members].sum() + added) / len(members)) if len(members) else 0.0
        if d is not None and k in ('R', 'L', 'F'):
            o[f'y_{k}'] = float(d @ r[steer])
    if d is not None and all(k in o for k in ('y_R', 'y_L', 'y_F')):
        ref = _cache['decoder_ref']            # intact (y_R - y_L)
        o['gain'] = (o['y_R'] - o['y_L']) / ref
        o['bias'] = 2 * (o['y_F'] - _cache['decoder_yF0']) / ref      # turning tendency in straight flight
        o['bias_RL'] = (o['y_R'] + o['y_L']) / ref
        o['gain_R'] = (o['y_R'] - o['y_F']) / _cache['decoder_ref_R']
        o['gain_L'] = (o['y_F'] - o['y_L']) / _cache['decoder_ref_L']
        o['decoder_schema'] = 2
    return o


def set_decoder(rates_R, rates_L, rates_F=None):
    """Define the steering decoder from intact-network responses (mean over seeds).
    The R-minus-L pattern over the 32 wing steering motor neurons is antisymmetrised
    across the midline so that a symmetric command projects to zero. V5 requires
    the intact forward-flow trial for separate directional reference spans.
    """
    if rates_F is None:
        raise ValueError('V5 directional calibration requires intact forward-flow rates')
    c = get_net(); G = c['G']; neu = c['neu']
    steer = np.array(G['steerMN_R'] + G['steerMN_L'])
    types = neu.type.values[steer]; sides = neu.somaSide.values[steer]
    mirror = np.array([np.flatnonzero((types == t) & (sides != s))[0] for t, s in zip(types, sides)])
    diff = rates_R[steer] - rates_L[steer]
    d = (diff - diff[mirror]) / 2.0
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Intact responses do not define a nonzero steering decoder')
    d = d / norm
    _cache['decoder'] = d; _cache['steer_idx'] = steer
    _cache['decoder_ref'] = float(d @ rates_R[steer] - d @ rates_L[steer])
    _cache['decoder_yF0'] = 0.0 if rates_F is None else float(d @ rates_F[steer])
    _cache['decoder_ref_R'] = float(d @ rates_R[steer]) - _cache['decoder_yF0']
    _cache['decoder_ref_L'] = _cache['decoder_yF0'] - float(d @ rates_L[steer])
    _validate_directional_refs(_cache['decoder_ref_R'], _cache['decoder_ref_L'])
    os.makedirs(PROC, exist_ok=True)
    np.savez(PROC + 'decoder.npz', d=d, steer_idx=steer, ref=_cache['decoder_ref'], yF0=_cache['decoder_yF0'], mirror=mirror,
             schema_version=2, gain_definition='state_forward_referenced_directional_intact_span',
             ref_R=_cache['decoder_ref_R'], ref_L=_cache['decoder_ref_L'],
             yR0=float(d @ rates_R[steer]), yL0=float(d @ rates_L[steer]))
    return d


def load_decoder():
    z = np.load(PROC + 'decoder.npz')
    _cache['decoder'] = z['d']; _cache['decoder_ref'] = float(z['ref']); _cache['decoder_yF0'] = float(z['yF0'])
    if not np.isfinite(_cache['decoder_ref']) or _cache['decoder_ref'] <= 0:
        raise ValueError('Decoder reference must be finite and positive')
    _cache['steer_idx'] = z['steer_idx']
    _cache['decoder_schema'] = int(z['schema_version']) if 'schema_version' in z else 1
    if 'ref_R' in z and 'ref_L' in z:
        _cache['decoder_ref_R'], _cache['decoder_ref_L'] = float(z['ref_R']), float(z['ref_L'])
    elif 'yR0' in z and 'yL0' in z:
        _cache['decoder_ref_R'] = float(z['yR0']) - _cache['decoder_yF0']
        _cache['decoder_ref_L'] = _cache['decoder_yF0'] - float(z['yL0'])
    else:
        raise ValueError('Legacy decoder lacks directional references; recalibrate with V5 intact R/L/F rates')
    _validate_directional_refs(_cache['decoder_ref_R'], _cache['decoder_ref_L'])


def _validate_directional_refs(ref_R, ref_L):
    """Require a resolvable, correctly signed response in each direction."""
    tolerance = max(1e-6, 1e-6 * abs(ref_R + ref_L))
    if not np.isfinite([ref_R, ref_L]).all() or min(ref_R, ref_L) <= tolerance:
        raise ValueError('Intact directional responses must both be positive and resolvable relative to forward flow')
