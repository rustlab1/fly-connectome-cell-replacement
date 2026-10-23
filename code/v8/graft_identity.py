"""V8 explicit non-target identity-pool extension. All donor-wiring operations match V5."""
"""
graft_typed.py
--------------
Type-level graft wiring: a graft cell is given a cell-type identity (its 'fate') and is wired like a
surviving neuron of that type, not like the particular neuron that died.

Wiring models compared in the paper:
  copy       each graft cell copies the exact partners of the lost neuron it replaces (experiments.make_graft,
             specificity 1; the upper bound: a transplanted cell cannot know which cell died)
  typed      each graft cell receives a fate; its partners are sampled from a surviving neuron of that type
             (a 'donor'): a same-side cell of the type, or the contralateral homologue with every partner
             mirrored across the midline (same type, opposite side, nearest mirrored soma position).
             Partners that died are redirected to the graft cells that replace them; partners that cannot
             be mapped are dropped. fate_accuracy is the fraction of graft cells whose fate is the type of
             the lost cell they stand in for; the others receive the type of a random other lost cell
             (misfated) and are wired like that type.
  proximity  no fate: the synapse budget of the lost cell is spent on random surviving neurons around the
             lesion (experiments.make_graft with specificity 0)

The mirror map is built once from the connectome annotation (type, soma side, soma position) and cached
in data/processed/mirror_map.npy.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
import experiments as ex

X_MID = 389.8   # midline (µm), median midpoint of the left and right soma clusters of 10,800 paired types


def build_mirror_map(force=False):
    path = ex.PROC + 'mirror_map.npy'
    if os.path.exists(path) and not force:
        return np.load(path)
    c = ex.get_net(); neu = c['neu']; N = len(neu)
    xyz = neu[['x_um', 'y_um', 'z_um']].values.astype(float)
    mirror = np.full(N, -1, np.int64)
    rng = np.random.default_rng(0)
    side = neu.somaSide.values; typ = neu.type.values
    mirror[side == 'M'] = np.flatnonzero(side == 'M')
    for t, sub in neu[neu.type.notna() & np.isin(side, ['L', 'R'])].groupby('type'):
        ii = sub.index.values; ss = side[ii]
        for a, b in (('L', 'R'), ('R', 'L')):
            src = ii[ss == a]; dst = ii[ss == b]
            if len(src) == 0 or len(dst) == 0:
                continue
            dpos = xyz[dst]; ok_d = ~np.isnan(dpos[:, 0])
            if ok_d.sum() >= 1:
                tree = cKDTree(dpos[ok_d]); dst_ok = dst[ok_d]
                spos = xyz[src].copy(); spos[:, 0] = 2 * X_MID - spos[:, 0]
                has = ~np.isnan(spos[:, 0])
                if has.any():
                    _, j = tree.query(spos[has]); mirror[src[has]] = dst_ok[j]
                if (~has).any():
                    mirror[src[~has]] = rng.choice(dst, (~has).sum())
            else:
                mirror[src] = rng.choice(dst, len(src))
    np.save(path, mirror)
    return mirror


def type_sign(neu):
    """majority sign per type (dict), used for the sign of a graft cell of a given fate"""
    s = neu.groupby('type').sign.agg(lambda v: 1 if (v > 0).mean() >= 0.5 else -1)
    return s.to_dict()


def make_graft_typed(lesion_idx, n_graft, templates, fate_accuracy=1.0, seed=0, center_um=None,
                     local_radius_um=60.0, strength=1.0, survival=1.0, strength_sd=0.0, identity_pool=None):
    """Wire n_graft graft cells by cell-type identity. templates: the lost neurons the graft cells stand in
    for (in order; decides fate when the cell is correctly fated). Internal edges
    proposed by both source and target donors use the mean of the two directional
    proposals, not their sum. A single donor proposal is retained at its weight.
    Repeated templates produce distinct cell pairs with full per-pair weight.
    Returns E (N_tot x N_tot CSC), graft signs, surviving templates, and statistics.
    E also carries sensory eligibility and population annotation metadata.
    """
    c = ex.get_net(); W = c['W']; neu = c['neu']; N = W.shape[0]; rng = np.random.default_rng(seed)
    Wcsr = ex._cache.setdefault('Wcsr', W.tocsr())
    mirror = build_mirror_map(); tsign = ex._cache.setdefault('type_sign', type_sign(neu))
    lesion_idx = np.asarray(lesion_idx); dead = np.zeros(N, bool); dead[lesion_idx] = True
    typ = neu.type.values; side = neu.somaSide.values; sign_t = neu.sign.values
    templ = np.asarray(templates, dtype=int)[:n_graft]
    if n_graft < 0 or len(templ) != n_graft or not np.isin(templ, lesion_idx).all():
        raise ValueError('Templates must provide the requested number of lost neurons')
    if not 0 <= fate_accuracy <= 1 or not 0 <= survival <= 1:
        raise ValueError('Fate accuracy and survival must lie between zero and one')
    if survival < 1.0:
        templ = templ[rng.random(len(templ)) < survival]
    n_graft = len(templ); N_tot = N + n_graft
    # fate: the type of the lost cell (correct) or the type of a random other lost cell (misfated)
    lost_types = typ[lesion_idx]; lost_types = lost_types[pd.notna(lost_types)]
    intended_correct = rng.random(n_graft) < fate_accuracy
    if identity_pool is not None:
        identity_pool=np.asarray(identity_pool,dtype=object)
        assert len(identity_pool)>0 and pd.notna(identity_pool).all()
        assert not np.isin(identity_pool,lost_types).any()
        fate=rng.choice(identity_pool,size=n_graft,replace=True)
    else:
        fate = []
        for i, t in enumerate(templ):
            alternatives = lost_types[lost_types != typ[t]]
            if intended_correct[i] and pd.notna(typ[t]):
                fate.append(typ[t])
            elif len(alternatives):
                fate.append(rng.choice(alternatives))
            else:
                fate.append(typ[t])
    fate = np.asarray(fate, dtype=object)
    correct = np.array([pd.notna(f) and f == typ[t] for f, t in zip(fate, templ)], dtype=bool)
    g_sign = np.array([tsign.get(f, sign_t[t]) for f, t in zip(fate, templ)], np.int8)
    sign_ext = np.concatenate([sign_t, g_sign])
    # lost neuron -> graft cells standing in for it (dead partners of a donor are redirected there)
    replaced_by = {}
    for gi, t in enumerate(templ):
        replaced_by.setdefault(int(t), []).append(N + gi)
    # donors: surviving neurons per type
    alive_idx = np.flatnonzero(~dead)
    donors = pd.Series(alive_idx).groupby(typ[alive_idx]).apply(lambda v: v.values).to_dict()
    # proximity pool for cells without a donor
    xyz = neu[['x_um', 'y_um', 'z_um']].values
    if center_um is None:
        center_um = np.nanmean(xyz[lesion_idx], axis=0)
    pool = c['soma_idx'][c['tree'].query_ball_point(center_um, local_radius_um)]; pool = pool[~dead[pool]]
    s_cell = strength * np.exp(strength_sd * rng.standard_normal(n_graft)) if strength_sd > 0 else np.full(n_graft, float(strength))
    R, C, V = [], [], []
    internal = {}
    annotation_indices = templ.copy()
    # Population labels follow assigned fate even if that fate has no surviving
    # wiring donor and therefore falls back to proximity connections.
    for gi, (t, f) in enumerate(zip(templ, fate)):
        if not correct[gi]:
            matches = np.flatnonzero(typ == f)
            same_side = matches[side[matches] == side[t]]
            if len(matches):
                annotation_indices[gi] = int(same_side[0] if len(same_side) else matches[0])
    stats = dict(n_cells=n_graft, n_correct_fate=int(correct.sum()), n_with_donor=0, n_mirror_donor=0, n_proximity=0, syn_kept=0, syn_total=0)

    def add(r, cc, v):
        R.append(np.asarray(r, np.int64)); C.append(np.asarray(cc, np.int64)); V.append(np.asarray(v, np.float32))

    def connect(g, partners, weights, direction):
        """connect graft cell g with (mapped) host partners; dead partners go to their graft replacements"""
        kept = 0
        for p_, w_ in zip(partners, weights):
            if p_ < 0:
                continue
            targets = [int(p_)] if not dead[p_] else replaced_by.get(int(p_), [])
            for q in targets:
                if q == g:
                    continue
                post, pre = (g, q) if direction == 'in' else (q, g)
                if post >= N and pre >= N:
                    proposals = internal.setdefault((post, pre), {})
                    proposals[direction] = proposals.get(direction, 0.0) + float(w_)
                else:
                    add([post], [pre], [w_ * sign_ext[pre]])
                kept += 1
        return kept

    for gi, (t, f) in enumerate(zip(templ, fate)):
        g = N + gi
        cand = donors.get(f, np.zeros(0, np.int64))
        if len(cand) == 0:
            # no surviving cell of this type: proximity wiring with the synapse budget of the lost cell
            stats['n_proximity'] += 1
            pre = Wcsr.indices[Wcsr.indptr[t]:Wcsr.indptr[t + 1]]; win = np.abs(Wcsr.data[Wcsr.indptr[t]:Wcsr.indptr[t + 1]])
            post = W.indices[W.indptr[t]:W.indptr[t + 1]]; wout = np.abs(W.data[W.indptr[t]:W.indptr[t + 1]])
            if len(pool):
                rp = rng.choice(pool, len(pre)); add(np.full(len(pre), g), rp, win * sign_t[rp])
                rq = rng.choice(pool, len(post)); add(rq, np.full(len(post), g), wout * g_sign[gi])
            stats['syn_total'] += int(win.sum() + wout.sum()); stats['syn_kept'] += int(win.sum() + wout.sum())
            continue
        stats['n_with_donor'] += 1
        same_side = cand[side[cand] == side[t]]
        d = int(rng.choice(same_side if len(same_side) else cand))
        use_mirror = side[d] != side[t] and side[t] in ('L', 'R') and side[d] in ('L', 'R')
        stats['n_mirror_donor'] += int(use_mirror)
        pre = Wcsr.indices[Wcsr.indptr[d]:Wcsr.indptr[d + 1]]; win = np.abs(Wcsr.data[Wcsr.indptr[d]:Wcsr.indptr[d + 1]])
        post = W.indices[W.indptr[d]:W.indptr[d + 1]]; wout = np.abs(W.data[W.indptr[d]:W.indptr[d + 1]])
        if use_mirror:
            pre = mirror[pre]; post = mirror[post]
        if not correct[gi]:
            annotation_indices[gi] = mirror[d] if use_mirror and mirror[d] >= 0 else d
        stats['syn_total'] += int(win.sum() + wout.sum())
        stats['syn_kept'] += int(win[pre >= 0].sum() + wout[post >= 0].sum())
        connect(g, pre, win, 'in'); connect(g, post, wout, 'out')
    stats['internal_edges_reconciled'] = sum(len(proposals) == 2 for proposals in internal.values())
    for (post, pre), proposals in internal.items():
        add([post], [pre], [np.mean(list(proposals.values())) * sign_ext[pre]])
    if R:
        Rr, Cc, Vv = np.concatenate(R), np.concatenate(C), np.concatenate(V)
        strengths = np.concatenate([np.ones(N), s_cell])
        fp, fq = strengths[Rr], strengths[Cc]
        both = (Rr >= N) & (Cc >= N); fac = np.where(both, np.sqrt(fp * fq), fp * fq).astype(np.float32)
        E = sp.csc_matrix((Vv * fac, (Rr, Cc)), shape=(N_tot, N_tot)); E.sum_duplicates()
    else:
        E = sp.csc_matrix((N_tot, N_tot), dtype=np.float32)
    stats['syn_kept_frac'] = round(stats['syn_kept'] / max(stats['syn_total'], 1), 3)
    E.assigned_types = fate.copy()
    E.stimulus_eligible = correct
    E.annotation_indices = annotation_indices
    return E, g_sign, templ, stats
