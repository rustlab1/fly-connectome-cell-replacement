"""
run_states.py  <specs.json>  <out.jsonl>  [n_workers]
Run a list of network states (intact / lesion / lesion+graft) in parallel and
append one JSON line of read-outs per state to <out.jsonl>.
Each spec: {"name":..., "seed":0, "lesion":{"center":[x,y,z],"n":1000} | null,
            "graft":{"n":500,"specificity":1.0,"composition":"matched","strength":1.0,
                     "local_radius":60.0,"seed":0} | null}
"""
import sys, json, os, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import experiments as ex


def prepare(spec):
    ex.get_net(); ex.load_decoder()
    lesion_idx = None; radius = None; center = None
    if spec.get('lesion'):
        center = spec['lesion']['center']
        lesion_idx, radius = ex.make_lesion(center, spec['lesion']['n'])
    graft = None
    if spec.get('graft'):
        g = spec['graft']
        templates = None
        if g.get('targeted') == 'ranked':
            # replace the optomotor core first: lost lobula plate tangential cells (HS, H2, VS) and descending
            # neurons; then the remaining visual projection and centrifugal neurons; then the rest
            net = ex.get_net(); neu = net['neu']; rng = np.random.default_rng(1000 + g.get('seed', 0))
            sc = neu.superclass.values[lesion_idx]
            lptc = np.isin(lesion_idx, np.array(net['G']['LPTC_R'] + net['G']['LPTC_L']))
            rank1 = lptc | (sc == 'descending_neuron')
            rank2 = ~rank1 & np.isin(sc, ['visual_projection', 'visual_centrifugal'])
            rest = ~rank1 & ~rank2
            templates = np.concatenate([rng.permutation(lesion_idx[rank1]), rng.permutation(lesion_idx[rank2]), rng.permutation(lesion_idx[rest])])
            if g.get('copies', 1) > 1:
                # k copies of each lost cell in rank order (core cells first, each repeated k times)
                templates = np.repeat(templates, int(g['copies']))
        elif g.get('targeted'):
            # replace the sensory-to-premotor interface first: descending, visual projection and visual
            # centrifugal neurons among the lost cells (random order within the set), then the rest
            neu = ex.get_net()['neu']; rng = np.random.default_rng(1000 + g.get('seed', 0))
            sc = neu.superclass.values[lesion_idx]
            prio = np.isin(sc, ['descending_neuron', 'visual_projection', 'visual_centrifugal'])
            templates = np.concatenate([rng.permutation(lesion_idx[prio]), rng.permutation(lesion_idx[~prio])])
        wiring = g.get('wiring', 'copy'); gstats = None
        if wiring == 'typed':
            # type-level wiring: fate from the lost cell (fate_accuracy) and partners from a surviving donor of that type
            import graft_typed as gt
            if templates is None:
                rng = np.random.default_rng(g.get('seed', 0)); templates = rng.choice(lesion_idx, min(g['n'], len(lesion_idx)), replace=False)
            E, gs, templ, gstats = gt.make_graft_typed(lesion_idx, g['n'], templates, fate_accuracy=g.get('fate_accuracy', 1.0),
                                                       seed=g.get('seed', 0), center_um=center, local_radius_um=g.get('local_radius', 2.0 * radius),
                                                       strength=g.get('strength', 1.0), survival=g.get('survival', 1.0), strength_sd=g.get('strength_sd', 0.0))
            graft = (E, gs, templ)
        else:
            graft = ex.make_graft(lesion_idx, g['n'], specificity=(0.0 if wiring == 'proximity' else g.get('specificity', 1.0)),
                                  composition=g.get('composition', 'matched'), strength=g.get('strength', 1.0),
                                  local_radius_um=g.get('local_radius', 2.0 * radius), seed=g.get('seed', 0),
                                  center_um=center, templates=templates, survival=g.get('survival', 1.0), strength_sd=g.get('strength_sd', 0.0),
                                  strength_mean_norm=g.get('strength_mean_norm', False))
    outdir = Path(ex.RES) / 'condition_indices'
    outdir.mkdir(exist_ok=True)
    label = f"{spec['name']}_s{spec.get('seed', 0)}"
    if lesion_idx is not None:
        np.save(outdir / f'{label}_lesion.npy', lesion_idx)
    if graft is not None:
        np.save(outdir / f'{label}_templates.npy', graft[2])
    return lesion_idx, radius, graft, gstats if graft is not None else None


def work(spec):
    lesion_idx, radius, graft, gstats = prepare(spec)
    gain = None
    if spec.get('gain_file'):
        gain = np.load(spec['gain_file'])
    out = ex.run_state(spec['name'], lesion_idx=lesion_idx, graft=graft, seed=spec.get('seed', 0),
                       save_rates=spec.get('save_rates', True), with_feed=spec.get('feed', False), gain_in=gain,
                       stim_rate=spec.get('stim_rate', ex.STIM_RATE), extra_params=spec.get('extra_params'))
    out['spec'] = spec
    if radius is not None:
        out['lesion_radius_um'] = radius
    if graft is not None:
        out['graft_nnz'] = int(graft[0].nnz)
        out['graft_frac_E'] = float((graft[1] > 0).mean())
        out['n_transplanted'] = int(spec['graft']['n'])
        out['n_graft'] = int(len(graft[2]))
        if gstats is not None:
            out['graft_stats'] = gstats
    return out


if __name__ == '__main__':
    specs = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line); done.add((d['name'], d['seed']))
    todo = [s for s in specs if (s['name'], s.get('seed', 0)) not in done]
    print(f'{len(todo)} states to run ({len(done)} already done)', flush=True)
    t0 = time.time()
    with Pool(n_workers) as pool, open(out_path, 'a') as f:
        for i, out in enumerate(pool.imap_unordered(work, todo)):
            f.write(json.dumps(out) + '\n'); f.flush()
            print(f'[{i+1}/{len(todo)}] {out["name"]} gain={out.get("gain", float("nan")):.2f} '
                  f'bias={out.get("bias", float("nan")):.2f} activeCB_R={out["activeCB_R"]}  '
                  f'({(time.time()-t0)/60:.1f} min)', flush=True)
