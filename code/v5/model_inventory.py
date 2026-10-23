"""Read-only model census and focal-lesion composition for V5 Methods.

This script loads annotations/connectivity and queries soma positions. It does
not run neural or flight simulations and does not modify the model inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
import experiments as ex
from adaptation import ranked_templates

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def counts(series):
    return {str(k): int(v) for k, v in series.fillna('<unclassified>').value_counts().sort_index().items()}


def main():
    net = ex.get_net(); neu = net['neu']; W = net['W']; groups = net['G']
    positions = neu[['x_um', 'y_um', 'z_um']].to_numpy()
    soma = np.isfinite(positions).all(axis=1)
    motor = neu.superclass.isin(['vnc_motor', 'cb_motor']).to_numpy()
    eligible = soma & ~motor
    sign = neu.sign.to_numpy()
    nt_sign = pd.crosstab(neu.consensus_nt.fillna('<unclassified>'), neu.sign)
    lptc = np.asarray(groups['LPTC_R'] + groups['LPTC_L'], dtype=int)
    descending = np.flatnonzero((neu.superclass == 'descending_neuron').to_numpy())
    core = np.union1d(lptc, descending)
    interface = np.flatnonzero(neu.superclass.isin(['descending_neuron', 'visual_projection', 'visual_centrifugal']).to_numpy())
    t4t5 = np.flatnonzero(neu.type.fillna('').str.match(r'^T[45][abcd]$').to_numpy())
    sites = pd.read_csv(Path(ex.RES) / 'lesion_sites.csv').set_index('site')
    lesions = []
    for site in (20, 25):
        row = sites.loc[site]
        center = [float(row.x), float(row.y), float(row.z)]
        for size in (1000, 2000, 4000, 16000):
            ix, radius = ex.make_lesion(center, size)
            selected = neu.iloc[ix]
            lesions.append(dict(site=site, n_lesion=size, center_um=center, radius_um=float(radius),
                                n_types=int(selected.type.nunique()), n_without_type=int(selected.type.isna().sum()),
                                superclass_counts=counts(selected.superclass), side_counts=counts(selected.somaSide),
                                neurotransmitter_counts=counts(selected.consensus_nt),
                                n_excitatory=int((selected.sign > 0).sum()), n_inhibitory=int((selected.sign < 0).sum()),
                                n_T4T5=int(np.isin(ix, t4t5).sum()), n_LPTC=int(np.isin(ix, lptc).sum()),
                                n_descending=int(np.isin(ix, descending).sum()), n_core=int(np.isin(ix, core).sum()),
                                n_interface=int(np.isin(ix, interface).sum()), n_motor=int(motor[ix].sum()),
                                selected_indices_sha256=hashlib.sha256(np.asarray(ix, dtype='<i8').tobytes()).hexdigest()))
    core_first_100 = []
    focal = sites.loc[20]
    focal_indices, _ = ex.make_lesion([focal.x, focal.y, focal.z], 2000)
    for seed in (0, 1, 2):
        selected = ranked_templates(focal_indices, seed)[:100]
        core_first_100.append(dict(seed=seed, n_graft=len(selected),
                                  n_LPTC=int(np.isin(selected, lptc).sum()),
                                  n_descending=int(np.isin(selected, descending).sum()),
                                  n_core=int(np.isin(selected, core).sum()),
                                  superclass_counts=counts(neu.iloc[selected].superclass),
                                  template_indices_sha256=hashlib.sha256(np.asarray(selected, dtype='<i8').tobytes()).hexdigest()))
    inventory = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='annotation/connectivity inventory only; no simulations',
                     n_neurons=len(neu), n_connections=int(W.nnz), exact_unsigned_synapses=int(np.abs(W.data.astype(np.float64)).sum()),
                     n_cell_type_labels=int(neu.type.nunique()), n_cells_without_type=int(neu.type.isna().sum()),
                     n_superclass_labels=int(neu.superclass.nunique()), n_class_labels=int(neu['class'].nunique()),
                     n_finite_somata=int(soma.sum()), n_x_position_present=int(neu.x_um.notna().sum()),
                     n_motoneurons=int(motor.sum()), n_motoneurons_with_soma=int((motor & soma).sum()),
                     n_lesion_eligible=int(eligible.sum()), n_lesion_pool_runtime=len(net['lesion_idx_pool']),
                     n_exc=int((sign > 0).sum()), n_inh=int((sign < 0).sum()), superclass_counts=counts(neu.superclass),
                     class_counts=counts(neu['class']), neurotransmitter_counts=counts(neu.consensus_nt),
                     neurotransmitter_sign_counts={str(k): {str(sign): int(v) for sign, v in row.items()} for k, row in nt_sign.iterrows()},
                     group_counts={name: len(values) for name, values in sorted(groups.items())},
                     stimulus_counts={name: len(ix) for name, ix in ex.stimuli(True).items()},
                     n_unique_driven_neurons=len(np.unique(np.concatenate(list(ex.stimuli(True).values())))),
                     n_LPTC=len(lptc), n_descending_total=len(descending), n_core=len(core), n_interface=len(interface),
                     W_SYN_mV=float(ex.W_SYN), stimulus_Hz=float(ex.STIM_RATE), trial_ms=ex.T_SIM,
                     measurement_ms=list(ex.T_MEAS), lesion_composition=lesions, site20_2000_core_first_100=core_first_100,
                     runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))
    original_neurons = pd.read_parquet(ROOT / 'data/processed/neurons.parquet')
    inventory['original_runtime_neuron_tables_equal'] = original_neurons.equals(neu)
    specs = json.loads((Path(ex.RES) / 'specs_states.json').read_text())
    inventory['static_design'] = dict(n_states=len(specs),
                                      category_counts=counts(pd.Series([s['category'] for s in specs])))
    paths = ['data/processed/neurons.parquet', 'data/v5/neurons.parquet', 'data/v5/W_csc.npz',
             'data/v5/cell_groups.json', 'data/v5/decoder.npz', 'results/v5/lesion_sites.csv',
             'code/01_prepare_data.py', 'code/v5/restore_connectivity.py', 'code/v5/flysim.py',
             'code/v5/flysim_fast.py', 'code/v5/experiments.py', 'code/v5/graft_typed.py',
             'code/v5/adaptation.py', 'code/v5/adaptation_protocol.py', 'code/v5/calibrate.py',
             'code/v5/protocol.py', 'code/v5/run_states.py', 'code/v5/summarize.py',
             'code/v5/flight.py', 'code/v5/flight_fast.py', 'results/v5/specs_states.json',
             'results/v5/protocol.json', 'tests/v5/test_model.py', 'tests/v5/test_adaptation.py',
             'code/v5/model_inventory.py']
    inventory['source_hashes'] = {path: sha256(ROOT / path) for path in paths}
    with np.load(ROOT / 'data/v5/decoder.npz', allow_pickle=False) as decoder:
        inventory['decoder'] = {key: np.asarray(decoder[key]).tolist() for key in ['schema_version', 'gain_definition', 'ref', 'ref_R', 'ref_L', 'yF0', 'yR0', 'yL0']}
    path = Path(ex.RES) / 'model_inventory.json'
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({key: value for key, value in inventory.items() if key not in ['source_hashes', 'class_counts', 'neurotransmitter_sign_counts', 'lesion_composition']}, indent=2))
    print(pd.DataFrame(lesions)[['site', 'n_lesion', 'radius_um', 'n_types', 'n_T4T5', 'n_LPTC', 'n_descending', 'n_core', 'n_interface', 'n_motor']].to_string(index=False))


if __name__ == '__main__':
    main()
