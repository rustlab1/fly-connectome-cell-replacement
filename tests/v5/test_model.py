"""Synthetic scientific invariants for the V5 simulator and graft construction."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'v5'))
import experiments as ex
import flysim
import graft_typed as gt


class PulseRng:
    def __init__(self, events):
        self.events, self.step = events, -1

    def random(self, size):
        self.step += 1
        result = np.ones(size)
        for slot in self.events.get(self.step, []):
            result[slot] = 0.0
        return result


def dense_reference(W, T, stimuli, events):
    """All-neuron reference without activity pruning or sparse propagation."""
    p = flysim.PARAMS
    dt = p['dt']; size = len(W); delay = round(p['t_delay'] / dt)
    voltage = np.zeros(size); current = np.zeros(size); refractory = np.zeros(size, int)
    pending = np.zeros((delay, size)); spikes = []
    membrane_decay = np.exp(-dt / p['tau_m']); synapse_decay = np.exp(-dt / p['tau_s'])
    factor = p['tau_s'] / (p['tau_s'] - p['tau_m'])
    for step in range(round(T / dt)):
        accepting = refractory == 0
        current[accepting] += pending[step % delay, accepting]
        pending[step % delay] = 0
        free = refractory == 0
        voltage[free] = voltage[free] * membrane_decay + current[free] * factor * (synapse_decay - membrane_decay)
        current[free] *= synapse_decay
        refractory[~free] -= 1
        voltage[np.asarray(stimuli)[events.get(step, [])]] += p['w_poi']
        fired = np.flatnonzero(voltage > p['v_th'])
        voltage[fired] = 0; current[fired] = 0
        refractory[fired] = round(p['t_ref'] / dt)
        refractory[np.intersect1d(fired, stimuli)] = 0
        for ix in fired:
            spikes.append((step * dt, int(ix)))
        pending[step % delay] += W[:, fired].sum(axis=1) * p['w_syn']
    return voltage, current, spikes


def fixture(W, signs=None, types=None, sides=None):
    W = sp.csc_matrix(np.asarray(W, np.float32)); n = W.shape[0]
    neu = pd.DataFrame({'x_um': np.arange(n, dtype=float), 'y_um': np.zeros(n), 'z_um': np.zeros(n),
                        'sign': signs if signs is not None else np.ones(n, int),
                        'type': types if types is not None else [f't{i}' for i in range(n)],
                        'somaSide': sides if sides is not None else ['R'] * n})
    groups = {name: [n - 1] for name in ['steerMN_R', 'steerMN_L', 'powerMN_R', 'powerMN_L',
             'haltereMN_R', 'haltereMN_L', 'legMN_R', 'legMN_L', 'DN_R', 'DN_L',
             'LPTC_R', 'LPTC_L', 'probMN', 'GF', 'TTMn', 'DLMn']}
    ex._cache.clear()
    ex._cache.update(W=W, neu=neu, G=groups, nonOL=np.ones(n, bool), soma_idx=np.arange(n),
                     tree=cKDTree(neu[['x_um', 'y_um', 'z_um']].values))
    return W, neu


class TestModel(unittest.TestCase):
    def tearDown(self):
        ex._cache.clear()

    def test_signed_decay_and_spiking_match_dense_reference(self):
        dense = np.asarray([[0, 0, 0], [-100, 0, 200], [0, 0, 0]], np.float32)
        events = {0: [0], 500: [1]}
        with patch.object(np.random, 'default_rng', return_value=PulseRng(events)):
            result = flysim.LIFNetReference(sp.csc_matrix(dense)).run(100, {100: np.array([0, 2])},
                       t_measure=(0, 100), record_spikes=True, return_state=True)
        v, g, spikes = dense_reference(dense, 100, [0, 2], events)
        np.testing.assert_allclose(result['voltage'], v, atol=1e-3)
        np.testing.assert_allclose(result['synaptic_state'], g, atol=1e-3)
        observed = list(zip(result['spike_t'], result['spike_i']))
        np.testing.assert_allclose(np.asarray(observed), np.asarray(spikes), atol=1e-5)
        self.assertGreater(result['rates'][1], 0)

    def test_excitation_and_inhibition_arriving_during_refractory_are_discarded(self):
        for second_weight in (1000., -1000.):
            with self.subTest(second_weight=second_weight):
                # Cell 0 drives a target spike at 2.3 ms. Cell 1's delayed event
                # arrives at 2.8 ms, inside the target's 2.2 ms refractory period.
                dense = np.zeros((3, 3), np.float32)
                dense[2, 0] = 1000; dense[2, 1] = second_weight
                events = {0: [0], 10: [1]}
                with patch.object(np.random, 'default_rng', return_value=PulseRng(events)):
                    result = flysim.LIFNetReference(sp.csc_matrix(dense)).run(10, {100: np.array([0, 1])},
                                     t_measure=(0, 10), record_spikes=True, return_state=True)
                target_times = result['spike_t'][result['spike_i'] == 2]
                np.testing.assert_allclose(target_times, [2.3], atol=1e-6)
                self.assertEqual(result['synaptic_state'][2], 0)
                self.assertEqual(result['voltage'][2], 0)

    def test_exact_reconstruction_including_internal_edges_and_autapse(self):
        W, _ = fixture([[2, 0, 3], [7, -5, 0], [0, -4, 0]], [1, -1, 1])
        E, signs, templ = ex.make_graft([0, 1], 2, templates=[0, 1])
        combined = sp.block_diag([W, sp.csc_matrix((2, 2))], format='csc') + E
        relabel = np.array([3, 4, 2])
        np.testing.assert_array_equal(combined[relabel][:, relabel].toarray(), W.toarray())
        self.assertEqual(E[4, 3], 7)
        self.assertEqual(E[3, 3], 2)

    def test_partial_replacement_omits_unreplaced_neurons(self):
        fixture([[0, 0, 3], [7, 0, 0], [0, -4, 0]], [1, -1, 1])
        E, _, _ = ex.make_graft([0, 1], 1, templates=[0])
        self.assertEqual(E[3, 2], 3)
        self.assertEqual(E[:2].nnz, 0)
        self.assertEqual(E[:, :2].nnz, 0)

    def test_repeated_templates_have_one_full_edge_per_cell_pair(self):
        fixture([[0, 0, 3], [7, 0, 0], [0, -4, 0]], [1, -1, 1])
        E, _, _ = ex.make_graft([0, 1], 4, templates=[0, 0, 1, 1])
        np.testing.assert_array_equal(E[[5, 6]][:, [3, 4]].toarray(), np.full((2, 2), 7))
        np.testing.assert_array_equal(E[[3, 4], 2].toarray(), np.full((2, 1), 3))

    def test_zero_survival_is_empty_graft(self):
        fixture([[0, 0], [7, 0]])
        E, signs, templ = ex.make_graft([0], 1, templates=[0], survival=0)
        self.assertEqual(E.shape, (2, 2)); self.assertEqual(E.nnz, 0)
        self.assertEqual(len(signs), 0); self.assertEqual(len(templ), 0)

    def test_exact_replacement_preserves_input_streams_population_readouts_and_rates(self):
        W, _ = fixture([[0, 0, 0], [0, 0, 0], [200, 250, 0]])
        original_stimuli = {'R': np.array([0]), 'L': np.array([1]), 'F': np.array([0, 1])}
        with patch.object(ex, 'stimuli', return_value=original_stimuli):
            intact, intact_rates = ex.run_state('intact', seed=2, save_rates=False, return_rates=True)
            graft = ex.make_graft([0, 1], 2, templates=[1, 0])
            np.testing.assert_array_equal(ex.graft_stimulus_indices([0, 1], [0, 1], graft), [4, 3])
            restored, restored_rates = ex.run_state('restored', lesion_idx=[0, 1], graft=graft,
                                                   seed=2, save_rates=False, return_rates=True)
        for key in original_stimuli:
            np.testing.assert_array_equal(restored_rates[key][[4, 3, 2]], intact_rates[key])
            for metric in ['active', 'activeCB', 'spikes', 'DN_R']:
                self.assertEqual(restored[f'{metric}_{key}'], intact[f'{metric}_{key}'])

    def test_removing_an_input_preserves_surviving_input_events(self):
        matrix = sp.csc_matrix((2, 2), dtype=np.float32)
        stim = {100: np.array([0, 1])}
        intact = flysim.LIFNet(matrix).run(300, stim, seed=17, t_measure=(0, 300), record_spikes=True)
        injured = flysim.LIFNet(matrix).run(300, stim, seed=17, dead=np.array([True, False]),
                                          t_measure=(0, 300), record_spikes=True)
        np.testing.assert_array_equal(intact['spike_t'][intact['spike_i'] == 1],
                                      injured['spike_t'][injured['spike_i'] == 1])
        self.assertFalse((injured['spike_i'] == 0).any())

    def test_partial_input_replacement_preserves_other_original_slots(self):
        fixture(np.zeros((3, 3)))
        graft = ex.make_graft([0, 1], 1, templates=[1])
        np.testing.assert_array_equal(ex.graft_stimulus_indices([0, 1, 2], [0, 1], graft), [0, 3, 2])

    def test_directional_decoder_uses_own_forward_response_and_calibrated_refs(self):
        _, neu = fixture([[0, 0], [0, 0]], types=['wing', 'wing'], sides=['R', 'L'])
        ex._cache['G']['steerMN_R'] = [0]; ex._cache['G']['steerMN_L'] = [1]
        rates = {'R': np.array([5., 1.]), 'L': np.array([1., 7.]), 'F': np.array([2., 1.])}
        with tempfile.TemporaryDirectory() as tmp, patch.object(ex, 'PROC', tmp + os.sep):
            ex.set_decoder(rates['R'], rates['L'], rates['F'])
            ex.load_decoder()
            result = ex.readouts(rates)
            self.assertAlmostEqual(result['gain_R'], 1)
            self.assertAlmostEqual(result['gain_L'], 1)
            self.assertAlmostEqual(result['bias'], 0)
            constant = ex.readouts({key: np.array([5., 1.]) for key in rates})
            self.assertAlmostEqual(constant['gain_R'], 0)
            self.assertAlmostEqual(constant['gain_L'], 0)
        with self.assertRaises(ValueError):
            ex._validate_directional_refs(0., 1.)

    def test_typed_internal_proposals_are_averaged(self):
        # Donor B proposes A->B with weight 10 via its input. Donor A proposes
        # the same replaced pair with weight 6 via its output. The edge is 8.
        dense = np.zeros((4, 4)); dense[1, 2] = 6; dense[3, 0] = 10
        fixture(dense, types=['A', 'B', 'A', 'B'])
        with patch.object(gt, 'build_mirror_map', return_value=np.arange(4)):
            E, _, _, stats = gt.make_graft_typed([0, 1], 2, [0, 1])
        self.assertEqual(E[5, 4], 8)
        self.assertEqual(stats['internal_edges_reconciled'], 1)
        self.assertTrue(E.stimulus_eligible.all())

    def test_typed_misfates_do_not_inherit_template_stimulus(self):
        fixture(np.zeros((4, 4)), types=['A', 'B', 'A', 'B'])
        with patch.object(gt, 'build_mirror_map', return_value=np.arange(4)):
            E, signs, templ, stats = gt.make_graft_typed([0, 1], 2, [0, 1], fate_accuracy=0)
        self.assertEqual(stats['n_correct_fate'], 0)
        # Original dead slots remain to consume matching random draws, but no
        # misfated replacement inherits either sensory slot.
        np.testing.assert_array_equal(ex.graft_stimulus_indices([0, 1], [0, 1], (E, signs, templ)), [0, 1])

    def test_typed_no_donor_population_annotation_follows_fate(self):
        fixture(np.zeros((3, 3)), types=['A', 'B', 'C'])
        with patch.object(gt, 'build_mirror_map', return_value=np.arange(3)):
            E, _, _, stats = gt.make_graft_typed([0, 1], 2, [0, 1], fate_accuracy=0)
        self.assertEqual(stats['n_proximity'], 2)
        np.testing.assert_array_equal(E.annotation_indices, [1, 0])

    def test_compiled_backend_matches_reference_for_mixed_graft_gain_state(self):
        if (np.zeros(1, np.float32) * np.float64(.99)).dtype != np.dtype(np.float64):
            self.skipTest('Compiled backend is validated for NumPy 2 scalar promotion')
        try:
            from flysim_fast import LIFNetCompiled
        except ImportError:
            self.skipTest('Optional Numba backend unavailable')
        rng = np.random.default_rng(717)
        signs = np.array([1, -1, 1, 1, -1, 1, 1, -1])
        dense = rng.integers(50, 500, size=(8, 8)).astype(np.float32) * signs[None, :]
        dense[rng.random(dense.shape) < .8] = 0
        W, _ = fixture(dense, signs)
        E, gs, templ = ex.make_graft([0, 1, 2], 3, templates=[1, 2, 0])
        kwargs = dict(n_extra=3, extra_csc=E, gain_in=rng.uniform(.5, 2, 11).astype(np.float32),
                      pre_sign=np.r_[signs, gs])
        dead = np.zeros(11, bool); dead[:3] = True
        ix = ex.graft_stimulus_indices(np.arange(4), [0, 1, 2], (E, gs, templ))
        outputs = [cls(W, **kwargs).run(100, {150: ix}, seed=12, dead=dead,
                         t_measure=(0, 100), record_spikes=True, return_state=True)
                   for cls in (flysim.LIFNetReference, LIFNetCompiled)]
        for key in outputs[0]:
            np.testing.assert_array_equal(outputs[0][key], outputs[1][key], err_msg=key)

    def test_factory_retains_explicit_reference_and_auto_backend(self):
        W = sp.csc_matrix((2, 2), dtype=np.float32)
        self.assertIs(type(flysim.LIFNet(W, backend='reference')), flysim.LIFNetReference)
        model = flysim.LIFNet(W, backend='auto')
        if (np.zeros(1, np.float32) * np.float64(.99)).dtype == np.dtype(np.float64):
            try:
                from flysim_fast import LIFNetCompiled
            except ImportError:
                self.assertIs(type(model), flysim.LIFNetReference)
            else:
                self.assertIs(type(model), LIFNetCompiled)
        else:
            self.assertIs(type(model), flysim.LIFNetReference)


if __name__ == '__main__':
    unittest.main()
