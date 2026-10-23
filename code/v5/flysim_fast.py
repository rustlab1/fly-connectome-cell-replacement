"""Optional, validated-candidate sparse propagation backend for the V5 model.

The reference simulator remains unchanged. This subclass replaces only sparse
spike accumulation. It preserves presynaptic iteration order and float64
accumulation followed by float32 conversion, matching NumPy bincount. Host and
graft matrices are accumulated separately, then combined in float32, as in the
reference backend. The reference module can be used without importing Numba.
"""
import numpy as np
from numba import njit
from flysim import LIFNetReference


@njit(cache=True)
def _gather_compiled(indptr, indices, data, pre, n_total):
    result = np.zeros(n_total, dtype=np.float64)
    for pre_index in pre:
        for edge in range(indptr[pre_index], indptr[pre_index + 1]):
            result[indices[edge]] += data[edge]
    return result.astype(np.float32)


class LIFNetFast(LIFNetReference):
    """Same model and API as LIFNet, with compiled sparse accumulation only."""

    def _propagate_raw(self, spk, N_tot):
        contrib = np.zeros(N_tot, np.float32)
        host = spk[spk < self.N]
        if host.size:
            contrib += _gather_compiled(self.indptr, self.indices, self.data, host, N_tot)
        if self.extra is not None and spk.size:
            contrib += _gather_compiled(*self.extra, spk, N_tot)
        return contrib


@njit(cache=True)
def _accumulate_ordered(indptr, indices, data, spikes, n_host, n_total, host_only, pre_exc, sign_filter):
    """Keep the exact accumulation order and conversion boundaries of _gather."""
    result = np.zeros(n_total, np.float64)
    for pre in spikes:
        if host_only and pre >= n_host:
            continue
        if sign_filter == 1 and not pre_exc[pre]:
            continue
        if sign_filter == -1 and pre_exc[pre]:
            continue
        for edge in range(indptr[pre], indptr[pre + 1]):
            result[indices[edge]] += data[edge]
    return result.astype(np.float32)


@njit(cache=True)
def _compiled_loop(host_ptr, host_idx, host_data, extra_ptr, extra_idx, extra_data,
                   n_host, n_total, n_steps, delay, refractory_steps, em, es, af,
                   threshold, w_poi, w_syn, dt, stim_idx, probabilities, alive,
                   s0, s1, rng, use_gain, gains, pre_exc, record, record_mask):
    voltage = np.zeros(n_total, np.float32)
    current = np.zeros(n_total, np.float32)
    refractory = np.zeros(n_total, np.int32)
    pending = np.zeros((delay, n_total), np.float32)
    active = np.zeros(n_total, np.bool_)
    is_stim = np.zeros(n_total, np.bool_)
    counts = np.zeros(n_total, np.int64)
    for cell in stim_idx:
        is_stim[cell] = True
    spike_times = [np.float32(0)]
    spike_cells = [np.int64(0)]
    spike_times.pop(); spike_cells.pop()
    difference = es - em
    tolerance = np.float32(1e-3)
    for step in range(n_steps):
        slot = step % delay
        # Delivery precedes integration, as in the reference backend.
        for cell in range(n_total):
            inc = pending[slot, cell]
            if inc != 0:
                if refractory[cell] == 0:
                    current[cell] = np.float32(current[cell] + inc)
                    active[cell] = True
                pending[slot, cell] = 0
            if active[cell]:
                if refractory[cell] == 0:
                    gg = current[cell]
                    a = np.float32(gg * np.float32(af))
                    voltage[cell] = np.float32(np.float64(voltage[cell]) * em + np.float64(a) * difference)
                    current[cell] = np.float32(np.float64(gg) * es)
                else:
                    refractory[cell] -= 1
        if len(stim_idx):
            randoms = rng.random(len(stim_idx))
            for j in range(len(stim_idx)):
                cell = stim_idx[j]
                if randoms[j] < probabilities[j] and alive[cell]:
                    # NumPy in-place addition to a float32 array casts the scalar.
                    voltage[cell] = np.float32(voltage[cell] + np.float32(w_poi))
                    active[cell] = True
        spikes = []
        for cell in range(n_total):
            if active[cell] and voltage[cell] > threshold and alive[cell]:
                spikes.append(cell)
                voltage[cell] = 0; current[cell] = 0
                refractory[cell] = 0 if is_stim[cell] else refractory_steps
                if s0 <= step < s1:
                    counts[cell] += 1
                if record and record_mask[cell]:
                    spike_times.append(np.float32(step * dt))
                    spike_cells.append(np.int64(cell))
        if len(spikes):
            if use_gain:
                excitatory = _accumulate_ordered(host_ptr, host_idx, host_data, spikes,
                                                n_host, n_total, True, pre_exc, 1)
                inhibitory = _accumulate_ordered(host_ptr, host_idx, host_data, spikes,
                                                n_host, n_total, True, pre_exc, -1)
                if len(extra_data):
                    excitatory += _accumulate_ordered(extra_ptr, extra_idx, extra_data, spikes,
                                                     n_host, n_total, False, pre_exc, 1)
                    inhibitory += _accumulate_ordered(extra_ptr, extra_idx, extra_data, spikes,
                                                     n_host, n_total, False, pre_exc, -1)
                for cell in range(n_total):
                    scaled = np.float32(excitatory[cell] * gains[cell])
                    total = np.float32(scaled + inhibitory[cell])
                    increment = np.float32(total * w_syn)
                    pending[slot, cell] = np.float32(pending[slot, cell] + increment)
            else:
                total = _accumulate_ordered(host_ptr, host_idx, host_data, spikes,
                                           n_host, n_total, True, pre_exc, 0)
                if len(extra_data):
                    total += _accumulate_ordered(extra_ptr, extra_idx, extra_data, spikes,
                                                n_host, n_total, False, pre_exc, 0)
                for cell in range(n_total):
                    increment = np.float32(total[cell] * w_syn)
                    pending[slot, cell] = np.float32(pending[slot, cell] + increment)
        if step % 10 == 0:
            for cell in range(n_total):
                active[cell] = active[cell] and (abs(voltage[cell]) > tolerance or
                                                 abs(current[cell]) > tolerance or refractory[cell] > 0)
                if not active[cell]:
                    voltage[cell] = 0; current[cell] = 0
    return counts, voltage, current, refractory, np.asarray(spike_times), np.asarray(spike_cells)


class LIFNetCompiled(LIFNetReference):
    """Complete loop backend, validated against the explicit NumPy reference."""
    backend = 'numba_compiled'

    def run(self, T, stim, dead=None, seed=0, t_measure=(100.0, None),
            record_spikes=False, record_idx=None, return_state=False):
        p = self.p; dt = p['dt']; n_total = self.N_tot
        n_steps = int(round(T / dt)); delay = int(round(p['t_delay'] / dt))
        refractory_steps = int(round(p['t_ref'] / dt))
        em, es = np.exp(-dt / p['tau_m']), np.exp(-dt / p['tau_s'])
        if (np.zeros(1, np.float32) * em).dtype != np.dtype(np.float64):
            raise RuntimeError('Compiled loop requires NumPy 2 scalar promotion; use the reference backend in this environment')
        af = p['tau_s'] / (p['tau_s'] - p['tau_m'])
        dead = np.zeros(n_total, bool) if dead is None else dead.astype(bool)
        stim_idx = np.concatenate([np.asarray(ix, int) for ix in stim.values()]) if stim else np.zeros(0, int)
        stim_rate = np.concatenate([np.full(len(ix), r, float) for r, ix in stim.items()]) if stim else np.zeros(0)
        probabilities = stim_rate * dt * 1e-3
        s0 = int(t_measure[0] / dt)
        s1 = int((T if t_measure[1] is None else t_measure[1]) / dt)
        record_mask = (np.ones(n_total, bool) if record_idx is None else np.isin(np.arange(n_total), record_idx))
        extra = self.extra if self.extra is not None else (np.zeros(n_total + 1, np.int32),
                  np.zeros(0, np.int32), np.zeros(0, np.float32))
        gains = self.gain_in if self.gain_in is not None else np.ones(n_total, np.float32)
        pre_exc = self.pre_exc if self.pre_exc is not None else np.ones(n_total, bool)
        counts, v, g, ref, times, cells = _compiled_loop(
            self.indptr, self.indices, self.data, *extra, self.N, n_total, n_steps, delay,
            refractory_steps, em, es, af, p['v_th'], p['w_poi'], np.float32(p['w_syn']), dt,
            stim_idx, probabilities, ~dead, s0, s1, np.random.default_rng(seed), self.gain_in is not None,
            gains, pre_exc, record_spikes, record_mask)
        result = {'rates': (counts / ((s1 - s0) * dt * 1e-3)).astype(np.float32)}
        if record_spikes:
            result.update(spike_t=times, spike_i=cells)
        if return_state:
            result.update(voltage=v, synaptic_state=g, refractory=ref)
        return result


def validate_backend(output_path=None):
    """Reproduce exact full-connectome checks without changing production output.

    Run as ``python code/v5/flysim_fast.py`` under the documented NumPy 2/Numba
    environment. This needs the V5 connectivity matrix and takes about a minute.
    """
    import hashlib
    import json
    import time
    from pathlib import Path
    import experiments as ex
    c = ex.get_net(); W = c['W']; N = W.shape[0]
    center = [292.0516907563026, 199.2039193277311, 301.10268907563017]
    lesion, _ = ex.make_lesion(center, 1000)
    graft = ex.make_graft(lesion, 1000, templates=lesion, center_um=center)
    gains = np.random.default_rng(923).uniform(.5, 2, N).astype(np.float32)
    rows = []
    cases = [('intact', None, None, None, 'R', 1), ('lesion', lesion, None, None, 'R', 2),
             ('exact_graft', lesion, graft, None, 'R', 2), ('homeostasis', lesion, None, gains, 'L', 3),
             ('graft_homeostasis', lesion, graft, gains, 'L', 3)]
    for label, lost, gg, gain, stimulus, seed in cases:
        n_extra = 0 if gg is None else len(gg[2])
        dead = np.zeros(N + n_extra, bool)
        if lost is not None:
            dead[lost] = True
        kwargs = dict(n_extra=n_extra, extra_csc=None if gg is None else gg[0], params={'w_syn': ex.W_SYN})
        if gain is not None:
            kwargs.update(gain_in=np.r_[gain, np.ones(n_extra, np.float32)],
                          pre_sign=np.r_[c['neu'].sign.values, np.zeros(0, np.int8) if gg is None else gg[1]])
        indices = ex.graft_stimulus_indices(ex.stimuli()[stimulus], lost, gg)
        for duration in (30., 300.):
            outputs = {}
            for backend, cls in [('reference', LIFNetReference), ('compiled_loop', LIFNetCompiled)]:
                start = time.perf_counter()
                outputs[backend] = cls(W, **kwargs).run(duration, {100.: indices}, dead=dead, seed=seed,
                         t_measure=(0, duration), record_spikes=True, return_state=True)
                row = dict(case=label, duration_ms=duration, backend=backend,
                           seconds=time.perf_counter() - start, spikes=len(outputs[backend]['spike_i']))
                rows.append(row); print(row, flush=True)
            for key in outputs['reference']:
                np.testing.assert_array_equal(outputs['reference'][key], outputs['compiled_loop'][key], err_msg=label + ' ' + key)
    path = Path(output_path) if output_path is not None else Path(ex.RES) / 'compiled_backend_check.json'
    files = ['flysim.py', 'flysim_fast.py', 'experiments.py']
    record = dict(exact_all_outputs=True, numpy=np.__version__, timings=rows,
                  source_hashes={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files})
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    validate_backend()
