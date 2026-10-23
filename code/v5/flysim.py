"""
flysim.py
---------
A minimal, fast leaky integrate-and-fire (LIF) simulator for the complete
male Drosophila CNS connectome (165,122 neurons, 25.6 M connections).

Model and parameters follow Shiu et al. 2024 (Nature, 'A Drosophila computational
brain model reveals sensorimotor processing'):
    dv/dt = (v_rest - v + g) / tau_m         (tau_m = 20 ms)
    dg/dt = -g / tau_s                       (tau_s = 5 ms)
    spike when v > v_rest + 7 mV, then v -> v_rest, g -> 0, refractory 2.2 ms
    a presynaptic spike adds  w_syn * (signed synapse count)  to g after 1.8 ms
    w_syn = 0.275 mV, stimulated neurons fire on Poisson events (150 Hz default)

The state is kept relative to rest (v_rest = 0).  Only 'active' neurons
(non-zero state) are integrated each step, which makes a whole-CNS simulation
cheap on a laptop-class machine.  Lesions are a boolean 'dead' mask; grafted
neurons are appended after the host neurons and wired with an extra sparse
matrix.  Everything is plain numpy.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import os

PARAMS = dict(dt=0.1, tau_m=20.0, tau_s=5.0, v_th=7.0, t_ref=2.2, t_delay=1.8,
              w_syn=0.275, w_poi=0.275 * 250, r_poi=150.0)


class LIFNetReference:
    backend = 'numpy_reference'
    def __init__(self, W_csc, n_extra=0, extra_csc=None, params=None, gain_in=None, pre_sign=None):
        """W_csc: host connectivity (post x pre, signed synapse counts).
        extra_csc: optional (N_tot x N_tot) CSC with graft connections.
        gain_in: optional per-neuron multiplicative gain on the excitatory synaptic input of each
                 postsynaptic neuron (homeostatic synaptic scaling); pre_sign (N_tot, +1/-1) says which
                 presynaptic neurons are excitatory. With gain_in=None the model is unchanged."""
        self.p = dict(PARAMS, **(params or {}))
        self.N = W_csc.shape[0]
        self.N_tot = self.N + n_extra
        self.indptr, self.indices, self.data = W_csc.indptr, W_csc.indices, W_csc.data
        self.extra = None
        if extra_csc is not None and extra_csc.nnz > 0:
            e = extra_csc.tocsc()
            self.extra = (e.indptr, e.indices, e.data.astype(np.float32))
        self.gain_in = None if gain_in is None else np.asarray(gain_in, np.float32)
        self.pre_exc = None if pre_sign is None else (np.asarray(pre_sign) > 0)

    # ---------------------------------------------------------------- run
    def run(self, T, stim, dead=None, seed=0, t_measure=(100.0, None),
            record_spikes=False, record_idx=None, return_state=False):
        """Simulate T ms.

        stim   : dict {rate_Hz: array of neuron indices}   Poisson-driven neurons
        dead   : boolean array (N_tot) of lesioned neurons (never spike)
        returns dict with firing rates (Hz) of every neuron in the measurement
        window and (optionally) spike times.
        """
        p = self.p
        dt = p['dt']
        n_steps = int(round(T / dt))
        e_m, e_s = np.exp(-dt / p['tau_m']), np.exp(-dt / p['tau_s'])
        A_fac = p['tau_s'] / (p['tau_s'] - p['tau_m'])
        n_ref = int(round(p['t_ref'] / dt))
        n_dly = int(round(p['t_delay'] / dt))
        w_syn = np.float32(p['w_syn'])
        rng = np.random.default_rng(seed)
        N_tot = self.N_tot

        v = np.zeros(N_tot, np.float32)
        g = np.zeros(N_tot, np.float32)
        ref = np.zeros(N_tot, np.int32)
        buf = np.zeros((n_dly, N_tot), np.float32)          # delay line
        counts = np.zeros(N_tot, np.int64)
        dead = np.zeros(N_tot, bool) if dead is None else dead.astype(bool)
        alive = ~dead
        stim_idx = np.concatenate([np.asarray(ix, int) for ix in stim.values()]) if stim else np.zeros(0, int)
        stim_rate = np.concatenate([np.full(len(ix), r, float) for r, ix in stim.items()]) if stim else np.zeros(0)
        stim_idx_alive = alive[stim_idx]
        # Consume a draw for every original stimulus slot, even if its neuron
        # died. Removing one input must not change the surviving inputs' events.
        p_ev = stim_rate * dt * 1e-3
        is_stim = np.zeros(N_tot, bool); is_stim[stim_idx] = True

        t0 = t_measure[0]; t1 = T if t_measure[1] is None else t_measure[1]
        s0, s1 = int(t0 / dt), int(t1 / dt)
        sp_t, sp_i = [], []
        rec_mask = None
        if record_spikes:
            rec_mask = np.ones(N_tot, bool) if record_idx is None else np.isin(np.arange(N_tot), record_idx)

        active = np.zeros(N_tot, bool)
        for step in range(n_steps):
            slot = step % n_dly
            # ---- deliver delayed synaptic input
            inc = buf[slot]
            nz = np.flatnonzero(inc)
            if nz.size:
                # Shiu/Brian2 marks both v and g "unless refractory". Synaptic
                # writes during that interval are discarded, not accumulated.
                receiving = nz[ref[nz] == 0]
                g[receiving] += inc[receiving]
                active[receiving] = True
                inc[nz] = 0.0
            # ---- integrate active, non-refractory neurons (exact for the linear system)
            a = np.flatnonzero(active)
            if a.size:
                r = ref[a]
                na = a[r == 0]
                if na.size:
                    gg = g[na]
                    A = gg * A_fac
                    v[na] = v[na] * e_m + A * (e_s - e_m)
                    g[na] = gg * e_s
                ra = a[r > 0]
                ref[ra] -= 1
            # ---- Poisson drive of stimulated neurons (event -> immediate spike)
            if stim_idx.size:
                ev = stim_idx[(rng.random(stim_idx.size) < p_ev) & stim_idx_alive]
                if ev.size:
                    v[ev] += p['w_poi']
                    active[ev] = True
            # ---- threshold
            a = np.flatnonzero(active)
            spk = a[(v[a] > p['v_th']) & alive[a]]
            if spk.size:
                v[spk] = 0.0; g[spk] = 0.0
                ref[spk] = np.where(is_stim[spk], 0, n_ref)
                if s0 <= step < s1:
                    counts[spk] += 1
                if rec_mask is not None:
                    keep = spk[rec_mask[spk]]
                    sp_t.append(np.full(keep.size, step * dt, np.float32)); sp_i.append(keep)
                # ---- propagate spikes into the delay line
                contrib = self._propagate(spk, N_tot)
                buf[(step + n_dly) % n_dly] += contrib * w_syn
            # Signed voltage and current must both decay, including inhibition.
            # Discard only states below the numerical tolerance, then zero them
            # so later input cannot reactivate a frozen residual.
            if step % 10 == 0:
                active &= (np.abs(v) > 1e-3) | (np.abs(g) > 1e-3) | (ref > 0)
                v[~active] = 0.0
                g[~active] = 0.0
        rates = counts / ((s1 - s0) * dt * 1e-3)
        out = dict(rates=rates.astype(np.float32))
        if return_state:
            out.update(voltage=v.copy(), synaptic_state=g.copy(), refractory=ref.copy())
        if rec_mask is not None:
            out['spike_t'] = np.concatenate(sp_t) if sp_t else np.zeros(0, np.float32)
            out['spike_i'] = np.concatenate(sp_i) if sp_i else np.zeros(0, int)
        return out

    # ------------------------------------------------------------ helpers
    def _propagate(self, spk, N_tot):
        if self.gain_in is None:
            return self._propagate_raw(spk, N_tot)
        # homeostatic scaling acts on excitatory synapses only: gather the excitatory and the
        # inhibitory presynaptic spikes separately and scale the excitatory input per postsynaptic cell
        exc = self.pre_exc[spk]
        return self._propagate_raw(spk[exc], N_tot) * self.gain_in + self._propagate_raw(spk[~exc], N_tot)

    def _propagate_raw(self, spk, N_tot):
        contrib = np.zeros(N_tot, np.float32)
        host = spk[spk < self.N]
        if host.size:
            contrib += _gather(self.indptr, self.indices, self.data, host, N_tot)
        if self.extra is not None and spk.size:
            contrib += _gather(*self.extra, spk, N_tot)
        return contrib


def LIFNet(W_csc, n_extra=0, extra_csc=None, params=None, gain_in=None, pre_sign=None, backend=None):
    """Construct the validated backend, retaining an explicit NumPy reference.

    ``FLY_SIM_BACKEND=reference`` selects the reference implementation.
    ``compiled`` requires the validated NumPy 2 promotion semantics and Numba.
    The default ``auto`` uses compiled when available and reference otherwise.
    Importing the compiled module is deferred until construction, so module
    imports have no circular initialization dependency.
    """
    selected = backend or os.environ.get('FLY_SIM_BACKEND', 'auto')
    if selected not in ('auto', 'reference', 'compiled'):
        raise ValueError('Backend must be auto, reference or compiled')
    cls = LIFNetReference
    wide = (np.zeros(1, np.float32) * np.float64(.99)).dtype == np.dtype(np.float64)
    if selected != 'reference' and wide:
        try:
            from flysim_fast import LIFNetCompiled
        except ImportError:
            if selected == 'compiled':
                raise
        else:
            cls = LIFNetCompiled
    elif selected == 'compiled':
        raise RuntimeError('Compiled backend requires NumPy 2 scalar promotion semantics')
    return cls(W_csc, n_extra=n_extra, extra_csc=extra_csc, params=params,
               gain_in=gain_in, pre_sign=pre_sign)


def _gather(indptr, indices, data, pre, N_tot):
    starts = indptr[pre]; lens = indptr[pre + 1] - starts
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(N_tot, np.float32)
    off = np.cumsum(lens) - lens
    pos = np.arange(tot) + np.repeat(starts - off, lens)
    return np.bincount(indices[pos], weights=data[pos], minlength=N_tot).astype(np.float32)


def load_net(processed=None):
    import pandas as pd, json
    processed = Path(processed) if processed is not None else Path(__file__).resolve().parents[2] / 'data' / 'v5'
    W = sp.load_npz(processed / 'W_csc.npz').tocsc()
    neu = pd.read_parquet(processed / 'neurons.parquet')
    groups = json.load(open(processed / 'cell_groups.json'))
    return W, neu, groups
