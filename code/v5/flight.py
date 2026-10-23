"""
flight.py
---------
Reduced closed-loop model of optomotor course control in flight.

The connectome model provides, for each network state, the yaw command evoked by
rightward and leftward optic flow (gain_R, gain_L; 1 = intact) and the turning
tendency evoked by the symmetric optic flow of straight flight (bias; in units of
the intact half-command).  Here those three numbers drive a first-order yaw
controller of a fly flying at constant speed through a turbulent world:

    tau_yaw * d(omega)/dt = -omega + omega_cmd + omega_gust
    omega_cmd = -G * g(omega) * w_sat * tanh(omega / w_sat) + B0 * bias

where omega > 0 is a rightward (clockwise) turn, g(omega) = gain_L for omega > 0
(a rightward turn is corrected by a leftward command) and gain_R for omega < 0.
Gusts are an Ornstein-Uhlenbeck process.  Heading and position are integrated
and course-holding performance is reported as the r.m.s. heading error and the
path straightness (net displacement / path length).

Flight motor drive.  The descending drive to the wing power motor neurons (`drive`,
1 = intact, clipped to [0.05, 1]) sets how well the fly can fly at all: forward speed
scales with the drive, and a gust of given speed deflects a slower fly more (the
apparent-wind angle scales with gust speed over airspeed), so the yaw perturbation
scales with 1/drive.  With drive = 1 the model is the pure yaw controller.
"""
import numpy as np

DEG = np.pi / 180.0
P = dict(tau_yaw=0.02, G=3.0, w_sat=300 * DEG, B0=150 * DEG, tau_gust=0.15, sigma_gust=100 * DEG,
         speed=0.3, dt=0.001)


def simulate(gain_R=1.0, gain_L=1.0, bias=0.0, T=8.0, seed=0, params=None, drive=1.0):
    p = dict(P, **(params or {}))
    d = float(np.clip(drive, 0.05, 1.0))
    p['speed'] = p['speed'] * d; p['sigma_gust'] = p['sigma_gust'] / d
    dt = p['dt']; n = int(T / dt)
    rng = np.random.default_rng(seed)
    omega = 0.0; gust = 0.0; psi = 0.0; x = y = 0.0
    out = np.zeros((n, 5), np.float32)          # t, psi, omega, x, y
    a = np.exp(-dt / p['tau_gust']); s = p['sigma_gust'] * np.sqrt(1 - a * a)
    for i in range(n):
        gust = a * gust + s * rng.standard_normal()
        g = gain_L if omega > 0 else gain_R
        cmd = -p['G'] * g * p['w_sat'] * np.tanh(omega / p['w_sat']) + p['B0'] * bias
        omega += dt / p['tau_yaw'] * (-omega + cmd + gust)
        psi += omega * dt
        x += p['speed'] * np.cos(psi) * dt; y += p['speed'] * np.sin(psi) * dt
        out[i] = (i * dt, psi, omega, x, y)
    return out


def performance(traj):
    psi = traj[:, 1]; x, y = traj[:, 3], traj[:, 4]
    rms_heading = float(np.sqrt(np.mean(psi ** 2)) / DEG)
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    straight = float(np.hypot(x[-1], y[-1]) / path) if path > 0 else 0.0
    return dict(rms_heading_deg=rms_heading, straightness=straight,
                mean_abs_omega_deg=float(np.mean(np.abs(traj[:, 2])) / DEG))


def evaluate(gain_R, gain_L, bias, n_seeds=20, T=10.0, drive=1.0):
    """Average course-holding performance over gust realisations."""
    perf = [performance(simulate(gain_R, gain_L, bias, T=T, seed=s, drive=drive)) for s in range(n_seeds)]
    return {k: float(np.mean([q[k] for q in perf])) for k in perf[0]} | \
           {k + '_sd': float(np.std([q[k] for q in perf])) for k in perf[0]}


if __name__ == '__main__':
    for name, g in dict(intact=(1, 1, 0), no_steering=(0, 0, 0), site29=(-0.45, 1.26, -1.05),
                        half=(0.5, 0.5, 0), bias_only=(1, 1, -1.0)).items():
        print(name, evaluate(*g, n_seeds=10))
