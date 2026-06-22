"""
sysid_unbalanced_disk.py
========================
System identification for the physical Unbalanced Disk hardware.

Runs four experiments in sequence:
  1. NOISE FLOOR  – u=0, disk at rest -> characterise omega sensor noise & quantisation
  2. STEP RESPONSES – series of voltage steps -> estimate Ku, time-constants, lag
  3. PRBS EXCITATION – pseudo-random binary sequence -> broadband frequency response
  4. GRAVITY TORQUE  – slow manual swing or zero-input free decay -> estimate omega0

All raw data is saved to  sysid_data.npz  so you can re-run the analysis offline.
Results and plots are saved as  sysid_results.png.

Usage
-----
    python sysid_unbalanced_disk_.py

Requirements: the UnbalancedDiskExp.py file must be on the Python path.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import correlate, welch
from collections import deque

from UnbalancedDiskExp import UnbalancedDisk_exp   # raw (th, omega) observations

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def send_and_record(env, u_seq, desc=""):
    """
    Play a voltage sequence and return arrays of (t, th, omega_raw, u_applied).
    u_seq : 1-D array of voltage values (one per dt step)
    """
    th_log, om_log, u_log, t_log = [], [], [], []
    t0 = time.time()
    print(f"  -> {desc}  ({len(u_seq)} steps × {env.dt*1000:.1f} ms = "
          f"{len(u_seq)*env.dt:.1f} s)")
    for k, u in enumerate(u_seq):
        obs, _, term, _, _ = env.step(np.float32(u))
        th_log.append(env.th)
        om_log.append(env.omega)   # raw hardware omega (no software filter)
        u_log.append(float(u))
        t_log.append(k * env.dt)
        if term:
            print("    [terminated early – encoder limit hit]")
            break
    return (np.array(t_log), np.array(th_log),
            np.array(om_log), np.array(u_log))


def wait_for_rest(env, timeout=45, tol=0.15):
    """Block until |omega| < tol for 10 consecutive samples or timeout."""
    print("  Waiting for disk to come to rest …", end="", flush=True)
    quiet = 0
    t0 = time.time()
    while time.time() - t0 < timeout:
        obs = env.get_obs()
        if abs(obs[1]) < tol:
            quiet += 1
            if quiet >= 10:
                print(" done.")
                return True
        else:
            quiet = 0
        time.sleep(env.dt)
    print(f" timed out (|ω| never stayed below {tol} for 10 samples — "
          f"continuing anyway).")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 – Noise floor  (u = 0, disk stationary at bottom)
# ─────────────────────────────────────────────────────────────────────────────

def exp_noise_floor(env, n_samples=600):
    """
    Record omega with zero input while disk sits at rest.
    Returns (t, th, omega) arrays and a dict of statistics.
    """
    print("\n[1] NOISE FLOOR — zero input, disk at rest")
    wait_for_rest(env)
    u_seq = np.zeros(n_samples)
    t, th, om, _ = send_and_record(env, u_seq, "noise floor capture")

    # ── theta noise ──────────────────────────────────────────────────────────
    th_deg = np.rad2deg(th)
    th_jumps = np.abs(np.diff(th_deg))
    nonzero_jumps = th_jumps[th_jumps > 1e-4]
    enc_res = float(np.min(nonzero_jumps)) if len(nonzero_jumps) > 0 else 0.0

    # theoretical: 2000 counts/rev -> 0.18 deg/count
    theoretical_enc_deg = 360.0 / 2000.0

    # Count how many samples have a non-zero step (encoder tick rate at rest)
    tick_rate_hz = float(np.sum(th_jumps > 1e-4) / (len(th) * env.dt))

    stats = {
        "omega_mean"      : float(np.mean(om)),
        "omega_std"       : float(np.std(om)),
        "omega_peak"      : float(np.max(np.abs(om))),
        # theta
        "th_mean_deg"     : float(np.mean(th_deg)),
        "th_std_deg"      : float(np.std(th_deg)),
        "enc_resolution_deg"    : enc_res,
        "enc_theoretical_deg"   : theoretical_enc_deg,
        "enc_tick_rate_hz"      : tick_rate_hz,   # how often encoder ticks at rest
        # omega derived from finite-diff on theta — what the controller really sees
        "omega_fd_std"    : float(np.std(np.diff(th) / env.dt)),
    }

    print(f"     omega: mean={stats['omega_mean']:+.4f}  "
          f"std={stats['omega_std']:.4f}  "
          f"peak={stats['omega_peak']:.4f}  rad/s")
    print(f"     theta std = {stats['th_std_deg']:.4f}°  "
          f"(expect ~{enc_res/2:.4f}° if pure quantisation)")
    print(f"     encoder step ≈ {enc_res:.4f}°  "
          f"(theoretical {theoretical_enc_deg:.4f}°/count)")
    print(f"     encoder tick rate at rest: {tick_rate_hz:.2f} Hz  "
          f"(should be ~0 if truly still)")
    print(f"     omega via finite-diff std: {stats['omega_fd_std']:.4f} rad/s  "
          f"<- this is what bad filters amplify")
    return t, th, om, stats


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 – Step responses  ->  Ku, lag, dominant time constant
# ─────────────────────────────────────────────────────────────────────────────

def exp_step_responses(env, amplitudes=(1.0, 2.0, 3.0, -1.0, -2.0, -3.0),
                       step_len=80, settle_steps=60):
    """
    Apply voltage steps of varying amplitude from rest near the bottom.
    Returns list of (t, th, om, u) tuples, one per step.
    """
    print("\n[2] STEP RESPONSES — estimating Ku, lag, friction")
    results = []
    for amp in amplitudes:
        wait_for_rest(env, tol=0.05)
        # short pre-roll at zero so we capture the true zero baseline
        pre = np.zeros(20)
        step = np.full(step_len, amp)
        post = np.zeros(settle_steps)
        seq = np.concatenate([pre, step, post])
        t, th, om, u = send_and_record(
            env, seq, f"step u={amp:+.1f} V")
        results.append((t, th, om, u))
    return results


def estimate_lag_from_steps(step_results, dt):
    """
    Cross-correlate each step's d(omega)/dt with the step input signal to
    find the round-trip actuation lag in samples.
    Returns median lag in seconds.
    """
    lags = []
    for t, th, om, u in step_results:
        dom = np.gradient(om, dt)
        # zero-mean both signals before cross-correlation
        a = u - np.mean(u)
        b = dom - np.mean(dom)
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        xcorr = correlate(b, a, mode="full")
        lag_idx = int(np.argmax(xcorr)) - (len(a) - 1)
        # only accept positive lags up to 10 steps
        if 0 <= lag_idx <= 10:
            lags.append(lag_idx)
    if lags:
        median_lag = float(np.median(lags))
        print(f"  Estimated actuation lag: {median_lag:.1f} steps  "
              f"= {median_lag*dt*1000:.1f} ms")
        return median_lag * dt
    else:
        print("  Lag estimation inconclusive (disk moved too fast?)")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 – PRBS broadband excitation  ->  frequency response
# ─────────────────────────────────────────────────────────────────────────────

def make_prbs(n, amplitude=2.0, seed=42):
    """n-length ±amplitude PRBS (pseudo-random binary sequence)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-amplitude, amplitude], size=n).astype(np.float32)


def exp_prbs(env, n_samples=1200, amplitude=2.5):
    """
    Apply a PRBS signal, return raw data for spectral analysis.
    Keep amplitude below umax to avoid termination.
    """
    print("\n[3] PRBS EXCITATION — broadband frequency response")
    wait_for_rest(env, tol=0.3)   # for PRBS we don't need perfect rest
    seq = make_prbs(n_samples, amplitude=amplitude)
    t, th, om, u = send_and_record(env, seq, f"PRBS ±{amplitude} V")
    return t, th, om, u


def compute_freq_response(u, om, dt, nperseg=256):
    """
    Estimate the input->omega frequency response via Welch's method (magnitude only).
    Returns (f, H_mag).
    """
    f, Puu = welch(u,  fs=1/dt, nperseg=nperseg)
    f, Pou = welch(om, fs=1/dt, nperseg=nperseg,
                   # cross-spectrum via trick: welch on (u+om) and (u-om)
                   )
    # Simple gain estimate: |H(f)| ≈ sqrt(P_oo / P_uu)
    f2, Poo = welch(om, fs=1/dt, nperseg=nperseg)
    H_mag = np.sqrt(np.clip(Poo / (Puu + 1e-12), 0, None))
    return f, H_mag


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 – Free decay  ->  omega0 (gravity), gamma (viscous friction)
# ─────────────────────────────────────────────────────────────────────────────

def exp_free_decay(env, n_samples=1000):
    """
    Give a short kick, then let the disk swing freely.
    The oscillation frequency near the bottom -> omega0.
    """
    print("\n[4] FREE DECAY — estimating omega0 and damping")
    wait_for_rest(env)
    # short kick to start oscillation
    kick = np.concatenate([np.full(8, 1.5), np.zeros(n_samples)])
    t, th, om, u = send_and_record(env, kick, "kick + free decay")
    return t, th, om, u


def estimate_omega0_from_decay(t, th):
    """
    Estimate natural frequency from zero-crossings of theta near the bottom.
    Works well when disk swings ±π/2 or less.
    """
    # detect sign changes in th (crossings through 0)
    signs = np.sign(th)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) < 4:
        print("  Not enough zero-crossings for omega0 estimate.")
        return None
    dt_half_periods = np.diff(t[crossings])
    period = 2 * np.mean(dt_half_periods)  # average full period
    omega0_est = 2 * np.pi / period
    print(f"  omega0 estimate (small-angle): {omega0_est:.3f} rad/s  "
          f"(nominal: 11.34 rad/s)")
    return omega0_est


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 – dt jitter  ->  real sample time variance
# ─────────────────────────────────────────────────────────────────────────────

def exp_dt_jitter(env, n_samples=300):
    """
    Measure actual wall-clock time per step to characterise timing jitter.
    The busy-wait loop in step() is not perfect — OS preemption causes spikes.
    Returns array of actual step durations in ms.
    """
    print("\n[5] DT JITTER — measuring actual step timing")
    durations = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        env.step(np.float32(0.0))
        durations.append(time.perf_counter() - t0)
    durations = np.array(durations) * 1000  # -> ms
    print(f"  nominal dt : {env.dt*1000:.2f} ms")
    print(f"  actual mean: {np.mean(durations):.3f} ms")
    print(f"  actual std : {np.std(durations):.3f} ms  <- jitter")
    print(f"  max spike  : {np.max(durations):.3f} ms")
    print(f"  99th pctile: {np.percentile(durations, 99):.3f} ms")
    return durations


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6 – Voltage dead zone  ->  minimum voltage that moves the motor
# ─────────────────────────────────────────────────────────────────────────────

def exp_dead_zone(env, v_range=None, steps_per_v=30):
    """
    Ramp voltage up from 0 in small steps, check if disk moves.
    Returns (voltages, moved_flags, threshold_voltage).
    """
    if v_range is None:
        v_range = np.arange(0.0, 1.6, 0.1)
    print("\n[6] DEAD ZONE — minimum voltage to overcome static friction")
    voltages, moved = [], []
    for v in v_range:
        wait_for_rest(env, timeout=20, tol=0.03)
        seq = np.full(steps_per_v, v, dtype=np.float32)
        t, th, om, u = send_and_record(env, seq, f"v={v:.2f}V")
        th_excursion = np.max(np.abs(th - th[0]))
        did_move = th_excursion > np.deg2rad(2.0)   # >2° counts as moving
        voltages.append(v)
        moved.append(did_move)
        print(f"  v={v:.2f} V  ->  {'MOVED' if did_move else 'still'}  "
              f"(Δθ={np.rad2deg(th_excursion):.2f}°)")
    voltages = np.array(voltages)
    moved = np.array(moved)
    threshold_idxs = np.where(moved)[0]
    v_threshold = float(voltages[threshold_idxs[0]]) if len(threshold_idxs) else float('nan')
    print(f"  Dead-zone threshold ≈ {v_threshold:.2f} V")
    return voltages, moved, v_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Parameter fitting — least squares on ODE
# ─────────────────────────────────────────────────────────────────────────────

def simulate_model(params, t_arr, th0, om0, u_arr, dt):
    """
    Euler-integrate the disk ODE with given parameters.
    params = [omega0, gamma, Ku, Fc]
    """
    omega0, gamma, Ku, Fc = params
    coulomb_omega = 0.001
    delta_th = 0.0
    th, om = th0, om0
    th_out, om_out = [th], [om]
    for u in u_arr[:-1]:
        friction = gamma * om + Fc * np.tanh(om / coulomb_omega)
        dom = -(omega0**2) * np.sin(th + delta_th) - friction + Ku * u
        om = om + dt * dom
        th = th + dt * om
        th_out.append(th)
        om_out.append(om)
    return np.array(th_out), np.array(om_out)


def fit_parameters(step_results, dt,
                   p0=(11.34, 1.33, 28.14, 6.06)):
    """
    Fit [omega0, gamma, Ku, Fc] by minimising prediction error on step data.
    Only uses segments where the disk stays within ±1.5π (linear-ish regime).
    """
    print("\n[FIT] Least-squares parameter fit …")

    def residuals(params):
        res = []
        for t, th, om, u in step_results:
            # only use first 60 steps to stay near small-angle
            n = min(60, len(t))
            th_sim, om_sim = simulate_model(
                params, t[:n], th[0], om[0], u[:n], dt)
            res.append((om_sim - om[:n]).ravel())
        return np.concatenate(res)

    bounds_lo = [5.0,  0.1, 10.0, 0.5]
    bounds_hi = [18.0, 5.0, 50.0, 20.0]
    try:
        result = least_squares(residuals, p0,
                               bounds=(bounds_lo, bounds_hi),
                               method="trf", max_nfev=500, verbose=0)
        omega0, gamma, Ku, Fc = result.x
        print(f"  omega0 = {omega0:.4f}  (nominal 11.34)")
        print(f"  gamma  = {gamma:.4f}  (nominal  1.33)")
        print(f"  Ku     = {Ku:.4f}    (nominal 28.14)")
        print(f"  Fc     = {Fc:.4f}    (nominal  6.06)")
        print(f"  cost   = {result.cost:.4f}")
        return dict(omega0=omega0, gamma=gamma, Ku=Ku, Fc=Fc)
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(noise_t, noise_th, noise_om,
                 step_results,
                 prbs_t, prbs_th, prbs_om, prbs_u,
                 decay_t, decay_th, decay_om,
                 noise_stats, lag_s, freq_f, freq_H,
                 fit_params, dt,
                 jitter_ms=None,
                 dz_voltages=None, dz_moved=None, dz_threshold=None):

    from scipy.stats import norm as sp_norm

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("Unbalanced Disk — System Identification Results", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(5, 3, hspace=0.60, wspace=0.38)

    # ── ROW 0: Omega noise ───────────────────────────────────────────────────

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(noise_t, noise_om, color='steelblue', lw=0.8, alpha=0.85)
    ax.axhline(noise_stats['omega_mean'], color='r', lw=1.2, ls='--',
               label=f"mean={noise_stats['omega_mean']:+.3f}")
    ax.fill_between(noise_t,
                    noise_stats['omega_mean'] - noise_stats['omega_std'],
                    noise_stats['omega_mean'] + noise_stats['omega_std'],
                    alpha=0.2, color='steelblue',
                    label=f"±1σ={noise_stats['omega_std']:.3f}")
    ax.set_title("ω noise (u=0, at rest)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ω [rad/s]")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(noise_om, bins=40, color='steelblue', edgecolor='white', alpha=0.85, density=True)
    x = np.linspace(noise_om.min(), noise_om.max(), 200)
    ax.plot(x, sp_norm.pdf(x, noise_stats['omega_mean'], noise_stats['omega_std']),
            'r-', lw=1.5, label="Gaussian fit")
    ax.set_title("ω noise histogram")
    ax.set_xlabel("ω [rad/s]"); ax.set_ylabel("Density")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    f_n, P_n = welch(noise_om, fs=1/dt, nperseg=min(128, len(noise_om)//4))
    ax.semilogy(f_n, P_n, color='steelblue', lw=1.0)
    ax.set_title("ω noise PSD")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Power [(rad/s)²/Hz]")
    ax.grid(alpha=0.3, which='both')

    # ── ROW 1: Theta noise ───────────────────────────────────────────────────

    th_deg = np.rad2deg(noise_th)
    th_centered = th_deg - np.mean(th_deg)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(noise_t, th_centered, color='coral', lw=0.8, alpha=0.85)
    ax.set_title(f"θ noise (u=0, at rest)  std={noise_stats['th_std_deg']:.4f}°")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("θ − mean [deg]")
    ax.grid(alpha=0.3)
    # annotate quantisation step
    if noise_stats['enc_resolution_deg'] > 0:
        ax.axhline( noise_stats['enc_resolution_deg']/2, color='gray', lw=0.8, ls=':',
                    label=f"enc step={noise_stats['enc_resolution_deg']:.3f}°")
        ax.axhline(-noise_stats['enc_resolution_deg']/2, color='gray', lw=0.8, ls=':')
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    # histogram of theta — should look like discrete spikes if quantisation-limited
    ax.hist(th_centered, bins=60, color='coral', edgecolor='white', alpha=0.85, density=True)
    ax.set_title("θ noise histogram\n(discrete spikes = encoder limited)")
    ax.set_xlabel("θ − mean [deg]"); ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    # omega derived from finite differences on theta — what a naive differentiator would see
    om_fd = np.diff(noise_th) / dt
    ax.plot(noise_t[1:], om_fd, color='coral', lw=0.7, alpha=0.8, label="finite-diff ω")
    ax.plot(noise_t, noise_om, color='steelblue', lw=0.7, alpha=0.7, label="hardware ω")
    ax.set_title(f"ω: hardware vs finite-diff on θ\n"
                 f"fd std={noise_stats['omega_fd_std']:.3f}  hw std={noise_stats['omega_std']:.3f}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ω [rad/s]")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── ROW 2: Step responses ────────────────────────────────────────────────

    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(step_results)))

    ax = fig.add_subplot(gs[2, 0])
    for (t, th, om, u), c in zip(step_results, colors):
        ax.plot(t, np.rad2deg(th), color=c, lw=1.0, alpha=0.85,
                label=f"u={u[len(u)//3]:+.0f}V")
    ax.set_title("Step responses — theta")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("θ [deg]")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for (t, th, om, u), c in zip(step_results, colors):
        ax.plot(t, om, color=c, lw=1.0, alpha=0.85)
    ax.set_title(f"Step responses — omega  (lag≈{lag_s*1000:.1f} ms)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ω [rad/s]")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    t0, th0, om0, u0 = step_results[0]
    ax.plot(t0, om0, 'k-', lw=1.2, label="Measured")
    if fit_params is not None:
        p = [fit_params['omega0'], fit_params['gamma'],
             fit_params['Ku'], fit_params['Fc']]
        _, om_sim = simulate_model(p, t0, th0[0], om0[0], u0, dt)
        ax.plot(t0, om_sim, 'r--', lw=1.2, label="Model fit")
    ax.set_title("Step u=+1V: measured vs fit")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ω [rad/s]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── ROW 3: PRBS + free decay ─────────────────────────────────────────────

    ax = fig.add_subplot(gs[3, 0])
    ax.semilogy(freq_f[1:], freq_H[1:], color='darkorange', lw=1.2)
    ax.set_title("Input->ω frequency response (gain)")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|H(f)|")
    ax.grid(alpha=0.3, which='both')

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(decay_t, np.rad2deg(decay_th), color='teal', lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(decay_t, decay_om, color='salmon', lw=0.8, alpha=0.7)
    ax.set_title("Free decay — omega0 & damping")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("θ [deg]", color='teal')
    ax2.set_ylabel("ω [rad/s]", color='salmon')
    ax.grid(alpha=0.3)

    # ── 3c. dt jitter ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    if jitter_ms is not None:
        ax.hist(jitter_ms, bins=40, color='mediumpurple', edgecolor='white', alpha=0.85)
        ax.axvline(dt * 1000, color='r', lw=1.5, ls='--', label=f"nominal {dt*1000:.1f} ms")
        ax.axvline(np.mean(jitter_ms), color='orange', lw=1.2, ls='-',
                   label=f"mean {np.mean(jitter_ms):.2f} ms")
        ax.set_title(f"Step timing jitter  σ={np.std(jitter_ms):.2f} ms")
        ax.set_xlabel("Actual step duration [ms]"); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Jitter not measured", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("dt jitter")
    ax.grid(alpha=0.3)

    # ── ROW 4: Dead zone + summary ───────────────────────────────────────────

    ax = fig.add_subplot(gs[4, 0])
    if dz_voltages is not None:
        bar_colors = ['#5ba85e' if m else '#c0392b' for m in dz_moved]
        ax.bar(dz_voltages, dz_moved.astype(float), width=0.08,
               color=bar_colors, edgecolor='white', alpha=0.85)
        if not np.isnan(dz_threshold):
            ax.axvline(dz_threshold, color='k', lw=1.5, ls='--',
                       label=f"threshold ≈ {dz_threshold:.2f} V")
        ax.set_title("Voltage dead zone")
        ax.set_xlabel("Input voltage [V]"); ax.set_ylabel("Moved? (0/1)")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, "Dead zone not measured", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Dead zone")

    # ── Summary text box ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, 1:])
    ax.axis('off')
    dz_str = (f"{dz_threshold:.2f} V" if dz_threshold is not None and not np.isnan(dz_threshold)
              else "not measured")
    jit_str = (f"σ={np.std(jitter_ms):.2f} ms, max={np.max(jitter_ms):.1f} ms"
               if jitter_ms is not None else "not measured")
    n_delay = max(1, int(round(lag_s / dt))) if lag_s > 0 else "?"
    lines = [
        "FULL SYSTEM ID SUMMARY",
        "═" * 46,
        "",
        "SENSOR NOISE (inject into sim for domain randomisation):",
        f"  ω  noise std   : {noise_stats['omega_std']:.4f} rad/s   <- add N(0,σ) to ω obs in sim",
        f"  ω  noise peak  : {noise_stats['omega_peak']:.4f} rad/s",
        f"  θ  std (raw)   : {noise_stats['th_std_deg']:.4f}°   <- mostly encoder quantisation",
        f"  θ  encoder step: {noise_stats['enc_resolution_deg']:.4f}°   (theoretical 0.18°/count)",
        f"  ω  via fin-diff: {noise_stats['omega_fd_std']:.4f} rad/s std  <- why hardware ω matters",
        "",
        "DYNAMICS:",
        f"  Actuation lag  : {lag_s*1000:.1f} ms  -> {n_delay} step(s) delay buffer in sim",
        f"  dt jitter      : {jit_str}",
        f"  Voltage dead zn: {dz_str}   <- add dead-zone to sim actuator",
        "",
        "FITTED PARAMETERS:",
    ]
    if fit_params:
        lines += [
            f"  omega0 = {fit_params['omega0']:.4f}   (nominal 11.34)",
            f"  gamma  = {fit_params['gamma']:.4f}   (nominal  1.33)",
            f"  Ku     = {fit_params['Ku']:.4f}   (nominal 28.14)",
            f"  Fc     = {fit_params['Fc']:.4f}   (nominal  6.06)",
        ]
    else:
        lines.append("  (fit failed — use nominal values)")
    lines += [
        "",
        "RL CHECKLIST  (all of these matter for PPO sim-to-real):",
        f"  [{'x' if noise_stats['omega_std']>0 else ' '}] ω obs noise in sim",
        f"  [{'x' if noise_stats['th_std_deg']>0 else ' '}] θ quantisation / obs noise in sim",
        f"  [{'x' if lag_s>0 else ' '}] action delay buffer in sim",
        f"  [{'x' if fit_params else ' '}] corrected system parameters",
        f"  [{'x' if dz_threshold and not np.isnan(dz_threshold) else ' '}] voltage dead zone in sim",
        f"  [{'x' if jitter_ms is not None else ' '}] dt jitter modelled (or ignored if <1ms)",
        "  [ ] domain randomisation ranges set from this data",
        "  [ ] filter alpha tuned to match hardware ω lag",
    ]
    ax.text(0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9))

    plt.savefig("sysid_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nPlot saved -> sysid_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Unbalanced Disk — System Identification")
    print("=" * 60)
    print("Make sure the disk is free to spin and USB is connected.\n")
    input("Press ENTER to start …")

    env = UnbalancedDisk_exp(umax=3.0, dt=0.025, render_mode=None)
    dt = env.dt

    # env.reset() does a 30s "wait until still" loop which we don't need here —
    # just initialize the attributes that step() requires.
    obs0 = env.get_obs()
    env.th_before = obs0[0]
    env._th_accumulated = 0.0
    env.omega_filtered = 0.0

    # ── Experiment 1: noise floor (θ AND ω) ──────────────────────────────────
    noise_t, noise_th, noise_om, noise_stats = exp_noise_floor(env, n_samples=600)

    # ── Experiment 2: step responses ─────────────────────────────────────────
    step_results = exp_step_responses(
        env,
        amplitudes=[1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        step_len=80, settle_steps=60
    )
    lag_s = estimate_lag_from_steps(step_results, dt)

    # ── Experiment 3: PRBS ───────────────────────────────────────────────────
    prbs_t, prbs_th, prbs_om, prbs_u = exp_prbs(env, n_samples=1200, amplitude=2.5)
    freq_f, freq_H = compute_freq_response(prbs_u, prbs_om, dt)

    # ── Experiment 4: free decay ─────────────────────────────────────────────
    decay_t, decay_th, decay_om, _ = exp_free_decay(env, n_samples=1000)
    omega0_decay = estimate_omega0_from_decay(decay_t, decay_th)

    # ── Experiment 5: dt jitter ──────────────────────────────────────────────
    jitter_ms = exp_dt_jitter(env, n_samples=300)

    # ── Experiment 6: voltage dead zone ──────────────────────────────────────
    dz_voltages, dz_moved, dz_threshold = exp_dead_zone(env)

    # ── Parameter fit ────────────────────────────────────────────────────────
    fit_params = fit_parameters(step_results, dt)

    # ── Save raw data ────────────────────────────────────────────────────────
    np.savez("sysid_data.npz",
             noise_t=noise_t, noise_th=noise_th, noise_om=noise_om,
             prbs_t=prbs_t, prbs_th=prbs_th, prbs_om=prbs_om, prbs_u=prbs_u,
             decay_t=decay_t, decay_th=decay_th, decay_om=decay_om,
             jitter_ms=jitter_ms,
             dz_voltages=dz_voltages, dz_moved=dz_moved,
             dt=dt)
    print("\nRaw data saved -> sysid_data.npz")

    # ── Plot everything ──────────────────────────────────────────────────────
    plot_results(
        noise_t, noise_th, noise_om,
        step_results,
        prbs_t, prbs_th, prbs_om, prbs_u,
        decay_t, decay_th, decay_om,
        noise_stats, lag_s, freq_f, freq_H,
        fit_params, dt,
        jitter_ms=jitter_ms,
        dz_voltages=dz_voltages, dz_moved=dz_moved, dz_threshold=dz_threshold,
    )

    env.close()

    # ── Print full RL checklist ───────────────────────────────────────────────
    n_delay = max(1, int(round(lag_s / dt)))
    print("\n" + "=" * 60)
    print("  RL TRAINING RECOMMENDATIONS")
    print("=" * 60)
    print(f"  1. ω obs noise     : σ ≈ {noise_stats['omega_std']:.4f} rad/s")
    print(f"     -> Add N(0, {noise_stats['omega_std']:.4f}) to ω in sim env")
    print(f"  2. θ obs noise     : σ ≈ {noise_stats['th_std_deg']:.4f}°  (quantisation)")
    print(f"     -> Add N(0, {np.deg2rad(noise_stats['th_std_deg']):.5f}) to θ in sim env")
    print(f"  3. Actuation lag   : ≈ {lag_s*1000:.1f} ms")
    print(f"     -> action_delay_buffer depth = {n_delay} step(s)")
    print(f"  4. Voltage dead zn : ≈ {dz_threshold:.2f} V")
    print(f"     -> clip sim action: u = 0 if |u| < {dz_threshold:.2f}")
    print(f"  5. dt jitter       : σ={np.std(jitter_ms):.2f} ms, max={np.max(jitter_ms):.1f} ms")
    if np.std(jitter_ms) > 2.0:
        print(f"     -> jitter is significant! add Uniform(±{np.std(jitter_ms):.1f}ms) dt noise in sim")
    else:
        print(f"     -> jitter is small, probably safe to ignore")
    if fit_params:
        print(f"  6. Update sim parameters:")
        print(f"       omega0={fit_params['omega0']:.4f}, gamma={fit_params['gamma']:.4f}, "
              f"Ku={fit_params['Ku']:.4f}, Fc={fit_params['Fc']:.4f}")
    print(f"  7. Filter alpha    : currently 0.30 in sincos env")
    print(f"     -> tune so filtered lag ≈ actuation lag (try 0.15–0.25)")
    print("=" * 60)
