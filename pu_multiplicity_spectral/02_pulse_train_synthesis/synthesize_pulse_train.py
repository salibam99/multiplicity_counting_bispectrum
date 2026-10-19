"""Module 02: stitch OpenMC single-neutron histories into a list-mode pulse train."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import DATA, PLOTS, StepLog, load_config, load_library, nu_moments  # noqa: E402


def synthesize(cfg, lib, T_meas, seed):
    rng = np.random.default_rng(seed)
    F = cfg["pu_sphere"]["F_true"]
    alpha = cfg["alpha"]
    p_nu = np.asarray(cfg["nu_sf_pu240"]["P_nu"], float)
    p_nu /= p_nu.sum()
    s1 = nu_moments(cfg)["s1"]
    off, t_lib, b_lib = lib["offsets"], lib["t"], lib["bank"]
    cnt = np.diff(off)
    n_hist = len(cnt)

    n_sf = rng.poisson(F * T_meas)
    n_an = rng.poisson(F * s1 * alpha * T_meas)
    t_sf = rng.uniform(0, T_meas, n_sf)
    nu = rng.choice(len(p_nu), size=n_sf, p=p_nu)
    t_an = rng.uniform(0, T_meas, n_an)
    # one "neutron" row per emitted source neutron
    birth = np.concatenate([np.repeat(t_sf, nu), t_an])
    parent = np.concatenate([np.repeat(np.arange(n_sf), nu), n_sf + np.arange(n_an)])
    etype = np.concatenate([np.zeros(nu.sum(), np.int8), np.ones(n_an, np.int8)])
    h = rng.integers(0, n_hist, size=len(birth))
    k = cnt[h]
    # expand each neutron into its detector captures
    rows = np.repeat(np.arange(len(h)), k)
    start = np.repeat(off[h], k)
    within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    idx = start + within
    ts = birth[rows] + t_lib[idx]
    o = np.argsort(ts, kind="stable")
    return dict(timestamps_sec=ts[o], bank_ids=b_lib[idx][o].astype(np.int8),
                parent_fission_ids=parent[rows][o], event_type=etype[rows][o],
                T_meas=T_meas, n_sf=n_sf, n_an=n_an, n_neutrons=len(h))


def rossi_alpha(ts, tmax=300e-6, bw=2e-6):
    """Histogram of all forward time differences t_j - t_i < tmax."""
    edges = np.arange(0, tmax + bw / 2, bw)
    h = np.zeros(len(edges) - 1)
    k = 1
    while True:
        d = ts[k:] - ts[:-k]
        d = d[d < tmax]
        if len(d) == 0:
            break
        h += np.histogram(d, edges)[0]
        k += 1
    return 0.5 * (edges[1:] + edges[:-1]), h


def fit_rossi(c, h, T_meas, bw, pre=3e-6):
    from scipy.optimize import curve_fit
    rate = h / (T_meas * bw)                       # pairs / s^2
    m = c > pre
    f = lambda x, A, tau, B: A * np.exp(-x / tau) + B  # noqa: E731
    p, cov = curve_fit(f, c[m], rate[m], p0=[rate[m][0] - rate[m][-1], 50e-6, rate[m][-1]])
    return rate, p, np.sqrt(np.diag(cov)), f


def plot(res, c, rate, p, f, S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(c * 1e6, rate, ".", ms=3, label="Rossi-α  R(Δt)")
    ax[0].plot(c * 1e6, f(c, *p), "r-", label=f"fit τ={p[1]*1e6:.1f} µs, B={p[2]:.4g}")
    ax[0].axhline(S**2, color="k", ls="--", label=f"S² = {S**2:.4g}")
    ax[0].set(xlabel="Δt [µs]", ylabel="pair rate density [s⁻²]", title="(a) Rossi-alpha")
    ax[0].legend(fontsize=8)
    ts, b = res["timestamps_sec"], res["bank_ids"]
    m = ts < ts[0] + 2e-3
    for i in range(3):
        ax[1].eventplot(ts[m & (b == i)] * 1e3, lineoffsets=i, linelengths=0.8, colors=f"C{i}")
    ax[1].set(yticks=[0, 1, 2], yticklabels=["X", "Y", "Z"], xlabel="t [ms]",
              title="(b) first 2 ms of the pulse stream")
    tb = np.arange(0, res["T_meas"] + 1e-9, 1.0)
    ax[2].plot(0.5 * (tb[1:] + tb[:-1]), np.histogram(ts, tb)[0], lw=0.8)
    ax[2].axhline(S, color="k", ls="--", label=f"S = {S:.0f} cps")
    ax[2].set(xlabel="t [s]", ylabel="counts / s", title="(c) count rate stability")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(T_meas=None, out=None, log_name="step_02", seed=None, rossi_span=60.0):
    cfg = load_config()
    lib = load_library()
    T_meas = T_meas or cfg["pulse_train"]["T_meas_s"]
    seed = cfg["pulse_train"]["seed"] if seed is None else seed
    log = StepLog(log_name)
    res = synthesize(cfg, lib, T_meas, seed)
    ts = res["timestamps_sec"]
    nu = nu_moments(cfg)
    F, a = cfg["pu_sphere"]["F_true"], cfg["alpha"]
    S_emp = len(ts) / T_meas
    S_theo = F * float(lib["eps"]) * float(lib["M_mc"]) * nu["s1"] * (1 + a)
    log.write(f"T_meas={T_meas} s: SF fissions={res['n_sf']}, (a,n)={res['n_an']}, "
              f"source neutrons={res['n_neutrons']}, pulses={len(ts)}")
    log.write(f"S_emp = {S_emp:.2f} cps, S_theo = F eps M nu_s1 (1+a) = {S_theo:.2f} cps, "
              f"rel diff = {S_emp/S_theo-1:+.4%}")
    span = min(rossi_span, T_meas)
    bw = 2e-6
    c, h = rossi_alpha(ts[ts < span], bw=bw)
    rate, p, perr, f = fit_rossi(c, h, span, bw)
    S_span = (ts < span).sum() / span
    log.write(f"Rossi-alpha (first {span:.0f} s): tau = {p[1]*1e6:.2f} +- {perr[1]*1e6:.2f} us "
              f"(step01 tau = {float(lib['tau_s'])*1e6:.2f} us), baseline B = {p[2]:.5g}, "
              f"S^2 = {S_span**2:.5g} (rel {p[2]/S_span**2-1:+.4%})")
    out = out or DATA / "02_list_mode_pulses.npz"
    np.savez_compressed(out, timestamps_sec=ts, bank_ids=res["bank_ids"],
                        parent_fission_ids=res["parent_fission_ids"],
                        event_type=res["event_type"], T_meas=T_meas)
    log.write(f"saved {out}")
    plot(res, c, rate, p, f, S_span, PLOTS / "step_02_rossi_alpha_and_pulse_stream.png")
    log.close()
    return dict(S_emp=S_emp, S_theo=S_theo, tau_rossi=p[1], tau_lib=float(lib["tau_s"]),
                B=p[2], S2=S_span**2, sorted=bool(np.all(np.diff(ts) >= 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=None)
    main(ap.parse_args().T)
