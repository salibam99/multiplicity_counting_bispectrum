"""Module 01: OpenMC fixed-source transport -> single-source-neutron event library.

For each source history i: list of (capture delay, bank) for He3 captures in the
3 azimuthal banks, plus the sphere net leakage (surface source) giving M_MC.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import (DATA, PLOTS, StepLog, bank_of, build_model,  # noqa: E402
                             load_config, read_collision_tracks, sphere_net_leakage)

PREDELAY_US = 3.0
PROMPT_CUT_S = 10e-3   # captures later than this come from OpenMC delayed neutrons


def run_transport(cfg, n_hist, tag, threads=None):
    run = DATA / "openmc_runs" / tag
    run.mkdir(parents=True, exist_ok=True)
    for f in run.glob("*.h5"):
        f.unlink()
    model = build_model(cfg, n_hist)
    model.run(cwd=run, output=False, threads=threads)
    n_hist = model.settings.particles * model.settings.batches
    return run, n_hist


def build_library(run, n_hist):
    ct, _ = read_collision_tracks(run)
    order = np.lexsort((ct["time"], ct["parent_id"]))
    ct = ct[order]
    n_delayed = int((ct["time"] > PROMPT_CUT_S).sum())
    ct = ct[ct["time"] <= PROMPT_CUT_S]
    hid = ct["parent_id"].astype(np.int64) - 1          # 0-based history index
    t = ct["time"].astype(np.float64)
    bank = bank_of(ct["r"]["x"], ct["r"]["y"])
    counts = np.bincount(hid, minlength=n_hist)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    M, n_out, n_in = sphere_net_leakage(run, n_hist)
    return dict(t=t, bank=bank, offsets=offsets, counts=counts, n_hist=n_hist,
                n_delayed=n_delayed, M_mc=M, n_out=n_out, n_in=n_in)


def die_away_fit(t, predelay_us=PREDELAY_US, tmax_us=250.0, bw_us=4.0):
    edges = np.arange(0, tmax_us + bw_us, bw_us)
    h, _ = np.histogram(t * 1e6, edges)
    c = 0.5 * (edges[1:] + edges[:-1])
    m = c > predelay_us
    f = lambda x, A, tau: A * np.exp(-(x - predelay_us) / tau)  # noqa: E731
    p, cov = curve_fit(f, c[m], h[m], p0=[h[m][0], 45.0],
                       sigma=np.sqrt(np.maximum(h[m], 1)))
    res = h[m] - f(c[m], *p)
    r2 = 1 - (res**2).sum() / ((h[m] - h[m].mean()) ** 2).sum()
    return dict(tau_us=p[1], tau_err_us=np.sqrt(cov[1, 1]), A=p[0], r2=r2,
                centers=c, hist=h, mask=m, fitfun=f, popt=p)


def empirical_gate_fractions(t, P_us, G_us, n=4_000_000, seed=0):
    """f_d, f_t from iid draws of the empirical capture-time distribution."""
    rng = np.random.default_rng(seed)
    P, G = P_us * 1e-6, G_us * 1e-6
    x = rng.choice(t, size=(n, 3))
    d = np.abs(x[:, 0] - x[:, 1])
    x = np.sort(x, axis=1)
    fd = np.mean((d >= P) & (d < P + G))
    ft = np.mean((x[:, 1] - x[:, 0] >= P) & (x[:, 2] - x[:, 0] < P + G))
    return float(fd), float(ft)


def analyse(lib, log):
    n, t, bank, cnt = lib["n_hist"], lib["t"], lib["bank"], lib["counts"]
    det_per_src = len(t) / n
    eps = det_per_src / lib["M_mc"]
    eps_b = np.bincount(bank, minlength=3) / n / lib["M_mc"]
    da = die_away_fit(t)
    log.write(f"N_hist = {n},  He3 captures = {len(t)},  detections/source = {det_per_src:.4f}")
    log.write(f"dropped {lib['n_delayed']} delayed-neutron captures (t > {PROMPT_CUT_S*1e3:.0f} ms), "
              f"{lib['n_delayed']/(len(t)+lib['n_delayed']):.4%} of all")
    log.write(f"sphere crossings out={lib['n_out']} in={lib['n_in']}  M_MC = {lib['M_mc']:.4f}")
    log.write(f"epsilon = {eps:.4f};  per bank = {np.round(eps_b, 4)}  "
              f"(rel. dev from eps/3: {np.round(eps_b/(eps/3)-1, 4)})")
    log.write(f"die-away tau = {da['tau_us']:.2f} +- {da['tau_err_us']:.2f} us  (R^2 = {da['r2']:.5f})")
    log.write(f"captures/history: mean {cnt.mean():.4f}, var {cnt.var():.4f}, "
              f"<n(n-1)>/2 = {(cnt*(cnt-1)).mean()/2:.4f}")
    return dict(eps=eps, eps_banks=eps_b, die_away=da, det_per_src=det_per_src)


def plot(lib, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    da = res["die_away"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    c, h, m = da["centers"], da["hist"], da["mask"]
    ax[0].semilogy(c, np.maximum(h, 0.5), ".", ms=3, label="He3 capture times")
    ax[0].semilogy(c[m], da["fitfun"](c[m], *da["popt"]), "r-",
                   label=rf"fit $\tau$={da['tau_us']:.1f} µs, $R^2$={da['r2']:.4f}")
    ax[0].axvline(PREDELAY_US, color="k", ls=":", label="pre-delay 3 µs")
    ax[0].set(xlabel="capture delay [µs]", ylabel="counts / 4 µs", title="(a) die-away")
    ax[0].legend(fontsize=8)
    cnt = lib["counts"]
    k = np.arange(cnt.max() + 1)
    pk = np.bincount(cnt) / len(cnt)
    eps = res["eps"]
    ax[1].bar(k, pk, color="C0", label=f"OpenMC (mean {cnt.mean():.3f})")
    ax[1].plot([0, 1], [1 - eps, eps], "ro--", label=f"no multiplication (ε={eps:.3f})")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="detections per source neutron", ylabel="probability",
              title=f"(b) per-source-neutron multiplicity, M_MC={lib['M_mc']:.3f}")
    ax[1].legend(fontsize=8)
    eb = res["eps_banks"]
    ax[2].bar(["X", "Y", "Z"], eb, color=["C0", "C1", "C2"])
    ax[2].axhline(eps / 3, color="k", ls="--", label="ε/3")
    for i, v in enumerate(eb):
        ax[2].text(i, v, f"{v:.4f}", ha="center", va="bottom")
    ax[2].set(ylabel="bank efficiency", title="(c) bank balance", ylim=(0, eps / 3 * 1.2))
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(n_hist=None, tag="step01_full", out=None, log_name="step_01", log_mode="w"):
    cfg = load_config()
    n_hist = n_hist or cfg["transport"]["n_hist_full"]
    log = StepLog(log_name, mode=log_mode)
    log.write(f"--- run_openmc_transport: n_hist={n_hist}, tag={tag} ---")
    run, n_hist = run_transport(cfg, n_hist, tag)
    lib = build_library(run, n_hist)
    res = analyse(lib, log)
    G = cfg["shift_register"]["gate_us"]
    gate = {}
    for P in cfg["shift_register"]["predelays_us"]:
        fd, ft = empirical_gate_fractions(lib["t"], P, G)
        tau = res["die_away"]["tau_us"]
        fd_exp = np.exp(-P / tau) * (1 - np.exp(-G / tau))
        gate[P] = (fd, ft)
        log.write(f"P={P:.1f} us G={G:.0f} us: empirical f_d={fd:.4f} f_t={ft:.4f} | "
                  f"exponential f_d={fd_exp:.4f} f_t={fd_exp**2:.4f}")
    out = out or DATA / "01_openmc_event_library.npz"
    np.savez_compressed(
        out, t=lib["t"], bank=lib["bank"], offsets=lib["offsets"], n_hist=n_hist,
        M_mc=lib["M_mc"], eps=res["eps"], eps_banks=res["eps_banks"],
        tau_s=res["die_away"]["tau_us"] * 1e-6, tau_r2=res["die_away"]["r2"],
        gate_P_us=np.array(list(gate.keys())), gate_fd=np.array([g[0] for g in gate.values()]),
        gate_ft=np.array([g[1] for g in gate.values()]))
    log.write(f"saved {out}")
    plot(lib, res, PLOTS / "step_01_geometry_and_die_away.png")
    log.close()
    return lib, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hist", type=int, default=None)
    a = ap.parse_args()
    main(a.n_hist)
