"""Method 2B: Welch PSD, 3-bank cross-PSD and 3-bank cross-bispectrum of the pulse train.

Signals: counts per dt bin in each bank, per segment of length T_seg, per-segment
mean removed (kills DC/accidental terms). With X(f) = sum_n x_n exp(-2 pi i f n dt):
  PSD      S_ss(f)            = <|X+Y+Z|^2>/T_seg        = S + 2 D_inf L(f)
  crossPSD S_c(f)             = Re<X*Y + Y*Z + Z*X>/T_seg = 2 D_inf L(f) * sum_{i<j} p_i p_j
  bispec   B(f1,f2)           = Re<mean_perm A(f1) B(f2) C*(f1+f2)>/T_seg
                              = 6 T_inf p_X p_Y p_Z Re[h(w1) h(w2) h*(w1+w2)]
with h(w) = 1/(1+i w tau), L = |h|^2, p_i = bank detection fractions.
Singles (single neutron -> one bank) and doubles (pair -> at most two banks)
cannot populate three distinct banks, so they vanish identically in B.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import (DATA, PLOTS, StepLog, effective_nu, invert_point_model,  # noqa: E402
                             load_config, load_library, reference_rates)

PERMS = list(itertools.permutations(range(3)))


def _take(A, k):
    """A(f_k) for signed bin index k of a real signal's rfft (A(-f) = A(f)*)."""
    v = A[:, np.abs(k)]
    return np.where(k >= 0, v, np.conj(v))


def accumulate(ts, banks, T_meas, dt=2e-6, T_seg=16.384e-3, f_psd_max=100e3,
               f_bis_max=10e3, stride=1, n_groups=10, chunk=48):
    nbin = int(round(T_seg / dt))
    nseg = int(T_meas // T_seg)
    df = 1.0 / T_seg
    kpsd = int(f_psd_max / df)
    kb = np.arange(stride, int(f_bis_max / df) + 1, stride)          # f1 > 0
    kb2 = np.concatenate([-kb[::-1], kb])                              # f2 of both signs
    kmax = max(kpsd, 2 * kb[-1])
    seg = np.floor(ts / T_seg).astype(np.int64)
    keep = seg < nseg
    ts, banks, seg = ts[keep], banks[keep], seg[keep]
    binx = np.minimum(((ts - seg * T_seg) / dt).astype(np.int64), nbin - 1)
    edges = np.linspace(0, nseg, n_groups + 1).astype(int)
    out = dict(psd=np.zeros((n_groups, kpsd + 1)), psd_bank=np.zeros((n_groups, 3, kpsd + 1)),
               cross=np.zeros((n_groups, kpsd + 1)), bis=np.zeros((n_groups, len(kb), len(kb2))),
               nseg=np.diff(edges), f=np.arange(kpsd + 1) * df, fb=kb * df, fb2=kb2 * df, T_seg=T_seg,
               dt=dt, bank_counts=np.bincount(banks, minlength=3))
    I1, I2 = np.meshgrid(kb, kb2, indexing="ij")
    for g in range(n_groups):
        for s0 in range(edges[g], edges[g + 1], chunk):
            s1 = min(s0 + chunk, edges[g + 1])
            a, b = np.searchsorted(seg, [s0, s1])
            idx = ((seg[a:b] - s0) * nbin + binx[a:b]) * 3 + banks[a:b]
            x = np.bincount(idx, minlength=(s1 - s0) * nbin * 3).astype(float)
            x = x.reshape(s1 - s0, nbin, 3)
            x -= x.mean(axis=1, keepdims=True)
            F = np.fft.rfft(x, axis=1)[:, : kmax + 1, :]
            X, Y, Z = F[..., 0], F[..., 1], F[..., 2]
            s = X + Y + Z
            out["psd"][g] += (np.abs(s[:, : kpsd + 1]) ** 2).sum(0) / T_seg
            out["psd_bank"][g] += (np.abs(F[:, : kpsd + 1, :]) ** 2).sum(0).T / T_seg
            c = np.conj(X) * Y + np.conj(Y) * Z + np.conj(Z) * X
            out["cross"][g] += c[:, : kpsd + 1].real.sum(0) / T_seg
            acc = np.zeros((s1 - s0, len(kb), len(kb2)), complex)
            for p in PERMS:
                A, B, C = F[..., p[0]], F[..., p[1]], F[..., p[2]]
                acc += A[:, I1] * _take(B, I2) * np.conj(_take(C, I1 + I2))
            out["bis"][g] += (acc.real / len(PERMS)).sum(0) / T_seg
        for k in ("psd", "psd_bank", "cross", "bis"):
            out[k][g] /= out["nseg"][g]
    return out


def lorentz(f, tau):
    return 1.0 / (1.0 + (2 * np.pi * f * tau) ** 2)


def h2(f, w, t1, t2):
    """Two-component die-away transfer function (1-w)/(1+iwt1) + w/(1+iwt2)."""
    W = 2j * np.pi * f
    return (1 - w) / (1 + W * t1) + w / (1 + W * t2)


def bis_kernel(f1, f2, tau, w=0.0, t2=None):
    t2 = tau if t2 is None else t2
    return (h2(f1, w, tau, t2) * h2(f2, w, tau, t2) * np.conj(h2(f1 + f2, w, tau, t2))).real


def fit_spectra(acc, psd, cross, bis, tau0=50e-6):
    """Zero-frequency amplitudes with a flexible two-component kernel.

    The plateau values (f -> 0) are kernel independent; a single Lorentzian
    biases them when the die-away / chain timing is not a pure exponential.
    The bispectrum gets its own free kernel, so non-iid triple timing (e.g.
    thermal-return fission chains) needs no gate-fraction calibration.
    """
    f, fb, fb2 = acc["f"], acc["fb"], acc["fb2"]
    p = acc["bank_counts"] / acc["bank_counts"].sum()
    pair = p[0] * p[1] + p[1] * p[2] + p[2] * p[0]
    trip = p[0] * p[1] * p[2]
    m = f > 0
    lo, hi = [0, 0, 1e-6, 1e-6], [np.inf, 1, 1e-3, 1e-2]
    k0 = [0.3, 0.8 * tau0, 1.5 * tau0]
    (A, B, w, t1, t2), _ = curve_fit(lambda x, A, B, w, a, b: A + B * np.abs(h2(x, w, a, b)) ** 2,
                                     f[m], psd[m], p0=[psd[m][-1], psd[m][1] - psd[m][-1]] + k0,
                                     bounds=([0] + lo, [np.inf] + hi))
    (C, off, wc, c1, c2), _ = curve_fit(
        lambda x, C, o, w, a, b: o + C * np.abs(h2(x, w, a, b)) ** 2, f[m], cross[m],
        p0=[cross[m][1], 0.0] + k0, bounds=([0, -np.inf] + lo[1:], [np.inf, np.inf] + hi[1:]))
    F1, F2 = np.meshgrid(fb, fb2, indexing="ij")
    ok = np.abs(F1 + F2) > 0.5 * (fb[1] - fb[0]) if len(fb) > 1 else np.ones_like(F1, bool)
    btot = bis / trip
    xy = np.vstack([F1[ok], F2[ok]])
    (K, wb, b1, b2), _ = curve_fit(lambda q, K, w, a, b: K * bis_kernel(q[0], q[1], a, w, b),
                                   xy, btot[ok], p0=[6 * B / 2 * 0.8] + k0,
                                   bounds=([0] + lo[1:], [np.inf] + hi[1:]))
    teff = lambda w_, a_, b_: (1 - w_) * a_ + w_ * b_  # noqa: E731
    return dict(S_psd=A, D_psd=B / 2, tau_psd=teff(w, t1, t2), D_cross=C / pair / 2,
                cross_offset=off / pair, tau_cross=teff(wc, c1, c2), T_bis=K / 6,
                tau_bis=teff(wb, b1, b2), kern_psd=(w, t1, t2), kern_cross=(wc, c1, c2),
                kern_bis=(wb, b1, b2), pair=pair, trip=trip, p=p)


def spectral_assay(ts, banks, T_meas, cfg, n_groups=10, tau0=50e-6):
    sp = cfg["spectral"]
    acc = accumulate(ts, banks, T_meas, dt=sp["dt_us"] * 1e-6, T_seg=sp["T_seg_ms"] * 1e-3,
                     f_bis_max=sp["bispec_fmax_kHz"] * 1e3, n_groups=n_groups)
    w = acc["nseg"][:, None]
    mean = lambda a: (a * w.reshape((-1,) + (1,) * (a.ndim - 1))).sum(0) / w.sum()  # noqa: E731
    fit = fit_spectra(acc, mean(acc["psd"]), mean(acc["cross"]), mean(acc["bis"]), tau0)
    grp = [fit_spectra(acc, acc["psd"][g], acc["cross"][g], acc["bis"][g], tau0)
           for g in range(n_groups)]
    err = {k: float(np.std([gg[k] for gg in grp], ddof=1) / np.sqrt(n_groups))
           for k in ("S_psd", "D_psd", "D_cross", "T_bis", "tau_psd", "tau_cross", "tau_bis",
                     "cross_offset")}
    return acc, fit, err, mean


def fit_1pole(acc, cross, bis, tau0=50e-6):
    """Single-Lorentzian / single-exponential kernel (spec form), for comparison."""
    f, fb, fb2 = acc["f"], acc["fb"], acc["fb2"]
    p = acc["bank_counts"] / acc["bank_counts"].sum()
    m = f > 0
    (C, t), _ = curve_fit(lambda x, C, t: C * lorentz(x, t), f[m], cross[m], p0=[cross[m][1], tau0])
    F1, F2 = np.meshgrid(fb, fb2, indexing="ij")
    ok = np.abs(F1 + F2) > 0.5 * (fb[1] - fb[0])
    btot = bis / (p[0] * p[1] * p[2])
    (K, tb), _ = curve_fit(lambda q, K, t: K * bis_kernel(q[0], q[1], t),
                           np.vstack([F1[ok], F2[ok]]), btot[ok], p0=[btot[ok].max(), tau0])
    pair = p[0] * p[1] + p[1] * p[2] + p[2] * p[0]
    return dict(D_cross=C / pair / 2, tau_cross=t, T_bis=K / 6, tau_bis=tb)


def plots(acc, fit, mean, path_b, path_c, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = acc["f"]
    m = f > 0
    psd, cross = mean(acc["psd"]), mean(acc["cross"])
    kp, kc, kb = fit["kern_psd"], fit["kern_cross"], fit["kern_bis"]
    Lp = np.abs(h2(f[m], *kp)) ** 2
    Lc = np.abs(h2(f[m], *kc)) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].semilogx(f[m] / 1e3, psd[m], ".", ms=2, alpha=0.5, label="PSD $S_{ss}$ (total)")
    ax[0].semilogx(f[m] / 1e3, fit["S_psd"] + 2 * fit["D_psd"] * Lp, "r-",
                   label=f"fit S={fit['S_psd']:.0f}, 2D∞={2*fit['D_psd']:.0f}")
    ax[0].semilogx(f[m] / 1e3, cross[m] / fit["pair"], ".", ms=2, alpha=0.5,
                   label="cross-PSD / Σp_ip_j  (= 3 S_cross)")
    ax[0].semilogx(f[m] / 1e3, 2 * fit["D_cross"] * Lc, "k-",
                   label=f"fit 2D∞={2*fit['D_cross']:.0f}, offset={fit['cross_offset']:.1f}")
    if ref is not None:
        ax[0].axhline(2 * ref["Dinf"], color="g", ls="--", lw=1, label=f"2D∞ reference {2*ref['Dinf']:.0f}")
    ax[0].axhline(fit["S_psd"], color="r", ls=":", lw=0.8)
    ax[0].axhline(0, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="f [kHz]", ylabel="two-sided spectral density [s⁻¹]",
              title="PSD (white floor S) vs 3-bank cross-PSD (zero floor)")
    ax[0].legend(fontsize=8)
    ax[1].loglog(f[m] / 1e3, psd[m] - fit["S_psd"], ".", ms=2, alpha=0.4, label="PSD − S")
    ax[1].loglog(f[m] / 1e3, cross[m] / fit["pair"], ".", ms=2, alpha=0.4, label="cross-PSD ×3")
    ax[1].loglog(f[m] / 1e3, 2 * fit["D_cross"] * Lc, "k-", label="2-component kernel fit")
    ax[1].loglog(f[m] / 1e3, 2 * fit["D_cross"] * lorentz(f[m], fit["tau_cross"]), "m--", lw=0.8,
                 label=f"Lorentzian τ_eff={fit['tau_cross']*1e6:.1f} µs")
    ax[1].axvline(1 / (2 * np.pi * fit["tau_cross"]) / 1e3, color="g", ls="--",
                  label=f"f_c = {1/(2*np.pi*fit['tau_cross'])/1e3:.2f} kHz")
    ax[1].set(xlabel="f [kHz]", ylim=(10, None), title="correlated part (log-log)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_b, dpi=130)
    plt.close(fig)

    fb, fb2 = acc["fb"], acc["fb2"]
    btot = mean(acc["bis"]) / fit["trip"]
    F1, F2 = np.meshgrid(fb, fb2, indexing="ij")
    model = 6 * fit["T_bis"] * bis_kernel(F1, F2, kb[1], kb[0], kb[2])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    v = 6 * fit["T_bis"]
    im = ax[0].pcolormesh(fb / 1e3, fb2 / 1e3, btot.T, shading="nearest", cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax[0], label="$B^{tot}_{xyz}$ [s⁻¹]")
    ax[0].set(xlabel="f1 [kHz]", ylabel="f2 [kHz]", title="3-bank cross-bispectrum (data)")
    im = ax[1].pcolormesh(fb / 1e3, fb2 / 1e3, (btot - model).T, shading="nearest", cmap="RdBu_r",
                          vmin=-v / 3, vmax=v / 3)
    fig.colorbar(im, ax=ax[1], label="residual")
    ax[1].set(xlabel="f1 [kHz]", ylabel="f2 [kHz]", title="data − fit")
    j = np.searchsorted(fb2, fb)
    d = btot[np.arange(len(fb)), j]
    ax[2].plot(fb / 1e3, d, "o", ms=2, alpha=0.6, label="diagonal $B^{tot}(f,f)$")
    ff = np.linspace(0, fb[-1], 400)
    ax[2].plot(ff / 1e3, 6 * fit["T_bis"] * bis_kernel(ff, ff, kb[1], kb[0], kb[2]), "r-",
               label=f"6T∞·Re[h(ω)²h*(2ω)], T∞={fit['T_bis']:.0f}")
    ax[2].plot(ff / 1e3, 6 * fit["T_bis"] * bis_kernel(ff, ff, fit["tau_bis"]), "m--", lw=0.8,
               label=f"exponential form Re[(1+iωτ)⁻²(1−2iωτ)⁻¹], τ={fit['tau_bis']*1e6:.1f} µs")
    if ref is not None:
        ax[2].axhline(6 * ref["Tinf"], color="g", ls="--", label=f"6T∞ reference = {6*ref['Tinf']:.0f}")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="f [kHz]", title="diagonal slice f1 = f2")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_c, dpi=130)
    plt.close(fig)


def run(pulses_path=DATA / "02_list_mode_pulses.npz", log_name="step_04b"):
    cfg = load_config()
    lib = load_library()
    ref = reference_rates(cfg, lib)
    d = np.load(pulses_path)
    ts, banks, T_meas = d["timestamps_sec"], d["bank_ids"], float(d["T_meas"])
    log = StepLog(log_name)
    acc, fit, err, mean = spectral_assay(ts, banks, T_meas, cfg)
    f1p = fit_1pole(acc, mean(acc["cross"]), mean(acc["bis"]))
    S_count = len(ts) / T_meas
    log.write(f"segments={acc['nseg'].sum()} x {acc['T_seg']*1e3:.3f} ms, dt={acc['dt']*1e6} us, "
              f"bank fractions={np.round(fit['p'], 5)}; bispectrum grid {len(acc['fb'])}x{len(acc['fb2'])}")
    log.write(f"reference (exact library moments): S={ref['S']:.2f} D_inf={ref['Dinf']:.2f} T_inf={ref['Tinf']:.2f}")
    log.write(f"PSD:   S_floor={fit['S_psd']:.2f}±{err['S_psd']:.2f} (counted {S_count:.2f}), "
              f"D_inf={fit['D_psd']:.2f}±{err['D_psd']:.2f}, tau_eff={fit['tau_psd']*1e6:.2f} us")
    log.write(f"cross: D_inf={fit['D_cross']:.2f}±{err['D_cross']:.2f}, kernel(w,t1,t2)={fit['kern_cross']}, "
              f"floor offset={fit['cross_offset']:.2f}±{err['cross_offset']:.2f} (expected 0)")
    log.write(f"bispec: T_inf={fit['T_bis']:.2f}±{err['T_bis']:.2f}, kernel(w,t1,t2)={fit['kern_bis']}")
    log.write(f"single-exponential kernel (spec form) for comparison: D_inf={f1p['D_cross']:.2f} "
              f"(tau {f1p['tau_cross']*1e6:.2f} us), T_inf={f1p['T_bis']:.2f} (tau {f1p['tau_bis']*1e6:.2f} us)")
    nu = effective_nu(cfg, lib)
    eps = float(lib["eps"])
    f240, sfg = cfg["pu_sphere"]["wt_frac_pu240"], cfg["pu_sphere"]["sf_rate_per_g240"]
    inv = invert_point_model(S_count, fit["D_cross"], fit["T_bis"], eps, 1.0, 1.0, nu, f240, sfg)
    inv_1p = invert_point_model(S_count, f1p["D_cross"], f1p["T_bis"], eps, 1.0, 1.0, nu, f240, sfg)
    grp = []
    for g in range(len(acc["nseg"])):
        fg = fit_spectra(acc, acc["psd"][g], acc["cross"][g], acc["bis"][g])
        grp.append(invert_point_model(S_count, fg["D_cross"], fg["T_bis"], eps, 1, 1, nu, f240, sfg))
    inv_err = {k: float(np.nanstd([g_[k] for g_ in grp], ddof=1) / np.sqrt(len(grp)))
               for k in ("M", "alpha", "m_eff", "m_pu")}
    m0 = cfg["pu_sphere"]["mass_g"]
    log.write(f"inversion (S counted, D_inf cross-PSD, T_inf bispectrum, f_d=f_t=1): M={inv['M']:.4f} "
              f"alpha={inv['alpha']:.4f} m_eff={inv['m_eff']:.3f} g m_Pu={inv['m_pu']:.2f}±"
              f"{inv_err['m_pu']:.2f} g ({inv['m_pu']/m0-1:+.2%})")
    log.write(f"inversion with single-exponential kernel: M={inv_1p['M']:.4f} alpha={inv_1p['alpha']:.4f} "
              f"m_Pu={inv_1p['m_pu']:.2f} g ({inv_1p['m_pu']/m0-1:+.2%})")
    res = dict(S=S_count, S_psd=fit["S_psd"], Dinf=fit["D_cross"], Dinf_psd=fit["D_psd"],
               Tinf=fit["T_bis"], tau_psd=fit["tau_psd"], tau_cross=fit["tau_cross"],
               tau_bis=fit["tau_bis"], cross_offset=fit["cross_offset"], err=err, inv=inv,
               inv_err=inv_err, one_pole=f1p, inv_one_pole=inv_1p)
    with open(DATA / "04b_psd_bispectrum_results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(DATA / "04b_spectra.npz", f=acc["f"], fb=acc["fb"], fb2=acc["fb2"],
                        psd=mean(acc["psd"]), cross=mean(acc["cross"]), bis=mean(acc["bis"]))
    plots(acc, fit, mean, PLOTS / "step_04b_psd_vs_cross_psd.png",
          PLOTS / "step_04c_cross_bispectrum_2d_and_diagonal.png", ref=ref)
    log.close()
    return res


if __name__ == "__main__":
    run()
