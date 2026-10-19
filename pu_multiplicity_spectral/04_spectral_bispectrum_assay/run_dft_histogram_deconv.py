"""Method 2A: DFT / characteristic-function deconvolution of shift-register histograms.

f(n) = (p * b)(n): the R+A distribution is the accidental A distribution
convolved with the distribution p(k) of correlated partners of the trigger.
p = IFFT(FFT f / FFT b);  r1 = sum k p(k),  r2 = sum C(k,2) p(k);
D = S r1, T = S r2 (the b1(f1-b1) cross-term is removed by the division).
Cross-check: Psi(theta) = ln f~ - ln b~ = ln G_p(z), z = e^{-i theta}; fitting
Psi as a polynomial in w = z - 1 gives factorial cumulants k1 = r1,
k2 = 2 r2 - r1^2.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "03_traditional_shift_register"))
from common.pu_model import (DATA, PLOTS, StepLog, effective_nu, invert_point_model,  # noqa: E402
                             load_config, load_library)
from run_shift_register import histos, shift_register  # noqa: E402


def deconvolve(f, b, n_fft=64):
    fn, bn = np.zeros(n_fft), np.zeros(n_fft)
    fn[: len(f)], bn[: len(b)] = f / f.sum(), b / b.sum()
    Ff, Fb = np.fft.fft(fn), np.fft.fft(bn)
    p = np.fft.ifft(Ff / Fb).real
    k = np.arange(n_fft)
    k = np.where(k < n_fft // 2, k, k - n_fft)   # negative lags = wrap-around noise
    r1 = (k * p).sum()
    r2 = (k * (k - 1) / 2 * p).sum()
    return p, r1, r2, Ff, Fb


def cgf_moments(Ff, Fb, n_fit=8, deg=6):
    n_fft = len(Ff)
    j = np.r_[1:n_fit + 1, -n_fit:0]
    theta = 2 * np.pi * j / n_fft
    psi = np.log(Ff[j % n_fft] / Fb[j % n_fft])
    w = np.exp(-1j * theta) - 1
    from math import factorial
    A =np.stack([w**q / factorial(q) for q in range(1, deg + 1)], 1)
    kap = np.linalg.lstsq(A, psi, rcond=None)[0].real
    k1, k2 = kap[0], kap[1]
    return k1, (k2 + k1**2) / 2, psi, theta


def run(pulses_path=DATA / "02_list_mode_pulses.npz", log_name="step_04a", n_seg=10):
    cfg = load_config()
    lib = load_library()
    sr = cfg["shift_register"]
    d = np.load(pulses_path)
    ts, T_meas = d["timestamps_sec"], float(d["T_meas"])
    S = len(ts) / T_meas
    eps, nu = float(lib["eps"]), effective_nu(cfg, lib)
    f240, sfg = cfg["pu_sphere"]["wt_frac_pu240"], cfg["pu_sphere"]["sf_rate_per_g240"]
    log = StepLog(log_name)
    results = {}
    for P_us in sr["predelays_us"]:
        i = list(lib["gate_P_us"]).index(P_us)
        fd, ft = float(lib["gate_fd"][i]), float(lib["gate_ft"][i])
        trig, nf, nb = shift_register(ts, P_us * 1e-6, sr["gate_us"] * 1e-6, sr["long_delay_us"] * 1e-6)
        f, b = histos(nf, nb)
        p, r1, r2, Ff, Fb = deconvolve(f, b, cfg["spectral"]["n_fft_hist"])
        c1, c2, psi, theta = cgf_moments(Ff, Fb)
        D, T = S * r1, S * r2
        inv = invert_point_model(S, D, T, eps, fd, ft, nu, f240, sfg)
        seg = np.minimum((trig / T_meas * n_seg).astype(int), n_seg - 1)
        sv = []
        for s in range(n_seg):
            m = seg == s
            fs, bs = histos(nf[m], nb[m])
            _, a1, a2, _, _ = deconvolve(fs, bs)
            Ss = m.sum() / (T_meas / n_seg)
            iv = invert_point_model(Ss, Ss * a1, Ss * a2, eps, fd, ft, nu, f240, sfg)
            sv.append([Ss * a1 / fd, Ss * a2 / ft, iv["M"], iv["alpha"], iv["m_eff"], iv["m_pu"]])
        e = np.std(sv, axis=0, ddof=1) / np.sqrt(n_seg)
        res = dict(P_us=P_us, S=S, r1=r1, r2=r2, r1_cgf=c1, r2_cgf=c2, D=D, T=T,
                   Dinf=D / fd, Tinf=T / ft, Dinf_err=e[0], Tinf_err=e[1], p=p[:12].tolist(),
                   p_neg_lag_sum=float(p[32:].sum()), inv=inv,
                   inv_err=dict(zip(("M", "alpha", "m_eff", "m_pu"), map(float, e[2:]))))
        results[f"P{P_us:g}"] = res
        log.write(f"-- P={P_us} us --  p(0..6) = {np.round(p[:7], 5)}  sum p = {p.sum():.6f}, "
                  f"wrap-around mass = {res['p_neg_lag_sum']:.2e}")
        log.write(f"r1={r1:.5f} r2={r2:.5f} | CGF fit: r1={c1:.5f} r2={c2:.5f}")
        log.write(f"D=S r1={D:.2f} T=S r2={T:.2f} -> D_inf={D/fd:.2f}±{e[0]:.2f} T_inf={T/ft:.2f}±{e[1]:.2f}")
        log.write(f"inversion: M={inv['M']:.4f} alpha={inv['alpha']:.4f} m_eff={inv['m_eff']:.3f} g "
                  f"m_Pu={inv['m_pu']:.2f}±{res['inv_err']['m_pu']:.2f} g "
                  f"({inv['m_pu']/cfg['pu_sphere']['mass_g']-1:+.2%})")
        if P_us == 3.0:
            plot(p, r1, r2, psi, theta, c1, c2, PLOTS / "step_04a_dft_deconvolved_pk.png")
    with open(DATA / "04a_dft_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    log.close()
    return results


def plot(p, r1, r2, psi, theta, c1, c2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import factorial
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    k = np.arange(7)
    ax[0].bar(k, p[:7])
    for kk in k:
        ax[0].text(kk, p[kk], f"{p[kk]:.4f}", ha="center", va="bottom", fontsize=8)
    ax[0].set(xlabel="k correlated partners in gate", ylabel="p(k)",
              title=f"IFFT(f̃/b̃): r1={r1:.4f}, r2={r2:.4f}")
    ax[1].bar(k[1:], p[1:7])
    ax[1].set_yscale("symlog", linthresh=1e-4)
    ax[1].set(xlabel="k", title="p(k), k ≥ 1 (symlog)")
    o = np.argsort(theta)
    ax[2].plot(theta[o], psi.real[o], "o-", label="Re Ψ(θ)")
    ax[2].plot(theta[o], psi.imag[o], "s-", label="Im Ψ(θ)")
    th = np.linspace(theta.min(), theta.max(), 200)
    w = np.exp(-1j * th) - 1
    k2 = 2 * c2 - c1**2
    approx = c1 * w + k2 * w**2 / factorial(2)
    ax[2].plot(th, approx.real, "k--", lw=0.8, label="κ1 w + κ2 w²/2")
    ax[2].plot(th, approx.imag, "k:", lw=0.8)
    ax[2].set(xlabel="θ", title=f"CGF Ψ = ln f̃ − ln b̃: r1={c1:.4f}, r2={c2:.4f}")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
