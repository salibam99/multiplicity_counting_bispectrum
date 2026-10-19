"""Module 03: digital multiplicity shift register + Böhnel point-model inversion."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import (DATA, PLOTS, StepLog, effective_nu, invert_point_model,  # noqa: E402
                             load_config, load_library, nu_moments)

N_BG_GATES = 8      # contiguous, non-overlapping background gates starting at L
N_SEG = 10          # time segments for the statistical error (segment scatter)


def shift_register(ts, P, G, L, nmax=25, n_bg=N_BG_GATES):
    """Exact sliding-window register: every pulse is a trigger.

    Foreground R+A gate (t+P, t+P+G], background A gates (t+L+jG, t+L+(j+1)G].
    Returns per-trigger foreground / background counts (bg averaged over gates
    as a histogram) and the trigger times.
    """
    t_end = ts[-1] - (L + n_bg * G)
    trig = ts[ts <= t_end]
    lo = np.searchsorted(ts, trig + P, side="right")
    hi = np.searchsorted(ts, trig + P + G, side="right")
    nf = np.minimum(hi - lo, nmax)
    nb = []
    for j in range(n_bg):
        a = np.searchsorted(ts, trig + L + j * G, side="right")
        b = np.searchsorted(ts, trig + L + (j + 1) * G, side="right")
        nb.append(np.minimum(b - a, nmax))
    return trig, nf, np.stack(nb, 1)


def histos(nf, nb, nmax=25):
    f = np.bincount(nf, minlength=nmax + 1)[: nmax + 1].astype(float)
    b = np.bincount(nb.ravel(), minlength=nmax + 1)[: nmax + 1].astype(float) / nb.shape[1]
    return f, b


def rates_from_histos(f, b, S):
    """Reduced factorial moments -> S, D, T (Ensslin)."""
    n = np.arange(len(f))
    fn, bn = f / f.sum(), b / b.sum()
    f1, b1 = (n * fn).sum(), (n * bn).sum()
    f2, b2 = (n * (n - 1) / 2 * fn).sum(), (n * (n - 1) / 2 * bn).sum()
    D = S * (f1 - b1)
    T = S * ((f2 - b2) - b1 * (f1 - b1))
    return dict(S=S, D=D, T=T, f1=f1, b1=b1, f2=f2, b2=b2, cross=S * b1 * (f1 - b1))


def segment_errors(trig, nf, nb, T_meas, fn, n_seg=N_SEG):
    """Scatter of a statistic fn(f_hist, b_hist, S_seg) over equal time segments."""
    seg = np.minimum((trig / T_meas * n_seg).astype(int), n_seg - 1)
    vals = []
    for s in range(n_seg):
        m = seg == s
        f, b = histos(nf[m], nb[m])
        vals.append(fn(f, b, m.sum() / (T_meas / n_seg)))
    vals = np.array(vals, float)
    return np.std(vals, axis=0, ddof=1) / np.sqrt(n_seg)


def assay(rates, eps, fd, ft, nu, cfg):
    f240 = cfg["pu_sphere"]["wt_frac_pu240"]
    sf = cfg["pu_sphere"]["sf_rate_per_g240"]
    return invert_point_model(rates["S"], rates["D"], rates["T"], eps, fd, ft, nu, f240, sf)


def run(pulses_path=DATA / "02_list_mode_pulses.npz", log_name="step_03", make_plot=True):
    cfg = load_config()
    lib = load_library()
    sr = cfg["shift_register"]
    G, L = sr["gate_us"] * 1e-6, sr["long_delay_us"] * 1e-6
    d = np.load(pulses_path)
    ts, T_meas = d["timestamps_sec"], float(d["T_meas"])
    eps = float(lib["eps"])
    nu_nom, nu_eff = nu_moments(cfg), effective_nu(cfg, lib)
    log = StepLog(log_name)
    log.write(f"pulses={len(ts)}, T={T_meas} s, G={G*1e6} us, L={L*1e6} us, {N_BG_GATES} bg gates")
    log.write(f"eps={eps:.5f}, M_MC={float(lib['M_mc']):.5f}, tau={float(lib['tau_s'])*1e6:.2f} us")
    log.write(f"nu nominal: {nu_nom}")
    log.write(f"nu effective (OpenMC implicit fission, calibrated from library): "
              f"i2={nu_eff['i2']:.4f} i3={nu_eff['i3']:.4f}")
    results = {}
    for P_us in sr["predelays_us"]:
        P = P_us * 1e-6
        trig, nf, nb = shift_register(ts, P, G, L, sr["n_max"])
        S = len(ts) / T_meas
        f, b = histos(nf, nb, sr["n_max"])
        r = rates_from_histos(f, b, S)
        i = list(lib["gate_P_us"]).index(P_us)
        fd_emp, ft_emp = float(lib["gate_fd"][i]), float(lib["gate_ft"][i])
        tau = float(lib["tau_s"])
        fd_exp = np.exp(-P / tau) * (1 - np.exp(-G / tau))
        err = segment_errors(trig, nf, nb, T_meas,
                             lambda f_, b_, s_: [rates_from_histos(f_, b_, s_)[k] for k in "SDT"])
        inv_eff = assay(r, eps, fd_emp, ft_emp, nu_eff, cfg)
        inv_nom = assay(r, eps, fd_emp, ft_emp, nu_nom, cfg)
        inv_exp = assay(r, eps, fd_exp, fd_exp**2, nu_eff, cfg)
        inv_err = segment_errors(
            trig, nf, nb, T_meas,
            lambda f_, b_, s_: [assay(rates_from_histos(f_, b_, s_), eps, fd_emp, ft_emp,
                                      nu_eff, cfg)[k] for k in ("M", "alpha", "m_eff", "m_pu")])
        res = dict(P_us=P_us, **{k: float(v) for k, v in r.items()},
                   S_err=float(err[0]), D_err=float(err[1]), T_err=float(err[2]),
                   fd=fd_emp, ft=ft_emp, fd_exp=float(fd_exp), ft_exp=float(fd_exp**2),
                   Dinf=r["D"] / fd_emp, Tinf=r["T"] / ft_emp,
                   Dinf_err=float(err[1]) / fd_emp, Tinf_err=float(err[2]) / ft_emp,
                   inv_eff=inv_eff, inv_nom=inv_nom, inv_expgate=inv_exp,
                   inv_err=dict(zip(("M", "alpha", "m_eff", "m_pu"), map(float, inv_err))),
                   f_hist=f.tolist(), b_hist=b.tolist())
        results[f"P{P_us:g}"] = res
        log.write(f"\n-- P = {P_us} us --")
        log.write(f"triggers={len(trig)}  f1={r['f1']:.5f} b1={r['b1']:.5f} f2={r['f2']:.5f} "
                  f"b2={r['b2']:.5f}  cross b1(f1-b1)={r['b1']*(r['f1']-r['b1']):.5f}")
        log.write(f"S={r['S']:.2f}±{err[0]:.2f}  D={r['D']:.2f}±{err[1]:.2f}  T={r['T']:.2f}±{err[2]:.2f} cps")
        log.write(f"f_d={fd_emp:.4f} f_t={ft_emp:.4f} (exp: {fd_exp:.4f}, {fd_exp**2:.4f})  "
                  f"D_inf={res['Dinf']:.2f}  T_inf={res['Tinf']:.2f}")
        for name, inv in (("effective nu", inv_eff), ("nominal nu", inv_nom),
                          ("eff nu, exp gate", inv_exp)):
            log.write(f"  inversion [{name:16s}]: M={inv['M']:.4f} alpha={inv['alpha']:.4f} "
                      f"m_eff={inv['m_eff']:.3f} g  m_Pu={inv['m_pu']:.2f} g "
                      f"({inv['m_pu']/cfg['pu_sphere']['mass_g']-1:+.2%})")
        log.write(f"  stat errors (eff nu): {res['inv_err']}")
    out = DATA / ("03_shift_register_results.json" if log_name == "step_03" else f"{log_name}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    log.write(f"saved {out}")
    if make_plot:
        plot(results, cfg, lib, PLOTS / "step_03_shift_register_histograms_and_inversion.png")
    log.close()
    return results


def plot(results, cfg, lib, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = results["P3"]
    f, b = np.array(r["f_hist"]), np.array(r["b_hist"])
    n = np.arange(len(f))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for a, log_ in zip(ax[:2], (False, True)):
        a.bar(n - 0.2, f / f.sum(), 0.4, label="f(n)  R+A")
        a.bar(n + 0.2, b / b.sum(), 0.4, label="b(n)  A")
        a.set(xlim=(-0.5, 14), xlabel="n in gate", ylabel="probability",
              title=f"P=3 µs, G=64 µs ({'log' if log_ else 'linear'})")
        if log_:
            a.set_yscale("log")
        a.legend()
    lab = ["S(f2-b2)", "S b1(f1-b1)", "T"]
    vals = [r["S"] * (r["f2"] - r["b2"]), r["cross"], r["T"]]
    ax[2].bar(lab, vals, color=["C0", "C3", "C2"])
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax[2].set(ylabel="cps", title="Triples cross-term breakdown")
    truth = [cfg["pu_sphere"]["mass_g"], float(lib["M_mc"]), cfg["alpha"]]
    labels = ["m_Pu / 300 g", "M / M_MC", "α / 0.5"]
    x = np.arange(3)
    for j, (key, nm) in enumerate((("inv_eff", "effective ν"), ("inv_nom", "nominal ν"))):
        inv = r[key]
        ax[3].bar(x + (j - 0.5) * 0.35, [inv["m_pu"] / truth[0], inv["M"] / truth[1],
                                         inv["alpha"] / truth[2]], 0.35, label=nm)
    e = r["inv_err"]
    ax[3].errorbar(x - 0.175, [r["inv_eff"]["m_pu"] / truth[0], r["inv_eff"]["M"] / truth[1],
                               r["inv_eff"]["alpha"] / truth[2]],
                   [e["m_pu"] / truth[0], e["M"] / truth[1], e["alpha"] / truth[2]],
                   fmt="none", ecolor="k", capsize=4)
    ax[3].axhline(1, color="k", ls="--")
    ax[3].set(xticks=x, xticklabels=labels, ylabel="recovered / true", ylim=(0, 1.6),
              title="Böhnel inversion (P=3 µs)")
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
