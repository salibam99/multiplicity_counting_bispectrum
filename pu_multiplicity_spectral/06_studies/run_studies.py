"""Step 06 studies: does the bispectrum method have a real case?

A  repeated measurements (8 seeds): bias vs spread for each method
B  Cf-252 calibration of f_d, f_t (as done on a real counter), transferred to the
   baseline item and to a moderated item (Pu in a 1.5 cm HDPE shell)
C  (alpha,n) rate scan: Triples precision of shift register vs bispectrum
D  ratio hybrid: f_t = T_SR / T_bis calibrated on the other runs (leave-one-out)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_lib import iid_gate_fractions, invert, run_pool, to_py  # noqa: E402
from common.pu_model import DATA, PLOTS, StepLog, changelog  # noqa: E402

M0 = 300.0


def jobs_for(smoke):
    T = 60.0 if smoke else 300.0
    nA, nP = (2, 2) if smoke else (8, 4)
    alphas = [2.0] if smoke else [0.0, 2.0, 5.0]
    J = [dict(study="A", lib="pu", alpha=0.5, T=T, seed=101 + i) for i in range(nA)]
    J += [dict(study="cal", lib="cf_cal", T=T, seed=201, spectral=False)]
    J += [dict(study="poly", lib="pu_poly", alpha=0.5, T=T, seed=301 + i) for i in range(nP)]
    J += [dict(study="C", lib="pu", alpha=a, T=T, seed=401 + i) for i, a in enumerate(alphas)]
    return J


def mass_sr(o, lib, fd, ft, alpha=None):
    s = o["sr"]
    return invert(s["S"], s["D"], s["T"], lib, fd, ft, alpha)["m_pu"]


def mass_sp(o, lib, alpha=None):
    return invert(o["sr"]["S"], o["sp"]["D"], o["sp"]["T"], lib, 1, 1, alpha)["m_pu"]


def mass_err(o, lib, fd, ft, which, alpha=None, n=200, seed=0):
    """1-sigma mass error by propagating the D, T stat errors (Gaussian sampling)."""
    rng = np.random.default_rng(seed)
    src = o["sr"] if which == "sr" else o["sp"]
    ms = []
    for _ in range(n):
        D = src["D"] + rng.normal() * src["D_err"]
        T = src["T"] + rng.normal() * src["T_err"]
        ms.append(invert(o["sr"]["S"], D, T, lib, fd, ft, alpha)["m_pu"])
    return float(np.nanstd(ms))


def analyse(res, log):
    out = {}
    by = lambda s: [r for r in res if r["job"]["study"] == s]  # noqa: E731
    A, cal, poly, C = by("A"), by("cal")[0], by("poly"), by("C")

    # ---------------- B: calibration with a Cf-252 source ----------------
    rc = cal["ref"]
    fd_cal, ft_cal = cal["sr"]["D"] / rc["Dinf"], cal["sr"]["T"] / rc["Tinf"]
    fd_cal_e, ft_cal_e = cal["sr"]["D_err"] / rc["Dinf"], cal["sr"]["T_err"] / rc["Tinf"]
    log.write(f"[B] Cf-252 calibration: f_d = {fd_cal:.4f} ± {fd_cal_e:.4f}, f_t = {ft_cal:.4f} ± {ft_cal_e:.4f}")
    out["cal"] = dict(fd=fd_cal, ft=ft_cal, fd_err=fd_cal_e, ft_err=ft_cal_e)
    gate = {}
    for name, runs, lib in (("pu", A, "pu"), ("pu_poly", poly, "pu_poly")):
        r0 = runs[0]["ref"]
        fd_true = np.mean([r["sr"]["D"] for r in runs]) / r0["Dinf"]
        ft_true = np.mean([r["sr"]["T"] for r in runs]) / r0["Tinf"]
        fd_iid, ft_iid = iid_gate_fractions(lib)
        gate[name] = dict(fd_true=fd_true, ft_true=ft_true, fd_iid=fd_iid, ft_iid=ft_iid)
        log.write(f"[B] {name:8s}: true f_d={fd_true:.4f} f_t={ft_true:.4f} | iid f_d={fd_iid:.4f} "
                  f"f_t={ft_iid:.4f} | Cf-cal f_t error {ft_cal/ft_true-1:+.2%}, iid f_t error {ft_iid/ft_true-1:+.2%}")
    out["gate"] = gate

    # ---------------- A + B + D: masses per run ----------------
    def table(runs, lib, tag):
        rows = []
        T_bis_sum = np.array([r["sp"]["T"] for r in runs])
        T_sr = np.array([r["sr"]["T"] for r in runs])
        D_sp = np.array([r["sp"]["D"] for r in runs])
        D_sr = np.array([r["sr"]["D"] for r in runs])
        fd_iid, ft_iid = iid_gate_fractions(lib)
        for i, r in enumerate(runs):
            o = np.arange(len(runs)) != i            # leave-one-out hybrid calibration
            ft_h = T_sr[o].sum() / T_bis_sum[o].sum()
            fd_h = D_sr[o].sum() / D_sp[o].sum()
            rows.append(dict(
                seed=r["job"]["seed"],
                sr_iid=mass_sr(r, lib, fd_iid, ft_iid), sr_cal=mass_sr(r, lib, fd_cal, ft_cal),
                bis=mass_sp(r, lib), hybrid=mass_sr(r, lib, fd_h, ft_h),
                err_sr=mass_err(r, lib, fd_cal, ft_cal, "sr"), err_bis=mass_err(r, lib, 1, 1, "sp"),
                T_ref=r["ref"]["Tinf"], T_sr_cal=r["sr"]["T"] / ft_cal, T_bis=r["sp"]["T"],
                T_bis_err=r["sp"]["T_err"], T_sr_err=r["sr"]["T_err"] / ft_cal))
        summ = {}
        for k in ("sr_iid", "sr_cal", "bis", "hybrid"):
            v = np.array([x[k] for x in rows])
            summ[k] = dict(mean=v.mean(), std=v.std(ddof=1), sem=v.std(ddof=1) / np.sqrt(len(v)),
                           bias_pct=100 * (v.mean() / M0 - 1), rms=np.sqrt(np.mean((v - M0) ** 2)))
            log.write(f"[{tag}] {k:7s}: mean m_Pu = {v.mean():7.2f} g  bias {100*(v.mean()/M0-1):+5.2f}% "
                      f"± {100*v.std(ddof=1)/np.sqrt(len(v))/M0:.2f}%   spread (1 run) {v.std(ddof=1):5.2f} g   "
                      f"RMS error {summ[k]['rms']:5.2f} g")
        log.write(f"[{tag}] mean in-run stat error: SR {np.mean([x['err_sr'] for x in rows]):.2f} g, "
                  f"bispectrum {np.mean([x['err_bis'] for x in rows]):.2f} g")
        return rows, summ

    out["A_rows"], out["A"] = table(A, "pu", "A baseline")
    out["poly_rows"], out["poly"] = table(poly, "pu_poly", "B moderated")

    # ---------------- C: alpha scan ----------------
    scan = []
    for r in sorted(C + A[:1], key=lambda x: x["job"]["alpha"]):
        a = r["job"]["alpha"]
        scan.append(dict(alpha=a, S=r["sr"]["S"], relT_sr=r["sr"]["T_err"] / r["sr"]["T"],
                         relT_bis=r["sp"]["T_err"] / r["sp"]["T"],
                         relD_sr=r["sr"]["D_err"] / r["sr"]["D"], relD_sp=r["sp"]["D_err"] / r["sp"]["D"],
                         biasT_sr=r["sr"]["T"] / ft_cal / r["ref"]["Tinf"] - 1,
                         biasT_bis=r["sp"]["T"] / r["ref"]["Tinf"] - 1))
        s = scan[-1]
        log.write(f"[C] alpha={a:3.1f} S={s['S']:8.0f}: sigma_T/T  SR {100*s['relT_sr']:5.2f}%  "
                  f"bispectrum {100*s['relT_bis']:6.2f}%  (ratio {s['relT_bis']/s['relT_sr']:.1f}); "
                  f"sigma_D/D SR {100*s['relD_sr']:.2f}% cross-PSD {100*s['relD_sp']:.2f}%")
    out["C"] = scan
    return out


def plot(o, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meth = [("sr_iid", "SR, iid f_t", "C1"), ("sr_cal", "SR, Cf-calibrated", "C0"),
            ("hybrid", "SR, hybrid f_t", "C2"), ("bis", "bispectrum", "C3")]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for j, (case, title) in enumerate((("A", "(a) baseline item, 8 × 300 s"), ("poly", "(b) moderated item, 4 × 300 s"))):
        rows = o[f"{case}_rows"]
        for i, (k, lab, c) in enumerate(meth):
            v = np.array([r[k] for r in rows])
            ax[j].plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, "o", color=c, alpha=0.6, ms=4)
            s = o[case][k]
            ax[j].errorbar(i + 0.3, s["mean"], s["sem"], fmt="s", color=c, capsize=4, ms=6)
        ax[j].axhline(M0, color="k", ls="--")
        ax[j].set(xticks=range(4), title=title, ylabel="m_Pu [g]")
        ax[j].set_xticklabels([m[1] for m in meth], fontsize=8)
    g = o["gate"]
    names = ["baseline", "moderated"]
    x = np.arange(2)
    ax[2].bar(x - 0.27, [g[n]["ft_true"] for n in ("pu", "pu_poly")], 0.27, label="true f_t", color="k")
    ax[2].bar(x, [o["cal"]["ft"]] * 2, 0.27, label="Cf-252 calibration", color="C0")
    ax[2].bar(x + 0.27, [g[n]["ft_iid"] for n in ("pu", "pu_poly")], 0.27, label="iid die-away", color="C1")
    ax[2].set(xticks=x, xticklabels=names, ylim=(0.38, 0.46), ylabel="f_t (P=3 µs, G=64 µs)",
              title="(c) is the Triples gate fraction transferable?")
    ax[2].legend(fontsize=8)
    C = o["C"]
    a = [c["alpha"] for c in C]
    ax[3].semilogy(a, [100 * c["relT_sr"] for c in C], "o-", color="C0", label="T∞ shift register")
    ax[3].semilogy(a, [100 * c["relT_bis"] for c in C], "s-", color="C3", label="T∞ bispectrum")
    ax[3].semilogy(a, [100 * c["relD_sr"] for c in C], "o--", color="C0", alpha=0.5, label="D∞ shift register")
    ax[3].semilogy(a, [100 * c["relD_sp"] for c in C], "s--", color="C3", alpha=0.5, label="D∞ cross-PSD")
    ax[3].set(xlabel="α  ((α,n) / SF neutrons)", ylabel="relative 1σ error [%] (300 s)",
              title="(d) precision vs (α,n) background")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    tag = "smoke" if a.smoke else "full"
    log = StepLog(f"step_06_studies_{tag}")
    jobs = jobs_for(a.smoke)
    log.write(f"{len(jobs)} synthetic measurements, {a.procs} processes")
    res = run_pool(jobs, a.procs)
    with open(DATA / f"06_studies_raw_{tag}.json", "w") as fh:
        json.dump(to_py(res), fh, indent=1)
    out = analyse(res, log)
    with open(DATA / f"06_studies_{tag}.json", "w") as fh:
        json.dump(to_py(out), fh, indent=1)
    plot(out, PLOTS / f"step_06_studies{'_smoke' if a.smoke else ''}.png")
    log.close()
    if a.smoke:
        assert all(np.isfinite(r["bis"]) for r in out["A_rows"]), "smoke: non-finite bispectrum mass"
        print("smoke_test_06 PASSED")
    else:
        changelog("Module 06 studies (case for the bispectrum)",
                  ["common/pu_model.py (variants cf_cal, pu_poly; load_library(path))",
                   "01_openmc_transport/run_openmc_transport.py (variant pass-through)",
                   "06_studies/run_variant_transport.py", "06_studies/study_lib.py", "06_studies/run_studies.py"],
                  ["Cf-252 calibration source and HDPE-shell moderated item transported with OpenMC (400k each).",
                   "A: 8 seeds; B: Cf-calibrated f_d/f_t transferred to baseline and moderated item; "
                   "C: alpha scan; D: leave-one-out ratio hybrid f_t = T_SR/T_bis."],
                  [f"{k}: bias {v['bias_pct']:+.2f}%, RMS {v['rms']:.1f} g" for k, v in out["A"].items()] +
                  [f"moderated {k}: bias {v['bias_pct']:+.2f}%, RMS {v['rms']:.1f} g" for k, v in out["poly"].items()])


if __name__ == "__main__":
    main()
