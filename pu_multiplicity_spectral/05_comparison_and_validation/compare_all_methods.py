"""Module 05: benchmark table + summary figure for all methods vs ground truth."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import (DATA, PLOTS, StepLog, changelog, effective_nu,  # noqa: E402
                             invert_point_model, load_config, load_library, reference_rates)

KEYS = ["S", "Dinf", "Tinf", "M", "alpha", "m_eff", "m_pu"]
LABELS = {"S": "Singles S (cps)", "Dinf": "Doubles D∞ (cps)", "Tinf": "Triples T∞ (cps)",
          "M": "Self-multiplication M", "alpha": "Impurity ratio α",
          "m_eff": "m_eff Pu-240 (g)", "m_pu": "m_Pu (g)"}


def collect():
    cfg = load_config()
    lib = load_library()
    ref = reference_rates(cfg, lib)
    nu = effective_nu(cfg, lib)
    pu = cfg["pu_sphere"]
    inv_ref = invert_point_model(ref["S"], ref["Dinf"], ref["Tinf"], float(lib["eps"]), 1, 1, nu,
                                 pu["wt_frac_pu240"], pu["sf_rate_per_g240"])
    sr = json.load(open(DATA / "03_shift_register_results.json"))["P3"]
    da = json.load(open(DATA / "04a_dft_results.json"))["P3"]
    sp = json.load(open(DATA / "04b_psd_bispectrum_results.json"))
    rows = {}
    rows["Ground truth"] = (dict(S=ref["S"], Dinf=ref["Dinf"], Tinf=ref["Tinf"], M=float(lib["M_mc"]),
                                 alpha=cfg["alpha"], m_eff=pu["m_eff_g"], m_pu=pu["mass_g"]), {})
    rows["Point model (exact rates)"] = (dict(S=ref["S"], Dinf=ref["Dinf"], Tinf=ref["Tinf"],
                                              **{k: inv_ref[k] for k in ("M", "alpha", "m_eff", "m_pu")}), {})
    rows["Shift register (P=3µs)"] = (
        dict(S=sr["S"], Dinf=sr["Dinf"], Tinf=sr["Tinf"], **{k: sr["inv_eff"][k] for k in ("M", "alpha", "m_eff", "m_pu")}),
        dict(S=sr["S_err"], Dinf=sr["Dinf_err"], Tinf=sr["Tinf_err"], **sr["inv_err"]))
    rows["2A DFT deconvolution"] = (
        dict(S=da["S"], Dinf=da["Dinf"], Tinf=da["Tinf"], **{k: da["inv"][k] for k in ("M", "alpha", "m_eff", "m_pu")}),
        dict(S=sr["S_err"], Dinf=da["Dinf_err"], Tinf=da["Tinf_err"], **da["inv_err"]))
    rows["2B PSD + bispectrum"] = (
        dict(S=sp["S"], Dinf=sp["Dinf"], Tinf=sp["Tinf"], **{k: sp["inv"][k] for k in ("M", "alpha", "m_eff", "m_pu")}),
        dict(S=sr["S_err"], Dinf=sp["err"]["D_cross"], Tinf=sp["err"]["T_bis"], **sp["inv_err"]))
    extra = dict(sr_nominal=sr["inv_nom"], sp_1pole=sp["inv_one_pole"], sp_1pole_rates=sp["one_pole"],
                 S_psd=sp["S_psd"], S_psd_err=sp["err"]["S_psd"], cross_offset=sp["cross_offset"],
                 cross_offset_err=sp["err"]["cross_offset"], nu_eff=nu, eps=float(lib["eps"]),
                 tau=float(lib["tau_s"]))
    return cfg, rows, extra


def table(rows):
    truth = rows["Ground truth"][0]
    names = list(rows)
    out = ["| Quantity | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in KEYS:
        cells = []
        for n in names:
            v, e = rows[n]
            s = f"{v[k]:.4g}" if k in ("M", "alpha") else f"{v[k]:.2f}" if k in ("m_eff", "m_pu") else f"{v[k]:.0f}"
            if k in e:
                s += f" ± {e[k]:.2g}"
            if n != "Ground truth":
                s += f" ({v[k]/truth[k]-1:+.1%})"
            cells.append(s)
        out.append(f"| {LABELS[k]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth = rows["Ground truth"][0]
    meth = [n for n in rows if n != "Ground truth"]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, k in zip(axs.ravel(), KEYS):
        v = [rows[n][0][k] for n in meth]
        e = [rows[n][1].get(k, 0) for n in meth]
        ax.bar(range(len(meth)), v, yerr=e, capsize=5, color=[f"C{i}" for i in range(len(meth))])
        ax.axhline(truth[k], color="k", ls="--", label="ground truth")
        lo = min(min(np.array(v) - np.array(e)), truth[k])
        hi = max(max(np.array(v) + np.array(e)), truth[k])
        pad = 0.25 * (hi - lo) + 1e-3 * abs(truth[k])
        ax.set(ylim=(lo - pad, hi + pad), title=LABELS[k], xticks=range(len(meth)))
        ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ") for m in meth], fontsize=7)
        for i, (vv, ee) in enumerate(zip(v, e)):
            ax.text(i, vv + ee, f"{vv/truth[k]-1:+.1%}", ha="center", va="bottom", fontsize=8)
    axs.ravel()[-1].axis("off")
    axs.ravel()[-1].text(0, 0.5, "Error bars: statistical (1σ) from\n10 independent time segments.\n"
                         "'Point model (exact rates)' = inversion of\nthe exact expected rates (validates\n"
                         "the inversion + effective ν calibration).", fontsize=10, va="center")
    axs.ravel()[0].legend()
    fig.suptitle("Pu sphere 300 g — traditional vs spectral multiplicity assay (OpenMC, 300 s list mode)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg, rows, ex = collect()
    log = StepLog("step_05_final_report")
    log.write("# Final report — passive neutron multiplicity & spectral/bispectrum Pu assay\n")
    log.write(f"Detector: eps = {ex['eps']:.4f}, die-away tau = {ex['tau']*1e6:.2f} us; "
              f"effective induced-fission moments nu_i2 = {ex['nu_eff']['i2']:.3f}, "
              f"nu_i3 = {ex['nu_eff']['i3']:.3f} (OpenMC implicit fission)\n")
    log.write(table(rows))
    log.write("\n## Notes")
    log.write(f"- PSD white floor S = {ex['S_psd']:.1f} ± {ex['S_psd_err']:.1f} cps (expected {rows['Ground truth'][0]['S']:.1f}); "
              f"cross-PSD floor offset = {ex['cross_offset']:.1f} ± {ex['cross_offset_err']:.1f} cps (vanishes).")
    n = ex["sr_nominal"]
    log.write(f"- Shift register with nominal Pu-239 moments (8.240, 17.321): m_Pu = {n['m_pu']:.1f} g "
              f"({n['m_pu']/300-1:+.1%}), M = {n['M']:.3f}, alpha = {n['alpha']:.3f} -> OpenMC implicit "
              f"fission does not reproduce induced-fission multiplicity; effective moments required.")
    o, orates = ex["sp_1pole"], ex["sp_1pole_rates"]
    log.write(f"- 2B with the single-exponential (spec) kernel: D_inf = {orates['D_cross']:.0f}, "
              f"T_inf = {orates['T_bis']:.0f} (biased low, chain timing not exponential); m_Pu = "
              f"{o['m_pu']:.1f} g (partial cancellation). The free two-component kernel removes the bias.")
    log.write("- Shift register / 2A T_inf use f_t from iid capture-time draws; correlated chain timing "
              "(thermal-return fission) makes this f_t ~3% too high -> T_inf ~3% low -> m_Pu ~+4%. "
              "The bispectrum needs no gate-fraction calibration.")
    log.write("- 2A reproduces the shift-register moments exactly (r1 = f1-b1, r2 = (f2-b2)-b1(f1-b1)): "
              "the deconvolution removes the accidental cross-term automatically.")
    plot(rows, PLOTS / "step_05_master_comparison.png")
    with open(DATA / "05_comparison.json", "w") as fh:
        json.dump({k: dict(value=v, err=e) for k, (v, e) in rows.items()}, fh, indent=1, default=float)
    log.close()
    changelog("Module 05 comparison",
              ["05_comparison_and_validation/compare_all_methods.py", "common/pu_model.py (reference_rates)"],
              ["Ground truth S, D_inf, T_inf from exact library factorial moments; M from sphere net leakage.",
               "Markdown benchmark table + 7-panel summary figure with 10-segment statistical errors."],
              [f"{k}: m_Pu = {v[0]['m_pu']:.2f} g ({v[0]['m_pu']/300-1:+.2%})" for k, v in rows.items()])


if __name__ == "__main__":
    main()
