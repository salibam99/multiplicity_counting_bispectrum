"""Smoke test 04: synthetic 3-bank stream with known D_inf, T_inf, tau=45 us.

Checks: cross-PSD -> 2 D_inf, bispectrum -> 6 T_inf (within 3%), cross-PSD floor ~ 0,
DFT deconvolution r1, r2 == shift-register moments; then runs 2A and 2B on OpenMC data.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "03_traditional_shift_register"))
from common.pu_model import PLOTS, StepLog, changelog, load_config, toy_stream  # noqa: E402
from run_dft_histogram_deconv import deconvolve, run as run_2a  # noqa: E402
from run_psd_bispectrum_assay import plots, run as run_2b, spectral_assay  # noqa: E402
from run_shift_register import histos, rates_from_histos, shift_register  # noqa: E402


def main():
    cfg = load_config()
    log = StepLog("step_04_smoke")
    tau, T = 45e-6, 300.0
    ts, bk, tr = toy_stream({1: 4000.0, 2: 1500.0, 3: 1500.0, 4: 300.0}, tau, T, seed=11)
    log.write(f"toy truth: S={tr['S']:.0f}  D_inf={tr['Dinf']:.0f}  T_inf={tr['Tinf']:.0f}  tau=45 us")
    acc, fit, err, mean = spectral_assay(ts, bk, T, cfg, tau0=tau)
    plots(acc, fit, mean, PLOTS / "step_04_smoke_toy_psd.png", PLOTS / "step_04_smoke_toy_bispectrum.png",
          ref=tr)
    log.write(f"PSD   : S={fit['S_psd']:.1f}  D_inf={fit['D_psd']:.1f}±{err['D_psd']:.1f}  tau={fit['tau_psd']*1e6:.2f}")
    log.write(f"cross : D_inf={fit['D_cross']:.1f}±{err['D_cross']:.1f}  tau={fit['tau_cross']*1e6:.2f}  "
              f"offset={fit['cross_offset']:.2f}±{err['cross_offset']:.2f}")
    log.write(f"bispec: T_inf={fit['T_bis']:.1f}±{err['T_bis']:.1f}  tau={fit['tau_bis']*1e6:.2f}")
    trig, nf, nb = shift_register(ts, 0.0, 64e-6, 2000e-6)
    f, b = histos(nf, nb)
    r = rates_from_histos(f, b, len(ts) / T)
    _, r1, r2, _, _ = deconvolve(f, b)
    log.write(f"DFT deconv: r1={r1:.6f} vs f1-b1={r['f1']-r['b1']:.6f};  "
              f"r2={r2:.6f} vs T/S={r['T']/r['S']:.6f}")
    checks = {
        f"PSD floor S={fit['S_psd']:.1f} vs {tr['S']:.0f} within 1%": abs(fit["S_psd"] / tr["S"] - 1) < 0.01,
        f"cross-PSD D_inf={fit['D_cross']:.1f} vs {tr['Dinf']:.0f} within 3%": abs(fit["D_cross"] / tr["Dinf"] - 1) < 0.03,
        f"PSD D_inf={fit['D_psd']:.1f} vs {tr['Dinf']:.0f} within 3%": abs(fit["D_psd"] / tr["Dinf"] - 1) < 0.03,
        f"bispectrum T_inf={fit['T_bis']:.1f} vs {tr['Tinf']:.0f} within 3%": abs(fit["T_bis"] / tr["Tinf"] - 1) < 0.03,
        f"cross floor {fit['cross_offset']:.2f} consistent with 0 (<3 sigma)":
            abs(fit["cross_offset"]) < 3 * err["cross_offset"] + 1e-9,
        f"tau_bis={fit['tau_bis']*1e6:.2f} us within 3% of 45": abs(fit["tau_bis"] / tau - 1) < 0.03,
        "DFT r1 == f1-b1 (1e-6)": abs(r1 - (r["f1"] - r["b1"])) < 1e-6,
        "DFT r2 == (f2-b2)-b1(f1-b1) (1e-6)": abs(r2 - r["T"] / r["S"]) < 1e-6,
    }
    log.close()
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    assert all(checks.values()), "smoke_test_04 toy stage FAILED - not running on OpenMC data"
    r2a = run_2a()
    r2b = run_2b()
    m0 = cfg["pu_sphere"]["mass_g"]
    checks[f"OpenMC 2A m_Pu={r2a['P3']['inv']['m_pu']:.2f} within 5%"] = abs(r2a["P3"]["inv"]["m_pu"] / m0 - 1) < 0.05
    checks[f"OpenMC 2B m_Pu={r2b['inv']['m_pu']:.2f} within 5%"] = abs(r2b["inv"]["m_pu"] / m0 - 1) < 0.05
    for k in list(checks)[-2:]:
        print(("PASS " if checks[k] else "FAIL ") + k)
    changelog("Module 04 spectral / bispectrum",
              ["04_spectral_bispectrum_assay/run_dft_histogram_deconv.py",
               "04_spectral_bispectrum_assay/run_psd_bispectrum_assay.py",
               "04_spectral_bispectrum_assay/smoke_test_04.py"],
              ["2A: p = IFFT(FFT f / FFT b), r1, r2 and CGF polynomial fit in w = e^{-i theta} - 1.",
               "2B: 2 us bins, 16.384 ms segments, per-segment mean removed; PSD, 3-bank cross-PSD, "
               "6-permutation-averaged cross-bispectrum on f1 in (0,10] kHz, f2 in [-10,10] kHz (stride 1), "
               "normalised by p_X p_Y p_Z (= spec's 4.5 x sum-over-permutations). Errors from 10 segment groups.",
               "Two-component kernel (1-w)/(1+iwt1)+w/(1+iwt2) for PSD/cross/bispectrum fits: removes "
               "the single-Lorentzian bias (-1.7% D, -7% T) caused by non-exponential chain timing.",
               "Spectral D_inf, T_inf are gate-free -> inversion with f_d = f_t = 1."],
              [("PASS " if v else "FAIL ") + k for k, v in checks.items()])
    assert all(checks.values()), "smoke_test_04 FAILED"
    print("smoke_test_04 PASSED")


if __name__ == "__main__":
    main()
