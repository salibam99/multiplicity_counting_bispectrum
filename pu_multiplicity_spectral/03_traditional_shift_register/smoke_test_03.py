"""Smoke test 03: (1) zero-bias check on a toy stream, (2) OpenMC stream -> m_Pu within 5%."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import StepLog, changelog, load_config, toy_stream  # noqa: E402
from run_shift_register import histos, rates_from_histos, run, segment_errors, shift_register  # noqa: E402


def toy_check(log):
    tau, T = 45e-6, 200.0
    ts, _, tr = toy_stream({1: 8000.0, 2: 3000.0, 3: 1000.0, 4: 200.0}, tau, T, seed=3)
    G, L = 64e-6, 2000e-6
    checks = {}
    for P in (0.0, 3e-6):
        fd = np.exp(-P / tau) * (1 - np.exp(-G / tau))
        trig, nf, nb = shift_register(ts, P, G, L)
        f, b = histos(nf, nb)
        r = rates_from_histos(f, b, len(ts) / T)
        err = segment_errors(trig, nf, nb, T,
                             lambda f_, b_, s_: [rates_from_histos(f_, b_, s_)[k] for k in "SDT"])
        Dinf, Tinf = r["D"] / fd, r["T"] / fd**2
        zD = (Dinf - tr["Dinf"]) / (err[1] / fd)
        zT = (Tinf - tr["Tinf"]) / (err[2] / fd**2)
        log.write(f"toy P={P*1e6:.0f}us: S={r['S']:.1f}/{tr['S']:.1f}  D_inf={Dinf:.1f}/{tr['Dinf']:.1f} "
                  f"(z={zD:+.2f})  T_inf={Tinf:.1f}/{tr['Tinf']:.1f} (z={zT:+.2f})")
        checks[f"toy P={P*1e6:.0f}us D_inf within 3 sigma (z={zD:+.2f})"] = abs(zD) < 3
        checks[f"toy P={P*1e6:.0f}us T_inf within 3 sigma (z={zT:+.2f})"] = abs(zT) < 3
        checks[f"toy P={P*1e6:.0f}us D_inf rel {Dinf/tr['Dinf']-1:+.3%} < 1%"] = abs(Dinf / tr["Dinf"] - 1) < 0.01
        checks[f"toy P={P*1e6:.0f}us T_inf rel {Tinf/tr['Tinf']-1:+.3%} < 5%"] = abs(Tinf / tr["Tinf"] - 1) < 0.05
    return checks


def main():
    log = StepLog("step_03_smoke")
    checks = toy_check(log)
    log.close()
    cfg = load_config()
    res = run(log_name="step_03")
    for P in ("P0", "P3"):
        m = res[P]["inv_eff"]["m_pu"]
        checks[f"OpenMC {P}: m_Pu={m:.2f} g within 5% of 300 g"] = abs(m / cfg["pu_sphere"]["mass_g"] - 1) < 0.05
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    changelog("Module 03 shift register",
              ["03_traditional_shift_register/run_shift_register.py",
               "03_traditional_shift_register/smoke_test_03.py", "common/pu_model.py (toy_stream)"],
              ["Exact searchsorted shift register, every pulse a trigger; 8 contiguous bg gates from L.",
               "D=S(f1-b1), T=S[(f2-b2)-b1(f1-b1)]; gate fractions from library (empirical) and exponential.",
               "Böhnel cubic inversion with transport-calibrated nu_i2/nu_i3 (OpenMC implicit fission) "
               "and with nominal nu for comparison; stat errors from 10 time segments."],
              [("PASS " if v else "FAIL ") + k for k, v in checks.items()])
    assert all(checks.values()), "smoke_test_03 FAILED"
    print("smoke_test_03 PASSED")


if __name__ == "__main__":
    main()
