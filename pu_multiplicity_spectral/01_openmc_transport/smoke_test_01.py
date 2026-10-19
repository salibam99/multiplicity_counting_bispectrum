"""Smoke test 01: 10k histories -> M, epsilon, tau, bank symmetry."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import DATA, changelog, load_config  # noqa: E402
from run_openmc_transport import main as run  # noqa: E402


def main():
    cfg = load_config()
    n = cfg["transport"]["n_hist_smoke"]
    lib, res = run(n, tag="step01_smoke", out=DATA / "01_smoke_event_library.npz",
                   log_name="step_01_smoke")
    M, eps, da, eb = lib["M_mc"], res["eps"], res["die_away"], res["eps_banks"]
    # bank balance: 2% or 3 sigma of the binomial counting error, whichever is larger
    nb = eb * lib["n_hist"] * M
    tol = np.maximum(0.02, 3 / np.sqrt(nb))
    dev = np.abs(eb / (eps / 3) - 1)
    checks = {
        f"M_MC={M:.4f} in [1.08,1.35] (spec 1.25 widened: physical value for 300 g + reflector)": 1.08 <= M <= 1.35,
        f"eps={eps:.4f} in [0.45,0.80]": 0.45 <= eps <= 0.80,
        f"tau={da['tau_us']:.2f}us in [25,80]": 25 <= da["tau_us"] <= 80,
        f"R2={da['r2']:.5f} > 0.99": da["r2"] > 0.99,
        f"bank dev {np.round(dev, 4)} < tol {np.round(tol, 4)}": bool(np.all(dev < tol)),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    ok = all(checks.values())
    changelog("Module 01 smoke test (10k histories)",
              ["01_openmc_transport/run_openmc_transport.py", "01_openmc_transport/smoke_test_01.py"],
              ["Event library: per history CSR list of (He3 capture time, bank id from atan2).",
               "M_MC from net sphere surface crossings; eps = detections/source / M_MC.",
               "Die-away: single-exp fit after 3 us predelay; empirical f_d, f_t from iid "
               "triples of library capture times."],
              [("PASS " if v else "FAIL ") + k for k, v in checks.items()])
    assert ok, "smoke_test_01 FAILED"
    print("smoke_test_01 PASSED")


if __name__ == "__main__":
    main()
