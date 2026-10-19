"""Smoke test 02: 30 s pulse train -> S vs Böhnel, Rossi-alpha tau and S^2 baseline, sorting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import DATA, changelog  # noqa: E402
from synthesize_pulse_train import main as synth  # noqa: E402


def main():
    r = synth(T_meas=30.0, out=DATA / "02_smoke_pulses.npz", log_name="step_02_smoke",
              seed=7, rossi_span=30.0)
    checks = {
        "timestamps strictly sorted": r["sorted"],
        f"S_emp={r['S_emp']:.1f} vs S_theo={r['S_theo']:.1f} within 2%":
            abs(r["S_emp"] / r["S_theo"] - 1) < 0.02,
        f"Rossi tau={r['tau_rossi']*1e6:.2f}us vs lib {r['tau_lib']*1e6:.2f}us within 5%":
            abs(r["tau_rossi"] / r["tau_lib"] - 1) < 0.05,
        f"Rossi baseline {r['B']:.5g} vs S^2 {r['S2']:.5g} within 1%":
            abs(r["B"] / r["S2"] - 1) < 0.01,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    changelog("Module 02 smoke test (30 s train)",
              ["02_pulse_train_synthesis/synthesize_pulse_train.py",
               "02_pulse_train_synthesis/smoke_test_02.py", "common/pu_model.py (effective_nu)"],
              ["Poisson SF births with nu~P(nu), Poisson (a,n) births with rate F nu_s1 alpha; "
               "each source neutron draws one OpenMC history (time offsets + bank).",
               "Rossi-alpha from all forward pair differences < 300 us, fit A exp(-t/tau)+B."],
              [("PASS " if v else "FAIL ") + k for k, v in checks.items()])
    assert all(checks.values()), "smoke_test_02 FAILED"
    print("smoke_test_02 PASSED")


if __name__ == "__main__":
    main()
