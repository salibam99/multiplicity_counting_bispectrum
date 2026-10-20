"""Study 06: OpenMC transport for the calibration source and the moderated item."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "01_openmc_transport"))
from common.pu_model import DATA, PLOTS, load_config  # noqa: E402
from run_openmc_transport import main as transport  # noqa: E402

VARIANTS = {"cf_cal": "Cf-252 calibration source, empty cavity",
            "pu_poly": "300 g Pu sphere inside a 1.5 cm HDPE shell (moderating item)"}


def main(n_hist=None, suffix=""):
    n_hist = n_hist or load_config()["transport"]["n_hist_full"]
    for v in VARIANTS:
        transport(n_hist, tag=f"step06_{v}{suffix}", out=DATA / f"06_lib_{v}{suffix}.npz",
                  log_name=f"step_06_transport_{v}{suffix}", variant=v,
                  plot_path=PLOTS / f"step_06_{v}{suffix}_die_away.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
