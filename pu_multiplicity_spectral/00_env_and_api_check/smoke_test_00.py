"""Smoke test 00: 100-particle production-filter run; collision_track + surface source."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.pu_model import (DATA, StepLog, build_model, changelog, load_config,  # noqa: E402
                             read_collision_tracks, sphere_net_leakage)


def main():
    log = StepLog("step_00", mode="a")
    log.write("--- smoke_test_00 ---")
    cfg = load_config()
    n = 100
    run = DATA / "openmc_runs" / "step00_smoke"
    run.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, n, batches=1)
    model.run(cwd=run, output=False)

    ct, files = read_collision_tracks(run)
    assert os.path.exists(files[0]), "collision_track.h5 not created"
    for k in ("time", "r", "cell_id", "event_mt", "nuclide_id", "parent_id"):
        assert k in ct.dtype.names, f"missing field {k}"
    assert len(ct) > 0, "no He3 captures recorded"
    assert np.all(ct["time"] > 0), "non-positive capture time"
    assert np.all(ct["event_mt"] == 101) and np.all(ct["nuclide_id"] == 1000020030)
    pid = ct["parent_id"]
    assert pid.min() >= 1 and pid.max() <= n, "parent_id is not a history index"
    M, n_out, n_in = sphere_net_leakage(run, n)
    assert n_out > 0 and M > 1.0, "surface tracking failed"

    log.write(f"He3 captures: {len(ct)} from {len(np.unique(pid))} histories; "
              f"det/source = {len(ct)/n:.3f}")
    log.write(f"time range [{ct['time'].min()*1e6:.3f}, {ct['time'].max()*1e6:.1f}] us")
    log.write(f"sphere surface crossings out={n_out} in={n_in}  ->  net leakage M = {M:.3f}")
    log.write("smoke_test_00 PASSED")
    log.close()
    changelog("Module 00 (env & API check)",
              ["config/assay_config.json", "common/pu_model.py",
               "00_env_and_api_check/check_openmc_api.py", "00_env_and_api_check/smoke_test_00.py"],
              ["Shared config + common module (geometry builder, point model, logging).",
               "collision_track filtered to He3 MT101 in bank cells 101-103 = detections; "
               "parent_id = source-history id used for grouping.",
               "Sphere escape via surf_source_write on surface 1; M = (out-in)/N_hist.",
               "OpenMC run as-is (implicit fixed-source fission)."],
              [f"100 hist: {len(ct)} captures, all t>0, MT101/He3, parent_id in [1,{n}]",
               f"net leakage M={M:.3f} (out={n_out}, in={n_in}) — PASS"])


if __name__ == "__main__":
    main()
