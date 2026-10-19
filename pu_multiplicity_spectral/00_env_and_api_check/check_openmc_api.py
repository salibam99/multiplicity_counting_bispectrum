"""Module 00: discover OpenMC version, cross sections and the collision_track API."""
import inspect
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openmc  # noqa: E402

from common.pu_model import DATA, StepLog, build_model, load_config, read_collision_tracks  # noqa: E402

PDG_NAMES = {1000020030: "He3", 1000010010: "H1", 1000060120: "C12", 1000060130: "C13",
             1000942390: "Pu239", 1000942400: "Pu240"}


def main():
    log = StepLog("step_00")
    cfg = load_config()
    log.write(f"python            : {sys.executable}")
    log.write(f"openmc.__version__: {openmc.__version__}")
    xs = os.environ.get("OPENMC_CROSS_SECTIONS")
    log.write(f"OPENMC_CROSS_SECTIONS: {xs}  (exists={xs is not None and os.path.exists(xs)})")
    lib = openmc.data.DataLibrary.from_xml(xs)
    have = {Path(l["path"]).stem for l in lib.libraries}
    for n in ("Pu239", "Pu240", "He3", "H1", "C12", "c_H_in_CH2"):
        log.write(f"  library {n:12s}: {'OK' if n in have else 'MISSING'}")

    src = inspect.getsource(openmc.Settings.collision_track.fset)
    keys = re.search(r"\(('cell_ids'.*?)\)\)", src, re.S).group(1)
    keys = [k.strip(" '\"\n") for k in keys.split(",")]
    log.write(f"Settings.collision_track accepted keys: {keys}")
    log.write(f"read_collision_track_file: {inspect.signature(openmc.read_collision_track_file)}")

    # 100-particle discovery run with NO reaction/nuclide filter to census MTs.
    run = DATA / "openmc_runs" / "step00_discovery"
    run.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, 100, batches=1)
    model.settings.collision_track = {"cell_ids": [1] + cfg["detector"]["bank_cell_ids"],
                                      "max_collisions": 1_000_000}
    model.run(cwd=run, output=False)
    ct, files = read_collision_tracks(run)
    log.write(f"collision_track files: {[os.path.basename(f) for f in files]}")
    log.write("collision_track structured dtype:")
    for name in ct.dtype.names:
        log.write(f"    {name:14s} {ct.dtype[name]}")
    log.write(f"n records = {len(ct)}, distinct parent_id (histories) = {len(set(ct['parent_id']))}")
    log.write("event_mt census (cell, nuclide, MT): count")
    cnt = Counter(zip(ct["cell_id"], ct["nuclide_id"], ct["event_mt"]))
    for (c, n, mt), v in sorted(cnt.items()):
        log.write(f"    cell {c:4d}  {PDG_NAMES.get(n, n)!s:7s}  MT {mt:4d}: {v}")
    log.write("NOTE: analog absorption is recorded as MT=101 (N_DISAPPEAR); He3(n,p) "
              "detections => cell in banks, nuclide He3, MT 101.")
    log.write("NOTE: fixed-source fission in OpenMC is implicit (fission sites at each "
              "fissionable collision); used as-is.")
    log.close()


if __name__ == "__main__":
    main()
