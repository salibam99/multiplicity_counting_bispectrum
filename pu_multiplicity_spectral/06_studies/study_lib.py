"""Shared worker for the step-06 studies: one synthetic measurement -> SR and spectral rates."""
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
for p in (HERE.parents[1], HERE.parents[1] / "02_pulse_train_synthesis",
          HERE.parents[1] / "03_traditional_shift_register", HERE.parents[1] / "04_spectral_bispectrum_assay"):
    sys.path.insert(0, str(p))
from common.pu_model import (DATA, effective_nu, invert_point_model, load_config,  # noqa: E402
                             load_library, reference_rates)
from run_psd_bispectrum_assay import spectral_assay  # noqa: E402
from run_shift_register import histos, rates_from_histos, segment_errors, shift_register  # noqa: E402
from synthesize_pulse_train import synthesize  # noqa: E402

CF252_P_NU = [0.00211, 0.02467, 0.12290, 0.27144, 0.30763, 0.18770, 0.06770, 0.01366, 0.00209]
LIBS = {"pu": DATA / "01_openmc_event_library.npz", "cf_cal": DATA / "06_lib_cf_cal.npz",
        "pu_poly": DATA / "06_lib_pu_poly.npz"}


def make_cfg(source="pu", alpha=None, F=None):
    cfg = load_config()
    if source == "cf_cal":
        cfg["nu_sf_pu240"]["P_nu"] = CF252_P_NU
        cfg["alpha"] = 0.0
        cfg["pu_sphere"]["F_true"] = F or 5000.0
    if alpha is not None:
        cfg["alpha"] = alpha
    return cfg


def measure(job):
    """job = dict(lib, source, alpha, T, seed, spectral) -> rates with 1-sigma stat errors."""
    cfg = make_cfg(job["lib"] if job["lib"] == "cf_cal" else "pu", job.get("alpha"))
    lib = load_library(LIBS[job["lib"]])
    r = synthesize(cfg, lib, job["T"], job["seed"])
    ts, bk = r["timestamps_sec"], r["bank_ids"]
    T = job["T"]
    out = dict(job=job, ref=reference_rates(cfg, lib), n_pulses=len(ts))
    trig, nf, nb = shift_register(ts, 3e-6, 64e-6, 2000e-6)
    f, b = histos(nf, nb)
    sr = rates_from_histos(f, b, len(ts) / T)
    e = segment_errors(trig, nf, nb, T, lambda f_, b_, s_: [rates_from_histos(f_, b_, s_)[k] for k in "SDT"])
    out["sr"] = dict(S=sr["S"], D=sr["D"], T=sr["T"], S_err=e[0], D_err=e[1], T_err=e[2])
    if job.get("spectral", True):
        _, fit, err, _ = spectral_assay(ts, bk, T, cfg)
        out["sp"] = dict(D=fit["D_cross"], T=fit["T_bis"], D_err=err["D_cross"], T_err=err["T_bis"],
                         kern_bis=[float(x) for x in fit["kern_bis"]])
    return out


def invert(S, D, T, lib_name, fd=1.0, ft=1.0, alpha=None):
    cfg = make_cfg("pu", alpha)
    lib = load_library(LIBS[lib_name])
    pu = cfg["pu_sphere"]
    return invert_point_model(S, D, T, float(lib["eps"]), fd, ft, effective_nu(cfg, lib),
                              pu["wt_frac_pu240"], pu["sf_rate_per_g240"])


def iid_gate_fractions(lib_name, P_us=3.0):
    lib = load_library(LIBS[lib_name])
    i = list(lib["gate_P_us"]).index(P_us)
    return float(lib["gate_fd"][i]), float(lib["gate_ft"][i])


def run_pool(jobs, procs=6):
    from multiprocessing import Pool
    with Pool(min(procs, len(jobs))) as pool:
        return pool.map(measure, jobs)


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


__all__ = ["copy", "measure", "invert", "iid_gate_fractions", "run_pool", "to_py", "make_cfg"]
