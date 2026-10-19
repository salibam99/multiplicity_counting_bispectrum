"""Shared helpers: config, OpenMC model builder, point-model equations, logging."""
import datetime
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
PLOTS = ROOT / "plots"
for _d in (DATA, LOGS, PLOTS):
    _d.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Config & derived constants
# --------------------------------------------------------------------------
def load_config():
    with open(ROOT / "config" / "assay_config.json") as f:
        cfg = json.load(f)
    pu = cfg["pu_sphere"]
    pu["radius_cm"] = (3 * pu["mass_g"] / (4 * np.pi * pu["density_g_cm3"])) ** (1 / 3)
    pu["m_eff_g"] = pu["mass_g"] * pu["wt_frac_pu240"]
    pu["F_true"] = pu["sf_rate_per_g240"] * pu["m_eff_g"]
    return cfg


def nu_moments(cfg):
    """Factorial moments used everywhere: <nu>, <nu(nu-1)>, <nu(nu-1)(nu-2)>.

    The SF moments are computed from the tabulated P(nu) actually sampled in
    step 02 (the rounded table differs from the quoted nu_s2/nu_s3 by
    0.5%/1.2%; using the sampled ones keeps the synthetic world self-consistent).
    """
    p = np.asarray(cfg["nu_sf_pu240"]["P_nu"], float)
    p = p / p.sum()
    n = np.arange(len(p))
    ni = cfg["nu_if_pu239"]
    return dict(
        s1=float((n * p).sum()),
        s2=float((n * (n - 1) * p).sum()),
        s3=float((n * (n - 1) * (n - 2) * p).sum()),
        i1=ni["nu_i1"], i2=ni["nu_i2"], i3=ni["nu_i3"],
    )


def load_library():
    return dict(np.load(DATA / "01_openmc_event_library.npz"))


def effective_nu(cfg, lib):
    """Transport-consistent induced-fission moments nu_i2, nu_i3.

    OpenMC fixed-source fission is implicit, so its chains do not follow the
    nominal Pu-239 P(nu). For a single starting neutron the detected-count
    factorial moments obey (point model, k = (M-1)/(nu_i1-1)):
        <n(n-1)>      = (eps M)^2 k nu_i2
        <n(n-1)(n-2)> = (eps M)^3 (k nu_i3 + 3 k^2 nu_i2^2)
    and are inverted here with nu_i1 nominal.
    """
    nu = nu_moments(cfg)
    c = np.diff(lib["offsets"]).astype(float)
    eM = c.mean()
    k = (lib["M_mc"] - 1) / (nu["i1"] - 1)
    i2 = (c * (c - 1)).mean() / (eM**2 * k)
    i3 = ((c * (c - 1) * (c - 2)).mean() / eM**3 - 3 * k**2 * i2**2) / k
    out = dict(nu)
    out.update(i2=float(i2), i3=float(i3))
    return out


def reference_rates(cfg, lib):
    """Exact expected S, D_inf, T_inf of the step-02 synthesis.

    Each source neutron draws an independent library history; with per-history
    detected-count factorial moments m1, m2 = <n(n-1)>, m3 = <n(n-1)(n-2)> and
    Q = nu_s1 (1 + alpha) primary neutrons per fission-equivalent:
      S = F Q m1,  D = F (nu_s2 m1^2 + Q m2)/2,
      T = F (nu_s3 m1^3/6 + nu_s2 m1 m2/2 + Q m3/6).
    """
    nu = nu_moments(cfg)
    c = np.diff(lib["offsets"]).astype(float)
    m1, m2, m3 = c.mean(), (c * (c - 1)).mean(), (c * (c - 1) * (c - 2)).mean()
    F, Q = cfg["pu_sphere"]["F_true"], nu["s1"] * (1 + cfg["alpha"])
    return dict(S=F * Q * m1, Dinf=F * (nu["s2"] * m1**2 + Q * m2) / 2,
                Tinf=F * (nu["s3"] * m1**3 / 6 + nu["s2"] * m1 * m2 / 2 + Q * m3 / 6))


# --------------------------------------------------------------------------
# Böhnel point model
# --------------------------------------------------------------------------
def point_model_rates(F, M, alpha, eps, fd, ft, nu):
    k = (M - 1) / (nu["i1"] - 1)
    S = F * eps * M * nu["s1"] * (1 + alpha)
    D = F * eps**2 * fd * M**2 / 2 * (nu["s2"] + k * nu["s1"] * (1 + alpha) * nu["i2"])
    T = F * eps**3 * ft * M**3 / 6 * (
        nu["s3"] + k * (3 * nu["s2"] * nu["i2"] + nu["s1"] * (1 + alpha) * nu["i3"])
        + 3 * k**2 * nu["s1"] * (1 + alpha) * nu["i2"] ** 2)
    return S, D, T


def invert_point_model(S, D, T, eps, fd, ft, nu, f240=None, sf_per_g=473.5):
    """Solve the point-model cubic for M >= 1, then F, alpha, m_eff (and m_Pu).

    Eliminating F*nu_s1*(1+alpha) = S/(eps M) and F from the D equation gives
      6T/(eps^3 ft) = 2D M (nu_s3 + 3 k nu_s2 nu_i2)/(eps^2 fd nu_s2)
                      + S M^2 k (nu_i3 - nu_i2 nu_s3/nu_s2)/eps ,  k=(M-1)/(nu_i1-1)
    which is a cubic a + bM + cM^2 + M^3 = 0.
    """
    P = np.polynomial.Polynomial
    Mp = P([0, 1])
    k = (Mp - 1) / (nu["i1"] - 1)
    rhs = (2 * D * Mp * (nu["s3"] + 3 * k * nu["s2"] * nu["i2"]) / (eps**2 * fd * nu["s2"])
           + S * Mp**2 * k * (nu["i3"] - nu["i2"] * nu["s3"] / nu["s2"]) / eps)
    poly = rhs - 6 * T / (eps**3 * ft)
    c = poly.coef / poly.coef[-1]          # normalise -> a + bM + cM^2 + M^3
    roots = np.roots(c[::-1])
    real = roots[np.abs(roots.imag) < 1e-9].real
    phys = real[real >= 1.0 - 1e-9]
    if len(phys) == 0:
        return dict(M=np.nan, F=np.nan, alpha=np.nan, m_eff=np.nan, m_pu=np.nan,
                    cubic=c.tolist())
    M = float(phys.min())
    kk = (M - 1) / (nu["i1"] - 1)
    F = (2 * D / (eps**2 * fd * M**2) - kk * nu["i2"] * S / (eps * M)) / nu["s2"]
    alpha = S / (eps * M * F * nu["s1"]) - 1
    m_eff = F / sf_per_g
    out = dict(M=M, F=F, alpha=alpha, m_eff=m_eff, cubic=c.tolist())
    out["m_pu"] = m_eff / f240 if f240 else np.nan
    return out


# --------------------------------------------------------------------------
# OpenMC model
# --------------------------------------------------------------------------
def build_model(cfg, n_hist, batches=None, seed=None):
    import openmc
    pu, det = cfg["pu_sphere"], cfg["detector"]
    batches = batches or cfg["transport"]["n_batches"]
    batches = max(1, min(batches, n_hist))

    m_pu = openmc.Material(material_id=1, name="Pu metal")
    m_pu.add_nuclide("Pu239", pu["wt_frac_pu239"], "wo")
    m_pu.add_nuclide("Pu240", pu["wt_frac_pu240"], "wo")
    m_pu.set_density("g/cm3", pu["density_g_cm3"])

    n_ch2 = det["hdpe_density_g_cm3"] / 14.0266 * 0.602214076   # molecules / b-cm
    m_det = openmc.Material(material_id=2, name="HDPE + He3")
    m_det.add_nuclide("H1", 2 * n_ch2)
    m_det.add_element("C", n_ch2)
    m_det.add_nuclide("He3", det["he3_atom_density_per_b_cm"])
    m_det.set_density("sum")
    m_det.add_s_alpha_beta("c_H_in_CH2")

    s_sph = openmc.Sphere(surface_id=1, r=pu["radius_cm"])
    s_cav = openmc.ZCylinder(surface_id=2, r=det["cavity_radius_cm"])
    s_cz0 = openmc.ZPlane(surface_id=3, z0=-det["cavity_half_height_cm"])
    s_cz1 = openmc.ZPlane(surface_id=4, z0=det["cavity_half_height_cm"])
    s_out = openmc.ZCylinder(surface_id=5, r=det["outer_radius_cm"], boundary_type="vacuum")
    s_z0 = openmc.ZPlane(surface_id=6, z0=-det["half_height_cm"], boundary_type="vacuum")
    s_z1 = openmc.ZPlane(surface_id=7, z0=det["half_height_cm"], boundary_type="vacuum")
    nb = det["n_banks"]
    th = 2 * np.pi * np.arange(nb) / nb
    planes = [openmc.Plane(surface_id=10 + i, a=-np.sin(t), b=np.cos(t), c=0.0, d=0.0)
              for i, t in enumerate(th)]

    cavity = -s_cav & +s_cz0 & -s_cz1
    body = -s_out & +s_z0 & -s_z1 & ~cavity
    cells = [openmc.Cell(cell_id=1, name="Pu sphere", fill=m_pu, region=-s_sph),
             openmc.Cell(cell_id=2, name="cavity", region=cavity & +s_sph)]
    for i in range(nb):
        cells.append(openmc.Cell(cell_id=det["bank_cell_ids"][i], name=f"bank {i}", fill=m_det,
                                 region=body & +planes[i] & -planes[(i + 1) % nb]))
    geom = openmc.Geometry(openmc.Universe(cells=cells))

    w = cfg["source_spectrum_watt_pu240_sf"]
    src = openmc.IndependentSource(
        space=openmc.stats.spherical_uniform(r_outer=pu["radius_cm"]),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Watt(a=w["a_MeV"] * 1e6, b=w["b_inv_MeV"] * 1e-6))

    st = openmc.Settings()
    st.run_mode = "fixed source"
    st.particles = int(np.ceil(n_hist / batches))
    st.batches = batches
    st.source = src
    st.survival_biasing = False
    st.seed = seed if seed is not None else cfg["transport"]["seed"]
    st.output = {"tallies": False}
    st.collision_track = {"cell_ids": det["bank_cell_ids"], "nuclides": ["He3"],
                          "reactions": [101], "max_collisions": 2_000_000,
                          "max_collision_track_files": 10_000}
    st.surf_source_write = {"surface_ids": [1], "max_particles": 50_000_000}
    return openmc.Model(geometry=geom, materials=openmc.Materials([m_pu, m_det]),
                        settings=st)


def read_collision_tracks(run_dir):
    import openmc
    files = sorted(glob.glob(os.path.join(run_dir, "collision_track*.h5")))
    if not files:
        raise FileNotFoundError(f"no collision_track*.h5 in {run_dir}")
    return np.concatenate([openmc.read_collision_track_file(f) for f in files]), files


def bank_of(x, y, nb=3):
    return np.floor((np.arctan2(y, x) % (2 * np.pi)) / (2 * np.pi / nb)).astype(np.int8)


def sphere_net_leakage(run_dir, n_hist):
    """Net (out - in) crossings of the Pu sphere surface per source neutron."""
    import h5py
    with h5py.File(os.path.join(run_dir, "surface_source.h5")) as f:
        sb = f["source_bank"][:]
    r = np.stack([sb["r"]["x"], sb["r"]["y"], sb["r"]["z"]], 1)
    u = np.stack([sb["u"]["x"], sb["u"]["y"], sb["u"]["z"]], 1)
    out = np.einsum("ij,ij->i", r, u) > 0
    w = sb["wgt"]
    return float((w[out].sum() - w[~out].sum()) / n_hist), int(out.sum()), int((~out).sum())


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
class StepLog:
    def __init__(self, name, mode="w"):
        self.path = LOGS / f"{name}.log"
        self.f = open(self.path, mode)
        self.write(f"=== {name}  {datetime.datetime.now().isoformat(timespec='seconds')} ===")

    def write(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def changelog(step, files, changes, smoke):
    block = (f"\n## {datetime.datetime.now().isoformat(timespec='seconds')} — {step}\n"
             f"**Files Created/Modified:** {', '.join(files)}\n\n"
             f"**Physics/Code Changes & Why:**\n" + "".join(f"- {c}\n" for c in changes) +
             f"\n**Smoke Test & Metrics:**\n" + "".join(f"- {s}\n" for s in smoke))
    print("\n" + "=" * 70 + block + "=" * 70)
    with open(LOGS / "CHANGELOG.md", "a") as f:
        f.write(block)


# --------------------------------------------------------------------------
# Synthetic toy stream with analytically known S, D_inf, T_inf
# --------------------------------------------------------------------------
def toy_stream(event_rates, tau, T_meas, seed=0, n_banks=3):
    """event_rates[m] = rate of m-neutron detected bursts (m = 1, 2, 3, ...).

    Each detected neutron gets an iid Exp(tau) delay and an iid uniform bank.
    Truth: S = sum m R_m, D_inf = sum C(m,2) R_m, T_inf = sum C(m,3) R_m.
    """
    from math import comb
    rng = np.random.default_rng(seed)
    ts, bk = [], []
    for m, R in event_rates.items():
        n = rng.poisson(R * T_meas)
        t0 = rng.uniform(0, T_meas, n)
        ts.append(np.repeat(t0, m) + rng.exponential(tau, n * m))
        bk.append(rng.integers(0, n_banks, n * m))
    ts, bk = np.concatenate(ts), np.concatenate(bk).astype(np.int8)
    o = np.argsort(ts)
    truth = dict(S=sum(m * R for m, R in event_rates.items()),
                 Dinf=sum(comb(m, 2) * R for m, R in event_rates.items()),
                 Tinf=sum(comb(m, 3) * R for m, R in event_rates.items()))
    return ts[o], bk[o], truth
