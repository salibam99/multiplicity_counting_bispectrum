"""Geometry figures for the report: XY / XZ slices of the OpenMC model and He-3 capture maps."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1] / "pu_multiplicity_spectral"
sys.path.insert(0, str(ROOT))
from common.pu_model import bank_of, build_model, load_config, read_collision_tracks  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
COLORS = {0: "#ffffff", 1: "#8b1a1a", 2: "#e8e8e8", 101: "#4c72b0", 102: "#dd8452", 103: "#55a868"}
NAMES = {1: "Pu sphere (300 g, r = 1.654 cm)", 2: "void cavity", 101: "bank X (HDPE + He-3)",
         102: "bank Y (HDPE + He-3)", 103: "bank Z (HDPE + He-3)", 0: "outside (vacuum)"}


def raster(cells, xs, ys, plane):
    ids = np.zeros((len(ys), len(xs)), int)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p = (x, y, 0.0) if plane == "xy" else (x, 0.0, y)
            for c in cells:
                if p in c.region:
                    ids[j, i] = c.id
                    break
    return ids


def show(ax, ids, xs, ys, xl, yl, title):
    keys = sorted(COLORS)
    lut = {k: n for n, k in enumerate(keys)}
    img = np.vectorize(lut.get)(ids)
    ax.imshow(img, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
              cmap=ListedColormap([COLORS[k] for k in keys]), vmin=0, vmax=len(keys) - 1,
              interpolation="nearest")
    ax.set(xlabel=xl, ylabel=yl, title=title, aspect="equal")


def main():
    cfg = load_config()
    model = build_model(cfg, 10)
    cells = list(model.geometry.get_all_cells().values())
    n = 401
    g = np.linspace(-22, 22, n)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    show(ax[0], raster(cells, g, g, "xy"), g, g, "x [cm]", "y [cm]", "XY slice at z = 0")
    show(ax[1], raster(cells, g, g, "xz"), g, g, "x [cm]", "z [cm]", "XZ slice at y = 0")
    for a in ax:
        a.add_patch(plt.Rectangle((-20, -20), 40, 40, fill=False, ec="k", lw=0.8))
    ax[0].annotate("Pu", (0, 0), (6, 9), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax[1].annotate("cavity r = 4 cm,\n|z| < 4 cm", (3.5, 3.5), (8, 12),
                   arrowprops=dict(arrowstyle="->"), fontsize=9)
    fig.legend(handles=[Patch(fc=COLORS[k], ec="k", label=NAMES[k]) for k in (1, 2, 101, 102, 103)],
               loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "geometry_slices.png", dpi=170)
    plt.close(fig)

    # zoom on the sphere and cavity
    z = np.linspace(-5, 5, 301)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    show(ax, raster(cells, z, z, "xz"), z, z, "x [cm]", "z [cm]", "Cavity detail (XZ, y = 0)")
    fig.tight_layout()
    fig.savefig(OUT / "geometry_zoom.png", dpi=170)
    plt.close(fig)

    # He-3 capture locations from the 400k-history run
    ct, _ = read_collision_tracks(ROOT / "data" / "openmc_runs" / "step01_full")
    ct = ct[ct["time"] < 10e-3]
    x, y, zz = ct["r"]["x"], ct["r"]["y"], ct["r"]["z"]
    b = bank_of(x, y)
    rng = np.random.default_rng(0)
    s = rng.choice(len(x), 40000, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for k, c in zip(range(3), ("#4c72b0", "#dd8452", "#55a868")):
        m = b[s] == k
        ax[0].scatter(x[s][m], y[s][m], s=0.4, c=c, label=f"bank {'XYZ'[k]}")
    ax[0].set(xlim=(-21, 21), ylim=(-21, 21), aspect="equal", xlabel="x [cm]", ylabel="y [cm]",
              title="He-3 capture positions (40k of 332k), coloured by bank")
    ax[0].legend(markerscale=12, fontsize=8, loc="upper right")
    r = np.hypot(x, y)
    h = ax[1].hist2d(r, zz, bins=[np.linspace(0, 20, 81), np.linspace(-20, 20, 161)], cmap="viridis")
    fig.colorbar(h[3], ax=ax[1], label="captures per bin")
    ax[1].set(xlabel="r [cm]", ylabel="z [cm]", aspect="equal",
              title="Capture density in (r, z): where neutrons are detected")
    fig.tight_layout()
    fig.savefig(OUT / "capture_maps.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
