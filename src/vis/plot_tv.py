import numpy as np
import matplotlib.pyplot as plt

def plotV_rank1(sysAbs, tv, ax=None):
    """
    Python equivalent of:
      function plotV_rank1(sysAbs, tv)
        X1hat = sysAbs{1}.states;
        X2hat = sysAbs{2}.states;
        imagesc(X1hat, X2hat, transpose(tv))
        set(gca, 'Ydir', 'normal')
        xlabel('x_1'); ylabel('x_2'); zlim([0, 1])
      end

    Assumes:
      - sysAbs is a dict keyed by dims (e.g., {0: ..., 1: ...})
      - sysAbs[d].states is a 1D array of grid centers
      - tv has shape (Nx, Ny), x-first (like MATLAB)
    """
    keys = sorted(sysAbs.keys())
    X1hat = np.asarray(sysAbs[keys[0]].states).ravel()
    X2hat = np.asarray(sysAbs[keys[1]].states).ravel()

    # axes extents from the state vectors (treat them as centers)
    x_min, x_max = float(X1hat.min()), float(X1hat.max())
    y_min, y_max = float(X2hat.min()), float(X2hat.max())

    Z = np.asarray(tv, dtype=float).T  # transpose to match MATLAB imagesc orientation

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z, origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        cmap="viridis",  # 0→blue, 1→yellow
        vmin=0.0, vmax=1.0
    )
    # plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("tv heatmap")
    return im
