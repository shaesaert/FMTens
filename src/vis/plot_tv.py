"""
Heatmap visualisation of a 2D satisfaction tensor.

Single function :func:`plotV_rank1`: renders ``tv`` (shape ``(Nx, Ny)``)
as an ``imshow`` heatmap with axes labelled by the per-dimension state
grids from ``sysAbs``. Matches the MATLAB ``imagesc`` orientation used
in the SysCoRe reference.
"""

import numpy as np
import matplotlib.pyplot as plt


def plotV_rank1(sysAbs, tv, ax=None):
    """
    Plot a 2D satisfaction tensor as a heatmap.

    Python equivalent of the MATLAB function::

        function plotV_rank1(sysAbs, tv)
            X1hat = sysAbs{1}.states;
            X2hat = sysAbs{2}.states;
            imagesc(X1hat, X2hat, transpose(tv))
            set(gca, 'Ydir', 'normal')
            xlabel('x_1'); ylabel('x_2'); zlim([0, 1])
        end

    Parameters
    ----------
    sysAbs : dict
        Per-dimension abstractions, keyed by integer dimension index.
        Each ``sysAbs[d].states`` must be a 1D array of grid centres.
        At least two keys are expected; the two smallest are taken as
        the x and y axes.
    tv : np.ndarray
        Satisfaction tensor of shape ``(Nx, Ny)``, x-first (MATLAB
        convention).
    ax : matplotlib.axes.Axes, optional
        Axes to plot into; a new ``(6, 5)`` figure is created if None.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist. Add a colorbar with
        ``fig.colorbar(im, ax=ax)`` from the caller if desired.
    """
    keys = sorted(sysAbs.keys())
    X1hat = np.asarray(sysAbs[keys[0]].states).ravel()
    X2hat = np.asarray(sysAbs[keys[1]].states).ravel()

    x_min, x_max = float(X1hat.min()), float(X1hat.max())
    y_min, y_max = float(X2hat.min()), float(X2hat.max())

    # Transpose to match the MATLAB imagesc orientation.
    Z = np.asarray(tv, dtype=float).T

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z, origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        cmap="viridis",        # 0 -> blue, 1 -> yellow
        vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("tv heatmap")
    return im