import numpy as np

def _axis_from_hx(hx):
    """hx may be np.array or [axis]; unify to extract 1D axis array."""
    return hx[0] if isinstance(hx, (list, tuple)) else hx


def _nx_of_axes(ax):
    if isinstance(ax, (list, tuple)):
        return int(np.prod([len(np.asarray(a).ravel()) for a in ax]))
    return len(np.asarray(ax).ravel())