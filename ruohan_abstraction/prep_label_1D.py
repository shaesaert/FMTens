import polytope as pc
import numpy as np

def _intervals_from_polytope_1d(Ppoly: pc.Polytope):
    """Convert a 1D polytope Ax <= b into a list of closed intervals [(lo, hi)],
    for use in label_axis_by_letters."""
    A = np.asarray(Ppoly.A, dtype=float).reshape(-1, 1)
    b = np.asarray(Ppoly.b, dtype=float).ravel()
    lo, hi = -np.inf, +np.inf
    for ai, bi in zip(A[:, 0], b):
        if ai > 0:        # inequality of the form x <= bi/ai
            hi = min(hi, bi / ai)
        elif ai < 0:      # inequality of the form x >= bi/ai
            lo = max(lo, bi / ai)
        else:             # 0 * x <= b
            if bi < 0:
                raise ValueError("Infeasible constraint: 0*x <= b with b < 0")
    return [(float(lo), float(hi))]
