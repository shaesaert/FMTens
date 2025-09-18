import numpy as np
import polytope as pc

# —— Multi-dimensional L (missing APs → True): detect directly on full 2D/multidim grid via polytopes
def _build_L_for_subsystem(axes, letters, ap_list, regions_by_ap, all_aps, tol=1e-12) -> np.ndarray:
    # Build full grid points X: (N, d)
    if isinstance(axes, (list, tuple)):
        arrs = [np.asarray(ax).ravel() for ax in axes]
    else:
        arrs = [np.asarray(axes).ravel()]
    meshes = np.meshgrid(*arrs, indexing='xy')
    X = np.stack([m.ravel() for m in meshes], axis=1)  # (N, d)
    N = X.shape[0]

    def _in_poly(poly: pc.Polytope, X_: np.ndarray) -> np.ndarray:
        A = np.asarray(poly.A, dtype=float); b = np.asarray(poly.b, dtype=float).ravel()
        return np.all((A @ X_.T) <= b[:, None] + tol, axis=0)

    # Masks for each AP; missing APs are set to True (don’t care)
    mask = {}
    ap_set = set(ap_list)
    for ap in ap_list:
        regs = regions_by_ap.get(ap, None)
        if regs is None:
            mask[ap] = np.zeros(N, dtype=bool);
            continue
        regs = regs if isinstance(regs, (list, tuple)) else [regs]
        m = np.zeros(N, dtype=bool)
        for P in regs:
            if P is not None:
                m |= _in_poly(P, X)
        mask[ap] = m
    for ap in all_aps:
        if ap not in mask:
            mask[ap] = np.ones(N, dtype=bool)

    # Evaluate each letter (supports !, &, |)
    rows = []
    for lab in letters:
        lab = str(lab).strip()
        if lab in ("1", "True", ""):
            rows.append(np.ones(N, dtype=float));
            continue
        expr = lab.replace("!", "~")
        val = eval(expr, {"__builtins__": {}}, mask)  # numpy bool vector
        rows.append(val.astype(float))
    return np.vstack(rows)  # (n_letters, N)