# viz_pospos_outer_avg.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any
import itertools

def _axes_list(hx):
    """hx may be [axis_p, axis_v, ...] or np.array; return a list of 1D arrays."""
    if isinstance(hx, (list, tuple)):
        return [np.asarray(ax).ravel() for ax in hx]
    return [np.asarray(hx).ravel()]

def _choose_q0(tree: Any) -> int:
    """Pick a non-accepting DFA state: prefer native 0, else the smallest native index."""
    skip = {tree.F1}
    if getattr(tree, "sink1", None) is not None:
        skip.add(tree.sink1)
    cands = [q for q in tree.S1 if q not in skip]
    if not cands:
        raise RuntimeError("No non-accepting states available (excluding qf/sink).")
    q0 = next((q for q in cands if tree._from1(q) == 0), None)
    return q0 if q0 is not None else min(cands, key=lambda q: tree._from1(q))

def _avg_over_nonpos_auto(v_flat: np.ndarray, axes: list, pos_dim: int = 0):
    """
    Try all combinations of:
      - reshape order: 'C' or 'F'
      - axis permutations: identity or swapped
    Average over all dims except the (permuted) position dim.
    Return the best candidate (largest std across position) and debug info.
    """
    sizes = [len(ax) for ax in axes]
    N = int(np.prod(sizes))
    if v_flat.size != N:
        raise ValueError(f"state-size mismatch: v_flat.size={v_flat.size}, sizes={sizes}")

    best = None
    best_std = -np.inf
    best_info = None

    D = len(sizes)
    perms = list(itertools.permutations(range(D)))  # e.g., [(0,1), (1,0)] for 2D

    for order in ("C", "F"):
        for perm in perms:
            sizes_perm = [sizes[i] for i in perm]
            pos_idx_in_perm = perm.index(pos_dim)

            arr = v_flat.reshape(sizes_perm, order=order)
            axes_to_avg = tuple(i for i in range(D) if i != pos_idx_in_perm)
            v_pos = np.asarray(arr.mean(axis=axes_to_avg))

            s = float(np.nanstd(v_pos))
            if s > best_std:
                best_std = s
                best = v_pos
                best_info = (order, perm, pos_idx_in_perm)

    return best, best_info  # v_pos, (order, perm, where_pos_is)

def plot_q0_outer_sum_posavg(
    tree: Any,
    sysAbs: Union[Dict[int, Any], list, tuple],
    q_native: Optional[int] = None,
    title: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    pos_dim_1: int = 0,      # which axis of hx[1] is "position"
    pos_dim_2: int = 0,      # which axis of hx[2] is "position"
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Collapse each subsystem's (p, v, ...) by averaging over all non-position dims,
    then for a chosen non-accepting DFA state q:
      - for each node n in q, take position-only vectors vpos_1, vpos_2
      - form outer product vpos_1 ⊗ vpos_2 and sum across nodes
    Visualize over (p1, p2).

    Returns:
      agg (len(p1) x len(p2)), p1_axis, p2_axis, q_selected (1-based)
    """
    assert getattr(tree, "dim", None) == 2, "This viz expects dim=2 (two subsystems)."

    # choose q
    if q_native is None:
        q0 = _choose_q0(tree)
    else:
        q0 = tree._to1(int(q_native))
        if q0 == tree.F1 or (getattr(tree, "sink1", None) is not None and q0 == tree.sink1):
            raise ValueError("Given q_native is accepting or sink; choose a non-accepting state.")
    print(f"chosen q0 (1-based): {q0}, native: {tree._from1(q0)}")

    # unpack axes
    if isinstance(sysAbs, dict):
        sa1, sa2 = sysAbs[1], sysAbs[2]
    else:
        sa1, sa2 = sysAbs[0], sysAbs[1]
    axes1 = _axes_list(sa1.hx)   # e.g., [pos_axis, vel_axis] (but not guaranteed)
    axes2 = _axes_list(sa2.hx)

    p1_axis = axes1[pos_dim_1]
    p2_axis = axes2[pos_dim_2]

    nodes = tree.Q.get(q0, [])
    if not nodes:
        raise RuntimeError("Selected DFA state has no nodes in the tree.")

    agg = None
    # for debug, track which reshape combo was selected for the *first* node
    debug_once = True

    for n in nodes:
        v1_flat = tree.V[0][n - 1, :]
        v2_flat = tree.V[1][n - 1, :]

        v1_pos, info1 = _avg_over_nonpos_auto(v1_flat, axes1, pos_dim=pos_dim_1)
        v2_pos, info2 = _avg_over_nonpos_auto(v2_flat, axes2, pos_dim=pos_dim_2)

        if debug_once:
            print(f"[reshape pick] dim1: order={info1[0]}, perm={info1[1]}, pos_at={info1[2]}; "
                  f"dim2: order={info2[0]}, perm={info2[1]}, pos_at={info2[2]}")
            debug_once = False

        out = np.multiply.outer(v1_pos, v2_pos)  # (len(p1_axis), len(p2_axis))
        agg = out if agg is None else (agg + out)

    print(f"agg (pos-pos) shape: {agg.shape}  min/max: {float(agg.min())}  {float(agg.max())}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        agg.T,  # rows -> p2, cols -> p1
        extent=(p1_axis[0], p1_axis[-1], p2_axis[0], p2_axis[-1]),
        origin="lower",
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel(r"position of system 1 ($p_1$)")
    ax.set_ylabel(r"position of system 2 ($p_2$)")
    ax.set_title(title or "Position–Position (avg over velocity)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("probability")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return agg, p1_axis, p2_axis, q0
