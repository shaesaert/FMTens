# viz_q0_outer.py
# Visualization: for a selected non-accepting DFA state q0, take all tree nodes,
# compute outer products across dimensions and sum them up.
# Display result on (x1, x2) grid with 0~1 color scale.
#
# Only supports dim=2 with each subsystem being 1D (consistent with your current code).
# Usage (inside an existing script/console with tree/sysAbs):
#   import viz_q0_outer as viz
#   viz.plot_q0_outer_sum(tree, sysAbs, title="q0 outer-product sum")
#
# To get numerical results:
#   agg, x1, x2, q0 = viz.plot_q0_outer_sum(tree, sysAbs, show=False)
#   # You can then process agg (shape=(len(x1), len(x2)))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any

def _axis_from_hx(hx: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """hx may be np.array or [axis]; unify to extract 1D axis array."""
    return hx[0] if isinstance(hx, (list, tuple)) else hx

def _choose_q0(tree: Any) -> int:
    """
    Choose a non-accepting q0 (tree internal 1-based index):
    - Prefer the state whose native index is 0 (if it exists)
    - Otherwise choose the non-accepting state with the smallest native index
    """
    skip = {tree.F1}
    if getattr(tree, "sink1", None) is not None:
        skip.add(tree.sink1)
    cands = [q for q in tree.S1 if q not in skip]
    if not cands:
        raise RuntimeError("No non-accepting states available (excluding qf/sink).")
    q0 = next((q for q in cands if tree._from1(q) == 0), None)
    if q0 is None:
        q0 = min(cands, key=lambda q: tree._from1(q))
    return q0

def plot_q0_outer_sum(
    tree: Any,
    sysAbs: Union[Dict[int, Any], list, tuple],
    q_native: Optional[int] = None,
    title: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    For all nodes in a given non-accepting DFA state q:
      - Take each node’s vectors v0, v1 in the two dimensions
      - Compute outer product v0 ⊗ v1 of shape (N1, N2)
      - Sum all such outer products across nodes
    Visualize result over (x1, x2) with imshow (color scale 0~1).

    Parameters
    ----------
    tree : DFATreeR1 instance
    sysAbs : {1: MDPModel, 2: MDPModel} or [sa1, sa2]
    q_native : if provided, choose DFA state by this native index (e.g. 0,1,2...);
               otherwise automatically select "non-accepting state with native=0",
               or the non-accepting state with smallest native index
    title : plot title
    vmin, vmax : color scale range (default 0~1)
    show : whether to call plt.show()
    save_path : if provided, save figure to this path (e.g. "q0_outer.png")

    Returns
    -------
    agg : np.ndarray, shape=(N1, N2)  summed outer products
    x1, x2 : np.ndarray  axis arrays for both dimensions
    q0 : int  selected DFA state (tree internal 1-based index)
    """
    # Only supports two 1D subsystems
    assert getattr(tree, "dim", None) == 2, "Visualization currently supports only dim=2 (two 1D subsystems)"

    # Select q0
    if q_native is not None:
        # Map native index to tree internal 1-based index
        q0 = tree._to1(int(q_native))
        if q0 == tree.F1 or (getattr(tree, "sink1", None) is not None and q0 == tree.sink1):
            raise ValueError("Given q_native is accepting or sink; choose a non-accepting state.")
    else:
        q0 = _choose_q0(tree)

    print(f"chosen q0 (1-based): {q0}, native: {tree._from1(q0)}")

    # Collect nodes in q0
    nodes = tree.Q.get(q0, [])
    if not nodes:
        raise RuntimeError("Selected q has no nodes in the tree.")

    # Aggregate outer products
    agg = None
    for n in nodes:
        v0 = tree.V[0][n - 1, :]  # (N1,)
        v1 = tree.V[1][n - 1, :]  # (N2,)
        out = np.multiply.outer(v0, v1)  # (N1, N2)
        agg = out if agg is None else (agg + out)

    print(f"agg shape: {agg.shape}  min/max: {float(agg.min())}  {float(agg.max())}")

    # Extract axes from two dimensions
    if isinstance(sysAbs, dict):
        sa1, sa2 = sysAbs[1], sysAbs[2]
    else:
        sa1, sa2 = sysAbs[0], sysAbs[1]
    x1 = _axis_from_hx(sa1.hx)  # 1D array
    x2 = _axis_from_hx(sa2.hx)  # 1D array

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        agg.T,  # transpose so rows->x2, cols->x1
        extent=(x1[0], x1[-1], x2[0], x2[-1]),
        origin="lower",
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title or f"Sum of outer-products over nodes in DFA state q={tree._from1(q0)}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("probability")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return agg, x1, x2, q0

# Runnable as a script: only prints usage instructions
if __name__ == "__main__":
    print(
        "This file visualizes the outer-product sum for q0.\n"
        "Use in an environment with existing tree/sysAbs:\n"
        "  import viz_q0_outer as viz\n"
        "  viz.plot_q0_outer_sum(tree, sysAbs, title='q0 outer sum')\n"
        "(Note: supports only dim=2 with each dimension being 1D)"
    )
