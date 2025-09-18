import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_binary_matrix(mat, title="", row_labels=None, outfile=None, x_tick_step=None):
    """
    Parameters
    ----------
    mat : list[list[int]] or np.ndarray
        Values must be 0/1. Shape (n_rows, n_cols).
    title : str
        Figure title.
    row_labels : list[str] or None
        Optional row labels (e.g., the letter names for each row).
    outfile : str or None
        Optional path to save the figure (e.g., "L1.png"). If None, only display without saving.
    x_tick_step : int or None
        Optional step for x-axis ticks (set to something like 100 when there are many columns).
    """
    A = np.asarray(mat)
    if A.ndim != 2:
        raise ValueError(f"mat must be 2-D, got shape {A.shape}")
    n_rows, n_cols = A.shape

    # Colors: 0 -> red, 1 -> green
    cmap = ListedColormap(["red", "green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    # Figure size adapts to the number of columns (but not too extreme)
    w = min(max(n_cols / 80.0, 8.0), 24.0)
    h = min(max(n_rows * 0.6, 2.5), 12.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)

    im = ax.imshow(A, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=12)

    # Row labels
    if row_labels is not None and len(row_labels) == n_rows:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)
    else:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([str(i) for i in range(n_rows)], fontsize=8)

    # Column ticks (sparsify when there are many columns)
    if x_tick_step is None:
        x_tick_step = 100 if n_cols > 200 else 20 if n_cols > 80 else 10
    xticks = np.arange(0, n_cols, x_tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=8, rotation=45)

    ax.set_xlabel("column (x index)")
    ax.set_ylabel("row (letter index)")
    ax.grid(False)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

