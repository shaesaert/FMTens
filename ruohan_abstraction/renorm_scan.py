# renorm_scan.py
import numpy as np
import types

def _sub_stochasticize_block(Bu, eps=1e-12, target=0.999, mode="cap"):
    """
    Sub-stochastic row processing:
      - Replace NaNs with 0
      - For all-zero rows: set a diagonal self-loop with value `target` (< 1)
      - For nonzero rows:
          mode="set": scale each row so its sum is exactly `target`
          mode="cap": if a row-sum > `target`, scale it down to `target`
                      (otherwise keep the original row; never enlarge)
    """
    Bu = np.asarray(Bu, dtype=float).copy()
    Bu[np.isnan(Bu)] = 0.0
    rsum = Bu.sum(axis=1, keepdims=True)

    # All-zero rows -> put `target` on the diagonal
    z = rsum <= eps
    if np.any(z):
        idx = np.where(z[:, 0])[0]
        Bu[idx, :] = 0.0
        Bu[idx, idx] = target
        rsum = Bu.sum(axis=1, keepdims=True)

    # Nonzero rows
    nz = ~z[:, 0]
    if np.any(nz):
        if mode == "set":
            Bu[nz, :] *= (target / rsum[nz])           # scale all nonzero rows to `target`
        else:  # "cap"
            scale = np.minimum(1.0, target / rsum[nz]) # only shrink, never enlarge
            Bu[nz, :] *= scale

    return Bu


def _stochasticize_block(Bu, eps=1e-12):
    """
    Row-stochasticize an (N×N) transition block:
      - replace NaNs with 0
      - for all-zero rows, set a self-loop (diagonal = 1)
      - normalize nonzero rows to sum to 1
    """
    Bu = np.asarray(Bu, dtype=float).copy()
    Bu[np.isnan(Bu)] = 0.0
    rsum = Bu.sum(axis=1)
    zero_rows = rsum <= eps
    if np.any(zero_rows):
        Bu[zero_rows, :] = 0.0
        idx = np.where(zero_rows)[0]
        for i in idx:
            Bu[i, i if i < Bu.shape[1] else 0] = 1.0
        rsum = Bu.sum(axis=1)
    nz = rsum > eps
    Bu[nz, :] /= rsum[nz, None]
    return Bu


def renorm_sysAbs_inplace(sysAbs_d, target=0.999, mode="cap"):
    N = sysAbs_d.block(0).shape[0]
    nu = int(np.prod(np.shape(getattr(sysAbs_d, "inputs", 1))))
    if nu < 1 and hasattr(sysAbs_d, "P"):
        nu = sysAbs_d.P.shape[1] // N
    assert nu >= 1, "Could not infer number of actions (nu)."

    blocks = [_sub_stochasticize_block(sysAbs_d.block(u), target=target, mode=mode) for u in range(nu)]
    P_new = np.hstack(blocks)
    if hasattr(sysAbs_d, "P"):
        sysAbs_d.P = P_new

    def _block(self, u, _blocks=blocks):
        return _blocks[u]
    sysAbs_d.block = types.MethodType(_block, sysAbs_d)


def scan_blocks(sysAbs_d, name=""):
    """
    Print per-action block row-sum min/max and whether NaNs are present.
    """
    B0 = sysAbs_d.block(0)
    N = B0.shape[0]
    nu = int(np.prod(np.shape(getattr(sysAbs_d, "inputs", 1)))) or (
        (sysAbs_d.P.shape[1] // N) if hasattr(sysAbs_d, "P") else 1
    )
    print(f"[scan] {name}: N={N}, nu={nu}")
    for u in range(nu):
        Bu = sysAbs_d.block(u).astype(float)
        rsum = Bu.sum(axis=1)
        print(f"  u={u:2d} rowsum min/max = {rsum.min():.6f} / {rsum.max():.6f}  nan={np.isnan(Bu).any()}")


def renorm_and_scan(sysAbs_list, names=None, title=None, target=0.999, mode="cap"):
    head = f" {title}" if title else ""
    print("BEFORE renorm" + head)
    for i, sa in enumerate(sysAbs_list, 1):
        nm = names[i-1] if names and i-1 < len(names) else f"agent{i}"
        scan_blocks(sa, nm)

    for sa in sysAbs_list:
        renorm_sysAbs_inplace(sa, target=target, mode=mode)

    print("AFTER renorm" + head)
    for i, sa in enumerate(sysAbs_list, 1):
        nm = names[i-1] if names and i-1 < len(names) else f"agent{i}"
        scan_blocks(sa, nm)
