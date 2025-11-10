#!/usr/bin/env python3
"""
Run D=20 for T in {1,2,3,4}, measure per-T memory (peak & end RSS) and wall time,
save results, and plot memory usage vs T (headless Agg backend).
"""

# -----------------------
# Stdlib + memory helpers
# -----------------------
import os
import gc
import time
import platform
from typing import Optional, Callable
from itertools import product
import csv

# (Optional) pin BLAS threads for reproducibility / lower memory jitter
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import resource  # POSIX (macOS/Linux)
    _HAS_RESOURCE = True
except Exception:
    _HAS_RESOURCE = False

def _current_rss_bytes():
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None

def _fmt_bytes_mib(b):
    mib = b / (1024**2)
    gib = b / (1024**3)
    return f"{mib:.2f} MiB ({gib:.3f} GiB)"

def _bytes_to_mib(b):
    return b / (1024**2)

# -----------------------
# Third-party + project imports (no GUI plotting)
# -----------------------
import numpy as np
import polytope as pc

from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label_eps
from src.dynprog.dfa_tree_r1 import DFATree

# Headless plotting (to keep plotting overhead minimal)
import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

# -----------------------
# Small helpers (unchanged)
# -----------------------
def nodes_with_q(G: DFATree, q: int):
    """Tree node ids whose DFA mode == q (via G.Q)."""
    return list(G.Q.get(int(q), []))

def robust_value_at_anchor_mul_then_sum(G: DFATree, qbar_vec, idx_vec):
    """
    Anchor value at idx_vec:
      sum_{(n_0,...,n_{D-1})}  prod_d V^{(d)}[n_d, idx_vec[d]],
    where n_d ranges over nodes_with_q(G, qbar_vec[d]).
    """
    D = G.dim
    assert len(qbar_vec) == D and len(idx_vec) == D

    per_dim = [nodes_with_q(G, int(qbar_vec[d])) for d in range(D)]
    if any(len(lst) == 0 for lst in per_dim):
        return 0.0

    total = 0.0
    for combo in product(*per_dim):
        term = 1.0
        for d, n in enumerate(combo):
            term *= float(G.V[d][n, idx_vec[d]])
            if term == 0.0:
                break
        total += term
    return float(total)

def fht_term(G: DFATree, T: int, qbar_vec, idx_vec) -> float:
    """
    Triple sum as exact: no algebraic shortcut
    """
    qbar = int(qbar_vec[0] if hasattr(qbar_vec, "__len__") else qbar_vec)
    idx_x = int(idx_vec[0]  if hasattr(idx_vec, "__len__")  else idx_vec)

    if 'Dl' not in G.tree.graph:
        G._recompute_levels()
    Dl = G.tree.graph['Dl']

    interested = set(G.Q.get(qbar, []))
    if not interested or T <= 0:
        return 0.0

    D = len(G.V)

    def node_val(z: int) -> float:
        prod = 1.0
        for d in range(D):
            v = float(G.V[d][z, idx_x])
            if v <= 0.0:
                return 0.0
            prod *= v
        return prod

    total = 0.0
    T_build = int(T)

    for h in range(0, T_build + 1):
        for npred in range(0, h):
            nodes_at_npred = Dl[npred][1] if npred < len(Dl) else []
            nodes = [z for z in nodes_at_npred if z in interested]
            for z in nodes:
                total += node_val(z)

    return float(total)

def make_uniform_pol_and_rho(DFA, sysAbs, L):
    dims = sorted(sysAbs.keys())
    nx_list = [sysAbs[d].N for d in dims]
    nu_list = [sysAbs[d].M for d in dims]
    nQ = len(DFA.S)
    pol = [
        [np.full((nx_list[d], nu_list[d]), 1.0 / nu_list[d], dtype=float) for d in range(len(dims))]
        for _ in range(nQ)
    ]
    rho = [np.full(nx_list[d], 1.0 / nx_list[d], dtype=float) for d in range(len(dims))]
    L_list = [L[d] for d in dims]
    return pol, rho, nx_list, L_list

# -----------------------
# One full sweep for a given dimension D (no GUI plotting)
# -----------------------
def sweep_for_dim(DIM: int,
                  nx_per_dim: int = 100,
                  nu_per_dim: int = 10,
                  anchor_centers = (40, 45, 50, 55, 59),
                  t_specs = None,
                  mem_callback: Optional[Callable[[], None]] = None):
    # Force t_specs to the provided list exactly
    if t_specs is None or len(t_specs) == 0:
        raise ValueError("Provide at least one t_spec.")

    delta_i = 0.0717  # per-dimension delta

    # Build continuous systems and safeset for this DIM
    s = np.array([[1], [-1]])
    bx = np.array([10, 10])  # X = [-10, 10]
    bu = np.array([2,  2])   # U = [-2,  2]
    bp1 = np.array([5, 5])
    P1 = pc.Polytope(s, bp1)

    # LTI params (identical per dim)
    A = {i: np.array([[0.9]]) for i in range(DIM)}
    B = {i: np.array([[0.5]]) for i in range(DIM)}
    C = {i: np.array([[1.0]]) for i in range(DIM)}
    Dmat = {i: np.array([[0.0]]) for i in range(DIM)}
    Bw = {i: np.array([[0.5]]) for i in range(DIM)}
    mu = {i: np.array([[0.0]]) for i in range(DIM)}
    sigma = {i: np.array([[1.0]]) for i in range(DIM)}

    sysLTI: dict[int, Optional[np.ndarray]] = {}
    safeset = ' '
    AP = set()
    for i in range(DIM):
        sys = LinModel(A[i], B[i], C[i], Dmat[i], Bw[i], mu=mu[i], sigma=sigma[i])
        sys.X = pc.Polytope(s, bx)
        sys.U = pc.Polytope(s, bu)
        sys.AP = [f'p1{i}']
        sys.regions = [P1]
        sysLTI[i] = sys
        safeset += f' & p1{i}'
        AP.update(sys.AP)
    safeset = safeset[3:]
    print(f"[D={DIM}] Safeset: {safeset}")

    # Anchors (same index across all dims)
    anchor_idx_list = [[k]*DIM for k in anchor_centers]

    rt_av_vals   = []
    apos_av_vals = []
    fht_avg_vals = []
    tbuild_vals  = []

    for t_spec in t_specs:
        print("\n" + "=" * 70)
        print(f"[D={DIM}] Building for t_spec = {t_spec}")
        print("=" * 70)

        ks = {0, t_spec}
        if t_spec >= 2:
            ks.add(t_spec - 1)

        terms = [f"{'X' * k}({safeset})" for k in sorted(ks)]
        formula = " & ".join(terms)
        print(f"[D={DIM}] LTL Formula (t_spec={t_spec}): {formula}")

        DFA = translate(formula)
        DFA, letters = dfa_manipulation(
            DFA,
            index_base=0,
            ensure_transitions=True,
            remove_qf_self_loop=True,
            verbose=False
        )

        # Abstraction
        sysAbs = {
            i: MDPModel.from_system(sysLTI[i], nx=nx_per_dim, nu=nu_per_dim,
                                    placement='centers', u_placement='endpoints',
                                    tol=1e-19, contract_sum=None, compute_P='1d')
            for i in range(DIM)
        }

        # Robust labeling
        eps_val = 0.1
        L = dim_label_eps(sysAbs, sysLTI, letters, eps=eps_val,
                          visualize=False, outdir=None, prefix="L_eps")

        # Policy/rho lists
        pol, rho, nx_list, L_list = make_uniform_pol_and_rho(DFA, sysAbs, L)

        # Anchor bounds sanity
        for d in range(DIM):
            for idx_set in anchor_idx_list:
                if idx_set[d] >= nx_list[d]:
                    raise ValueError(f"[D={DIM}] anchor idx {idx_set[d]} out of range for dim {d} (nx={nx_list[d]}).")

        # DFA state and transition table
        q0 = int(np.asarray(DFA.S0).ravel()[0])
        Trans = np.asarray(DFA.trans, dtype=int)

        # Build trees (no prune)
        T_build = int(t_spec)

        # RT deltas
        delta_VI_vecs_rt  = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]
        delta_pol_vecs_rt = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]
        # APoS deltas
        delta_VI_vecs_apos  = [np.zeros(sysAbs[d].N, dtype=float) for d in range(DIM)]
        delta_pol_vecs_apos = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]

        # ---- RT ----
        G_rt = DFATree(
            DFA, sysAbs, pol, nx_list, L_list,
            delta_VI=delta_VI_vecs_rt, delta_pol=delta_pol_vecs_rt,
            pol_mode='rt', VI_mode='rt'
        ).initiate()
        for it in range(1, T_build + 1):
            G_rt.maxpolicy(rho)
            G_rt.update_tree()
            G_rt.grow()
            if mem_callback: mem_callback()

        # ---- APoS ----
        G_apos = DFATree(
            DFA, sysAbs, pol, nx_list, L_list,
            delta_VI=delta_VI_vecs_apos, delta_pol=delta_pol_vecs_apos,
            pol_mode='apos', VI_mode='apos'
        ).initiate()
        for it in range(1, T_build + 1):
            if hasattr(G_apos, "set_iter"):
                try:
                    G_apos.set_iter(it)
                except Exception:
                    pass
            G_apos.maxpolicy(rho)
            G_apos.update_tree()
            G_apos.grow()
            if mem_callback: mem_callback()

        # ---- Evaluate per-anchor, then average ----
        vals_rt   = []
        vals_apos = []
        term_fhts = []

        for idx_vec in anchor_idx_list:
            qbar = []
            valid = True
            for d in range(DIM):
                col = np.asarray(L_list[d][:, idx_vec[d]], dtype=bool)
                if not np.any(col):
                    print(f"[D={DIM} t_spec={t_spec}] WARNING: no active letter at idx={idx_vec[d]} for dim {d}; skip anchor.")
                    valid = False
                    break
                l = int(np.where(col)[0][0])
                qbar.append(int(Trans[q0, l]))
            if not valid:
                continue

            V_rt_k = robust_value_at_anchor_mul_then_sum(G_rt, qbar, idx_vec)
            vals_rt.append(V_rt_k)

            V_apos_base_k = robust_value_at_anchor_mul_then_sum(G_apos, qbar, idx_vec)
            TermFHT_k     = fht_term(G_apos, T_build, qbar, idx_vec)
            term_fhts.append(TermFHT_k)
            expr_k        = V_apos_base_k - T_build * (1.0 - (1.0 - delta_i)**DIM) \
                            + (1.0 - (1.0 - delta_i)**DIM) * TermFHT_k
            V_apos_k      = max(0.0, expr_k)
            vals_apos.append(V_apos_k)

        if len(vals_rt) == 0:
            rt_av = np.nan
            apos_av = np.nan
            fht_av = np.nan
            print(f"[D={DIM} t_spec={t_spec}] No valid anchors; averages are NaN.")
        else:
            rt_av = float(np.mean(vals_rt))
            apos_av = float(np.mean(vals_apos))
            fht_av = float(np.mean(term_fhts))
            print(f"[D={DIM} t_spec={t_spec}] AVG over anchors: RT_AV={rt_av:.6e}  APoS_AV={apos_av:.6e}")

        rt_av_vals.append(rt_av)
        apos_av_vals.append(apos_av)
        fht_avg_vals.append(fht_av)
        tbuild_vals.append(T_build)

        if mem_callback: mem_callback()

    # pack results (returned for reference; file saved by caller if desired)
    return dict(
        DIM=DIM, t_specs=np.asarray(tbuild_vals),
        V_rt_avg=np.asarray(rt_av_vals, float),
        V_apos_avg=np.asarray(apos_av_vals, float),
        FHT_avg=np.asarray(fht_avg_vals, float),
        T_builds=np.asarray(tbuild_vals, int)
    )

# -----------------------
# Main: D=20, T in {1,2,3,4}; collect per-T memory & plot
# -----------------------
if __name__ == "__main__":
    if not _HAS_PSUTIL:
        raise RuntimeError("This script requires 'psutil' to track per-T RSS. Please install it: pip install psutil")

    DIM = 20
    anchor_centers = (40, 45, 50, 55, 59)
    nx_per_dim = 100
    nu_per_dim = 10

    T_values = [1, 2, 3, 4]
    peak_mib_list = []
    end_mib_list  = []
    wall_s_list   = []

    # Run each T in the SAME process, but track per-T local peak via callback
    for T in T_values:
        print(f"\n########## Running D={DIM}, T={T} ##########")
        start_time = time.perf_counter()
        baseline_b = _current_rss_bytes()
        if baseline_b is None:
            raise RuntimeError("psutil returned no RSS; cannot measure per-T memory.")

        # Use a mutable container to avoid 'nonlocal' (which doesn't work at module scope)
        local_peak_b = [baseline_b]  # list of one element

        def _update_local_peak():
            b = _current_rss_bytes()
            if b is not None and b > local_peak_b[0]:
                local_peak_b[0] = b

        # Run one sweep for this T
        res = sweep_for_dim(
            DIM,
            nx_per_dim=nx_per_dim,
            nu_per_dim=nu_per_dim,
            anchor_centers=anchor_centers,
            t_specs=[T],
            mem_callback=_update_local_peak
        )

        # Capture end-of-T RSS BEFORE cleanup (represents steady RSS after compute)
        end_b = _current_rss_bytes()
        wall_s = time.perf_counter() - start_time

        # Store metrics
        peak_mib_list.append(_bytes_to_mib(local_peak_b[0]))
        end_mib_list.append(_bytes_to_mib(end_b if end_b is not None else local_peak_b[0]))
        wall_s_list.append(wall_s)

        # Save the per-T NPZ (optional)
        np.savez(
            f"anchor_values_vs_tspec_multi_anchors_D{DIM}_T{T}.npz",
            t_specs=res["t_specs"],
            anchor_centers=np.asarray(anchor_centers, dtype=int),
            V_rt_avg=res["V_rt_avg"],
            V_apos_avg=res["V_apos_avg"],
            FHT_avg=res["FHT_avg"],
            T_builds=res["T_builds"]
        )
        print(f"[D={DIM} T={T}] Peak RSS (local): {_fmt_bytes_mib(local_peak_b[0])}")
        if end_b is not None:
            print(f"[D={DIM} T={T}] End RSS: {_fmt_bytes_mib(end_b)}")
        print(f"[D={DIM} T={T}] Wall time: {wall_s:.2f} s")

        # Cleanup to minimize carryover
        del res
        gc.collect()

    # -----------------------
    # Save CSV with memory/time
    # -----------------------
    csv_path = f"memory_usage_D{DIM}_T1-4.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "Peak_RSS_MiB", "End_RSS_MiB", "Wall_time_s"])
        for T, p, e, wtime in zip(T_values, peak_mib_list, end_mib_list, wall_s_list):
            w.writerow([T, f"{p:.2f}", f"{e:.2f}", f"{wtime:.2f}"])
    print(f"Saved memory/time table to {csv_path}")

    # -----------------------
    # Plot memory vs T (headless Agg)
    # -----------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(T_values, peak_mib_list, marker="o", linewidth=2.0, label="Peak RSS (MiB)")
    ax.plot(T_values, end_mib_list,  marker="s", linewidth=2.0, linestyle="--", label="End RSS (MiB)")
    ax.set_xlabel("Specification horizon T")
    ax.set_ylabel("Memory (MiB)")
    ax.set_xticks(T_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = f"memory_vs_T_D{DIM}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved memory plot to {out_png}")

    # -----------------------
    # Final overall report (whole process)
    # -----------------------
    if _HAS_RESOURCE:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            peak_b = int(ru.ru_maxrss)
        elif platform.system() == "Linux":
            peak_b = int(ru.ru_maxrss) * 1024
        else:
            peak_b = None
        if peak_b is not None:
            print(f"[OVERALL] Process Peak RSS: {_fmt_bytes_mib(peak_b)}")
