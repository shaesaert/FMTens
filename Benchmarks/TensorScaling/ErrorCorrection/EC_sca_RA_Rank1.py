# compare_targetU_safe.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
from importlib import reload

# --- optional dev reload ---
def dev_reload():
    import src.models.linmodel as LinModel_mod
    import src.models.mdpmodel as mdpmodel_mod
    import src.abstraction.utils.labeling as dim_label_mod
    import src.dynprog.dfa_tree_r1 as DFATree_mod
    reload(LinModel_mod); reload(mdpmodel_mod); reload(dim_label_mod); reload(DFATree_mod)

# project imports
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.abstraction.utils.pc_utils import Pc

# your existing parser; reuse for sigma/delta
from src.config.delta_ui import parse_delta_list as parse_list

# ---------- small helpers ----------
def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} ").strip()
    except EOFError:
        ans = ""
    return ans if ans else default

def ask_yes_no(prompt: str, default: bool) -> bool:
    tag = "[Y/n]" if default else "[y/N]"
    ans = ask(f"{prompt} {tag}:", "Y" if default else "N").lower()
    return ans in ("y", "yes") if ans else default

def _fmt_vec(vals):
    return "[" + ", ".join(f"{float(v):g}" for v in vals) + "]"

def parse_Ts(s: str, default: str = "1:100:1"):
    """
    Parse Ts string as either 'start:end[:step]' (inclusive end) or CSV '1,5,10'.
    """
    s = (s or "").strip() or default
    if ":" in s:
        parts = [int(x.strip()) for x in s.split(":")]
        if len(parts) == 2:
            start, end = parts; step = 1
        elif len(parts) == 3:
            start, end, step = parts
        else:
            raise ValueError("Ts format must be 'start:end[:step]' or CSV.")
        if step == 0:
            raise ValueError("Ts step cannot be 0.")
        if end < start and step > 0:
            raise ValueError("Ts end must be >= start when step>0.")
        return list(range(start, end + 1, step))
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty Ts list.")
    return vals

def set_uniform_Pxx_for_all_states(G: DFATree, pol):
    """Fill Pxx using dense policy pol[q][d] for all non-final/sink states."""
    skip = {int(G.DFA.F)}
    if hasattr(G.DFA, "sink") and getattr(G.DFA, "sink") is not None:
        skip.add(int(G.DFA.sink))
    for q in set(G.DFA.S) - skip:
        for d in range(G.dim):
            P_src = G.sysAbs[d].P_flat if getattr(G.sysAbs[d], "P_flat", None) is not None else G.sysAbs[d].P
            G.Pxx[q][d] = Pc(P_src, pol[q][d])

def run_one_iteration(G: DFATree, rho, use_maxpolicy: bool, pol, prune_tol: float):
    if use_maxpolicy:
        G.maxpolicy(rho)
    else:
        set_uniform_Pxx_for_all_states(G, pol)
    G.update_tree()
    if prune_tol and prune_tol > 0:
        G.prune(prune_tol, 'leafs')

# ---------- build system/spec for arbitrary dim with (safeset) U ((targetset) & (safeset)) ----------
def build_system_and_spec(dim: int, sigma_vals):
    """
    For each dimension i, create APs p1i (safe) and p2i (target).
    safeset  = p10 & p11 & ... & p1{dim-1}
    targetset= p20 | p21 | ... | p2{dim-1}
    formula  = (safeset) U ( (targetset) & (safeset) )
    """
    # dynamics (all dims identical here)
    A  = {i: np.array([[1.0]]) for i in range(dim)}
    B  = {i: np.array([[1.0]]) for i in range(dim)}
    C  = {i: np.array([[1.0]]) for i in range(dim)}
    Dm = {i: np.array([[0.0]]) for i in range(dim)}
    Bw = {i: np.array([[1.0]]) for i in range(dim)}
    mu = {i: np.array([0.0])  for i in range(dim)}
    sigma = {i: float(sigma_vals[i]) * np.eye(1) for i in range(dim)}

    # state / input bounds
    s_mat = np.array([[1], [-1]])
    bx = np.array([10, 10])
    bu = np.array([2, 2])

    # AP regions (same for all dims)
    bp1 = np.array([5, 5])  # safe
    bp2 = np.array([2, 2])  # target
    P1 = pc.Polytope(s_mat, bp1)
    P2 = pc.Polytope(s_mat, bp2)

    # build systems + APs
    sysLTI = {}
    for i in range(dim):
        sys = LinModel(A[i], B[i], C[i], Dm[i], Bw[i], mu=mu[i], sigma=sigma[i])
        sys.X = pc.Polytope(s_mat, bx)
        sys.U = pc.Polytope(s_mat, bu)
        sys.AP = [f"p1{i}", f"p2{i}"]  # p1= safe, p2= target
        sys.regions = [P1, P2]
        sysLTI[i] = sys

    safeset   = " & ".join(f"p1{i}" for i in range(dim)) if dim > 0 else ""
    targetset = " | ".join(f"p2{i}" for i in range(dim)) if dim > 0 else ""

    formula = f" ({safeset}) U ( ({targetset}) & ({safeset}) )"
    DFA = translate(formula)
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0, ensure_transitions=True, remove_qf_self_loop=True,
        verbose=True
    )
    return sysLTI, DFA, letters

def build_abs_label_policy(sysLTI, DFA, letters, nx: int, nu: int):
    dim = len(sysLTI)
    sysAbs = {
        i: MDPModel.from_system(sysLTI[i], nx=nx, nu=nu, placement='centers',
                                u_placement='endpoints', tol=1e-19,
                                 contract_sum=None, compute_P='1d')
        for i in range(dim)
    }
    L = dim_label(sysAbs, sysLTI, letters, visualize=True)
    dims = sorted(sysAbs.keys())
    nxv  = [sysAbs[d].N for d in dims]
    nuv  = [sysAbs[d].M for d in dims]
    nQ   = len(DFA.S)
    pol = [[np.full((nxv[d], nuv[d]), 1.0/nuv[d], dtype=float) for d in range(len(dims))]
           for _ in range(nQ)]
    rho = [np.full(nxv[d], 1.0/nxv[d], dtype=float) for d in range(len(dims))]
    return sysAbs, L, pol, rho, dims

# Try calling compute_tv_from_tree with memory guard if your local version supports it.
def _compute_tv_safe(G, DFA, L):
    from src.dynprog.utils.treebasedV import compute_tv_from_tree
    try:
        # If you added the guard earlier, this will work:
        return compute_tv_from_tree(G, DFA, L, max_elements=50_000_000)
    except TypeError:
        # Fallback for older signature without kwargs
        return compute_tv_from_tree(G, DFA, L)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare (robustness-aware & a-posteriori) vs normal tree for target∧safe until spec.")
    ap.add_argument("--dim", type=int, default=None,
                    help="Number of 1D subsystems (enter here or you'll be prompted).")
    ap.add_argument("--nx", type=int, default=10, help="States per dimension in abstraction.")
    ap.add_argument("--nu", type=int, default=5,   help="Actions per dimension in abstraction.")
    ap.add_argument("--delta", type=str, default="", help="Δ per-dim: single or CSV (e.g., '0.001' or '0.001,0.001,...').")
    ap.add_argument("--sigma", type=str, default="", help="Σ per-dim (cov scalar): single or CSV (default 1.0).")
    ap.add_argument("--Ts", type=str, default="", help="T sweep: 'start:end[:step]' or CSV list (default '1:100:1').")
    ap.add_argument("--prune-tol", type=float, default=0.0, help="Pruning tol for leafs (0 disables pruning).")
    ap.add_argument("--apply-maxpolicy", action="store_true", help="Apply greedy policy improvement each iteration.")
    ap.add_argument("--dev-reload", action="store_true")
    args, _ = ap.parse_known_args()

    if args.dev_reload:
        dev_reload()

    # ----- choose dim (# of 1D subsystems) -----
    if args.dim is None:
        dim_str = ask("Enter dim (# of 1D subsystems) [default 9]:", "9")
        D_dims = max(1, int(dim_str))
    else:
        D_dims = max(1, int(args.dim))

    # ----- choose Sigma (per-dim scalars) -----
    if args.sigma:
        sigma_vals = parse_list(args.sigma, D_dims, default=1.0)
    else:
        sigma_str = ask(f"Enter Sigma (single or {D_dims}-CSV) [default 1.0]:", "1.0")
        sigma_vals = parse_list(sigma_str, D_dims, default=1.0)

    # build sys/spec with chosen sigma
    sysLTI, DFA, letters = build_system_and_spec(D_dims, sigma_vals)
    sysAbs, L, pol, rho, dims = build_abs_label_policy(sysLTI, DFA, letters, nx=args.nx, nu=args.nu)

    # lists (keep your original order & names)
    nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
    L_list  = [L[k]       for k in sorted(sysAbs.keys())]
    dims_sorted = sorted(sysAbs.keys())

    # ----- choose Ts sweep -----
    if args.Ts:
        Ts_list = parse_Ts(args.Ts, default="1:100:1")
    else:
        Ts_str = ask("Enter T sweep (e.g. '1:100:1' or '1,5,10') [default 1:100:1]:", "1:100:1")
        Ts_list = parse_Ts(Ts_str, default="1:100:1")

    # ----- choose delta -----
    if args.delta:
        delta_sys = parse_list(args.delta, D_dims, default=0.001)
    else:
        s = ask(f"Enter delta (single value or {D_dims}-CSV) [default 0.001]:", "0.001")
        delta_sys = parse_list(s, D_dims, default=0.001)

    # ----- whether to apply maxpolicy each iteration -----
    use_max = args.apply_maxpolicy or ask_yes_no("Apply policy improvement (maxpolicy) each iteration?", True)

    print(f"\nConfig → dim={D_dims}, nx={args.nx}, nu={args.nu}, sigma={sigma_vals}, delta={delta_sys}, "
          f"apply_maxpolicy={use_max}, Ts={Ts_list}, prune_tol={args.prune_tol}")

    # delta vectors for trees
    delta_zero_vecs = [np.zeros(sysAbs[k].N, dtype=float) for k in dims_sorted]
    delta_vecs      = [np.full(sysAbs[k].N, delta_sys[i], dtype=float) for i, k in enumerate(dims_sorted)]

    # trees
    G0 = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_zero_vecs)  # normal
    Gd = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_vecs)       # robust
    G0.initiate(); Gd.initiate()

    diff_robust_vs_normal = []  # || tv_delta - tv_0 ||_inf
    diff_apos_vs_normal   = []  # || tv_before - tv_after ||_inf

    # iterate over Ts_list; we advance one step per element
    for T_run in Ts_list:
        # step both trees once (consistent control choice)
        run_one_iteration(G0, rho, use_max, pol, args.prune_tol)
        run_one_iteration(Gd, rho, use_max, pol, args.prune_tol)

        # tvs (guard against huge allocations if your compute_tv_from_tree has a guard)
        tv0 = _compute_tv_safe(G0, DFA, L)
        tvd = _compute_tv_safe(Gd, DFA, L)

        # Robustness-aware vs Normal
        diff_robust_vs_normal.append(float(np.max(np.abs(tvd - tv0))))

        # A-posteriori vs Normal on G0
        tv_after = apply_delta_correction_apos(
            tv0, G0, DFA, L, sysAbs, delta_sys=delta_sys, T=T_run
        )
        diff_apos_vs_normal.append(float(np.max(np.abs(tv0 - tv_after))))

        # grow for next outer iteration
        G0.grow(); Gd.grow()

    # --- annotations ---
    sig_s = _fmt_vec(sigma_vals)
    del_s = _fmt_vec(delta_sys)
    pol_s = "maxpolicy" if use_max else "uniform (fixed)"

    # --- Plot: Robustness-aware vs Normal ---
    fig, ax = plt.subplots()
    ax.plot(Ts_list, diff_robust_vs_normal, marker='o')
    ax.set_xlabel('T')
    ax.set_ylabel(r'$||\,tv_{\Delta}-tv_{0}\,||_\infty$')
    ax.set_title(' $\Delta$-correction (inf norm) of scalability RA specification\n'
                 'as a function of the specification horizon computed by robustness-aware tree')
    ax.text(0.02, 0.98,
            f"dim={D_dims}\nσ={sig_s}\nΔ={del_s}\npolicy={pol_s}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', alpha=0.1, linewidth=0))
    ax.grid(True)

    # --- Plot: A-posteriori vs Normal ---
    fig, ax = plt.subplots()
    ax.plot(Ts_list, diff_apos_vs_normal, marker='o')
    ax.set_xlabel('T')
    ax.set_ylabel(r'$||\,tv_{\mathrm{before}}-tv_{\mathrm{after}}\,||_\infty$')
    ax.set_title(' $\Delta$-correction (inf norm) of scalability RA specification\n'
                 'as a function of the specification horizon computed by aposteriori correction')
    ax.text(0.02, 0.98,
            f"dim={D_dims}\nσ={sig_s}\nΔ={del_s}\npolicy={pol_s}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', alpha=0.1, linewidth=0))
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
