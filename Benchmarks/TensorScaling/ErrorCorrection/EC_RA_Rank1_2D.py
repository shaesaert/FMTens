# user define Sigma, delta^i, iteration steps, maxpolicy or not.
#
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
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.abstraction.utils.pc_utils import Pc

# reuse your delta parser for sigma too
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
    Parse Ts string as either:
      - 'start:end' (inclusive), or 'start:end:step'
      - CSV list like '1,5,10'
    """
    s = (s or "").strip()
    if not s:
        s = default
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
        # inclusive end
        return list(range(start, end + 1, step))
    else:
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
    G.prune(prune_tol, 'leafs')

# ---------- build system/spec ----------
def build_system_and_spec(sigma_vals):
    """
    sigma_vals: list of per-dim scalars; each converted to 1x1 covariance (since dims are 1D here)
    """
    A = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
    B = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    C = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    Dm= {0: np.array([[0.0]]), 1: np.array([[0.0]])}
    Bw = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    mu = {0: np.array([0.0]),   1: np.array([0.0])}
    sigma = {0: float(sigma_vals[0]) * np.eye(1),
             1: float(sigma_vals[1]) * np.eye(1)}

    sysLTI = {
        0: LinModel(A[0], B[0], C[0], Dm[0], Bw[0], mu=mu[0], sigma=sigma[0]),
        1: LinModel(A[1], B[1], C[1], Dm[1], Bw[1], mu=mu[1], sigma=sigma[1]),
    }

    DFA = translate('( (!p2 | !p3 ) U p1)')
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0, ensure_transitions=True, remove_qf_self_loop=True,
        replacements={"!p1 & !p2": "p3&!p1&!p2", "!p1 & !p3": "!p3&!p1"},
        desired_order=['p1', 'p3&!p1&!p2', '!p3&!p1'],
        verbose=True
    )
    # Regions / APs
    s  = np.array([[1], [-1]])
    bx = np.array([5, 20]); bu = np.array([5, 5])
    bp1 = np.array([5, 0]); bp2 = np.array([0, 5]); bp3 = np.array([-15, 20])
    P1 = pc.Polytope(s, bp1); P2 = pc.Polytope(s, bp2); P3 = pc.Polytope(s, bp3)
    for i in [0, 1]:
        sysLTI[i].X = pc.Polytope(s, bx)
        sysLTI[i].U = pc.Polytope(s, bu)
    sysLTI[0].regions = [P1, P2]; sysLTI[1].regions = [P3]
    sysLTI[0].AP = ['p1', 'p2'];  sysLTI[1].AP = ['p3']
    return sysLTI, DFA, letters

def build_abs_label_policy(sysLTI, DFA, letters):
    sysAbs = {
        0: MDPModel.from_system(sysLTI[0], nx=1000, nu=5, placement='centers',
                                u_placement='endpoints', tol=1e-19,
                                 contract_sum=None, compute_P='1d'),
        1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5, placement='centers',
                                u_placement='endpoints', tol=1e-19,
                                 contract_sum=None, compute_P='1d'),
    }
    L = dim_label(sysAbs, sysLTI, letters, visualize=True)
    dims = sorted(sysAbs.keys())
    nx   = [sysAbs[d].N for d in dims]
    nu   = [sysAbs[d].M for d in dims]
    nQ   = len(DFA.S)
    pol = [[np.full((nx[d], nu[d]), 1.0/nu[d], dtype=float) for d in range(len(dims))]
           for _ in range(nQ)]
    rho = [np.full(nx[d], 1.0/nx[d], dtype=float) for d in range(len(dims))]
    return sysAbs, L, pol, rho, dims

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare robustness-aware & a-posteriori vs normal tree.")
    ap.add_argument("--delta", type=str, default="", help="delta: single or CSV (e.g., '0.001' or '0.001,0.001')")
    ap.add_argument("--Ts", type=str, default="", help="T sweep: 'start:end[:step]' or CSV list, e.g. '1:100:1' or '1,5,10'")
    ap.add_argument("--sigma", type=str, default="", help="Sigma (covariance scalar per dim): single or CSV (default 1.0)")
    ap.add_argument("--prune-tol", type=float, default=0.005, help="pruning tolerance for leafs")
    ap.add_argument("--apply-maxpolicy", action="store_true", help="apply policy improvement each iteration")
    ap.add_argument("--dev-reload", action="store_true")
    args, _ = ap.parse_known_args()

    if args.dev_reload:
        dev_reload()

    # ----- choose Sigma (per-dim scalars) -----
    D_dims = 2  # your example has two 1D subsystems
    if args.sigma:
        sigma_vals = parse_list(args.sigma, D_dims, default=1.0)
    else:
        sigma_str = ask(f"Enter Sigma (single or {D_dims}-CSV) [default 1.0]:", "1.0")
        sigma_vals = parse_list(sigma_str, D_dims, default=1.0)

    # build sys/spec with chosen sigma
    sysLTI, DFA, letters = build_system_and_spec(sigma_vals)
    sysAbs, L, pol, rho, dims = build_abs_label_policy(sysLTI, DFA, letters)

    # lists (keep your original order & names)
    nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
    L_list  = [L[k]       for k in sorted(sysAbs.keys())]
    dims_sorted = sorted(sysAbs.keys())
    D = len(dims_sorted)

    # ----- choose Ts sweep -----
    if args.Ts:
        Ts_list = parse_Ts(args.Ts, default="1:100:1")
    else:
        Ts_str = ask("Enter T sweep (e.g. '1:100:1' or '1,5,10') [default 1:100:1]:", "1:100:1")
        Ts_list = parse_Ts(Ts_str, default="1:100:1")

    # ----- choose delta -----
    if args.delta:
        delta_sys = parse_list(args.delta, D, default=0.001)
    else:
        s = ask(f"Enter delta (single value or {D} comma-separated) [default 0.001]:", "0.001")
        delta_sys = parse_list(s, D, default=0.001)

    # ----- whether to apply maxpolicy each iteration -----
    use_max = args.apply_maxpolicy or ask_yes_no("Apply policy improvement (maxpolicy) each iteration?", True)

    print(f"\nConfig → sigma={sigma_vals}, delta={delta_sys}, apply_maxpolicy={use_max}, Ts={Ts_list}, prune_tol={args.prune_tol}")

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
    for idx, T_run in enumerate(Ts_list, start=1):
        # step both trees once
        run_one_iteration(G0, rho, use_max, pol, args.prune_tol)
        run_one_iteration(Gd, rho, use_max, pol, args.prune_tol)

        # tvs
        tv0 = compute_tv_from_tree(G0, DFA, L)
        tvd = compute_tv_from_tree(Gd, DFA, L)

        diff_robust_vs_normal.append(float(np.max(np.abs(tvd - tv0))))

        # a-posteriori vs normal on G0, with current “horizon index” T_run
        tv_after = apply_delta_correction_apos(
            tv0, G0, DFA, L, sysAbs, delta_sys=delta_sys, T=T_run
        )
        diff_apos_vs_normal.append(float(np.max(np.abs(tv0 - tv_after))))

        # grow for next outer iteration
        G0.grow(); Gd.grow()

    # plots
    sig_s = _fmt_vec(sigma_vals)
    del_s = _fmt_vec(delta_sys)
    pol_s = "maxpolicy" if use_max else "uniform (fixed)"

    plt.figure()
    plt.plot(Ts_list, diff_robust_vs_normal, marker='o')
    ax = plt.gca()
    plt.xlabel('T'); plt.ylabel(r'$||\,tv_{\Delta}-tv_{0}\,||_\infty$')
    plt.title(' $\Delta$-correction (inf norm) of reach-avoid2D specification\n'
              'as a function of the specification horizon computed by robustness-aware tree')
    ax.text(0.02, 0.98,
            f"σ={sig_s}\nΔ={del_s}\npolicy={pol_s}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', alpha=0.1, linewidth=0.0))
    plt.grid(True)

    plt.figure()
    plt.plot(Ts_list, diff_apos_vs_normal, marker='o')
    ax = plt.gca()
    plt.xlabel('T'); plt.ylabel(r'$||\,tv_{\mathrm{before}}-tv_{\mathrm{after}}\,||_\infty$')
    plt.title(' $\Delta$-correction (inf norm) of reach-avoid2D specification\n'
              'as a function of the specification horizon computed by aposteriori correction')
    ax.text(0.02, 0.98,
            f"σ={sig_s}\nΔ={del_s}\npolicy={pol_s}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', alpha=0.1, linewidth=0.0))
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
