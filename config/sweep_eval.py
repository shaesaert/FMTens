# src/iv_eval.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np

from src.dynprog.dfa_tree_r1 import DFATree


# -----------------------
# Core evaluation primitives (generic)
# -----------------------
def nodes_with_q(G: DFATree, q: int) -> List[int]:
    return list(G.Q.get(int(q), []))


def derive_qbar_for_anchor(
    DFA: Any,
    L_list: Sequence[Any],
    idx_vec: Sequence[int],
) -> Optional[List[int]]:
    """
    For each agent d:
      pick the active letter index l at abstract state idx_vec[d],
      then qbar[d] = DFA.trans[q0, l]
    Returns None if any dim has no active letter.
    """
    q0 = int(np.asarray(DFA.S0).ravel()[0])
    Trans = np.asarray(DFA.trans, dtype=int)
    qbar: List[int] = []

    for d in range(len(L_list)):
        col = np.asarray(L_list[d][:, idx_vec[d]], dtype=bool)
        if not np.any(col):
            return None
        l = int(np.where(col)[0][0])
        qbar.append(int(Trans[q0, l]))
    return qbar


def robust_value_at_anchor_mul_then_sum(G: DFATree, qbar_vec: Sequence[int], idx_vec: Sequence[int]) -> float:
    """
    Anchor value:
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


def fht_term_exact(G: DFATree, T_build: int, qbar: int, idx_x: int) -> float:
    """
    Exact triple loops:
      for h=0..T_build:
        for npred=0..h-1:
          for z in nodes_at_depth_npred with DFA label qbar:
             sum prod_d V[d][z, idx_x]
    (Uses Dl depth cache.)
    """
    if "Dl" not in G.tree.graph:
        G._recompute_levels()
    Dl = G.tree.graph["Dl"]

    interested = set(G.Q.get(int(qbar), []))
    if not interested or T_build <= 0:
        return 0.0

    D = len(G.V)

    def node_val(z: int) -> float:
        prodv = 1.0
        for d in range(D):
            v = float(G.V[d][z, idx_x])
            if v <= 0.0:
                return 0.0
            prodv *= v
        return prodv

    total = 0.0
    for h in range(0, int(T_build) + 1):
        for npred in range(0, h):
            nodes_at_npred = Dl[npred][1] if npred < len(Dl) else []
            for z in nodes_at_npred:
                if z in interested:
                    total += node_val(z)
    return float(total)


# -----------------------
# Tree building helper (generic)
# -----------------------
def build_tree_no_prune(
    *,
    DFA: Any,
    sysAbs: Dict[int, Any],
    pol: List[List[np.ndarray]],
    rho: List[np.ndarray],
    nx_list: List[int],
    L_list: List[Any],
    delta_VI_vecs: List[np.ndarray],
    delta_pol_vecs: List[np.ndarray],
    pol_mode: str,
    VI_mode: str,
    T_build: int,
) -> DFATree:
    """
    Minimal build loop: initiate -> for it=1..T_build: maxpolicy, update_tree, grow
    (No prune; matches your IV script’s behavior.)
    """
    G = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_VI_vecs,
        delta_pol=delta_pol_vecs,
        pol_mode=pol_mode,
        VI_mode=VI_mode,
    ).initiate()

    for it in range(1, int(T_build) + 1):
        if hasattr(G, "set_iter"):
            try:
                G.set_iter(it)
            except Exception:
                pass
        G.maxpolicy(rho)
        G.update_tree()
        G.grow()

    return G


# -----------------------
# Sweep evaluation (generic)
# -----------------------
@dataclass
class SweepResult:
    DIM: int
    t_specs: np.ndarray
    anchor_centers: np.ndarray
    V_rt_avg: np.ndarray
    V_apos_avg: np.ndarray
    FHT_avg: np.ndarray
    T_builds: np.ndarray


def evaluate_dim_sweep(
    *,
    DIM: int,
    t_specs: Sequence[int],
    anchor_centers: Sequence[int],
    eps_val: float,
    delta_i: float,
    make_system_and_safeset,     # callable(cfg, DIM) -> (sysLTI, safeset, s)
    make_formula,               # callable(cfg, safeset, t_spec) -> str
    cfg_obj: Any,
    abs_cfg: Any,
    build_dfa_from_formula,      # callable(formula, spec_manip_cfg) -> (DFA, letters)
    spec_manip_cfg: Any,
    prepare_pipeline_func,       # typically src.pipeline.prepare_pipeline
    save_npz_path: str | None = None,
) -> SweepResult:
    """
    Generic sweep for one DIM.
    Assumes per-agent 1D abstraction indices are valid for anchor_centers.
    APoS correction expression is left to caller; here we compute:
      - RT average anchor value
      - APoS base anchor value
      - FHT average
    Caller can combine into final APoS-corrected expression.
    """
    sysLTI, safeset, _s = make_system_and_safeset(cfg_obj, DIM)

    anchor_idx_list = [[int(k)] * DIM for k in anchor_centers]
    Delta = 1.0 - (1.0 - float(delta_i)) ** int(DIM)

    rt_av_vals: List[float] = []
    apos_av_vals: List[float] = []
    fht_av_vals: List[float] = []
    tbuild_vals: List[int] = []

    for t_spec in t_specs:
        formula = make_formula(cfg_obj, safeset, int(t_spec))
        DFA, letters = build_dfa_from_formula(formula, spec_manip_cfg)

        sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline_func(
            sysLTI=sysLTI,
            DFA=DFA,
            letters=letters,
            abs_cfg=abs_cfg,
            eps_val=eps_val,
        )

        # anchor bounds sanity
        for d in range(DIM):
            for idx_vec in anchor_idx_list:
                if idx_vec[d] >= nx_list[d]:
                    raise ValueError(f"[D={DIM}] anchor idx {idx_vec[d]} out of range for dim {d} (nx={nx_list[d]}).")

        T_build = int(getattr(cfg_obj, "T_build")(int(t_spec)) if hasattr(cfg_obj, "T_build") else int(t_spec))
        tbuild_vals.append(T_build)

        # delta vectors (RT vs APoS, matches your script)
        delta_VI_rt = [np.full(sysAbs[d].N, float(delta_i), dtype=float) for d in range(DIM)]
        delta_pol_rt = [np.full(sysAbs[d].N, float(delta_i), dtype=float) for d in range(DIM)]

        delta_VI_apos = [np.zeros(sysAbs[d].N, dtype=float) for d in range(DIM)]
        delta_pol_apos = [np.full(sysAbs[d].N, float(delta_i), dtype=float) for d in range(DIM)]

        # build trees
        G_rt = build_tree_no_prune(
            DFA=DFA, sysAbs=sysAbs, pol=pol, rho=rho, nx_list=nx_list, L_list=L_list,
            delta_VI_vecs=delta_VI_rt, delta_pol_vecs=delta_pol_rt,
            pol_mode="rt", VI_mode="rt", T_build=T_build
        )
        G_apos = build_tree_no_prune(
            DFA=DFA, sysAbs=sysAbs, pol=pol, rho=rho, nx_list=nx_list, L_list=L_list,
            delta_VI_vecs=delta_VI_apos, delta_pol_vecs=delta_pol_apos,
            pol_mode="apos", VI_mode="apos", T_build=T_build
        )

        # evaluate anchors
        vals_rt: List[float] = []
        vals_apos_corr: List[float] = []
        vals_fht: List[float] = []

        for idx_vec in anchor_idx_list:
            qbar = derive_qbar_for_anchor(DFA, L_list, idx_vec)
            if qbar is None:
                continue

            V_rt = robust_value_at_anchor_mul_then_sum(G_rt, qbar, idx_vec)
            V_apos_base = robust_value_at_anchor_mul_then_sum(G_apos, qbar, idx_vec)

            # FHT uses scalar qbar, idx_x (you used the first entry)
            TermFHT = fht_term_exact(G_apos, T_build, qbar=int(qbar[0]), idx_x=int(idx_vec[0]))

            # your corrected expression:
            expr = float(V_apos_base) - float(T_build) * Delta + Delta * float(TermFHT)
            V_apos_corr = max(0.0, expr)

            vals_rt.append(float(V_rt))
            vals_apos_corr.append(float(V_apos_corr))
            vals_fht.append(float(TermFHT))

        rt_av_vals.append(float(np.mean(vals_rt)) if vals_rt else np.nan)
        apos_av_vals.append(float(np.mean(vals_apos_corr)) if vals_apos_corr else np.nan)
        fht_av_vals.append(float(np.mean(vals_fht)) if vals_fht else np.nan)

    res = SweepResult(
        DIM=int(DIM),
        t_specs=np.asarray(list(t_specs), dtype=int),
        anchor_centers=np.asarray(list(anchor_centers), dtype=int),
        V_rt_avg=np.asarray(rt_av_vals, dtype=float),
        V_apos_avg=np.asarray(apos_av_vals, dtype=float),
        FHT_avg=np.asarray(fht_av_vals, dtype=float),
        T_builds=np.asarray(tbuild_vals, dtype=int),
    )

    if save_npz_path:
        np.savez(
            save_npz_path,
            t_specs=res.t_specs,
            anchor_centers=res.anchor_centers,
            V_rt_avg=res.V_rt_avg,
            V_apos_avg=res.V_apos_avg,
            FHT_avg=res.FHT_avg,
            T_builds=res.T_builds,
        )

    return res