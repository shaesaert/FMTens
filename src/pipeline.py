# src/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import polytope as pc

from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.abstraction.utils.labeling import dim_label_eps
from src.dynprog.dfa_tree_r1 import DFATree
from src.vis.dfa_tree_viz import plot_tree_layered
from src.dynprog.utils.treebasedV import compute_tv_from_tree


AgentKey = Hashable
PerAgent = Dict[AgentKey, Any]


# -----------------------------
# Small utilities (generic)
# -----------------------------
def _sorted_agent_keys(sysLTI_or_sysAbs: Mapping[AgentKey, Any]) -> List[AgentKey]:
    # stable order for lists
    return sorted(sysLTI_or_sysAbs.keys(), key=lambda x: (str(type(x)), str(x)))


def _broadcast_to_agents(
    value: Any, agent_keys: Sequence[AgentKey], name: str = "value"
) -> Dict[AgentKey, Any]:
    """
    Broadcast an int/float/str/etc. to all agents, or accept dict keyed by agent.
    """
    if isinstance(value, dict):
        missing = [k for k in agent_keys if k not in value]
        if missing:
            raise KeyError(f"{name} missing keys: {missing}")
        return {k: value[k] for k in agent_keys}
    else:
        return {k: value for k in agent_keys}


def _as_list_per_agent(
    value: Union[float, int, Sequence[float], Mapping[AgentKey, Union[float, Sequence[float]]]],
    agent_keys: Sequence[AgentKey],
    default_dim: int,
    name: str,
) -> Dict[AgentKey, List[float]]:
    """
    For things like delta scalars:
      - if float/int: broadcast to all agents, and to all dims (default_dim)
      - if list/tuple: treat as per-dim list, broadcast to all agents
      - if dict: per agent -> float or per-dim list
    """
    if isinstance(value, (int, float)):
        per_agent = {k: [float(value)] * default_dim for k in agent_keys}
        return per_agent

    if isinstance(value, (list, tuple, np.ndarray)):
        lst = [float(x) for x in value]
        if len(lst) != default_dim:
            raise ValueError(f"{name} length {len(lst)} != default_dim {default_dim}")
        return {k: lst[:] for k in agent_keys}

    if isinstance(value, dict):
        out: Dict[AgentKey, List[float]] = {}
        for k in agent_keys:
            v = value[k]
            if isinstance(v, (int, float)):
                out[k] = [float(v)] * default_dim
            else:
                lst = [float(x) for x in v]
                if len(lst) != default_dim:
                    raise ValueError(f"{name}[{k}] length {len(lst)} != default_dim {default_dim}")
                out[k] = lst
        return out

    raise TypeError(f"Unsupported type for {name}: {type(value)}")


def _make_delta_vectors(
    per_agent_dim_scalars: Dict[AgentKey, List[float]],
    sysAbs: Mapping[AgentKey, Any],
) -> Dict[AgentKey, np.ndarray]:
    """
    Convert per-agent per-dim delta scalars into a vector of length N for each agent.
    In your original code, delta vectors were per-state (length N), not per-control.
    We keep that: for each agent k, return shape (N,) filled with a single scalar.
    If you want per-state varying deltas, you can pass an array instead.
    """
    out: Dict[AgentKey, np.ndarray] = {}
    for k in _sorted_agent_keys(sysAbs):
        N = sysAbs[k].N
        # Here we take the first scalar as the agent scalar (like your 1D case).
        # For higher-dimensional agents, you typically still use ONE abstraction grid per agent;
        # delta is applied per abstract state => one scalar is enough.
        scalar = float(per_agent_dim_scalars[k][0])
        out[k] = np.full(N, scalar, dtype=float)
    return out


def _uniform_policy_and_rho(sysAbs: Mapping[AgentKey, Any], DFA) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[int]]:
    """
    Uniform random policy per DFA state, per agent.
    pol[q][i] = (N_i, M_i)
    rho[i]    = (N_i,)
    """
    agent_keys = _sorted_agent_keys(sysAbs)
    nx_list = [sysAbs[k].N for k in agent_keys]
    nu_list = [sysAbs[k].M for k in agent_keys]
    nQ = len(DFA.S)

    pol: List[List[np.ndarray]] = []
    for _q in range(nQ):
        per_q: List[np.ndarray] = []
        for i, k in enumerate(agent_keys):
            per_q.append(np.full((nx_list[i], nu_list[i]), 1.0 / nu_list[i], dtype=float))
        pol.append(per_q)

    rho: List[np.ndarray] = []
    for i, _k in enumerate(agent_keys):
        rho.append(np.full(nx_list[i], 1.0 / nx_list[i], dtype=float))

    return pol, rho, nx_list


# -----------------------------
# Public API expected by you
# -----------------------------
def build_regions_from_cfg(s: np.ndarray, region_cfg: Any) -> Any:
    """
    Generic region builder.

    Supports:
     Generic list/dict:
         - if region_cfg is list/tuple of b-vectors: returns list[Polytope]
         - if region_cfg is dict[name -> b-vector]: returns dict[name -> Polytope]
    """


    if isinstance(region_cfg, (list, tuple)):
        return [pc.Polytope(s, np.array(b, dtype=float)) for b in region_cfg]

    if isinstance(region_cfg, dict):
        return {name: pc.Polytope(s, np.array(b, dtype=float)) for name, b in region_cfg.items()}

    raise TypeError(f"Unsupported region_cfg type: {type(region_cfg)}")


def build_system_with_ap(
    *,
    sys_cfg: Any,
    bx_by_agent: Mapping[AgentKey, np.ndarray],
    bu_by_agent: Mapping[AgentKey, np.ndarray],
    s_by_agent: Mapping[AgentKey, np.ndarray],
    regions_by_agent: Mapping[AgentKey, Sequence[pc.Polytope]],
    ap_by_agent: Mapping[AgentKey, Sequence[str]],
) -> Dict[AgentKey, LinModel]:
    """
    Generic system builder for ANY number of agents, each any dimension.

    You provide:
      - sys_cfg.models_by_agent[k] (or sys_cfg.A_by_agent/B_by_agent/etc.)
      - bx_by_agent[k], bu_by_agent[k], s_by_agent[k]
      - regions_by_agent[k] : list of polytopes for that agent
      - ap_by_agent[k]      : list of AP names aligned with regions
    Returns:
      sysLTI[k] as LinModel with .X .U .regions .AP attached.

    Expected sys_cfg format (choose ONE of these patterns):
      Pattern A (recommended):
        sys_cfg.models_by_agent = {
          k: dict(A=..., B=..., C=..., D=..., Bw=..., mu=..., sigma=...)
        }
      Pattern B:
        sys_cfg.A_by_agent, sys_cfg.B_by_agent, ... are dicts keyed by agent.
    """
    agent_keys = _sorted_agent_keys(bx_by_agent)

    # Fetch model params
    if hasattr(sys_cfg, "models_by_agent"):
        models = sys_cfg.models_by_agent
        get = lambda k, name: models[k][name]
    else:
        get = lambda k, name: getattr(sys_cfg, f"{name}_by_agent")[k]

    sysLTI: Dict[AgentKey, LinModel] = {}
    for k in agent_keys:
        sysLTI[k] = LinModel(
            get(k, "A"), get(k, "B"), get(k, "C"), get(k, "D"), get(k, "Bw"),
            mu=get(k, "mu"), sigma=get(k, "sigma")
        )

        sysLTI[k].X = pc.Polytope(s_by_agent[k], np.array(bx_by_agent[k], dtype=float))
        sysLTI[k].U = pc.Polytope(s_by_agent[k], np.array(bu_by_agent[k], dtype=float))
        sysLTI[k].regions = list(regions_by_agent.get(k, []))
        sysLTI[k].AP = list(ap_by_agent.get(k, []))

    return sysLTI


def prepare_pipeline(
    *,
    sysLTI: Mapping[AgentKey, LinModel],
    DFA: Any,
    letters: Any,
    abs_cfg: Any,
    eps_val: float,
) -> Tuple[Dict[AgentKey, Any], Dict[AgentKey, Any], List[Any], List[List[np.ndarray]], List[np.ndarray], List[int]]:
    """
    abstraction -> labeling -> uniform policy/rho

    abs_cfg.nx and abs_cfg.nu can be:
      - int  (broadcast)
      - dict keyed by agent

    Returns:
      sysAbs, L, L_list, pol, rho, nx_list
    """
    agent_keys = _sorted_agent_keys(sysLTI)

    nx_by_agent = _broadcast_to_agents(getattr(abs_cfg, "nx"), agent_keys, "abs_cfg.nx")
    nu_by_agent = _broadcast_to_agents(getattr(abs_cfg, "nu"), agent_keys, "abs_cfg.nu")

    placement = getattr(abs_cfg, "placement", "centers")
    u_placement = getattr(abs_cfg, "u_placement", "endpoints")
    tol = getattr(abs_cfg, "tol", 1e-19)
    contract_sum = getattr(abs_cfg, "contract_sum", None)
    compute_P = getattr(abs_cfg, "compute_P", None)

    sysAbs: Dict[AgentKey, Any] = {}
    for k in agent_keys:
        sysAbs[k] = MDPModel.from_system(
            sysLTI[k],
            nx=int(nx_by_agent[k]),
            nu=int(nu_by_agent[k]),
            placement=placement,
            u_placement=u_placement,
            tol=tol,
            contract_sum=contract_sum,
            compute_P=compute_P,
        )

    L = dim_label_eps(
        sysAbs,
        sysLTI,
        letters,
        eps=eps_val,
        visualize=False,
        outdir=None,
        prefix="L_eps",
    )

    L_list = [L[k] for k in agent_keys]
    pol, rho, nx_list = _uniform_policy_and_rho(sysAbs, DFA)
    return sysAbs, L, L_list, pol, rho, nx_list


def run_tree(
    *,
    DFA: Any,
    sysAbs: Mapping[AgentKey, Any],
    sysLTI: Mapping[AgentKey, LinModel],
    L: Mapping[AgentKey, Any],
    L_list: List[Any],
    pol: List[List[np.ndarray]],
    rho: List[np.ndarray],
    nx_list: List[int],
    run_cfg: Any,
    compute_tv: bool = True,   # <-- NEW
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Run DFATree loop for ANY number of agents.

    If compute_tv=False, returns (G, None) and skips compute_tv_from_tree (+ a-posteriori correction).
    """
    agent_keys = _sorted_agent_keys(sysAbs)
    n_agents = len(agent_keys)

    VI_mode = getattr(run_cfg, "VI_mode", "rt")
    pol_mode = getattr(run_cfg, "pol_mode", "rt")

    default_dim = n_agents

    delta_VI_scalars = _as_list_per_agent(
        getattr(run_cfg, "delta_VI_scalars", 0.0),
        agent_keys,
        default_dim=default_dim,
        name="delta_VI_scalars",
    )
    delta_pol_scalars = _as_list_per_agent(
        getattr(run_cfg, "delta_pol_scalars", 0.0),
        agent_keys,
        default_dim=default_dim,
        name="delta_pol_scalars",
    )
    apos_corr_scalars = _as_list_per_agent(
        getattr(run_cfg, "apos_correction_delta_scalars", getattr(run_cfg, "delta_pol_scalars", 0.0)),
        agent_keys,
        default_dim=default_dim,
        name="apos_correction_delta_scalars",
    )

    delta_VI_vecs_by_agent = _make_delta_vectors(delta_VI_scalars, sysAbs)
    delta_pol_vecs_by_agent = _make_delta_vectors(delta_pol_scalars, sysAbs)

    delta_VI_vecs = [delta_VI_vecs_by_agent[k] for k in agent_keys]
    delta_pol_vecs = [delta_pol_vecs_by_agent[k] for k in agent_keys]

    G = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_VI_vecs,
        delta_pol=delta_pol_vecs,
        pol_mode=pol_mode,
        VI_mode=VI_mode,
    )
    G.initiate()

    T = int(getattr(run_cfg, "T", 1))
    prune_tol = float(getattr(run_cfg, "prune_tol", 5e-3))
    visualize_tree = bool(getattr(run_cfg, "visualize_tree", False))

    do_progress_check = bool(getattr(run_cfg, "do_progress_check", True))
    idx_bounds = getattr(run_cfg, "idx_bounds", None)
    K_samp = int(getattr(run_cfg, "K_samp", 2000))
    tol_growth = float(getattr(run_cfg, "tol_growth", 1e-8))
    fixed_seed = int(getattr(run_cfg, "fixed_seed", 114514))

    prev_score = None
    it_final = 0
    for it in range(1, T + 1):
        it_final = it
        G.set_iter(it)
        G.maxpolicy(rho)
        G.update_tree()
        G.prune(prune_tol, "leafs")
        G.grow()

        if do_progress_check and idx_bounds is not None:
            score, gain, stop_now = G.progress_check(
                idx_bounds=idx_bounds,
                prev_score=prev_score,
                K=K_samp,
                tol_growth=tol_growth,
                seed=fixed_seed,
                it=it,
            )
            prev_score = score
            if stop_now:
                break

        if visualize_tree:
            plot_tree_layered(G)

    # ---- NEW: optionally skip tv computation ----
    if not compute_tv:
        return G, None

    max_elements_tv = int(getattr(run_cfg, "max_elements_tv", 50_000_000))
    tv, _ = compute_tv_from_tree(G, DFA, L, max_elements=max_elements_tv)

    if pol_mode == "apos":
        from src.dynprog.utils.v_apos import apply_delta_correction_apos
        delta_sys = [apos_corr_scalars[k][0] for k in agent_keys]
        tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys=delta_sys, T=it_final + 1)

    return G, tv