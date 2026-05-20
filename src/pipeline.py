"""
Pipeline orchestration for stochastic multi-agent controller synthesis.

Top-level glue connecting the algorithmic stages:

    1. System construction: build per-agent :class:`LinModel` instances
       with state/input polytopes, regions, and APs from a configuration
       object (:func:`build_system_with_ap`).
    2. Abstraction + labeling: build per-agent :class:`MDPModel`
       abstractions and compute robust per-AP labels with margin
       ``eps`` (:func:`prepare_pipeline`).
    3. DFA-constrained value iteration: run :class:`DFATree` on the
       product of per-agent abstractions and the specification DFA, with
       optional pruning, growth, and progress checks
       (:func:`run_tree`).
    4. Trace value computation: optional
       :func:`compute_tv_from_tree` extracts the trace value tensor.
    5. A-posteriori correction: optional δ-correction of the trace
       value when ``pol_mode == "apos"``.

Agent-keyed inputs use a stable sort order via
:func:`_sorted_agent_keys`. Per-agent scalars/lists can be passed as a
single value (broadcast), a per-dim list, or a dict keyed by agent.

Configuration objects (``sys_cfg``, ``abs_cfg``, ``run_cfg``) are
duck-typed: required attributes are read directly; optional ones via
``getattr(..., default)``.
"""

from __future__ import annotations

from typing import (
    Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union,
)

import numpy as np
import polytope as pc

from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.vis.dfa_tree_viz import plot_tree_layered


AgentKey = Hashable


# === private utilities ================================================

def _sorted_agent_keys(d: Mapping[AgentKey, Any]) -> List[AgentKey]:
    """Stable, deterministic ordering of agent keys (by type then repr)."""
    return sorted(d.keys(), key=lambda x: (str(type(x)), str(x)))


def _broadcast_to_agents(
    value: Any,
    agent_keys: Sequence[AgentKey],
    name: str = "value",
) -> Dict[AgentKey, Any]:
    """
    Broadcast a scalar value to every agent, or accept a dict already
    keyed by agent. Dict inputs must contain all ``agent_keys``.
    """
    if isinstance(value, dict):
        missing = [k for k in agent_keys if k not in value]
        if missing:
            raise KeyError(f"{name} missing keys: {missing}")
        return {k: value[k] for k in agent_keys}
    return {k: value for k in agent_keys}


def _as_list_per_agent(
    value: Union[
        float, int, Sequence[float],
        Mapping[AgentKey, Union[float, Sequence[float]]],
    ],
    agent_keys: Sequence[AgentKey],
    default_dim: int,
    name: str,
) -> Dict[AgentKey, List[float]]:
    """
    Normalise a delta-like configuration value into a per-agent list.

    Accepted forms:
        - scalar (int/float): broadcast to all agents, length ``default_dim``.
        - list/tuple/ndarray of length ``default_dim``: broadcast to all agents.
        - dict keyed by agent, where each value is either a scalar or a
          length-``default_dim`` sequence.
    """
    if isinstance(value, (int, float)):
        return {k: [float(value)] * default_dim for k in agent_keys}

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
                    raise ValueError(
                        f"{name}[{k}] length {len(lst)} != default_dim {default_dim}"
                    )
                out[k] = lst
        return out

    raise TypeError(f"Unsupported type for {name}: {type(value)}")


def _make_delta_vectors(
    per_agent_dim_scalars: Dict[AgentKey, List[float]],
    sysAbs: Mapping[AgentKey, Any],
) -> Dict[AgentKey, np.ndarray]:
    """
    Convert per-agent per-dim delta scalars into shape-``(N_k,)`` arrays.

    Each agent ``k`` produces an array of length ``sysAbs[k].N`` filled
    with the first scalar of ``per_agent_dim_scalars[k]``. The delta is
    applied per abstract state, so a single scalar per agent suffices.
    """
    out: Dict[AgentKey, np.ndarray] = {}
    for k in _sorted_agent_keys(sysAbs):
        N = sysAbs[k].N
        scalar = float(per_agent_dim_scalars[k][0])
        out[k] = np.full(N, scalar, dtype=float)
    return out


def _uniform_policy_and_rho(
    sysAbs: Mapping[AgentKey, Any],
    DFA: Any,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[int]]:
    """
    Uniform random policy and uniform initial state distribution.

    Returns
    -------
    pol : list of list of np.ndarray
        Indexed ``pol[q][i]``; each entry is a stochastic matrix of
        shape ``(N_i, M_i)``.
    rho : list of np.ndarray
        Per-agent uniform initial state distribution of shape ``(N_i,)``.
    nx_list : list of int
        Per-agent state counts ``N_i``.
    """
    agent_keys = _sorted_agent_keys(sysAbs)
    nx_list = [sysAbs[k].N for k in agent_keys]
    nu_list = [sysAbs[k].M for k in agent_keys]
    nQ = len(DFA.S)

    pol: List[List[np.ndarray]] = [
        [
            np.full((nx_list[i], nu_list[i]), 1.0 / nu_list[i], dtype=float)
            for i in range(len(agent_keys))
        ]
        for _q in range(nQ)
    ]

    rho: List[np.ndarray] = [
        np.full(nx_list[i], 1.0 / nx_list[i], dtype=float)
        for i in range(len(agent_keys))
    ]

    return pol, rho, nx_list


# === public API ========================================================

def build_regions_from_cfg(s: np.ndarray, region_cfg: Any) -> Any:
    """
    Build polytope regions from a configuration value.

    Parameters
    ----------
    s : np.ndarray
        Shared ``A`` matrix (state-space half-space normals).
    region_cfg : list, tuple, or dict
        - list/tuple of ``b`` vectors -> list of :class:`pc.Polytope`.
        - dict ``{name: b}`` -> dict of named :class:`pc.Polytope`.

    Returns
    -------
    list or dict
        Polytopes, in the same container shape as ``region_cfg``.
    """
    if isinstance(region_cfg, (list, tuple)):
        return [pc.Polytope(s, np.array(b, dtype=float)) for b in region_cfg]
    if isinstance(region_cfg, dict):
        return {
            name: pc.Polytope(s, np.array(b, dtype=float))
            for name, b in region_cfg.items()
        }
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
    Build per-agent :class:`LinModel` instances with regions and APs.

    Supports any number of agents and any per-agent state dimension.

    Parameters
    ----------
    sys_cfg : object
        Source of per-agent system matrices. Two patterns supported:

        Pattern A (recommended): ``sys_cfg.models_by_agent[k]`` returns
        a mapping with keys ``A, B, C, D, Bw, mu, sigma``.

        Pattern B: ``sys_cfg.{A,B,C,D,Bw,mu,sigma}_by_agent[k]``.

    bx_by_agent, bu_by_agent : mapping
        Per-agent right-hand-side ``b`` vectors for the state and input
        polytopes, respectively.
    s_by_agent : mapping
        Per-agent left-hand-side ``A`` matrices for the polytopes
        (shared between ``X`` and ``U``).
    regions_by_agent : mapping
        Per-agent list of :class:`pc.Polytope` regions for AP labels.
    ap_by_agent : mapping
        Per-agent list of atomic proposition names, aligned with
        ``regions_by_agent``.

    Returns
    -------
    dict
        ``sysLTI[k]`` -> :class:`LinModel` with attached ``X``, ``U``,
        ``regions``, ``AP`` attributes.
    """
    agent_keys = _sorted_agent_keys(bx_by_agent)

    if hasattr(sys_cfg, "models_by_agent"):
        models = sys_cfg.models_by_agent

        def _get(k, name):
            return models[k][name]
    else:
        def _get(k, name):
            return getattr(sys_cfg, f"{name}_by_agent")[k]

    sysLTI: Dict[AgentKey, LinModel] = {}
    for k in agent_keys:
        sysLTI[k] = LinModel(
            _get(k, "A"), _get(k, "B"), _get(k, "C"), _get(k, "D"), _get(k, "Bw"),
            mu=_get(k, "mu"), sigma=_get(k, "sigma"),
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
) -> Tuple[
    Dict[AgentKey, Any],
    Dict[AgentKey, Any],
    List[Any],
    List[List[np.ndarray]],
    List[np.ndarray],
    List[int],
]:
    """
    Run abstraction, labeling, and uniform policy construction.

    Parameters
    ----------
    sysLTI : mapping
        Per-agent :class:`LinModel` instances.
    DFA : object
        Specification DFA (used here only for its state count via the
        downstream uniform-policy helper).
    letters : object
        Letter set used by :func:`dim_label`.
    abs_cfg : object
        Abstraction configuration. Required attributes:

            - ``nx`` : int or dict keyed by agent — per-agent state bin counts.
            - ``nu`` : int or dict keyed by agent — per-agent input counts.

        Optional attributes (with defaults): ``placement="centers"``,
        ``u_placement="endpoints"``, ``tol=1e-19``, ``compute_P``.
    eps_val : float
        Robustness margin passed to :func:`dim_label`.

    Returns
    -------
    sysAbs : dict
        Per-agent :class:`MDPModel` abstractions.
    L : dict
        Per-agent label arrays.
    L_list : list
        ``L`` reindexed by sorted agent keys.
    pol, rho, nx_list :
        Uniform-policy outputs; see :func:`_uniform_policy_and_rho`.
    """
    agent_keys = _sorted_agent_keys(sysLTI)

    nx_by_agent = _broadcast_to_agents(abs_cfg.nx, agent_keys, "abs_cfg.nx")
    nu_by_agent = _broadcast_to_agents(abs_cfg.nu, agent_keys, "abs_cfg.nu")

    placement = getattr(abs_cfg, "placement", "centers")
    u_placement = getattr(abs_cfg, "u_placement", "endpoints")
    tol = getattr(abs_cfg, "tol", 1e-19)
    compute_P = getattr(abs_cfg, "compute_P", None)

    def _maybe_int(v):
        """Convert scalar to int; pass sequences through as int lists."""
        if isinstance(v, (list, tuple, np.ndarray)):
            return [int(x) for x in np.asarray(v).ravel()]
        return int(v)

    sysAbs: Dict[AgentKey, Any] = {}
    for k in agent_keys:
        sysAbs[k] = MDPModel.from_system(
            sysLTI[k],
            nx=_maybe_int(nx_by_agent[k]),
            nu=_maybe_int(nu_by_agent[k]),
            placement=placement,
            u_placement=u_placement,
            tol=tol,
            compute_P=compute_P,
        )

    L = dim_label(
        sysAbs, sysLTI, letters,
        eps=eps_val, visualize=False, outdir=None, prefix="L_eps",
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
    compute_tv: bool = True,
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Run the DFA-tree value iteration loop.

    Parameters
    ----------
    DFA, sysAbs, sysLTI, L, L_list, pol, rho, nx_list :
        Outputs of :func:`prepare_pipeline`.
    run_cfg : object
        Loop configuration. All attributes are optional with defaults:

            - ``VI_mode`` ("rt"), ``pol_mode`` ("rt").
            - ``delta_VI_scalars`` (0.0), ``delta_pol_scalars`` (0.0).
            - ``apos_correction_delta_scalars`` (defaults to
              ``delta_pol_scalars``).
            - ``T`` (1) — outer iteration count.
            - ``prune_tol`` (5e-3), ``visualize_tree`` (False).
            - ``do_progress_check`` (True), ``idx_bounds`` (None),
              ``K_samp`` (2000), ``tol_growth`` (1e-8),
              ``fixed_seed`` (114514).
            - ``max_elements_tv`` (5e7).
    compute_tv : bool, default True
        If False, skip :func:`compute_tv_from_tree` and any a-posteriori
        correction; return ``(G, None)``.

    Returns
    -------
    G : DFATree
        The grown DFA tree.
    tv : np.ndarray or None
        Trace value tensor, or None if ``compute_tv`` is False.
    """
    agent_keys = _sorted_agent_keys(sysAbs)
    n_agents = len(agent_keys)

    VI_mode = getattr(run_cfg, "VI_mode", "rt")
    pol_mode = getattr(run_cfg, "pol_mode", "rt")

    # === per-agent delta scalars =====================================
    delta_VI_scalars = _as_list_per_agent(
        getattr(run_cfg, "delta_VI_scalars", 0.0),
        agent_keys, default_dim=n_agents, name="delta_VI_scalars",
    )
    delta_pol_scalars = _as_list_per_agent(
        getattr(run_cfg, "delta_pol_scalars", 0.0),
        agent_keys, default_dim=n_agents, name="delta_pol_scalars",
    )
    apos_corr_scalars = _as_list_per_agent(
        getattr(run_cfg, "apos_correction_delta_scalars",
                getattr(run_cfg, "delta_pol_scalars", 0.0)),
        agent_keys, default_dim=n_agents, name="apos_correction_delta_scalars",
    )

    delta_VI_vecs_by_agent = _make_delta_vectors(delta_VI_scalars, sysAbs)
    delta_pol_vecs_by_agent = _make_delta_vectors(delta_pol_scalars, sysAbs)
    delta_VI_vecs = [delta_VI_vecs_by_agent[k] for k in agent_keys]
    delta_pol_vecs = [delta_pol_vecs_by_agent[k] for k in agent_keys]

    # === build and grow the tree =====================================
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
            score, _, stop_now = G.progress_check(
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

    if not compute_tv:
        return G, None

    # === trace value and optional a-posteriori correction ============
    max_elements_tv = int(getattr(run_cfg, "max_elements_tv", 50_000_000))
    tv, _ = compute_tv_from_tree(G, DFA, L, max_elements=max_elements_tv)

    if pol_mode == "apos":
        delta_sys = [apos_corr_scalars[k][0] for k in agent_keys]
        tv = apply_delta_correction_apos(
            tv, G, DFA, L, sysAbs, delta_sys=delta_sys, T=it_final + 1,
        )

    return G, tv