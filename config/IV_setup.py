# config/safe_rank1_6d_setup.py
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import polytope as pc

from src.models.linmodel import LinModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation


@dataclass
class Safe6DConfig:
    dim: int = 20
    t_spec: int = 7

    # system bounds
    bx: np.ndarray = field(default_factory=lambda: np.array([10, 10]))
    bu: np.ndarray = field(default_factory=lambda: np.array([2, 2]))

    # AP region
    bp1: np.ndarray = field(default_factory=lambda: np.array([5, 5]))

    # abstraction config passed into prepare_pipeline(...)
    abs_cfg: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(
        nx=100,
        nu=5,
        placement="centers",
        u_placement="endpoints",
        tol=1e-19,
        contract_sum=None,
        compute_P="1d",
    ))

    # run config passed into run_tree(...)
    run_cfg: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(
        prefer_backend=None,
        eps_val=0.0,

        # tree controls
        T=4,                     # usually t_spec - 1
        VI_mode="rt",
        pol_mode="rt",

        # scalar or per-dim list; let your pipeline normalize if needed
        delta_VI=0.001,
        delta_pol=0.001,
        apos_correction_delta=0.001,

        prune_tol=5e-3,
        prune_mode=None,         # keep None if pruning stays off
        tol_growth=1e-8,

        compute_tv=False,
    ))


def get_safe6d_config() -> Safe6DConfig:
    cfg = Safe6DConfig()
    cfg.run_cfg.T = cfg.t_spec - 1
    if cfg.run_cfg.apos_correction_delta is None:
        cfg.run_cfg.apos_correction_delta = cfg.run_cfg.delta_pol
    return cfg


def apply_mpl_style(prefer_backend: Optional[str] = None) -> None:
    if prefer_backend:
        import matplotlib
        matplotlib.use(prefer_backend)


def build_sysLTI_safe6d(cfg: Safe6DConfig):
    dim = cfg.dim
    s = np.array([[1], [-1]])
    P1 = pc.Polytope(s, cfg.bp1)

    sysLTI = {}

    for i in range(dim):
        A = np.array([[1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[0]])
        Bw = np.array([[1]])
        mu = np.array([[0]])
        sigma = np.array([[1]])

        sys = LinModel(A, B, C, D, Bw, mu=mu, sigma=sigma)
        sys.X = pc.Polytope(s, cfg.bx)
        sys.U = pc.Polytope(s, cfg.bu)

        sys.AP = [f"p1{i}"]
        sys.regions = [P1]
        sysLTI[i] = sys

    return sysLTI


def build_dfa_safe6d(cfg: Safe6DConfig):
    safeset = " & ".join(f"p1{i}" for i in range(cfg.dim))
    ks = range(cfg.t_spec + 1)
    terms = [f"{'X' * k}({safeset})" for k in ks]
    formula = " & ".join(terms)

    print(f"Safeset: {safeset}")
    print(f"[D={cfg.dim}] LTL Formula (t_spec={cfg.t_spec}): {formula}")

    DFA = translate(formula)
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0,
        ensure_transitions=True,
        remove_qf_self_loop=True,
        verbose=True,
    )
    return DFA, letters