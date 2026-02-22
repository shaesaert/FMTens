# Config/IFAC26_IV_Config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple, Union, Any
import platform
import numpy as np


AgentKey = Hashable


def apply_mpl_style(prefer_backend: str = "QtAgg") -> None:
    """
    Choose a GUI backend BEFORE importing pyplot, then apply TeX + bold styling.
    """
    import matplotlib as mpl
    try:
        mpl.use(prefer_backend, force=True)
    except Exception:
        if platform.system() == "Darwin":
            mpl.use("MacOSX", force=True)
        else:
            mpl.use("TkAgg", force=True)

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })


@dataclass
class SpecManipulationConfig:
    index_base: int = 0
    ensure_transitions: bool = True
    remove_qf_self_loop: bool = True
    verbose: bool = False
    replacements: Dict[str, str] | None = None
    desired_order: List[str] | None = None


@dataclass
class AbstractionConfig:
    nx: Union[int, Dict[AgentKey, int]] = 100
    nu: Union[int, Dict[AgentKey, int]] = 10
    placement: str = "centers"
    u_placement: str = "endpoints"
    tol: float = 1e-19
    contract_sum: Any = None
    compute_P: str = "1d"


@dataclass
class IVConfig:
    # sweep
    dims: Tuple[int, ...] = (4, 20)
    t_specs: Tuple[int, ...] = tuple(range(1, 8))
    anchor_centers: Tuple[int, ...] = (40, 45, 50, 55, 59)

    # system polytope (1D interval in each agent)
    bx: Tuple[float, float] = (10.0, 10.0)   # X=[-10,10] as s=[1;-1], b=[10,10]
    bu: Tuple[float, float] = (2.0, 2.0)     # U=[-2,2]
    bp_safe: Tuple[float, float] = (5.0, 5.0)  # safe region b for each agent: [-5,5]

    # identical LTI per agent (1D)
    A: float = 0.9
    B: float = 0.5
    Bw: float = 0.5

    # robustness + deltas
    eps_val: float = 0.1
    delta_i: float = 0.0717

    # tree build horizon policy
    # (your script used T_build = t_spec)
    def T_build(self, t_spec: int) -> int:
        return int(t_spec)

    # DFA manipulation defaults
    spec_manip: SpecManipulationConfig = SpecManipulationConfig()

    # abstraction defaults
    abs_cfg: AbstractionConfig = AbstractionConfig()

    # plotting/saving
    prefer_backend: str = "QtAgg"
    save_prefix: str = "anchor_values_vs_tspec_multi_anchors"
    save_combined_npz: str = "anchor_values_vs_tspec_multi_anchors_D4_D20.npz"
    plot1_png: str = "anchor_values_vs_tspec_multi_anchors_D4_D20.png"
    plot2_png: str = "fht_vs_tbuild_D4_D20.png"


def get_ifac26_iv_config() -> IVConfig:
    return IVConfig()


# -----------------------
# Factories: system + formula
# -----------------------
def make_iv_sysLTI_and_safeset(cfg: IVConfig, DIM: int):
    """
    Build DIM agents, each 1D, each with AP=['p1{i}'] and region=[P_safe].
    Returns:
      sysLTI (dict[int, LinModel]), safeset_str, s (shared halfspace normal)
    """
    from src.models.linmodel import LinModel
    import polytope as pc

    s = np.array([[1.0], [-1.0]])
    bx = np.array(cfg.bx, dtype=float)
    bu = np.array(cfg.bu, dtype=float)
    P_safe = pc.Polytope(s, np.array(cfg.bp_safe, dtype=float))

    A = np.array([[cfg.A]], dtype=float)
    B = np.array([[cfg.B]], dtype=float)
    C = np.array([[1.0]], dtype=float)
    Dm = np.array([[0.0]], dtype=float)
    Bw = np.array([[cfg.Bw]], dtype=float)
    mu = np.array([[0.0]], dtype=float)
    sigma = np.array([[1.0]], dtype=float)

    sysLTI = {}
    for i in range(DIM):
        sys = LinModel(A, B, C, Dm, Bw, mu=mu, sigma=sigma)
        sys.X = pc.Polytope(s, bx)
        sys.U = pc.Polytope(s, bu)
        sys.AP = [f"p1{i}"]
        sys.regions = [P_safe]
        sysLTI[i] = sys

    safeset = " & ".join(f"p1{i}" for i in range(DIM))
    return sysLTI, safeset, s


def make_iv_formula(cfg: IVConfig, safeset: str, t_spec: int) -> str:
    """
    Your sparse-horizon 'finite' formula:
      ks = {0, t_spec, t_spec-1 if >=2}
      formula = ∧_{k in ks} X^k(safeset)
    """
    ks = {0, int(t_spec)}
    if t_spec >= 2:
        ks.add(int(t_spec) - 1)
    terms = [f"{'X' * k}({safeset})" for k in sorted(ks)]
    return " & ".join(terms)



def build_dfa_from_formula(formula: str, spec_manip_cfg: Any) -> Tuple[Any, Any]:
    """
    Generic DFA builder from a formula + spec manipulation config.
    Only uses translate + dfa_manipulation; does not depend on any specific case study.
    """
    from src.specifications.translate import translate
    from src.specifications.utils.dfa_tool import dfa_manipulation

    DFA = translate(formula)
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=getattr(spec_manip_cfg, "index_base", 0),
        ensure_transitions=getattr(spec_manip_cfg, "ensure_transitions", True),
        remove_qf_self_loop=getattr(spec_manip_cfg, "remove_qf_self_loop", True),
        replacements=getattr(spec_manip_cfg, "replacements", None),
        desired_order=getattr(spec_manip_cfg, "desired_order", None),
        verbose=getattr(spec_manip_cfg, "verbose", False),
    )
    return DFA, letters

