# configBM/Config_PD_8D.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional
import platform
import numpy as np
import polytope as pc


def apply_mpl_style(prefer_backend: str = "QtAgg") -> None:
    """
    Choose backend BEFORE importing pyplot; apply TeX styling.
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
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    })


@dataclass
class AbstractionConfig:
    nx: int = 1000
    nu: int = 10
    placement: str = "centers"
    u_placement: str = "endpoints"
    tol: float = 1e-19
    contract_sum = None
    compute_P: str = "1d"


@dataclass
class TreeRunConfig:
    T: int = 20
    # spec_horizon: int = 15
    prune_tol: float = 5e-3  # only used if you later add pruning
    eps_val: float = 0.1

    VI_mode: str = "rt"
    pol_mode: str = "rt"
    delta_VI_scalars: float = 0.002
    delta_pol_scalars: float = 0.002

    # 2D diagonal slice region in center coordinates
    x_region: Tuple[float, float] = (-20.0, 5.0)
    y_region: Tuple[float, float] = (-20.0, 5.0)
    dims_x: Tuple[int, ...] = (0, 2, 4, 6)
    dims_y: Tuple[int, ...] = (1, 3, 5, 7)

    # visualization toggles
    visualize_labeling: bool = True
    plot_heatmap: bool = False
    save_png: Optional[str] = None
    prefer_backend: str = "QtAgg"


@dataclass
class OverlayBoxCfg:
    xy: Tuple[float, float]
    w: float
    h: float
    label: str
    color: str
    ls: str = "--"
    lw: float = 2.0
    text_xy: Optional[Tuple[float, float]] = None


@dataclass
class PD8DConfig:
    abs_cfg: AbstractionConfig = AbstractionConfig()
    run_cfg: TreeRunConfig = TreeRunConfig()

    # system polytopes (shared)
    s: np.ndarray = np.array([[1.0], [-1.0]])
    bx: Tuple[float, float] = (5.0, 20.0)
    bu: Tuple[float, float] = (5.0, 5.0)

    # region b-vectors (intervals)
    bp1: Tuple[float, float] = (-15.0, 20.0)  # p1: [-20,-15]
    bp2: Tuple[float, float] = (5.0, 0.0)     # p2: [0,5]
    bp3: Tuple[float, float] = (0.0, 5.0)     # p3: [-5,0]
    bp4: Tuple[float, float] = (-8.0, 20.0)   # p4: [-20,-8]
    bp5: Tuple[float, float] = (5.0, 8.0)     # p5: [-8,5]
    bp6: Tuple[float, float] = (-5.0, 10.0)   # p6: [-10,-5]

    # LTI parameters (same for all 8 dims)
    A: float = 0.9
    B: float = 0.5
    Bw: float = 0.5

    # point sets (8D: x0,y0,x1,y1,x2,y2,x3,y3)
    point_sets: Tuple[Tuple[float, ...], ...] = (
        (-17.8375,  3.2875, -16.8625,  3.2875,  1.4125,  3.2875,  3.3625,  3.2875),
        ( -9.4625, -3.8625,  -6.3375, -3.8625, -9.4625, -1.5125, -6.3375, -1.5125),
        (-18.0875, -7.2875, -16.8625, -7.2875, -13.5375, -7.2875, -11.8625, -7.2875),
        (  1.4125,  1.2875,   3.3625,  1.2875, -17.8375,  1.2875, -16.8625,  1.2875),
    )

    # overlays
    overlay_boxes: Tuple[OverlayBoxCfg, ...] = (
        OverlayBoxCfg(xy=(-20, 0),   w=25, h=5,  label="Package pickup", color="black",  ls="--", text_xy=(-19.5, 4.5)),
        OverlayBoxCfg(xy=(-20, -20), w=12, h=5,  label="Deliver-A and B",    color="purple", ls="--", text_xy=(-19.5, -15.5)),
        OverlayBoxCfg(xy=(-8, -20),  w=13, h=5,  label="Deliver-C and D",    color="orange", ls="--", text_xy=(-7.7, -15.5)),
        OverlayBoxCfg(xy=(-10, -5),  w=5,  h=5,  label="Package loss",   color="red",    ls="--", text_xy=(-9.8, -0.2)),
        OverlayBoxCfg(xy=(-20, -15), w=25, h=10, label="Safe1",          color="green",  ls=":",  text_xy=(-19.5, -5.5)),
        OverlayBoxCfg(xy=(-20, -5),  w=10, h=5,  label="Safe2",          color="blue",   ls=":",  text_xy=(-19.5, -0.2)),
        OverlayBoxCfg(xy=(-5, -5),   w=10, h=5,  label="Safe2",          color="blue",   ls=":",  text_xy=(-4.8, -0.2)),
    )


def build_sysLTI_pd8d(cfg: PD8DConfig):
    """
    Build 8 agents (0..7), each 1D. Attach AP + regions as in your original script.
    """
    from src.models.linmodel import LinModel

    s = np.array(cfg.s, dtype=float)
    bx = np.array(cfg.bx, dtype=float)
    bu = np.array(cfg.bu, dtype=float)

    def P(b):
        return pc.Polytope(s, np.array(b, dtype=float))

    P1 = P(cfg.bp1)  # p1
    P2 = P(cfg.bp2)  # p2
    P3 = P(cfg.bp3)  # p3
    P4 = P(cfg.bp4)  # p4
    P5 = P(cfg.bp5)  # p5
    P6 = P(cfg.bp6)  # p6

    A = np.array([[cfg.A]], dtype=float)
    B = np.array([[cfg.B]], dtype=float)
    C = np.array([[1.0]], dtype=float)
    Dm = np.array([[0.0]], dtype=float)
    Bw = np.array([[cfg.Bw]], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma = np.eye(1)

    sysLTI = {i: LinModel(A, B, C, Dm, Bw, mu=mu, sigma=sigma) for i in range(8)}
    for i in range(8):
        sysLTI[i].X = pc.Polytope(s, bx)
        sysLTI[i].U = pc.Polytope(s, bu)

    # regions per agent (exact mapping)
    sysLTI[0].regions = [P4, P6];     sysLTI[0].AP = ["p4", "p6"]
    sysLTI[1].regions = [P1, P2, P3]; sysLTI[1].AP = ["p1", "p2", "p3"]

    sysLTI[2].regions = [P4, P6];     sysLTI[2].AP = ["p4", "p6"]
    sysLTI[3].regions = [P1, P2, P3]; sysLTI[3].AP = ["p1", "p2", "p3"]

    sysLTI[4].regions = [P5, P6];     sysLTI[4].AP = ["p5", "p6"]
    sysLTI[5].regions = [P1, P2, P3]; sysLTI[5].AP = ["p1", "p2", "p3"]

    sysLTI[6].regions = [P5, P6];     sysLTI[6].AP = ["p5", "p6"]
    sysLTI[7].regions = [P1, P2, P3]; sysLTI[7].AP = ["p1", "p2", "p3"]

    return sysLTI


def build_manual_dfa_pd8d():
    """
    Your manual DFA builder: returns (DFA, letters) after dfa_manipulation.
    """
    import networkx as nx
    from src.specifications.translate import translate
    from src.specifications.utils.dfa_tool import dfa_manipulation

    DFA = translate("p1")  # base object

    DFA.AP = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    DFA.S  = [0, 1, 2]
    DFA.F  = 0
    DFA.S0 = [1]
    DFA.nr_states = len(DFA.S)
    DFA.automaton_state = DFA.S0[0]
    DFA.delta = [{}]

    DFA.transitions = [
        (0, 0, {'condition': '1',               'label': '1'}),
        (1, 1, {'condition': '!p2',             'label': '!p2'}),
        (1, 2, {'condition': 'p2',              'label': 'p2'}),
        (2, 1, {'condition': 'p3&p6',           'label': 'p3&p6'}),
        (2, 2, {'condition': 'p2',              'label': 'p2'}),
        (2, 2, {'condition': '!p3&!p1&!p2',     'label': '!p3&!p1&!p2'}),
        (2, 2, {'condition': 'p3&!p6',          'label': 'p3&!p6'}),
        (2, 0, {'condition': 'p1&p4&p5',        'label': 'p1&p4&p5'}),
    ]

    DFA.graph = nx.MultiDiGraph()
    DFA.graph.add_nodes_from(DFA.S)
    for u, v, data in DFA.transitions:
        DFA.graph.add_edge(u, v, **data)

    desired_order = ['1', '!p1&!p2', 'p2', 'p3&p6', '!p3&!p1&!p2', 'p3&!p6', 'p1&p4&p5', '!p2']
    replacements = {
        'true': '1',
        '!p1 & !p2': '!p1&!p2',
        'p2': 'p2',
        'p3 & p6': 'p3&p6',
        '!p3 & !p1 & !p2': '!p3&!p1&!p2',
        'p3 & !p6': 'p3&!p6',
        'p1 & p4 & p5': 'p1&p4&p5',
        '!p2': '!p2',
    }

    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0,
        ensure_transitions=True,
        remove_qf_self_loop=True,
        replacements=replacements,
        desired_order=desired_order,
        verbose=True,
    )
    return DFA, letters


def get_pd8d_config() -> PD8DConfig:
    return PD8DConfig()