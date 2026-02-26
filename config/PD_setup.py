# configBM/Config_PD_8D.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional
import platform
import numpy as np
import polytope as pc
# README: We consider 4 robots (robot A, B, C, D), each robot's state [px,py] (decoupled) positions on x- and y-axis.
#
#        State space (and output space) for each robot: [-20,5] X [-20,5]
#
#        Interested regions:
#           Pick-up:                                            [-20,5] X [0,5]
#           Package-loss:                                       [-10,-5] X [-10,-5]
#           Delivery-point A:                                   [-20,-14] X [-20,-15]
#           Delivery-point B:                                   [-14,-8] X [-20,-15]
#           Delivery-point C:                                   [-8,-2] X [-20,-15]
#           Delivery-point D:                                   [-2,5] X [-20,-15]
#           Safe zone 1 (for all robots to be together within): [-20,5] X [-5,5]
#           Safe zone 2 (for all robots to be together within): [-20,5] X [-10,-5]
#           Safe zone 3 (for all robots to be together within): [-20,5] X [-15,-10]
#
#        Package-delivery specification:
#           " All robots pick up packages together in [Pick-up],
#             and then robot A deliver to [Delivery-point A];
#             robot B deliver to [Delivery-point B];
#             robot C deliver to [Delivery-point C];
#             robot D deliver to [Delivery-point D].
#
#             When all robots are equipped with packages, if any of them enters [Package-loss],
#             then all robots lose their packages meaning that all robots have to enter [Pick-up] again to re-pick packages.
#
#             For robots to successfully pick up and deliver packages, all robots have to stay close together, meaning that
#             all robots have to be together in one of the safe zones (e.g. all in [Safe zone 1]). "



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
    T: int = 15
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
    bp2: Tuple[float, float] = (5.0, 0)       # p2: [0,5]
    bp3: Tuple[float, float] = (-14.0, 20.0)  # p3: [-20,-14]
    bp4: Tuple[float, float] = (-8.0, 14.0)   # p4: [-14,-8]
    bp5: Tuple[float, float] = (-2.0, 8.0)    # p5: [-8,-2]
    bp6: Tuple[float, float] = (5.0, 2.0)     # p6: [-2,5]

    bp7: Tuple[float, float] = (-5, 10.0)     # p7:[-10,-5]
    bp8: Tuple[float, float] = (-5, 10.0)     # p8:[-10,-5]
    bp9: Tuple[float, float] = (-5, 10.0)     # p9:[-10,-5]
    bp10: Tuple[float, float] = (-5, 10.0)    # p10:[-10,-5]

    bp11: Tuple[float, float] = (-5.0, 10.0)  # p11:[-10,-5]
    bp12: Tuple[float, float] = (-5.0, 10.0)  # p12:[-10,-5]
    bp13: Tuple[float, float] = (-5.0, 10.0)  # p13:[-10,-5]
    bp14: Tuple[float, float] = (-5.0, 10.0)  # p14:[-10,-5]

    bp15: Tuple[float, float] = (5.0, 5.0)    # p15:[-5,5]
    bp16: Tuple[float, float] = (-10.0, 15.0)   # p16:[-15,-10]



    # LTI parameters (same for all 8 dims)
    A: float = 0.9
    B: float = 0.5
    Bw: float = 0.5

    # point sets (8D: x0,y0,x1,y1,x2,y2,x3,y3)
    point_sets: Tuple[Tuple[float, ...], ...] = (
        (-17.8375,  3.2875, -16.8625,  3.2875,  1.4125,  3.2875,  3.3625,  3.2875),
        # ( -9.4625, -3.8625,  -6.3375, -3.8625, -9.4625, -1.5125, -6.3375, -1.5125),
        (-18.0875, -7.2875, -16.8625, -7.2875, -13.5375, -7.2875, -11.8625, -7.2875),
        (  1.4125,  1.2875,   3.3625,  1.2875, -17.8375,  1.2875, -16.8625,  1.2875),
        (-17.8375, -1.2875, -16.8625, -1.2875, 1.4125, -1.2875, 3.3625, -1.2875),

    )

    overlay_boxes: Tuple[OverlayBoxCfg, ...] = (
        # Pick-up: [-20, 5] x [0, 5]
        OverlayBoxCfg(
            xy=(-20, 0), w=25, h=5,
            label="Pick-up", color="black", ls="--",
            text_xy=(-19.5, 4.5)
        ),

        # Deliver zones: x2 in [-20, -15]
        OverlayBoxCfg(
            xy=(-20, -20), w=6, h=5,
            label="Deliver-A", color="purple", ls="--",
            text_xy=(-19.8, -15.5)
        ),
        OverlayBoxCfg(
            xy=(-14, -20), w=6, h=5,
            label="Deliver-B", color="orange", ls="--",
            text_xy=(-13.8, -15.5)
        ),
        OverlayBoxCfg(
            xy=(-8, -20), w=6, h=5,
            label="Deliver-C", color="brown", ls="--",
            text_xy=(-7.8, -15.5)
        ),
        OverlayBoxCfg(
            xy=(-2, -20), w=7, h=5,
            label="Deliver-D", color="teal", ls="--",
            text_xy=(-1.8, -15.5)
        ),

        # Package loss: [-10, -5] x [-10, -5]
        OverlayBoxCfg(
            xy=(-10, -10), w=5, h=5,
            label="Package loss", color="red", ls="--",
            text_xy=(-9.8, -5.2)
        ),

        # Safe1: [-20, 5] x [-5, 5]
        OverlayBoxCfg(
            xy=(-20, -5), w=25, h=10,
            label="Safe1", color="green", ls=":",
            text_xy=(-19.5, 4.5)
        ),

        # Safe2: [-20, -10] x [-10, -5]
        OverlayBoxCfg(
            xy=(-20, -10), w=10, h=5,
            label="Safe2", color="blue", ls=":",
            text_xy=(-19.5, -5.2)
        ),
        # Safe2: [-5, 5] x [-10, -5]
        OverlayBoxCfg(
            xy=(-5, -10), w=10, h=5,
            label="Safe2", color="blue", ls=":",
            text_xy=(-4.8, -5.2)
        ),

        # Safe3: [-20, 5] x [-15, -10]
        OverlayBoxCfg(
            xy=(-20, -15), w=25, h=5,
            label="Safe3", color="cyan", ls=":",
            text_xy=(-19.5, -10.2)
        ),
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
    P7 = P(cfg.bp7)  # p7
    P8 = P(cfg.bp8)  # p8
    P9 = P(cfg.bp9)  # p9
    P10 = P(cfg.bp10)# p10
    P11 = P(cfg.bp11)# p11
    P12 = P(cfg.bp12)# p12
    P13 = P(cfg.bp13)# p13
    P14 = P(cfg.bp14)# p14
    P15 = P(cfg.bp15)
    P16 = P(cfg.bp16)





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
    sysLTI[0].regions = [P3, P7];                    sysLTI[0].AP = ["p3", "p7"]
    sysLTI[1].regions = [P1, P2, P11, P15, P16];     sysLTI[1].AP = ["p1", "p2", "p11", "p15", "p16"]

    sysLTI[2].regions = [P4, P8];                    sysLTI[2].AP = ["p4", "p8"]
    sysLTI[3].regions = [P1, P2, P12, P15, P16];     sysLTI[3].AP = ["p1", "p2", "p12", "p15", "p16"]

    sysLTI[4].regions = [P5, P9];                    sysLTI[4].AP = ["p5", "p9"]
    sysLTI[5].regions = [P1, P2, P13, P15, P16];     sysLTI[5].AP = ["p1", "p2", "p13", "p15", "p16"]

    sysLTI[6].regions = [P6, P10];                   sysLTI[6].AP = ["p6", "p10"]
    sysLTI[7].regions = [P1, P2, P14, P15, P16];     sysLTI[7].AP = ["p1", "p2", "p14", "p15", "p16"]

    return sysLTI


def build_manual_dfa_pd8d():
    """
    Your manual DFA builder: returns (DFA, letters) after dfa_manipulation.
    """
    import networkx as nx
    from src.specifications.translate import translate
    from src.specifications.utils.dfa_tool import dfa_manipulation

    DFA = translate("p1")  # base object

    DFA.AP = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16']
    DFA.S  = [0, 1, 2]
    DFA.F  = 0
    DFA.S0 = [1]
    DFA.nr_states = len(DFA.S)
    DFA.automaton_state = DFA.S0[0]
    DFA.delta = [{}]

    DFA.transitions = [
        (0, 0, {'condition': '1', 'label': '1'}),  # F->F

        (1, 1, {'condition': '!p2', 'label': '!p2'}),  # S0->S0 robot not pick up

        (1, 2, {'condition': 'p2', 'label': 'p2'}), # S0->S1 all robots pickup

        (2, 2, {'condition': 'p15', 'label': 'p15'}),  # S1->S1 no robot loses package
        (2, 2, {'condition': 'p16', 'label': 'p16'}),  # S1->S1 no robot loses package
        (2, 2, {'condition': 'p11&p12&p13&p14&!p7&!p8&!p9&!p10', 'label': 'p11&p12&p13&p14&!p7&!p8&!p9&!p10'}),  # S1->S1 no robot loses package

        # (2, 2, {'condition': 'p7&p8&p9&p10&!p1&!p11&!p12&!p13&!p14', 'label': 'p7&p8&p9&p10&!p1&!p11&!p12&!p13&!p14'}),  # S1->S1  no robot loses package
        # (2, 2, {'condition': '!p1&!p7&!p8&!p9&!p10', 'label': '!p1&!p7&!p8&!p9&!p10'}),   # S1->S1 no robot loses package

        (2, 1, {'condition': 'p7&p11', 'label': 'p7&p11'}), # robot A loses package
        (2, 1, {'condition': 'p8&p12', 'label': 'p8&p12'}),  # robot B loses package
        (2, 1, {'condition': 'p9&p13', 'label': 'p9&p13'}),  # robot C loses package
        (2, 1, {'condition': 'p10&p14', 'label': 'p10&p14'}),  # robot D loses package

        (2, 0, {'condition': 'p1&p3&p4&p5&p6', 'label': 'p1&p3&p4&p5&p6'}), # all robots deliver to the right place

    ]

    DFA.graph = nx.MultiDiGraph()
    DFA.graph.add_nodes_from(DFA.S)
    for u, v, data in DFA.transitions:
        DFA.graph.add_edge(u, v, **data)

    desired_order = ['1', '!p2', 'p2', 'p15', 'p16', 'p11&p12&p13&p14&!p7&!p8&!p9&!p10', 'p7&p11', 'p8&p12', 'p9&p13', 'p10&p14', 'p1&p3&p4&p5&p6']
    replacements = {
        'true': '1',
        '!p2': '!p2',
        'p2': 'p2',

        # 'p7&p8&p9&p10&!p1&!p11&!p12&!p13&!p14': 'p7 & p8 & p9 & p10 & !p1 & !p11 & !p12 & !p13 & !p14',
        # '!p1&!p7&!p8&!p9&!p10': '!p1 & !p7 & !p8 & !p9 & !p10',
        'p15': 'p15',
        'p16': 'p16',
        'p11&p12&p13&p14&!p7&!p8&!p9&!p10': 'p11 & p12 & p13 & p14 & !p7 & !p8 & !p9 & !p10',

        'p7&p11': 'p7 & p11',
        'p8&p12': 'p8 & p12',
        'p9&p13': 'p9 & p13',
        'p10&p14': 'p10 & p14',

        'p1&p3&p4&p5&p6': 'p1 & p3 & p4 & p5 & p6',

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