# Config/IFAC26_RA_Config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple, Union, Any
import numpy as np

AgentKey = Hashable


def apply_mpl_style(backend: str = "QtAgg") -> None:
    """
    Apply the IFAC26 RA plotting style. Call this BEFORE importing matplotlib.pyplot.
    """
    import matplotlib as mpl
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.usetex": True,
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
    })
    mpl.use(backend)


@dataclass
class SystemConfig:
    models_by_agent: Dict[AgentKey, Dict[str, np.ndarray]]
    s_by_agent: Dict[AgentKey, np.ndarray]
    bx_by_agent: Dict[AgentKey, np.ndarray]
    bu_by_agent: Dict[AgentKey, np.ndarray]
    region_b_by_agent: Dict[AgentKey, Dict[str, Union[List[float], Tuple[float, ...], np.ndarray]]]
    ap_label_by_region_by_agent: Dict[AgentKey, Dict[str, str]]


@dataclass
class AbstractionConfig:
    nx: Union[int, Dict[AgentKey, int]] = 1000
    nu: Union[int, Dict[AgentKey, int]] = 10
    placement: str = "centers"
    u_placement: str = "endpoints"
    tol: float = 1e-19
    contract_sum: Any = None
    compute_P: str = "1d"


@dataclass
class SpecConfig:
    formula: str
    replacements: dict
    desired_order: list
    verbose: bool = True


@dataclass
class RunConfig:
    eps_val: float = 0.1

    T: int = 6
    prune_tol: float = 5e-3
    visualize_tree: bool = False

    do_progress_check: bool = True
    idx_bounds: List[Tuple[int, int]] = None
    K_samp: int = 2000
    tol_growth: float = 1e-8
    fixed_seed: int = 114514

    VI_mode: str = "rt"
    pol_mode: str = "rt"

    delta_VI_scalars: Union[float, int, List[float], Tuple[float, ...], Dict[AgentKey, Union[float, List[float]]]] = (0.002, 0.002)
    delta_pol_scalars: Union[float, int, List[float], Tuple[float, ...], Dict[AgentKey, Union[float, List[float]]]] = (0.002, 0.002)
    apos_correction_delta_scalars: Union[float, int, List[float], Tuple[float, ...], Dict[AgentKey, Union[float, List[float]]]] = (0.0, 0.0)

    max_elements_tv: int = 50_000_000

    plot_tv: bool = True
    print_memory: bool = True

    # Plot backend lives here now
    mpl_backend: str = "QtAgg"


def get_ifac26_ra_configs():
    A = np.array([[0.9]])
    B = np.array([[0.5]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    Bw = np.array([[0.5]])
    mu = np.array([0.0])
    sigma = np.eye(1)

    models_by_agent = {
        0: dict(A=A, B=B, C=C, D=D, Bw=Bw, mu=mu, sigma=sigma),
        1: dict(A=A, B=B, C=C, D=D, Bw=Bw, mu=mu, sigma=sigma),
    }

    s = np.array([[1.0], [-1.0]])
    s_by_agent = {0: s, 1: s}

    bx = np.array([5.0, 20.0])
    bu = np.array([5.0, 5.0])
    bx_by_agent = {0: bx, 1: bx}
    bu_by_agent = {0: bu, 1: bu}

    region_b_by_agent = {
        0: {"P1": (5.0, 0.0), "P2": (0.0, 5.0)},
        1: {"P3": (-15.0, 20.0)},
    }
    ap_label_by_region_by_agent = {
        0: {"P1": "p1", "P2": "p2"},
        1: {"P3": "p3"},
    }

    sys_cfg = SystemConfig(
        models_by_agent=models_by_agent,
        s_by_agent=s_by_agent,
        bx_by_agent=bx_by_agent,
        bu_by_agent=bu_by_agent,
        region_b_by_agent=region_b_by_agent,
        ap_label_by_region_by_agent=ap_label_by_region_by_agent,
    )

    abs_cfg = AbstractionConfig(nx=1000, nu=10, compute_P="1d")

    spec_cfg = SpecConfig(
        formula="( (!p2 | !p3 ) U p1)",
        replacements={
            "!p1 & !p2": "p3&!p1&!p2",
            "!p1 & !p3": "!p3&!p1",
        },
        desired_order=["p1", "p3&!p1&!p2", "!p3&!p1"],
        verbose=True,
    )

    run_cfg = RunConfig(
        eps_val=0.1,
        T=6,
        prune_tol=5e-3,
        visualize_tree=False,
        do_progress_check=True,
        idx_bounds=[(0, 798), (599, 699)],
        K_samp=2000,
        tol_growth=1e-8,
        fixed_seed=114514,
        VI_mode="rt",
        pol_mode="rt",
        delta_VI_scalars=(0.002, 0.002),
        delta_pol_scalars=(0.002, 0.002),
        apos_correction_delta_scalars=(0.0, 0.0),
        plot_tv=True,
        print_memory=True,
        mpl_backend="QtAgg",
    )

    return sys_cfg, abs_cfg, run_cfg, spec_cfg