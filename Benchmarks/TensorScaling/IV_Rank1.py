
from __future__ import annotations

# Optional dev reloads
DEV_RELOAD = False
if DEV_RELOAD:
    from importlib import reload
    import src.models.linmodel as LinModel_mod
    import src.models.mdpmodel as mdpmodel_mod
    import src.abstraction.utils.labeling as dim_label_mod
    import src.dynprog.dfa_tree_r1 as DFATree_mod

    reload(LinModel_mod)
    reload(mdpmodel_mod)
    reload(dim_label_mod)
    reload(DFATree_mod)

from config.IV_setup import (
    get_safe6d_config,
    apply_mpl_style,
    build_sysLTI_safe6d,
    build_dfa_safe6d,
)

from src.pipeline import prepare_pipeline, run_tree


def main():
    cfg = get_safe6d_config()
    run_cfg = cfg.run_cfg
    abs_cfg = cfg.abs_cfg

    # ---- matplotlib style / backend before pyplot ----
    apply_mpl_style(getattr(run_cfg, "prefer_backend", None))

    # ---- build system + DFA from config ----
    sysLTI = build_sysLTI_safe6d(cfg)
    DFA, letters = build_dfa_safe6d(cfg)

    # ---- abstraction + labeling + uniform policy/rho ----
    sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
        sysLTI=sysLTI,
        DFA=DFA,
        letters=letters,
        abs_cfg=abs_cfg,
        eps_val=run_cfg.eps_val,
    )

    # ---- tree / optional tv ----
    G, tv = run_tree(
        DFA=DFA,
        sysAbs=sysAbs,
        sysLTI=sysLTI,
        L=L,
        L_list=L_list,
        pol=pol,
        rho=rho,
        nx_list=nx_list,
        run_cfg=run_cfg,
        compute_tv=getattr(run_cfg, "compute_tv", False),
    )

    # ---- summary ----
    print("\nRun finished.")
    print(f"  dim           = {len(sysLTI)}")
    print(f"  |Q|           = {len(DFA.S)}")
    print(f"  horizon T     = {run_cfg.T}")
    print(f"  VI_mode       = {run_cfg.VI_mode}")
    print(f"  pol_mode      = {run_cfg.pol_mode}")
    print(f"  eps_val       = {run_cfg.eps_val}")
    print(f"  compute_tv    = {getattr(run_cfg, 'compute_tv', False)}")
    if tv is not None:
        print(f"  tv.shape      = {getattr(tv, 'shape', None)}")

    return G, tv


if __name__ == "__main__":
    main()