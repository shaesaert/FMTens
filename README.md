# FMTens

Tensor computations for model checking stochastic continuous-space systems.

FMTens performs correct-by-design controller synthesis for stochastic
multi-agent systems. It abstracts continuous-state linear systems into finite
Markov Decision Processes, compiles a temporal-logic specification into a
deterministic finite automaton (DFA), and runs a tensor-structured value
iteration over the DFA × abstraction product to synthesise controllers with
formal satisfaction guarantees. The tensor decomposition keeps the computation
tractable as the number of agents and the state dimension grow.

## Pipeline

Synthesis proceeds in five stages, orchestrated by `src/pipeline.py`:

1. **System construction** — per-agent continuous LTI models with state/input
   polytopes and atomic propositions. (`LinModel`, `build_system_with_ap`)
2. **Specification → DFA** — an LTL specification is translated to a DFA and
   normalised to a 0-based form. (`translate`, `dfa_manipulation`)
3. **Abstraction & labeling** — each system is gridded into a finite
   `MDPModel`, and atomic propositions are labeled over the grid with a
   robustness margin. (`MDPModel.from_system`, `dim_label`)
4. **DFA-tree value iteration** — a `DFATree` is grown backwards from the
   accepting DFA mode over the product of the per-agent abstractions, with
   iterative pruning and growth. (`DFATree`)
5. **Trace value & correction** — the satisfaction-probability tensor is
   extracted, with an optional a-posteriori correction.
   (`compute_tv_from_tree`, `apply_delta_correction_apos`)

## Algorithm modes

Both the value iteration and the policy-improvement steps support two modes,
selected independently:

- **`rt` (robust)** — subtracts a per-state robustness margin (`delta_VI` /
  `delta_pol`) and clamps at zero. Conservative; no depth weighting.
- **`apos` (a-posteriori)** — no subtraction during iteration; depth weighting
  is applied via `gamma ** l_n`, with the correction folded in afterwards.

These are set through `VI_mode` and `pol_mode` on `DFATree`, or via the run
configuration passed to `pipeline.run_tree`.

## Repository layout

```
src/
  abstraction/utils/      gridding, labeling, transition construction
    labeling.py                          atomic-proposition labeling (dim_label)
    pc_utils.py                          policy composition (Pc)
    abstraction_grid_helper.py           uniform gridding (make_uniform_grid)
    abstraction_transition_builder.py    transition-matrix construction
    abstraction_tensor_helper.py         tensor / dynamics helpers
    tensor_transition_probability_2d.py  separable 2-D transition operator
  models/                 model data structures
    linmodel.py                          continuous LTI model (LinModel)
    mdpmodel.py                          discrete MDP abstraction (MDPModel)
    utils/
      abstraction_factory.py             MDPModel.from_system backend
      abstraction_shape_helper.py        flat <-> block layout reshaping
  dynprog/                value iteration
    dfa_tree_r1.py                       DFATree: tree-based value iteration
    utils/                               trace value & a-posteriori correction
  specifications/         specification handling
    translate.py                         LTL -> DFA via Spot
    utils/dfa_tool.py                    DFA manipulation + Windows fallback
  vis/                    plotting utilities
  pipeline.py             top-level orchestration

tests/                    pytest suite
tutorials/                worked end-to-end examples (IV20D, PD8D, RA2D, RA4D, RA6D)
paper_artifacts/          scripts reproducing the paper figures
```

The `utils` split follows a deliberate boundary: `abstraction/utils` holds the
gridding and transition-computation logic, while `models/utils` holds the model
constructor and the layout-reshaping helpers.

## Installation

FMTens targets Python 3.10. The core dependencies are `numpy`, `scipy`,
`networkx`, `polytope`, and `matplotlib`. DFA construction additionally relies
on [Spot](https://spot.lre.epita.fr/), most easily installed via conda-forge:

```bash
conda create -n fmtens_env python=3.10
conda activate fmtens_env
conda install -c conda-forge spot
pip install numpy scipy networkx polytope matplotlib
```

> **Windows note.** Spot may not be able to install on Windows. On Windows, comment out
> the `translate(...)` call in the tutorials and use the portable `SimpleDFA`
> fallback in `src/specifications/utils/dfa_tool.py` instead.

To run the test suite, additionally install `pytest`:

```bash
pip install pytest
pytest tests/
```

## Running

Run any tutorial as a module from the repository root:

```bash
python -m tutorials.RA2D
```

Available tutorials: `IV20D`, `PD8D`, `RA2D`, `RA4D`, `RA6D`.

## Where to start reading

`src/pipeline.py` is the spine — its `build_system_with_ap`, `prepare_pipeline`,
and `run_tree` functions walk through the five stages above. Pair it with a
tutorial (e.g. `tutorials/RA2D.py`) as a worked example.

## Citation

If you use FMTens in your research, please cite:

> R. Wang, S. Liu, Z. Sun, and S. Haesaert, "Correct-by-Design Control Synthesis
> of Stochastic Multi-agent Systems: a Robust Tensor-based Solution,"
> arXiv:2511.06873, 2026.

```bibtex
@misc{wang2026correctbydesigncontrolsynthesisstochastic,
      title={Correct-by-Design Control Synthesis of Stochastic Multi-agent Systems: a Robust Tensor-based Solution},
      author={Ruohan Wang and Siyuan Liu and Zhiyong Sun and Sofie Haesaert},
      year={2026},
      eprint={2511.06873},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2511.06873},
}
```

## License

FMTens is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

