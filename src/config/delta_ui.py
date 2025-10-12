# src/config/delta_ui.py
from typing import List, Tuple
import numpy as np

def parse_delta_list(s: str, D: int, default: float = 0.001) -> List[float]:
    """
    Parse 's' as either a single float or D comma-separated floats.
    Empty string -> uses 'default' for all D dims.
    """
    s = "" if s is None else str(s)
    vals = [float(x.strip()) for x in s.split(",")] if s.strip() else [default]
    if len(vals) == 1:
        vals = vals * D
    if len(vals) != D:
        raise ValueError(f"Need {D} value(s), got {len(vals)}.")
    if any(v < 0 for v in vals):
        raise ValueError("Delta values must be non-negative.")
    return vals

def ask_mode(input_fn=input, default: str = "none") -> str:
    """
    Ask user: none / robusttree / apos. Returns one of: 'none','robusttree','apos'.
    """
    prompt = "\nUse delta? [n] none / [r] robusttree / [a] apos  [n/r/a]"
    while True:
        ans = input_fn(f"{prompt}: ").strip().lower()
        if ans in ("", "n", "none"):
            return "none"
        if ans in ("r", "robusttree"):
            return "robusttree"
        if ans in ("a", "apos"):
            return "apos"
        print("Please enter n/r/a.")

def interactive_delta(sysAbs, default: float = 0.001):
    """
    Full interactive flow. Returns:
      mode, delta_sys(list[float]), delta_list_tree(list[np.ndarray]), apos(int), dims_sorted(list[int])
    """
    dims_sorted = sorted(sysAbs.keys())
    D = len(dims_sorted)

    mode = ask_mode()
    if mode == "none":
        delta_sys = [0.0] * D
        delta_list_tree = [np.zeros(sysAbs[k].N, dtype=float) for k in dims_sorted]
        apos = 0
    else:
        # ask magnitudes
        while True:
            s = input(f"Enter delta (single value or {D} comma-separated) [default {default}]: ")
            try:
                delta_sys = parse_delta_list(s, D, default=default)
                break
            except ValueError as e:
                print(e)

        if mode == "robusttree":
            delta_list_tree = [
                np.full(sysAbs[k].N, delta_sys[i], dtype=float)
                for i, k in enumerate(dims_sorted)
            ]
            apos = 0
        else:  # 'apos'
            delta_list_tree = [np.zeros(sysAbs[k].N, dtype=float) for k in dims_sorted]
            apos = 1

    return mode, delta_sys, delta_list_tree, apos, dims_sorted
