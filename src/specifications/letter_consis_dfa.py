import numpy as np
import re

def _norm_label(s: str) -> str:
    # Normalize label: strip spaces and collapse multiple spaces into one
    return re.sub(r"\s+", " ", str(s).strip())


def letters_from_dfa_consistent(dfa):
    """
    Same letter collection logic as DFATreeR1:
    prioritize transitions if available, otherwise use graph.
    """
    from collections import OrderedDict
    seen = OrderedDict()
    if hasattr(dfa, "transitions") and dfa.transitions is not None:
        for _, _, data in dfa.transitions:
            lab = _norm_label(data.get("label", data.get("condition", "1")))
            if lab not in seen:
                seen[lab] = True
    elif hasattr(dfa, "graph"):
        for _, _, data in dfa.graph.edges(data=True):
            lab = _norm_label(data.get("label", data.get("condition", "1")))
            if lab not in seen:
                seen[lab] = True
    else:
        # Default to "1" if neither transitions nor graph exist
        seen["1"] = True
    return list(seen.keys())
