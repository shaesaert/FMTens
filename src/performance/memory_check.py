# memory_check.py
import sys
import gc
import types
import numpy as np

# --- helpers to identify "code-ish" objects we don't want to descend into ---
_CODE_TYPES = (
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    type,  # classes
)

def _safe_iter_slots(o):
    """
    Yield (slot_name, slot_value) pairs for objects with __slots__.
    Handles the cases:
      - __slots__ missing
      - __slots__ = None
      - __slots__ = "string"
      - __slots__ = single str
      - __slots__ = tuple/list
    Silently skips slots we can't getattr.
    """
    if not hasattr(o, "__slots__"):
        return
    slots = o.__slots__
    if slots is None:
        return

    # if someone wrote __slots__ = "abc", treat that as ['abc']? No.
    # We only iterate if it's a real iterable of slot names, not just str/bytes.
    if isinstance(slots, (str, bytes)):
        slots_iter = [slots]
    else:
        try:
            slots_iter = list(slots)
        except TypeError:
            # not actually iterable
            return

    for slot in slots_iter:
        # slot should be attribute name strings
        if not isinstance(slot, str):
            continue
        if hasattr(o, slot):
            try:
                yield slot, getattr(o, slot)
            except Exception:
                # e.g. property that raises on access
                continue


def _sizeof_obj(o, seen_ids):
    """
    Recursively estimate size of object `o` in bytes.

    Rules:
    - numpy arrays: use nbytes
    - built-in containers: recurse into them
    - objects with __dict__ / __slots__: recurse into their contents
    - skip modules / functions / classes etc. (we don't consider that "workspace data")
    - protect against cycles via seen_ids
    """
    oid = id(o)
    if oid in seen_ids:
        return 0
    seen_ids.add(oid)

    # Skip code-ish objects. We don't count imported modules, functions, classes,
    # etc. in the "data footprint" for the paper.
    if isinstance(o, _CODE_TYPES):
        return 0

    # numpy arrays (and array-likes with nbytes)
    if hasattr(o, "nbytes") and isinstance(getattr(o, "nbytes"), (int, np.integer)):
        return int(o.nbytes)

    # base shallow size
    try:
        size = sys.getsizeof(o)
    except Exception:
        size = 0

    # dict: dive keys + values
    if isinstance(o, dict):
        for k, v in o.items():
            size += _sizeof_obj(k, seen_ids)
            size += _sizeof_obj(v, seen_ids)

    # list / tuple / set / frozenset
    elif isinstance(o, (list, tuple, set, frozenset)):
        for item in o:
            size += _sizeof_obj(item, seen_ids)

    else:
        # objects with attributes in __dict__
        if hasattr(o, "__dict__"):
            try:
                size += _sizeof_obj(o.__dict__, seen_ids)
            except Exception:
                pass

        # objects that use __slots__
        for slot_name, slot_val in _safe_iter_slots(o):
            size += _sizeof_obj(slot_val, seen_ids)

    return size


def measure_workspace_bytes(namespace_dict):
    """
    Rough analog of MATLAB `whos` + sum(bytes).

    Call this with `globals()` (or `locals()`) in *your experiment script*,
    AFTER you've built tv, G, etc.

    We skip dunder names like __builtins__ to avoid counting Python internals,
    and we skip code objects (modules, functions, classes).
    """
    seen_ids = set()
    total = 0

    for name, obj in namespace_dict.items():
        # skip obvious interpreter globals
        if name.startswith("__"):
            continue
        # optional: skip the imported module objects at top level
        if isinstance(obj, _CODE_TYPES):
            continue
        total += _sizeof_obj(obj, seen_ids)

    return total


def print_workspace_memory(namespace_dict, label="workspace"):
    # run GC so we don't count garbage waiting to die (closer to steady-state usage)
    gc.collect()
    total_bytes = measure_workspace_bytes(namespace_dict)
    total_mb = total_bytes / 1e6  # MB in 10^6 bytes, to match your MATLAB style
    print(f"[Memory:{label}] ~{total_mb:.3f} MB ({total_bytes} bytes)")
