import numpy as np
from polytope import Polytope  # Or use your preferred polytope library

def overtake_spec(sys):
    """
    Function to define specification using scLTL

    Parameters
    ----------
    sys : dict or object
        Dynamical system to evaluate

    Returns
    -------
    tuple
        formula : str
            Formulated scLTL specification
        sys : dict or object
            Updated dynamical system including the
            atomic proposition and polytope regions
    """

    # Vector to extract the values from all the states
    xe_xo = np.array([1, 0, 0, 0])
    ye_yo = np.array([0, 1, 0, 0])

    y_thres = 1.8
    safe_dist = 5

    # Goal of the vehicle
    # P1: Ego vehicle in left lane
    P1 = Polytope(
        A=np.array([ye_yo, -ye_yo]),
        b=np.array([y_thres, y_thres])
    )

    # P2: Safe distance constraint
    P2 = Polytope(
        A=np.array([-xe_xo]),
        b=np.array([-safe_dist])
    )

    goal = "(p1 & p2)"

    # Keep safe x-distance (front and back)
    # P3: Rear distance constraint
    P3 = Polytope(
        A=np.array([xe_xo]),
        b=np.array([-safe_dist])
    )

    x_dist = "(!p1 | (p3 | p2))"

    # Update system with regions and atomic propositions
    if hasattr(sys, 'regions'):
        sys.regions = [P1, P2, P3]
    else:
        sys['regions'] = [P1, P2, P3]

    if hasattr(sys, 'AP'):
        sys.AP = ['p1', 'p2', 'p3']
    else:
        sys['AP'] = ['p1', 'p2', 'p3']

    # Construct the formula string
    formula_str = f"{x_dist} U ({goal} & {x_dist})"
    formula = str(formula_str)

    return formula, sys



# Example usage
if __name__ == "__main__":
    # Example with dictionary-based system
    sys_dict = {}
    formula, updated_sys = overtake_spec_simple(sys_dict)

    print("Formula:", formula)
    print("Atomic Propositions:", updated_sys['AP'])
    print("Number of regions:", len(updated_sys['regions']))

    # Example with object-based system
    class System:
        def __init__(self):
            pass

    sys_obj = System()
    formula, updated_sys = overtake_spec_simple(sys_obj)

    print("\nObject-based system:")
    print("Formula:", formula)
    print("Atomic Propositions:", updated_sys.AP)