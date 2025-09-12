import numpy as np
import itertools
from typing import Union, List, Tuple, Optional, Any


def grid_input_space(lu: int, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """
    GRIDINPUTSPACE grids the input space

    uhat = grid_input_space(lu, Uspace) grids the input space described by a
    rectangular polyhedron Uspace by selecting a finite number of points (lu^dim) inside USpace.

    uhat = grid_input_space(lu, Uspace, interface=0) is equivalent to the above

    uhat = grid_input_space(lu, lb, ub) grids the input space described by a
    rectangular polyhedron with lowerbound lb and upperbound ub
    by selecting a finite number of points inside USpace.

    uhat, InputSpace = grid_input_space(lu, Uspace, interface=1, Act, Fb) considers interface
    function u = uhat +  K(x-xhat), where Act ∈ [0,1] specifies the part of
    the input space Uspace used for the actuation and
    Fb ∈ [0,1] specifies the part of the input space Uspace used for
    feedback. That is, uhat ∈ Act*Uspace, K(x-xhat) ∈ Fb*Uspace.
    The abstract input space is constructed based on the smaller input space
    based on the part for Actuation. In this case, InputSpace is supplied as
    an output of the function

    Parameters
    ----------
    lu : int
        Number of abstract inputs in each direction
    InputSpace : object or list
        Input space, either as a polyhedron or by specifying the
        lower- and upperbound of the rectangular polyhedron.

    Returns
    -------
    uhat : np.ndarray
        Finite number of inputs
    InputSpace : object (optional)
        Continuous input space. If the interface is set to option 1
        u=uhat+K(x-xhat), then InputSpace[0] is the original space,
        InputSpace[1] is the part used for actuation and InputSpace[2] is the
        part used for feedback.

    Keyword Arguments
    -----------------
    interface : int, optional (default=0)
        Set the interface function: 0: u=uhat, 1: u=uhat+K(x-xhat)
    order : bool, optional (default=False)
        Order the abstract input space uhat by increasing absolute value
    act_part : float, optional (default=0.75)
        Part of input space used for actuation when interface=1
    fb_part : float, optional (default=0.25)
        Part of input space used for feedback when interface=1

    Example
    -------
    Simple 2D input space:
    >>> import numpy as np
    >>> # Simulate a simple rectangular polyhedron
    >>> lb = np.array([-1, -1])
    >>> ub = np.array([1, 1])
    >>> lu = 3  # number of abstract inputs in each direction
    >>> uhat = grid_input_space(lu, lb, ub)
    >>> # output equals: uhat = [[-1  0  1 -1  0  1 -1  0  1],
    >>>                         [-1 -1 -1  0  0  0  1  1  1]]

    Copyright 2022 Birgit van Huijgevoort b.c.v.huijgevoort@tue.nl
    Translated to Python 2024
    """

    print('<---- Start finite-state abstraction')

    # Parse keyword arguments
    interface = kwargs.get('interface', 0)
    order = kwargs.get('order', False)
    act_part = kwargs.get('act_part', 0.75)
    fb_part = kwargs.get('fb_part', 0.25)

    # If interface is specified but act_part/fb_part are not provided through args
    if interface > 0 and len(args) > 2:
        act_part = args[2] if len(args) > 2 else 0.75
        fb_part = args[3] if len(args) > 3 else (1 - act_part)
    elif interface > 0:
        fb_part = 1 - act_part

    # Determine input type and extract bounds
    if len(args) >= 1:
        first_arg = args[0]

        # Check if input is a list/cell of polyhedra
        if isinstance(first_arg, list) and len(first_arg) > 1:
            InputSpace = first_arg
            InputSpace_working = InputSpace[1]  # select part for actuation
            poly = True
        # Check if input is a single polyhedron-like object with vertices
        elif hasattr(first_arg, 'vertices') or hasattr(first_arg, 'V'):
            InputSpace = first_arg
            InputSpace_working = InputSpace
            poly = True
        # Check if we have lower and upper bounds as separate arguments
        elif len(args) >= 2 and isinstance(first_arg, (list, np.ndarray)):
            poly = False
            lb = np.array(args[0])
            ub = np.array(args[1])
        else:
            raise ValueError("Invalid input format. Expected polyhedron object or lb, ub arrays")
    else:
        raise ValueError("Insufficient arguments provided")

    # Adjust InputSpace based on interface function
    if interface > 0 and poly:
        InputSpace_temp = InputSpace
        InputSpace = [None, None, None]
        InputSpace[0] = InputSpace_temp
        InputSpace = divide_input_space(InputSpace, act_part, fb_part)

    # Compute lower-bound and upperbound if working with polyhedron
    if poly:
        if isinstance(InputSpace, list):
            InputSpace_act = InputSpace[1]
        else:
            InputSpace_act = InputSpace

        # Extract vertices - assuming the object has a vertices attribute or V
        if hasattr(InputSpace_act, 'vertices'):
            vertices = InputSpace_act.vertices
        elif hasattr(InputSpace_act, 'V'):
            vertices = InputSpace_act.V
        else:
            raise AttributeError("Polyhedron object must have 'vertices' or 'V' attribute")

        vertices = np.array(vertices)
        dim = vertices.shape[1]
        lb = np.min(vertices, axis=0)
        ub = np.max(vertices, axis=0)
    else:
        # Lower and upper bounds are given as input
        lb = np.array(lb)
        ub = np.array(ub)
        dim = len(lb)

        if len(lb) != len(ub):
            raise ValueError('The lower- and upperbound of the input are not the same size')

    # Construct abstract input space
    grids = []
    for i in range(dim):
        grids.append(np.linspace(lb[i], ub[i], lu))

    # Create all combinations (equivalent to MATLAB's combvec)
    mesh_grids = np.meshgrid(*grids, indexing='ij')
    uhat = np.column_stack([grid.ravel() for grid in mesh_grids]).T

    if order:
        # Sort inputs based on absolute value
        # Calculate the norm for each column (point)
        norms = np.linalg.norm(uhat, axis=0)
        sorted_indices = np.argsort(norms)
        uhat = uhat[:, sorted_indices]

    # Return based on interface setting
    if interface > 0:
        return uhat, InputSpace
    else:
        return uhat


def divide_input_space(InputSpace: List, act_part: float, fb_part: float) -> List:
    """
    Placeholder for DivideInputSpace function.
    This function would need to be implemented based on the original MATLAB version.

    Parameters
    ----------
    InputSpace : List
        List containing input spaces
    act_part : float
        Actuation part factor
    fb_part : float
        Feedback part factor

    Returns
    -------
    List
        Divided input spaces
    """
    # This is a placeholder implementation
    # The actual implementation would depend on the DivideInputSpace function
    print("Warning: divide_input_space is a placeholder function")
    return InputSpace


# Example usage and test
if __name__ == "__main__":
    # Simple 2D input space example
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    lu = 3

    uhat = grid_input_space(lu, lb, ub)
    print("Grid points:")
    print(uhat)

    # Test with ordering
    uhat_ordered = grid_input_space(lu, lb, ub, order=True)
    print("\nOrdered grid points:")
    print(uhat_ordered)