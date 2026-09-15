import numpy as np

def aggregate_tv_position(tree, Np, Nv):
    """
    Per-agent velocity-averaged satisfaction probability for a 2-agent
    system. Each agent's flat state encodes (position, velocity) as
    k = p + Np * v (position fast, velocity slow, F-order).

    Returns
    -------
    np.ndarray
        (Np, Np) array. result[p1, p2] is the uniform average over
        (v1, v2) of the joint satisfaction probability at the joint
        state (k1 = p1 + Np*v1, k2 = p2 + Np*v2). Axis 0 is agent 0's
        position; axis 1 is agent 1's position.

    Notes
    -----
    Uses the CPD structure of tv at q0 to avoid materialising the
    (Np*Nv) x (Np*Nv) joint tensor:

        tv(k1, k2) = sum_l L1[l, k1] L2[l, k2] *
                     sum_{n in Q[trans(q0, l)]} V1[n, k1] V2[n, k2]

    Averaging over (v1, v2) factorises across agents, so for each
    (letter l, candidate node n) the contribution to the (Np, Np)
    result is the outer product of two length-Np per-agent reductions.
    """
    if tree.dim != 2:
        raise ValueError(f"expected 2 agents; got tree.dim={tree.dim}")

    N = Np * Nv
    for d in range(2):
        if tree.nx[d] != N:
            raise ValueError(
                f"agent {d}: tree.nx[{d}]={tree.nx[d]} != Np*Nv={N}"
            )

    # initial DFA mode
    S0 = list(np.asarray(tree.DFA.S0).ravel())
    if len(S0) != 1:
        raise ValueError(f"expected a single initial DFA state; got S0={S0}")
    q0 = int(S0[0])

    # sink (skip letters that lead to it)
    sink = getattr(tree.DFA, "sink", None)
    sink = int(sink) if sink is not None else None

    trans = np.asarray(tree.DFA.trans, dtype=int)
    num_letters = trans.shape[1]

    result = np.zeros((Np, Np), dtype=float)

    for l in range(num_letters):
        q_next = int(trans[q0, l])
        if sink is not None and q_next == sink:
            continue
        cand_nodes = tree.Q.get(q_next, [])
        if not cand_nodes:
            continue

        # Per-agent stack of A_d[n, p] for n in cand_nodes, shape (n_cand, Np).
        # A_d[n, p] = sum_v L_d[l, p + Np*v] * V_d[n, p + Np*v].
        A_stacks = []
        for d in range(2):
            L_grid = np.asarray(tree.L[d][l, :], dtype=float).reshape(
                Np, Nv, order="F"
            )                                                # (Np, Nv)
            A_stack = np.empty((len(cand_nodes), Np), dtype=float)
            for i, n in enumerate(cand_nodes):
                V_grid = np.asarray(
                    tree.V[d][n, :], dtype=float
                ).reshape(Np, Nv, order="F")                 # (Np, Nv)
                A_stack[i, :] = (L_grid * V_grid).sum(axis=1)  # (Np,)
            A_stacks.append(A_stack)

        # sum_{n in cand} A1[n, :] (outer) A2[n, :]  ==  A1.T @ A2
        result += A_stacks[0].T @ A_stacks[1]

    result /= float(Nv) * float(Nv)
    return result

"""
tv_conditional_on_fixed_agent
=============================

Companion to ``aggregate_tv_position``. Where ``aggregate_tv_position``
averages each agent's state along an internal velocity axis to produce
a (Np, Np) heatmap over the joint position grid, this function instead
**conditions** one agent's state to a user-supplied subset of cells
(for example, agent 1's p1-labelled cells) and returns the satisfaction
probability over the **other** agent's full state grid as a 2D heatmap.

Math (2-agent CPD form):

    tv(k0, k1) = sum_l L0[l, k0] L1[l, k1] *
                 sum_{n in Q[trans(q0, l)]} V0[n, k0] V1[n, k1]

Conditioning on k1 in a mask M of size n_M, taking a uniform average:

    tv_cond(k0) = (1/n_M) * sum_{k1 in M} tv(k0, k1)
                = sum_l L0[l, k0] * sum_n V0[n, k0] * M1[l, n]

    M1[l, n] = (1/n_M) * sum_{k1 in M} L1[l, k1] * V1[n, k1]

Result is reshaped to the free agent's per-axis grid shape using F-order
(consistent with the convention `k = i_axis0 + N_axis0 * i_axis1` used
throughout this codebase).
"""


def tv_conditional_on_fixed_agent(
    tree,
    fixed_agent: int,
    fixed_mask,
    free_agent_shape,
):
    """
    Satisfaction probability over the free agent's 2D state, with the
    other agent's state marginalised uniformly over a user-supplied
    cell mask.

    Parameters
    ----------
    tree : DFATree
        Trained DFA-tree for a 2-agent system.
    fixed_agent : int (0 or 1)
        Index of the agent to marginalise.
    fixed_mask : array_like of bool or 0/1
        Length-`tree.nx[fixed_agent]` mask selecting the cells of the
        fixed agent to average over. Coerced to bool internally.
    free_agent_shape : tuple of int
        Per-axis grid shape (N_a, N_b) of the free agent's state. Must
        satisfy `N_a * N_b == tree.nx[free_agent]`. The flat index
        convention is `k = i_a + N_a * i_b` (axis 0 fast, axis 1 slow,
        F-order), matching the abstraction.

    Returns
    -------
    np.ndarray
        Shape `free_agent_shape`. Entry `[i_a, i_b]` is the uniform
        average of `tv(k_free=i_a + N_a*i_b, k_fixed)` over `k_fixed`
        in `fixed_mask`.

    Notes
    -----
    The fixed-agent average is uniform across the masked cells. To
    instead condition on a single representative point, pass a mask
    that's True at exactly one index.
    """
    if tree.dim != 2:
        raise ValueError(f"expected 2 agents; got tree.dim={tree.dim}")
    if fixed_agent not in (0, 1):
        raise ValueError(f"fixed_agent must be 0 or 1; got {fixed_agent}")

    free_agent = 1 - fixed_agent
    N_free = tree.nx[free_agent]
    N_fixed = tree.nx[fixed_agent]

    N_a, N_b = int(free_agent_shape[0]), int(free_agent_shape[1])
    if N_a * N_b != N_free:
        raise ValueError(
            f"free_agent_shape product {N_a * N_b} != "
            f"tree.nx[{free_agent}]={N_free}"
        )

    fixed_mask = np.asarray(fixed_mask).astype(bool).ravel()
    if fixed_mask.shape != (N_fixed,):
        raise ValueError(
            f"fixed_mask shape {fixed_mask.shape} != ({N_fixed},)"
        )
    n_mask = int(fixed_mask.sum())
    if n_mask == 0:
        raise ValueError("fixed_mask has no True entries")

    # Initial DFA mode
    S0 = list(np.asarray(tree.DFA.S0).ravel())
    if len(S0) != 1:
        raise ValueError(f"expected a single initial DFA state; got S0={S0}")
    q0 = int(S0[0])

    sink = getattr(tree.DFA, "sink", None)
    sink = int(sink) if sink is not None else None

    trans = np.asarray(tree.DFA.trans, dtype=int)
    num_letters = trans.shape[1]

    result = np.zeros(N_free, dtype=float)

    for l in range(num_letters):
        q_next = int(trans[q0, l])
        if sink is not None and q_next == sink:
            continue
        cand_nodes = tree.Q.get(q_next, [])
        if not cand_nodes:
            continue

        L_free_l = np.asarray(
            tree.L[free_agent][l, :], dtype=float
        )                                                # (N_free,)
        L_fixed_l_in = np.asarray(
            tree.L[fixed_agent][l, fixed_mask], dtype=float
        )                                                # (n_mask,)

        for n in cand_nodes:
            V_fixed_n_in = np.asarray(
                tree.V[fixed_agent][n, fixed_mask], dtype=float
            )                                            # (n_mask,)

            # M[l, n] = mean over k in mask of L_fixed[l, k] * V_fixed[n, k]
            M_ln = float((L_fixed_l_in * V_fixed_n_in).sum()) / n_mask
            if M_ln == 0.0:
                continue

            V_free_n = np.asarray(
                tree.V[free_agent][n, :], dtype=float
            )                                            # (N_free,)
            result += M_ln * L_free_l * V_free_n

    return result.reshape(N_a, N_b, order="F")


def tv_joint_same_state(tree, agent_shape):
    """
    Joint satisfaction probability tv(k, ..., k) — all agents starting at
    the same abstract state k — reshaped onto the per-agent 2D grid.

    Math (n-agent CPD form evaluated on the diagonal):
        tv_diag(k) = sum_l ( prod_d L_d[l, k] )
                       * sum_{n in Q[trans(q0, l)]} ( prod_d V_d[n, k] )
    """
    import numpy as np

    n_agents = tree.dim
    if n_agents < 2:
        raise ValueError(f"need at least 2 agents; got tree.dim={n_agents}")
    if not all(tree.nx[d] == tree.nx[0] for d in range(n_agents)):
        sizes = [tree.nx[d] for d in range(n_agents)]
        raise ValueError(f"all agents must share grid size; got {sizes}")
    N = tree.nx[0]

    N_a, N_b = int(agent_shape[0]), int(agent_shape[1])
    if N_a * N_b != N:
        raise ValueError(f"agent_shape product {N_a * N_b} != tree.nx[0]={N}")

    S0 = list(np.asarray(tree.DFA.S0).ravel())
    if len(S0) != 1:
        raise ValueError(f"expected a single initial DFA state; got S0={S0}")
    q0 = int(S0[0])

    sink = getattr(tree.DFA, "sink", None)
    sink = int(sink) if sink is not None else None

    trans = np.asarray(tree.DFA.trans, dtype=int)
    num_letters = trans.shape[1]

    result = np.zeros(N, dtype=float)

    for l in range(num_letters):
        q_next = int(trans[q0, l])
        if sink is not None and q_next == sink:
            continue
        cand_nodes = tree.Q.get(q_next, [])
        if not cand_nodes:
            continue

        # prod_d L_d[l, k] across all agents
        l_coinc = np.ones(N, dtype=float)
        for d in range(n_agents):
            l_coinc *= np.asarray(tree.L[d][l, :], dtype=float)
        if l_coinc.sum() == 0.0:
            continue

        v_sum = np.zeros(N, dtype=float)
        for n in cand_nodes:
            v_prod = np.ones(N, dtype=float)
            for d in range(n_agents):
                v_prod *= np.asarray(tree.V[d][n, :], dtype=float)
            v_sum += v_prod
        result += l_coinc * v_sum

    return result.reshape(N_a, N_b, order="F")

def tv_conditional_on_fixed_agents(
    tree,
    free_agent: int,
    fixed_masks: dict,
    free_agent_shape,
):
    """
    Satisfaction probability over the free agent's 2D state, with every
    other agent marginalised uniformly over a user-supplied cell mask.

    Math (CPD form for n agents):
        tv(k_0, ..., k_{n-1}) = Σ_l ( Π_d L_d[l, k_d] )
                                 * Σ_n_node ( Π_d V_d[n, k_d] )
    Uniform conditioning on masks M_d for d != free_agent factorises:
        tv_cond(k_a) = Σ_l L_a[l, k_a]
                       * Σ_n V_a[n, k_a] * Π_{d != a} M_d[l, n]
    where  M_d[l, n] = mean_{k_d in M_d} L_d[l, k_d] * V_d[n, k_d].

    Args
    ----
    tree              : DFATree (n agents)
    free_agent        : int — index of the free agent.
    fixed_masks       : dict {agent_idx: bool/0-1 array of length tree.nx[agent_idx]}
                        Must cover every agent except `free_agent`.
    free_agent_shape  : tuple — per-axis grid shape of the free agent's state.

    Returns
    -------
    np.ndarray of shape `free_agent_shape`, F-order.
    """
    if not 0 <= free_agent < tree.dim:
        raise ValueError(
            f"free_agent must be in 0..{tree.dim-1}; got {free_agent}"
        )
    fixed_agents = set(fixed_masks.keys())
    if free_agent in fixed_agents:
        raise ValueError("free_agent must not appear in fixed_masks")
    if fixed_agents | {free_agent} != set(range(tree.dim)):
        raise ValueError(
            f"fixed_masks must cover all agents except {free_agent}; "
            f"got {sorted(fixed_agents)} for tree.dim={tree.dim}"
        )

    N_free = tree.nx[free_agent]
    N_a, N_b = int(free_agent_shape[0]), int(free_agent_shape[1])
    if N_a * N_b != N_free:
        raise ValueError(
            f"free_agent_shape product {N_a * N_b} != "
            f"tree.nx[{free_agent}]={N_free}"
        )

    # Validate / normalise each mask up front.
    mask_data = {}
    for d, mask in fixed_masks.items():
        m = np.asarray(mask).astype(bool).ravel()
        if m.shape != (tree.nx[d],):
            raise ValueError(
                f"fixed_masks[{d}] shape {m.shape} != ({tree.nx[d]},)"
            )
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"fixed_masks[{d}] has no True entries")
        mask_data[d] = (m, n)

    # Initial DFA mode
    S0 = list(np.asarray(tree.DFA.S0).ravel())
    if len(S0) != 1:
        raise ValueError(f"expected a single initial DFA state; got S0={S0}")
    q0 = int(S0[0])

    sink = getattr(tree.DFA, "sink", None)
    sink = int(sink) if sink is not None else None

    trans = np.asarray(tree.DFA.trans, dtype=int)
    num_letters = trans.shape[1]

    result = np.zeros(N_free, dtype=float)

    for l in range(num_letters):
        q_next = int(trans[q0, l])
        if sink is not None and q_next == sink:
            continue
        cand_nodes = tree.Q.get(q_next, [])
        if not cand_nodes:
            continue

        L_free_l = np.asarray(tree.L[free_agent][l, :], dtype=float)
        if L_free_l.sum() == 0.0:
            continue

        # Precompute L_d[l, mask] for each fixed agent — same across nodes.
        Ld_l_in = {
            d: np.asarray(tree.L[d][l, mask], dtype=float)
            for d, (mask, _) in mask_data.items()
        }

        for n in cand_nodes:
            # Product Π_d M_d[l, n] over fixed agents.
            M_prod = 1.0
            for d, (mask, n_mask) in mask_data.items():
                Vd_n_in = np.asarray(tree.V[d][n, mask], dtype=float)
                M_d = float((Ld_l_in[d] * Vd_n_in).sum()) / n_mask
                M_prod *= M_d
                if M_prod == 0.0:
                    break
            if M_prod == 0.0:
                continue

            V_free_n = np.asarray(tree.V[free_agent][n, :], dtype=float)
            result += M_prod * L_free_l * V_free_n

    return result.reshape(N_a, N_b, order="F")