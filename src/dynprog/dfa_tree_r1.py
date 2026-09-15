"""
DFATree — backward DFA-product tree for value iteration on multi-agent MDPs.

This module defines :class:`DFATree`, the data structure that powers
correct-by-design controller synthesis over the product of:

    - a 0-based deterministic finite automaton (DFA) representing the
      LTL specification, and
    - one or more per-agent :class:`MDPModel` abstractions.

The tree is rooted at the accepting DFA mode (node 0) and grown
backwards through DFA predecessors (one ancestor per (state, letter)
edge). Value iteration propagates from leaves to the root and
supports two algorithm modes selected independently:

    - ``VI_mode``: ``"rt"`` (robust, subtracts ``delta_VI`` and
      clamps at 0) or ``"apos"`` (no subtraction during VI;
      correction applied a-posteriori).
    - ``pol_mode``: ``"rt"`` (robust policy improvement using
      ``delta_pol``) or ``"apos"`` (a-posteriori weighting via
      ``zeta`` and ``gamma^{l_n}``).

Conventions:
    - DFA.S must be ``[0, 1, ..., |S|-1]`` (consecutive, 0-based).
    - Internal graph node ids are also 0-based, with root = 0.
    - Edge label ``l`` on an edge ``(p -> c)`` is a 0-based column
      index into the per-dimension labeling matrices ``L[d]``.
"""
from typing import List, Dict, Optional, Union

import networkx as nx
import numpy as np
from scipy.sparse import issparse
from scipy import sparse
from ..abstraction.utils.pc_utils import Pc


class DFATree:
    """
    DFA tree assuming a 0-based DFA:
      - DFA.S == [0,1,...,nq-1]
      - DFA.F (accepting) and optional DFA.sink are 0-based ints
      - DFA.trans has shape (|S|, |act|) with 0-based target states

    Internal graph node ids are 0-based:
      - root node = 0 (accepting mode)
      - children are added as 1, 2, ...
    """

    # === construction =====================================================
    def __init__(self, DFA, sysAbs, pol, nx_list, L,
                 delta_VI: Optional[Union[List[np.ndarray], tuple]] = None,
                 delta_pol: Optional[Union[List[np.ndarray], tuple]] = None,
                 pol_mode: Optional[str] = None,
                 VI_mode: Optional[str] = None,
                 iter_idx: Optional[int] = None):
        self.DFA = DFA
        self.iter_idx = int(iter_idx) if iter_idx is not None else 0 # iteration
        # Normalize sysAbs / nx / L to aligned lists (allow dicts keyed 0..D-1)
        if isinstance(sysAbs, dict):
            self.dim_keys = sorted(sysAbs.keys())
            self.sysAbs = [sysAbs[k] for k in self.dim_keys]
        else:
            self.dim_keys = list(range(len(sysAbs)))
            self.sysAbs = list(sysAbs)

        if isinstance(nx_list, dict):
            self.nx = [nx_list[k] for k in self.dim_keys]
        else:
            self.nx = list(nx_list)

        if isinstance(L, dict):
            self.L = [L[k] for k in self.dim_keys]
        else:
            self.L = list(L)

        self.dim = len(self.sysAbs)
        # pol[q][d] is (N, nu) dense or csr one-hot (or None -> uniform)
        self.pol = pol
        self.depths: Dict[int, int] = {}

        # store zeta per node for analysis, to be removed when done with analysis
        self.zeta_node: Dict[int, float] = {}
        # discount factor and per-node gamma^l_n (for analysis)
        self.gamma: float = 0.99
        self.gamma_pow_node: Dict[int, float] = {}

        # === deltas for VI and policy ===
        def _normalize_delta_like(name, inp):
            if inp is None:
                # default: zeros
                return [np.zeros(self.nx[d], dtype=float) for d in range(self.dim)]
            if not isinstance(inp, (list, tuple)) or len(inp) != self.dim:
                raise ValueError(f"{name} must be list/tuple with length == dim ({self.dim})")
            out = []
            for d in range(self.dim):
                vec = np.asarray(inp[d], dtype=float).ravel()
                if vec.size != self.nx[d]:
                    raise ValueError(f"{name}[{d}] length {vec.size} != N_d ({self.nx[d]})")
                out.append(np.maximum(vec, 0.0))
            return out

        self.delta_VI  = _normalize_delta_like("delta_VI",  delta_VI)
        self.delta_pol = _normalize_delta_like("delta_pol", delta_pol)

        # === policy mode dispatch ===
        if pol_mode is None:
            self.pol_mode = "rt"   # default: robusttree-like policy improvement
        else:
            pm = pol_mode.lower()
            if pm not in {"rt", "apos"}:
                raise ValueError("pol_mode must be 'rt' or 'apos'")
            self.pol_mode = pm

        # === VI mode dispatch ===
        if VI_mode is None:
            self.VI_mode = "rt"    # default: robust VI (subtract delta_VI)
        else:
            vm = VI_mode.lower()
            if vm not in {"rt", "apos"}:
                raise ValueError("VI_mode must be 'rt' or 'apos'")
            self.VI_mode = vm

        print(f"[DFATree] Initialized (pol_mode={self.pol_mode}, VI_mode={self.VI_mode})")

        # Validate DFA is 0-based, consecutive
        S = list(np.asarray(DFA.S).ravel())
        if not (min(S) == 0 and max(S) == len(S) - 1 and len(set(S)) == len(S)):
            raise ValueError(f"DFA.S must be 0..|S|-1 (0-based consecutive). Got: {S}")

        # Per-DFA-state controlled transitions (computed lazily)
        self.Pxx: List[List[Optional[np.ndarray]]] = [
            [None for _ in range(self.dim)] for _ in range(len(DFA.S))
        ]

        # Value tables per dimension (rows indexed by graph node id)
        self.V: List[np.ndarray] = [np.zeros((0, self.nx[d])) for d in range(self.dim)]

        # Graph + bookkeeping
        self.tree: nx.DiGraph = nx.DiGraph()
        self.leafs: List[int] = []
        self.Q: Dict[int, List[int]] = {int(q): [] for q in DFA.S}  # DFA state q -> list of node ids
        self.Dl: List[List[List[int]]] = []

    # === helpers ==========================================================
    def _recompute_levels(self) -> None:
        """
        Build Dl as per-depth buckets, using G.Q for DFA labeling:
          Dl[depth][0] = nodes at this depth with q == DFA.F   (qf bucket)
          Dl[depth][1] = nodes at this depth with q != DFA.F   (q0/non-final bucket)
        """
        Dl = []
        self.depths = {}

        if 0 in self.tree:
            # depth for every node
            lengths = nx.single_source_shortest_path_length(self.tree, 0)
            self.depths.update({int(n): int(d) for n, d in lengths.items()})

            if lengths:
                maxd = max(lengths.values())
                Dl = [[[], []] for _ in range(maxd + 1)]

                F = int(self.DFA.F)
                # sets for fast membership
                qf_nodes = set(self.Q.get(F, []))
                # everything else is "q0/non-final" bucket; you may refine if you track an explicit q0
                # collect all nodes across non-final states
                non_final_nodes = set()
                for q, nodes in self.Q.items():
                    if int(q) != F:
                        non_final_nodes.update(nodes)

                # assign by depth using the sets above
                for n, d in lengths.items():
                    if n in qf_nodes:
                        Dl[d][0].append(int(n))
                    elif n in non_final_nodes:
                        Dl[d][1].append(int(n))
                    else:
                        # Fallback: if a node isn't in Q-mapping (shouldn't happen), use its stored q
                        q = self.Lq(n)
                        (Dl[d][0] if q == F else Dl[d][1]).append(int(n))

                # stable ordering
                for level in Dl:
                    level[0].sort()
                    level[1].sort()

        self.Dl = Dl
        self.tree.graph['Dl'] = Dl

    def Lq(self, n: int) -> int:
        """Return DFA state (0-based) stored on node n."""
        return int(self.tree.nodes[n]["q"])

    def _nu_of_dim(self, d: int) -> int:
        """
        Number of actions for dimension d inferred from the transition container.

        Supports:
          - 1D: self.sysAbs[d].P is a dense ndarray shaped (N, N*nu)
          - 2D/tensor: self.sysAbs[d].P is an object exposing .P_det (sparse/dense)
        """
        Pobj = getattr(self.sysAbs[d], "P")

        # 2D/tensor wrapper: use its P_det
        if hasattr(Pobj, "P_det"):
            P_det = Pobj.P_det  # may be scipy.sparse or ndarray
            # get shape without forcing dense
            N, NU = P_det.shape
            if NU % N != 0:
                raise ValueError(f"P_det must be (N, N*nu); got {P_det.shape}")
            return NU // N

        # 1D dense path (unchanged)
        P_flat = np.asarray(Pobj, dtype=float)  # (N, N*nu)
        N, NU = P_flat.shape
        if NU % N != 0:
            raise ValueError(f"P must be (N, N*nu); got {P_flat.shape}")
        return NU // N

    # === initialization ===================================================
    def initiate(self) -> "DFATree":
        """
        Build the initial tree:
          - node 0 is the accepting mode (q = DFA.F)
          - its children are all predecessors (s, l) such that trans[s, l] == F
        Initialize V with root rows set to 1 (per dimension), others 0.
        """
        F = int(self.DFA.F)
        trans = np.asarray(self.DFA.trans, dtype=int)

        # Add root
        self.tree = nx.DiGraph()
        self.tree.add_node(0, q=F)

        # Add all predecessors of F as children of root
        S_rows, L_cols = np.where(trans == F)  # 0-based (sources s, letters l)
        nid = 1
        self.leafs = []
        for s, l in zip(S_rows, L_cols):
            self.tree.add_node(nid, q=int(s))
            self.tree.add_edge(0, nid, l=int(l))  # store label as 0-based column index
            self.leafs.append(nid)
            nid += 1

        # Initialize value tables: root row = 1, others = 0
        n_nodes = self.tree.number_of_nodes()
        for d in range(self.dim):
            self.V[d] = np.zeros((n_nodes, self.nx[d]), dtype=float)
            self.V[d][0, :] = 1.0

        # Build Q-mapping
        self.Q = {int(q): [] for q in self.DFA.S}
        self.Q[F].append(0)
        for n in self.leafs:
            self.Q[self.Lq(n)].append(n)

        self._recompute_levels()

        return self

    # === growth ===========================================================
    def grow(self, *args) -> None:
        """
        Expand all current leaves.
        Optionally: grow('number', k) -> only expand top-k leaves ranked by product of max V across dims.
        """
        leafs_old = list(self.leafs)
        if len(args) >= 2 and args[0] == "number":
            k = int(args[1])
            scores = []
            for leaf in self.leafs:
                s = 1.0
                for d in range(self.dim):
                    s *= float(np.max(self.V[d][leaf, :]))
                scores.append(s)
            order = np.argsort(scores)[::-1]
            leafs_old = [self.leafs[i] for i in order[: min(k, len(order))]]

        for n in leafs_old:
            self.growleaf(n)

    def growleaf(self, n: int) -> None:
        """Expand a single leaf by adding all its DFA predecessors as children."""
        if n not in self.leafs:
            raise ValueError("node is not a leaf node")

        q1 = self.Lq(n)
        trans = np.asarray(self.DFA.trans, dtype=int)

        # All predecessors of q1: pairs (s, l) with trans[s,l] == q1
        S_rows, L_cols = np.where(trans == q1)

        # Append children
        maxnode = self.tree.number_of_nodes() - 1
        new_nodes = []
        for i, (s, l) in enumerate(zip(S_rows, L_cols)):
            nid = maxnode + i + 1
            self.tree.add_node(nid, q=int(s))
            self.tree.add_edge(n, nid, l=int(l))   # 0-based label column
            new_nodes.append(nid)

        # Update leaf set
        self.leafs.remove(n)
        self.leafs.extend(new_nodes)

        # Extend value tables to accommodate new nodes (rows index == node ids)
        for d in range(self.dim):
            self.V[d] = np.vstack([self.V[d], np.zeros((len(new_nodes), self.nx[d]))])

        # Update Q mapping
        for nid in new_nodes:
            self.Q[self.Lq(nid)].append(nid)

        # Update levels Dl
        self._recompute_levels()

    # === dynamic programming =============================================
    def update_node_value(self, n: int) -> None:
        """
        One child->parent propagation step for node n (skip root).
        v_child = (χ_l ⊙ v_parent) @ Pxx[q]
        """
        if n == 0:
            return
        parents = list(self.tree.predecessors(n))
        if not parents:
            return

        p = parents[0]
        l = int(self.tree.edges[p, n]["l"])  # 0-based label column
        q = self.Lq(n)                        # DFA mode at node n (0-based)

        # skip accepting / sink states
        skip = {int(self.DFA.F)}
        if hasattr(self.DFA, "sink") and getattr(self.DFA, "sink") is not None:
            skip.add(int(self.DFA.sink))
        if q in skip:
            return

        for d in range(self.dim):
            v_parent = self.V[d][p, :]
            v_parent_row = v_parent.reshape(1, -1)

            mask = self.L[d][l, :].astype(float)     # (N,)
            mask_row = mask.reshape(1, -1)  # (1, N)

            elemul = v_parent_row * mask_row

            # naive computation, one line, works for 1d case:
            # vx = elemul @ self.Pxx[q][d]

            if self.sysAbs[0].dim == 1:
                mat = self.Pxx[q][d]
                vx = elemul @ mat
            elif self.sysAbs[0].dim == 2:
                mat = self.Pxx[q][d]
                # EQUIVALENCE: vx = mat.mtimes(elemul)
                vx = elemul @ mat.stoch

            else:
                raise ValueError(f"{self.sysAbs[0].dim}>2 is not supported.")

            # self.V[d][n, :] = vx.ravel()

            # --- VI_mode branching ---
            if self.VI_mode == "rt":
                # robust VI: subtract delta_VI, clamp at 0
                self.V[d][n, :] = np.maximum(vx.ravel() - self.delta_VI[d], 0.0)
            elif self.VI_mode == "apos":
                # a-posteriori VI: no subtraction here
                self.V[d][n, :] = vx.ravel()
            else:
                raise RuntimeError(f"Unknown VI_mode: {self.VI_mode}")

    def update_tree(self) -> None:
        """Propagate values deepest→root, skipping the root itself."""
        for n in sorted(self.tree.nodes, reverse=True):
            if n == 0:
                continue
            self.update_node_value(n)

    def Q_n_apos(self, n: int):
        """
        Per-dimension arrays Qv[d] with shape (N_d, nu_d).

        Also computes the per-node weighting used by the a-posteriori
        policy. See the remark in the body: gamma ** l_n is what drives the
        policy; the zeta weighting is kept (commented out) for reference.
        """
        # parent and label
        parents = list(self.tree.predecessors(n))
        if not parents:
            return [np.zeros((self.nx[d], self._nu_of_dim(d)), dtype=float)
                    for d in range(self.dim)]
        nparent = parents[0]
        l = int(self.tree.edges[nparent, n]["l"])  # 0-based label

        # parent depth of node n
        dep_n = int(self.depths.get(n, 0))
        l_n = max(dep_n, 0)

        # === gamma weighting: gamma ** l_n  (this is what drives the policy) ===
        gamma_pow = float(self.gamma ** l_n)
        self.gamma_pow_node[n] = gamma_pow

        # === zeta weighting: 1 + delta_bold * (iter_idx - l_n) ===
        # REMARK: zeta and gamma ** l_n are mathematically different node
        # weightings, but they were observed to give the same resulting
        # policy here. maxpolicy uses gamma ** l_n; the zeta computation is
        # kept below (commented out) for reference / analysis.
        # iter_idx = self.iter_idx
        # delta_scalars = []
        # for dd in range(self.dim):
        #     deltad = np.asarray(self.delta_pol[dd], dtype=float).ravel()
        #     delta_scalars.append(float(deltad.max() if deltad.size else 0.0))
        # delta_bold = 1.0 - float(np.prod([1.0 - s for s in delta_scalars]))
        # zeta = 1.0 + delta_bold * (iter_idx - l_n)
        # self.zeta_node[n] = float(zeta)
        zeta = None  # not used in policy; see remark above


        # ----- build Qv and scale by zeta -----
        Qv: List[np.ndarray] = []
        for d in range(self.dim):
            N = self.nx[d]
            P_attr = getattr(self.sysAbs[d], "P", None)

            mask_row = np.asarray(self.L[d][l, :], dtype=float).reshape(1, -1)
            v_parent_row = np.asarray(self.V[d][nparent, :], dtype=float).reshape(1, -1)
            w = mask_row * v_parent_row  # (1, N)

            if self.sysAbs[0].dim == 1:
                P_flat = P_attr  # (N, N*nu)
                nu = P_flat.shape[1] // N
                prod = np.asfortranarray(w) @ P_flat  # (1, N*nu)
                Qv_d = np.reshape(prod, (N, nu), order='F')
                # Qv_d *= zeta
                Qv.append(Qv_d)

            elif self.sysAbs[0].dim == 2:
                nu = self._nu_of_dim(d)
                prod = np.asfortranarray(w) @ P_attr.stoch
                Qv_d = np.reshape(prod, (N, nu), order='F')
                # Qv_d *= zeta
                Qv.append(Qv_d)

            else:
                raise ValueError(f"{self.sysAbs[0].dim}>2 is not supported.")

        return Qv, zeta, l_n

    # === Q-values for policy improvement =================================
    def Q_n(self, n: int) -> List[np.ndarray]:
        """
        Return per-dimension arrays Vxa[d] with shape (N, nu), where
        Vxa[d] = ( L[d](l,:) .* V[d](nparent,:) ) @ P_flat[d],
        reshaped to (N, nu).

        If node n has no parent (e.g., the root), returns zeros of the
        correct shapes for each dimension.
        """
        # find parent and label l on edge (parent -> n)
        parents = list(self.tree.predecessors(n))
        if not parents:
            # Root or detached: return zeros with correct shapes
            return [np.zeros((self.nx[d], self._nu_of_dim(d)), dtype=float)
                    for d in range(self.dim)]

        nparent = parents[0]
        l = int(self.tree.edges[nparent, n]["l"])  # 0-based label index

        Qv: List[np.ndarray] = []
        for d in range(self.dim):
            N = self.nx[d]

            # Get flat transition P_flat[d] with shape (N, N*nu)
            # P_flat = np.asarray(getattr(self.sysAbs[d], "P"), dtype=float, order='F')

            P_attr = getattr(self.sysAbs[d], "P", None)
            if self.sysAbs[0].dim == 1:
                P_flat = P_attr
                nu = P_flat.shape[1] // N
                mask_row = np.asarray(self.L[d][l, :], dtype=float).reshape(1, -1)
                v_parent_row = np.asarray(self.V[d][nparent, :], dtype=float).reshape(1, -1)
                w = mask_row * v_parent_row
                prod = np.asfortranarray(w) @ P_flat
                Qv_d = np.reshape(prod, (N, nu), order='F')
                # Qv_d = np.maximum(Qv_d - 0.001, 0.0)
                Qv.append(Qv_d)

            elif self.sysAbs[0].dim == 2:
                nu = self._nu_of_dim(d)
                mask_row = np.asarray(self.L[d][l, :], dtype=float).reshape(1, -1)
                v_parent_row = np.asarray(self.V[d][nparent, :], dtype=float).reshape(1, -1)
                w = mask_row * v_parent_row
                # EQUIVALENCE: prod = P_attr.mtimes( np.asfortranarray(w)  )
                prod = np.asfortranarray(w) @ P_attr.stoch
                Qv_d = np.reshape(prod, (N, nu), order='F')
                Qv.append(Qv_d)

            else:
                raise ValueError(f"{self.sysAbs[0].dim}>2 is not supported.")

        return Qv

    def set_iter(self, i: int) -> None:
        self.iter_idx = int(i)

    # === policy improvement ==============================================
    def maxpolicy(self, rho):
        """
        Greedy policy improvement with two modes controlled by self.pol_mode:

          pol_mode == "rt":
              - robust-tree style
              - uses Q_n(n), then subtracts delta_pol[d] before aggregation

          pol_mode == "apos":
              - a-posteriori style
              - uses Q_n(n) only to compute c[d],
                and Q_n_apos(n) (with zeta) for the contributions

        In both cases:
          - c[d] is computed with a UNIFORM policy (or previous policy if available)
          - greedy argmax per (q,d) is used to update self.Pxx[q][d]
        """
        mode = self.pol_mode
        print(f"[maxpolicy] Using policy mode: {mode}")
        if mode not in {"rt", "apos"}:
            raise RuntimeError(f"Unknown pol_mode: {mode}")

        num_states = len(self.DFA.S)
        new_pol = np.empty((num_states, self.dim), dtype=object)

        # states to skip (final/sink)
        skip = {int(self.DFA.F)}
        if hasattr(self.DFA, "sink") and getattr(self.DFA, "sink") is not None:
            skip.add(int(self.DFA.sink))

        # helper: dense uniform policy for a dimension d
        def uniform_pol_dense(d: int) -> np.ndarray:
            N = self.nx[d]
            nu = self._nu_of_dim(d)
            return np.full((N, nu), 1.0 / nu, dtype=float)

            # helper: choose the transition argument expected by Pc for dimension d.
            #   1D agents pass the dense (N, N*nu) flat matrix on sysAbs[d].P_flat;
            #   2D agents pass the TransitionProbability2D operator on sysAbs[d].P.
        def _transition_arg(d: int):
            return self.sysAbs[d].P_flat if self.sysAbs[d].dim == 1 else self.sysAbs[d].P

            # === main loop over DFA states ===

        # === main loop over DFA states ===
        for q in set(self.DFA.S) - skip:
            q_int = int(q)

            # If there's no tree nodes for this DFA state, keep previous policy (or uniform)
            if not self.Q[q_int]:
                for d in range(self.dim):
                    prev_pol = None
                    if self.pol is not None:
                        prev_pol = self.pol[q_int][d]

                    if prev_pol is None:
                        prev_pol = uniform_pol_dense(d)

                    new_pol[q_int][d] = prev_pol
                    # keep Pxx consistent with that carried-over policy
                    self.Pxx[q_int][d] = Pc(_transition_arg(d), prev_pol)
                continue

            # accumulate per-dimension scores
            Vxa = [np.zeros((self.nx[d], self._nu_of_dim(d)), dtype=float)
                   for d in range(self.dim)]

            # === iterate over all tree nodes n with DFA state q ===
            for n in self.Q[q_int]:
                # base Q (used for computing c[d] in both modes)
                Qv = self.Q_n(n)  # list of arrays, each (N_d, nu_d)

                if mode == "rt":
                    # --- robust pre-subtraction using delta_pol ---
                    for d in range(self.dim):
                        Qv[d] = np.maximum(Qv[d] - self.delta_pol[d][:, None], 0.0)

                # constants c[d] using (previous) policy or uniform
                c = np.zeros(self.dim, dtype=float)

                for d in range(self.dim):
                    pol_mat = None
                    if self.pol is not None:
                        pol_mat = self.pol[q_int][d]

                    # turn policy into dense (N_d, nu_d)
                    if pol_mat is None:
                        pol_dense = uniform_pol_dense(d)
                    elif issparse(pol_mat):
                        pol_dense = pol_mat.toarray()
                    else:
                        pol_dense = np.asarray(pol_mat, dtype=float)

                    # elementwise weighting of Q
                    vec = np.sum(Qv[d] * pol_dense, axis=1)  # (N_d,)
                    c[d] = float(np.asarray(rho[d]).ravel() @ vec)  # scalar

                # scale[d] = prod(c) / c[d] (handle zeros safely)
                prod_c = float(np.prod(c)) if self.dim > 0 else 1.0
                scale = np.divide(prod_c, c,
                                  out=np.zeros_like(c),
                                  where=(c != 0))

                # choose which Q to use for the contributions
                if mode == "apos":
                    # a-posteriori: use Q_n_apos (already includes zeta)
                    Q_eff, zeta, l_n = self.Q_n_apos(n)
                else:
                    # robust-tree: use the robust-subtracted Qv
                    Q_eff = Qv


                # accumulate contribution to Vxa[d]
                for d in range(self.dim):
                    contrib = Q_eff[d] * scale[d]
                    # make absolutely sure both sides are arrays
                    contrib = np.asarray(contrib, dtype=float)
                    if mode == "apos":
                        # Vxa[d] = Vxa[d] + contrib * zeta
                        Vxa[d] = Vxa[d] + contrib * (self.gamma ** l_n)
                    else:
                        Vxa[d] = Vxa[d] + contrib



            # === greedy argmax per dimension -> update ONLY Pxx ===
            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)  # best action per abstract state
                rows = np.arange(Vxa[d].shape[0])
                pol_d = sparse.csr_matrix(
                    (np.ones_like(rows, dtype=float), (rows, I)),
                    shape=Vxa[d].shape
                )
                new_pol[q_int][d] = pol_d
                self.Pxx[q_int][d] = Pc(_transition_arg(d), pol_d)

        # === keep policies for final / sink states ===
        for q in skip:
            q_int = int(q)
            if q_int < num_states:
                for d in range(self.dim):
                    if self.pol is not None:
                        new_pol[q_int][d] = self.pol[q_int][d]
                    # Pxx[q_int][d] is irrelevant for final/sink, but keep existing value

        self.pol = new_pol
        return self.Pxx


    # === pruning / relabeling ============================================
    def findSubtree(self, n: int, nodeIDs: Optional[List[int]] = None) -> List[int]:
        """Collect all nodes in the subtree rooted at n (post-order)."""
        if nodeIDs is None:
            nodeIDs = []
        for n_next in list(self.tree.successors(n)):
            nodeIDs = self.findSubtree(n_next, nodeIDs)
        nodeIDs.append(n)
        return nodeIDs

    def prune(self, tol: float, *args) -> None:
        """
        Prune nodes whose product over dimensions of max V along the row is < tol.
        Usage:
            prune(tol)            -> consider ALL nodes
            prune(tol, 'leafs')   -> consider ONLY current leaf nodes
        """
        use_leafs = (len(args) >= 1) and (args[0] == 'leafs')

        # rows to evaluate
        if use_leafs:
            rows = np.asarray(self.leafs, dtype=int)
        else:
            rows = np.arange(self.tree.number_of_nodes(), dtype=int)

        if rows.size == 0:
            return

        # per-dimension max over columns, for the chosen rows
        # max_vals[d] has shape (len(rows),)
        max_vals = []
        for d in range(self.dim):
            Vd = np.asarray(self.V[d], dtype=float)
            # guard if V has fewer rows (shouldn't happen, but safe)
            nrows = min(Vd.shape[0], rows.max() + 1) if rows.size else 0
            sel = rows[rows < nrows]
            if sel.size == 0:
                max_vals.append(np.zeros(rows.shape[0], dtype=float))
                continue

            md = np.zeros(rows.shape[0], dtype=float)
            md[np.isin(rows, sel)] = np.max(Vd[sel, :], axis=1)
            max_vals.append(md)

        # product across dimensions (elementwise)
        prod_vals = np.ones(rows.shape[0], dtype=float)
        for md in max_vals:
            prod_vals *= md

        # nodes to prune: product < tol
        idx = np.where(prod_vals < tol)[0]
        if idx.size == 0:
            return

        if use_leafs:
            nodeids = [int(self.leafs[i]) for i in idx]
        else:
            nodeids = [int(rows[i]) for i in idx]

        print(f"Pruning nodeids = {nodeids}")
        self.removeBranch(nodeids)


    def removeBranch(self, nodes: List[int]) -> None:
        """
        Remove subtrees rooted at given nodes. After deletion, re-label the remaining
        graph nodes to keep ids contiguous (0..N-1) and keep V/Q/leafs in sync.
        """
        # 1) collect nodes to delete
        to_delete: List[int] = []
        for n in nodes:
            to_delete = list(set(to_delete).union(self.findSubtree(n, [])))

        # 2) delete from graph
        for n in sorted(to_delete, reverse=True):
            if n in self.tree:
                self.tree.remove_node(n)

        # 3) relabel remaining nodes to 0..N-1
        remaining = sorted(list(self.tree.nodes))
        mapping = {old: new for new, old in enumerate(remaining)}
        nx.relabel_nodes(self.tree, mapping, copy=False)

        # 4) rebuild leafs and Q using new ids
        self.leafs = [mapping[leaf] for leaf in self.leafs if leaf in mapping]

        # # also add parents who just lose a child as leaves to avoid drop in EC convergence
        # self.leafs = [n for n in self.tree.nodes
        #               if self.tree.out_degree(n) == 0 and n != 0]

        new_Q = {int(q): [] for q in self.DFA.S}
        for q, lst in self.Q.items():
            new_Q[q] = [mapping[n] for n in lst if n in mapping]
        self.Q = new_Q

        # 5) rebuild V rows in new order
        for d in range(self.dim):
            old_V = self.V[d]
            new_V = np.zeros((len(remaining), self.nx[d]), dtype=float)
            for old in remaining:
                new = mapping[old]
                new_V[new, :] = old_V[old, :]
            self.V[d] = new_V

        # update levels Dl
        self._recompute_levels()

    # === stopping criterion: region mean =================================
    def approx_region_mean_sampling(
            self,
            idx_bounds,
            K: int = 2000,
            seed: int | None = None,
    ) -> float:
        """
        Monte Carlo estimate of the average 'tv-like' value over a rectangular
        region of the joint state grid, WITHOUT building the full tv tensor.

        This matches the logic you were using in MATLAB conceptually:
          value(x) = max_{n in candidate_nodes}  Π_d V[d][n, x_d]

        where:
          - candidate_nodes = all tree nodes whose DFA mode is "not done yet"
            (i.e. skip accepting and sink states)
          - V[d][n, :] is the value row for node n in dimension d
          - x_d is the abstract state index in dimension d

        We then average value(x) over K random samples x drawn uniformly from
        the specified hyper-rectangle.

        Parameters
        ----------
        idx_bounds : list[tuple[int,int]]
            Per-dimension inclusive index ranges.

            You can provide fewer than self.dim ranges.
            For any missing dimension d, we default to the entire domain [0, N_d-1].

            Example (2D region in a higher-D system):
                idx_bounds = [(0, 798), (599, 699)]
            If self.dim == 5, dims 2..4 will automatically use (0, N_d-1).

        K : int
            Number of Monte Carlo samples.

        seed : int | None
            RNG seed. IMPORTANT: keep this FIXED across iterations if you're
            using this as a convergence/stopping metric, so noise doesn't
            trigger early stopping.

        Returns
        -------
        float
            Estimated mean of value(x) over that region.
        """
        rng = np.random.default_rng(seed)

        D = self.dim  # number of independent subsystems / dims
        N_per_dim = self.nx  # list of grid sizes per dim, length D

        # -------- normalize / clip bounds per dimension --------
        # build lo[d], hi[d] for d=0..D-1
        lo = np.zeros(D, dtype=int)
        hi = np.zeros(D, dtype=int)

        for d in range(D):
            if d < len(idx_bounds):
                lo_d, hi_d = idx_bounds[d]
            else:
                # no bound provided for this dim -> use full range
                lo_d, hi_d = 0, N_per_dim[d] - 1

            # clip to valid range
            lo_d = max(0, min(N_per_dim[d] - 1, int(lo_d)))
            hi_d = max(0, min(N_per_dim[d] - 1, int(hi_d)))

            if hi_d < lo_d:
                # empty region in this dim ⇒ region volume is 0 ⇒ mean = 0
                return 0.0

            lo[d] = lo_d
            hi[d] = hi_d

        # -------- gather candidate nodes (non-accepting, non-sink DFA states) --------
        skip_states = {int(self.DFA.F)}
        if hasattr(self.DFA, "sink") and getattr(self.DFA, "sink") is not None:
            skip_states.add(int(self.DFA.sink))

        cand_nodes = []
        for q in self.DFA.S:
            q_int = int(q)
            if q_int in skip_states:
                continue
            cand_nodes.extend(self.Q.get(q_int, []))

        # if (pathologically) empty, fall back to *all* existing nodes so we don't just return 0
        if not cand_nodes:
            cand_nodes = list(self.tree.nodes)

        # deduplicate to avoid double-counting same node
        cand_nodes = list({int(n) for n in cand_nodes})

        if not cand_nodes:
            return 0.0

        # -------- Monte Carlo sampling --------
        total_val = 0.0
        K_int = int(K)

        for _ in range(K_int):
            # sample one joint abstract state index x = (i_0, ..., i_{D-1})
            idx_vec = [
                rng.integers(low=lo[d], high=hi[d] + 1)
                for d in range(D)
            ]

            # compute value(x) = max_n prod_d V[d][n, idx_vec[d]]
            best_val = 0.0
            for n in cand_nodes:
                prod_val = 1.0
                # multiply across dims
                for d in range(D):
                    v_d = float(self.V[d][n, idx_vec[d]])
                    if v_d <= 0.0:
                        prod_val = 0.0
                        break
                    prod_val *= v_d

                if prod_val > best_val:
                    best_val = prod_val

            total_val += best_val

        mean_est = total_val / float(K_int)
        return float(mean_est)

    def progress_check(
            self,
            idx_bounds,
            prev_score: float | None,
            K: int,
            tol_growth: float,
            seed: int | None = None,
            it: int | None = None,
    ):
        """
        Convenience helper for the stopping heuristic:
          1. compute current approx region-mean score
          2. compute gain vs prev_score (if any)
          3. decide whether to stop (gain <= tol_growth)

        Returns:
            score        : float
            gain         : float or None
            stop_now     : bool
        Also prints the debug lines for you.
        """
        score = self.approx_region_mean_sampling(
            idx_bounds=idx_bounds,
            K=K,
            seed=seed,
        )

        # always print score
        print(f"  approx region-mean score = {score:.6e}")

        if prev_score is None:
            # first iteration: no gain, don't stop
            return score, None, False

        gain = score - prev_score
        print(f"  gain since last iter    = {gain:.3e}")

        if gain <= tol_growth:
            # plateau → suggest stop
            if it is not None:
                print(f"  -> stopping early at iter {it} (plateau)")
            else:
                print("  -> stopping early (plateau)")
            return score, gain, True

        return score, gain, False

    # === plotting ========================================================
    def plot(self, use_letters: bool = False) -> None:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.tree, seed=0)
        nx.draw_networkx_nodes(self.tree, pos)
        nx.draw_networkx_labels(self.tree, pos, {n: self.tree.nodes[n]["q"] for n in self.tree.nodes})
        nx.draw_networkx_edges(self.tree, pos, arrows=True)
        if use_letters and hasattr(self.DFA, "act"):
            elabs = {(u, v): self.DFA.act[self.tree.edges[u, v]["l"]] for (u, v) in self.tree.edges}
        else:
            elabs = {(u, v): self.tree.edges[u, v]["l"] for (u, v) in self.tree.edges}
        nx.draw_networkx_edge_labels(self.tree, pos, elabs, font_size=9)
        plt.axis("off")
        plt.show()


    # === label helper ====================================================
    @staticmethod
    def num2label(act: List[str], nodes_l: List[int]) -> List[str]:
        """Map label indices (0-based) to strings."""
        return [act[i] for i in nodes_l]