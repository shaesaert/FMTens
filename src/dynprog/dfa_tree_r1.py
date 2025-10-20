# dfa_tree_r1.py
import copy
from operator import index
from typing import List, Dict, Any, Optional, Union

import networkx as nx
import numpy as np
from fontTools.varLib.builder import VarData_CalculateNumShorts
from scipy.sparse import csr_matrix, issparse
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

    # ---------- construction ----------
    def __init__(self, DFA, sysAbs, pol, nx_list, L, delta=None):
        self.DFA = DFA

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

        # === [optional Delta] ===
        if delta is None:
            self.delta: List[np.ndarray] = [np.zeros(self.nx[d], dtype=float) for d in range(self.dim)]
        elif isinstance(delta, (list, tuple)):
            if len(delta) != self.dim:
                raise ValueError(f"delta list length {len(delta)} != dim {self.dim}")
            self.delta = []
            for d in range(self.dim):
                vec = np.asarray(delta[d], dtype=float).ravel()
                if vec.size != self.nx[d]:
                    raise ValueError(f"delta[{d}] length {vec.size} != N_d ({self.nx[d]})")
                # clamp negatives just in case
                self.delta.append(np.maximum(vec, 0.0))
        else:
            raise TypeError("delta must be None or a list/tuple of vectors (one per dimension)")

        # === end of delta

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

    # ---------- helpers ----------
    def _recompute_levels(self) -> None:
        """
        Build Dl as per-depth buckets, using G.Q for DFA labeling:
          Dl[depth][0] = nodes at this depth with q == DFA.F   (qf bucket)
          Dl[depth][1] = nodes at this depth with q != DFA.F   (q0/non-final bucket)
        """
        Dl = []
        if 0 in self.tree:
            # depth for every node
            lengths = nx.single_source_shortest_path_length(self.tree, 0)
            if lengths:
                maxd = max(lengths.values())
                Dl = [[[], []] for _ in range(maxd + 1)]

                F = int(self.DFA.F)
                # sets for fast membership
                qf_nodes = set(self.Q.get(F, []))
                # everything else is “q0/non-final” bucket; you may refine if you track an explicit q0
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
                        # Fallback: if a node isn’t in Q-mapping (shouldn’t happen), use its stored q
                        q = self.Lq(n)
                        (Dl[d][0] if q == F else Dl[d][1]).append(int(n))

                # stable ordering
                for level in Dl:
                    level[0].sort()
                    level[1].sort()

        self.Dl = Dl
        self.tree.graph['Dl'] = Dl

    # def _recompute_levels(self) -> None:
    #     """
    #     Recompute Dl: Dl[k] = list of node ids at graph distance k from root (0).
    #     Also mirrors it onto self.tree.graph['Dl'] for convenient external access.
    #     """
    #     Dl: List[List[int]] = []
    #     if 0 in self.tree:
    #         lengths = nx.single_source_shortest_path_length(self.tree, 0)
    #         if lengths:
    #             maxd = max(lengths.values())
    #             Dl = [[] for _ in range(maxd + 1)]
    #             for n, d in lengths.items():
    #                 Dl[d].append(int(n))
    #             for lst in Dl:
    #                 lst.sort()
    #     self.Dl = Dl
    #     self.tree.graph['Dl'] = Dl

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

    def _as_dense_policy(self, q: int, d: int) -> np.ndarray:
        """Return pol[q][d] as dense (N,nu). If None, uniform."""
        N = self.nx[d]
        nu = self._nu_of_dim(d)
        P = self.pol[q][d]
        if P is None:
            return np.full((N, nu), 1.0 / nu, dtype=float)
        if issparse(P):
            arr = P.toarray()
        else:
            arr = np.asarray(P, dtype=float)
        if arr.shape != (N, nu):
            raise ValueError(f"policy shape incompatible: got {arr.shape}, need ({N},{nu})")
        return arr

    # ---------- initialization ----------
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

    # ---------- growth ----------
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

    # ---------- dynamic programming ----------
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

            self.V[d][n, :] = np.maximum(vx.ravel() - self.delta[d], 0.0)




    def update_tree(self) -> None:
        """Propagate values deepest→root, skipping the root itself."""
        for n in sorted(self.tree.nodes, reverse=True):
            if n == 0:
                continue
            self.update_node_value(n)

    # ---------- Q-values for policy improvement ----------
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
                nu = P_attr.a
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


    # ---------- policy improvement ----------
    def maxpolicy(self, rho):
        """
        Stateless greedy improvement:
          - uses UNIFORM policy to compute c[d]
          - computes greedy actions per (q,d)
          - updates ONLY self.Pxx[q][d] in-place
        """
        import numpy as np
        import scipy.sparse as sparse

        num_states = len(self.DFA.S)
        pol = np.empty((num_states, self.dim), dtype=object)

        # states to skip (final/sink)
        skip = {int(self.DFA.F)}
        if hasattr(self.DFA, "sink") and getattr(self.DFA, "sink") is not None:
            skip.add(int(self.DFA.sink))

        # helper: dense uniform policy for a dimension d
        def uniform_pol_dense(d: int) -> np.ndarray:
            N = self.nx[d]
            nu = self._nu_of_dim(d)
            return np.full((N, nu), 1.0 / nu, dtype=float)

        for q in set(self.DFA.S) - skip:
            # If there's no Q for this state, refresh cache with uniform policy and continue
            if not self.Q[q]:
                for d in range(self.dim):
                    P_src = self.sysAbs[d].P_flat if self.sysAbs[d].P_flat is not None else self.sysAbs[d].P
                    self.Pxx[q][d] = Pc(P_src, uniform_pol_dense(d))
                    # self.Pxx[q][d] = Pc(self.sysAbs[d].P_flat, uniform_pol_dense(d))
                continue

            # accumulate per-dimension scores
            Vxa = [np.zeros((self.nx[d], self._nu_of_dim(d)), dtype=float)
                   for d in range(self.dim)]

            # iterate over all n in Q[q]
            for n in self.Q[q]:
                Qv = self.Q_n(n)  # list of arrays, each (N_d, nu_d)

                for d in range(self.dim):
                    Qv[d] = np.maximum(Qv[d] - self.delta[d][:, None], 0.0)

                # constants c[d] using UNIFORM policy
                c = np.zeros(self.dim, dtype=float)
                for d in range(self.dim):
                    pol_dense = uniform_pol_dense(d)  # (N_d, nu_d)
                    vec = np.sum(Qv[d] * pol_dense, axis=1)  # (N_d,)
                    c[d] = float(np.asarray(rho[d]).ravel() @ vec)  # scalar

                # scale[d] = prod(c) / c[d] (handle zeros safely)
                prod_c = float(np.prod(c)) if self.dim > 0 else 1.0
                # where c[d] == 0, use 0.0 to avoid Inf; we'll also nan_to_num below
                scale = np.divide(prod_c, c, out=np.zeros_like(c), where=(c != 0))

                # accumulate contribution to Vxa[d]
                for d in range(self.dim):
                    contrib = np.nan_to_num(
                        Qv[d] * scale[d],
                        nan=0.0, posinf=0.0, neginf=0.0
                    )  # (N_d, nu_d)
                    Vxa[d] += contrib

            # greedy argmax per dimension -> update ONLY Pxx
            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)  # choose best action per state row
                rows = np.arange(Vxa[d].shape[0])
                pol[q][d] = sparse.csr_matrix(
                    (np.ones_like(rows, dtype=float), (rows, I)),
                    shape=Vxa[d].shape
                )
                P_src = self.sysAbs[d].P_flat if self.sysAbs[d].P_flat is not None else self.sysAbs[d].P
                self.Pxx[q][d] = Pc(P_src, pol[q][d])

        return self.Pxx

    # ---------- pruning / relabeling ----------
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

    # ---------- plotting ----------
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


    # ---------- label helper ----------
    @staticmethod
    def num2label(act: List[str], nodes_l: List[int]) -> List[str]:
        """Map label indices (0-based) to strings."""
        return [act[i] for i in nodes_l]
