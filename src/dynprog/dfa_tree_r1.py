# dfa_tree_r1.py
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, issparse
from typing import List, Dict, Any, Optional, Tuple, Union


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
    def __init__(self, DFA, sysAbs, pol, nx_list, L):
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
        self.pol = pol  # pol[q][d] is (N, nu) dense or csr one-hot

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

    # ---------- helpers ----------
    def Lq(self, n: int) -> int:
        """Return DFA state (0-based) stored on node n."""
        return int(self.tree.nodes[n]["q"])

    def _nu_of_dim(self, d: int) -> int:
        """Number of actions for dimension d inferred from flat transition shape."""
        P_flat = np.asarray(getattr(self.sysAbs[d], "P"), dtype=float)  # (N, N*nu)
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

    # ---------- dynamic programming ----------
    def update_node_value(self, n: int) -> None:
        """
        One child->parent propagation step for node n (skip root).
        v_child(x) = L[l](x) * ( v_parent @ Pxx[q] )(x)
        """
        if n == 0:
            return
        parents = list(self.tree.predecessors(n))
        if not parents:
            return

        p = parents[0]
        l = int(self.tree.edges[p, n]["l"])  # 0-based label column
        q = self.Lq(n)                        # DFA mode at node n (0-based)

        for d in range(self.dim):
            # Lazily compute Pxx for this DFA mode & dimension from current policy
            if self.Pxx[q][d] is None:
                self.Pxx[q][d] = self.Pc(self.sysAbs[d], self.pol[q][d])

            v_parent = self.V[d][p, :]                     # (N,)
            mask = self.L[d][l, :].astype(float)           # (N,)
            self.V[d][n, :] = (mask * v_parent) @ self.Pxx[q][d]

    def update_tree(self) -> None:
        """Propagate values deepest→root, skipping the root itself."""
        for n in sorted(self.tree.nodes, reverse=True):
            if n == 0:
                continue
            self.update_node_value(n)

    # ---------- Q-values for policy improvement ----------
    def Q_n(self, n: int) -> List[np.ndarray]:
        """
        Return per-dimension Q-tables for node n:
        Q[d] has shape (N, nu) and column u is the masked value using block B_u.
        """
        parents = list(self.tree.predecessors(n))
        if not parents:
            return [np.zeros((self.nx[d], self._nu_of_dim(d))) for d in range(self.dim)]
        p = parents[0]
        l = int(self.tree.edges[p, n]["l"])                # 0-based
        out: List[np.ndarray] = []
        for d in range(self.dim):
            N = self.nx[d]
            P_flat = np.asarray(getattr(self.sysAbs[d], "P"), dtype=float)  # (N, N*nu)
            nu = P_flat.shape[1] // N
            blocks = P_flat.reshape(N, N, nu, order="C")
            v_parent = self.V[d][p, :]                     # (N,)
            mask = self.L[d][l, :].astype(float)           # (N,)
            Q = np.zeros((N, nu), dtype=float)
            for u in range(nu):
                Bu = blocks[:, :, u]                       # (N,N)
                Q[:, u] = mask * (v_parent @ Bu)           # (N,)
            out.append(Q)
        return out

    def maxpolicy(self, rho: List[np.ndarray]) -> List[List[csr_matrix]]:
        """
        Greedy, per-DFA-state, per-dimension improvement (same structure as your MATLAB version).
        Updates self.pol and self.Pxx for all q ∉ {F, sink}.
        """
        nq = len(self.DFA.S)
        new_pol: List[List[csr_matrix]] = [[None for _ in range(self.dim)] for _ in range(nq)]
        skip = {int(self.DFA.F)}
        if hasattr(self.DFA, "sink") and getattr(self.DFA, "sink") is not None:
            skip.add(int(self.DFA.sink))

        for q in set(self.DFA.S) - skip:
            if not self.Q[q]:
                continue

            Vxa = [np.zeros((self.nx[d], self._nu_of_dim(d)), dtype=float) for d in range(self.dim)]
            for n in self.Q[q]:
                Qv = self.Q_n(n)  # List[(N,nu)]

                # constants c[d]
                c = np.zeros(self.dim, dtype=float)
                for d in range(self.dim):
                    pol_dense = self._as_dense_policy(q, d)     # (N,nu)
                    vec = np.sum(Qv[d] * pol_dense, axis=1)     # (N,)
                    c[d] = float(np.asarray(rho[d]).ravel() @ vec)

                prod_c = np.prod(c) if self.dim > 0 else 1.0
                for d in range(self.dim):
                    if self.dim == 1:
                        scale = 1.0
                    else:
                        scale = (prod_c / c[d]) if c[d] != 0 else 0.0
                    Vxa[d] += Qv[d] * scale

            # Greedy per-dimension argmax -> one-hot; update Pxx
            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)
                rows = np.arange(self.nx[d])
                onehot = csr_matrix((np.ones(self.nx[d]), (rows, I)),
                                    shape=(self.nx[d], self._nu_of_dim(d)))
                new_pol[q][d] = onehot
                self.Pxx[q][d] = self.Pc(self.sysAbs[d], onehot)

        self.pol = new_pol
        return new_pol

    # ---------- pruning / relabeling ----------
    def findSubtree(self, n: int, nodeIDs: Optional[List[int]] = None) -> List[int]:
        """Collect all nodes in the subtree rooted at n (post-order)."""
        if nodeIDs is None:
            nodeIDs = []
        for n_next in list(self.tree.successors(n)):
            nodeIDs = self.findSubtree(n_next, nodeIDs)
        nodeIDs.append(n)
        return nodeIDs

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

    # ---------- Pc (controlled transitions) ----------
    @staticmethod

    def Pc(sys_or_P: Union[Any, np.ndarray],
           pol: Union[np.ndarray, csr_matrix],
           *,
           return_container: bool = False):
        """
        MATLAB-equivalent 'Pc':
          Plarge = P_flat .* vec(pol)'   (vec is column-major)
          PC     = sum_u Plarge[:, u*N : (u+1)*N]

        Inputs
        ------
        sys_or_P : object with one of {P_det, Prob, P} shaped (N, N*nu), or a flat ndarray (N, N*nu)
        pol      : (N, nu) dense or CSR policy (one-hot or stochastic)
        return_container : if True and sys_or_P is an object, return a shallow copy with the chosen
                           source field (P_det/Prob/P) replaced by the N×N controlled matrix. Otherwise
                           return the N×N ndarray (default, recommended for DFATree use).

        Returns
        -------
        PC : (N, N) ndarray if return_container=False
             or a container with .P_det / .Prob / .P replaced by PC when return_container=True.
        """
        # --- pick the source flat matrix in this priority: P_det -> Prob -> P ---
        src_name = None
        if isinstance(sys_or_P, np.ndarray):
            S = np.asarray(sys_or_P, dtype=float)          # (N, N*nu)
        else:
            for name in ("P_det", "Prob", "P"):
                if hasattr(sys_or_P, name):
                    S = np.asarray(getattr(sys_or_P, name), dtype=float)
                    src_name = name
                    break
            else:
                raise ValueError("Pc: sys_or_P must be ndarray or have one of fields: P_det, Prob, P.")

        N, NU = S.shape
        if NU % N != 0:
            raise ValueError(f"Pc: expected flat shape (N, N*nu). Got {S.shape}.")
        nu = NU // N

        # --- policy -> dense (N, nu) ---
        if issparse(pol):
            pol_dense = pol.toarray()
        else:
            pol_dense = np.asarray(pol, dtype=float)
        if pol_dense.shape != (N, nu):
            raise ValueError(f"Pc: policy must be shape (N,{nu}), got {pol_dense.shape}.")

        # --- MATLAB's vec(pol) is column-major (Fortran) ---
        w = np.reshape(pol_dense, (N * nu,), order="F")    # length N*nu

        # --- Plarge = S .* w' (column-wise scaling) ---
        Plarge = S * w[None, :]                            # (N, N*nu)

        # --- Sum N-wide blocks across actions ---
        PC = np.zeros((N, N), dtype=float)
        for u in range(nu):
            PC += Plarge[:, u * N : (u + 1) * N]

        if return_container and not isinstance(sys_or_P, np.ndarray):
            # Shallow copy with updated field (MATLAB style)
            out = type(sys_or_P).__new__(type(sys_or_P))
            out.__dict__.update(sys_or_P.__dict__)
            setattr(out, src_name, PC)
            return out

        return PC

    # ---------- label helper ----------
    @staticmethod
    def num2label(act: List[str], nodes_l: List[int]) -> List[str]:
        """Map label indices (0-based) to strings."""
        return [act[i] for i in nodes_l]
