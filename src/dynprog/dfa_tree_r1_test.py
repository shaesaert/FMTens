# src/dynprog/dfa_tree.py
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Union
from scipy.sparse import csr_matrix, issparse


ArrayLike = Union[np.ndarray, csr_matrix]


class DFATree:
    """
    DFA-guided tree for PRE-style dynamic programming.

    Conventions
    -----------
    - Nodes are numbered 1..M, root == 1.
    - self.tree edges are oriented parent -> child.
    - DFA.S is [1..nq] (1-based); DFA.F is accepting (1-based); optional DFA.sink.
    - DFA.trans is (nq x nLetter) int matrix with 1-based targets.
    - L[d] is (nLetter x N) mask for dimension d (bool/0-1).
    - sysAbs[d] provides: .P / .Prob / .P_det  (N x (N*nu))  OR  .block(u)->(N,N).
    - pol[q-1][d] is (N x nu) CSR one-hot or dense prob. distribution.

    Flow you asked for
    ------------------
    maxpolicy(rho)   -> computes pol & Pxx (for affected q)
    update_tree()    -> recompute values deepest -> ... -> level 1 (root NOT updated)
    grow()           -> expand current leaves; new children have V=0 and proper DFA labels
    """

    # ---------- ctor ----------
    def __init__(
        self,
        DFA: Any,
        sysAbs: List[Any],
        pol: List[List[ArrayLike]],
        nx_list: List[int],
        L: List[np.ndarray],
    ) -> None:
        # DFA basics
        self.DFA = DFA
        self.S: List[int] = list(self.DFA.S)
        if self.S != list(range(1, len(self.S) + 1)):
            raise ValueError("DFA.S must be 1..|S| (1-based consecutive integers).")
        self.F: int = int(self.DFA.F)
        self.sink: int = int(getattr(self.DFA, "sink", 0) or 0)
        # --- letter names & order (for compatibility with caller) ---
        if hasattr(self.DFA, "act") and self.DFA.act:
            # Use DFA.act if present (this is what your script compares to)
            self.letters = list(self.DFA.act)
        else:
            # Fallback: infer number of letters from DFA.trans
            if hasattr(self.DFA, "trans"):
                m = int(np.asarray(self.DFA.trans).shape[1])
                self.letters = [f"l{i + 1}" for i in range(m)]
            else:
                self.letters = []
        # Optional helper: label -> column index
        self.lab2idx = {lab: i for i, lab in enumerate(self.letters)}

        # systems & sizes
        self.sysAbs: List[Any] = list(sysAbs)
        self.dim: int = len(self.sysAbs)
        self.nx: List[int] = list(nx_list)
        self.L: List[np.ndarray] = L

        # number of actions per dim
        self.nu: List[int] = []
        for sa in self.sysAbs:
            inu = 0
            if hasattr(sa, "inputs"):
                inu = int(np.prod(np.shape(sa.inputs)))
            if inu <= 0:
                P_flat = self._get_flat_P(sa)
                N, NU = P_flat.shape
                assert NU % N == 0, "Flat P must be (N, N*nu)."
                inu = NU // N
            self.nu.append(inu)

        # policies (q-1, d)
        self.pol: List[List[Optional[ArrayLike]]] = [
            [None for _ in range(self.dim)] for _ in self.S
        ]
        if pol:
            for q_idx in range(min(len(pol), len(self.S))):
                for d in range(min(len(pol[q_idx]), self.dim)):
                    self.pol[q_idx][d] = pol[q_idx][d]

        # controlled transitions cache (N x N)
        self.Pxx: List[List[Optional[np.ndarray]]] = [
            [None for _ in range(self.dim)] for _ in self.S
        ]

        # graph + bookkeeping
        self.tree: nx.DiGraph = nx.DiGraph()
        self.leafs: List[int] = []
        self.Q: Dict[int, List[int]] = {int(q): [] for q in self.S}  # q -> nodes

        # value buffers V[d]: (nNodes x N)
        self.V: List[np.ndarray] = [np.zeros((0, self.nx[d])) for d in range(self.dim)]

    # ---------- helpers ----------
    @staticmethod
    def _get_flat_P(sys_or_P):
        import numpy as np
        if isinstance(sys_or_P, np.ndarray):
            return np.asarray(sys_or_P, dtype=float)
        for name in ("P", "Prob", "P_det"):  # <— CHANGED ORDER: P first
            if hasattr(sys_or_P, name):
                return np.asarray(getattr(sys_or_P, name), dtype=float)
        if hasattr(sys_or_P, "block"):
            B0 = sys_or_P.block(0)
            N = B0.shape[0]
            nu = 1
            if hasattr(sys_or_P, "inputs"):
                nu = int(np.prod(np.shape(sys_or_P.inputs)))
            blocks = [sys_or_P.block(u) for u in range(nu)]
            return np.hstack(blocks)
        raise AttributeError("sysAbs[d] must expose .P/.Prob/.P_det or .block(u).")

    def _block_matrix(self, d: int, u: int) -> np.ndarray:
        sa = self.sysAbs[d]
        if hasattr(sa, "block"):
            return np.asarray(sa.block(u), dtype=float)
        P_flat = self._get_flat_P(sa)
        N = self.nx[d]
        return P_flat[:, u * N : (u + 1) * N]

    def _edge_label_0based(self, p: int, c: int) -> int:
        return int(self.tree.edges[p, c]["l"]) - 1

    def Lq(self, n: int) -> int:
        return int(self.tree.nodes[n]["q"])

    # ---------- tree build ----------
    def initiate(self) -> "DFATree":
        """Root at F; children are states with transitions to F (reverse edges)."""
        self.tree = nx.DiGraph()
        self.tree.add_node(1, q=self.F)

        trans = np.asarray(self.DFA.trans, dtype=int)  # (nq x nLetter)
        S, l = np.where(trans == self.F)               # 0-based
        node_id = 2
        self.leafs = []
        for s0, lab0 in zip(S, l):
            self.tree.add_node(node_id, q=int(s0 + 1))
            self.tree.add_edge(1, node_id, l=int(lab0 + 1))
            self.leafs.append(node_id)
            node_id += 1

        # init V
        nnode = self.tree.number_of_nodes()
        for d in range(self.dim):
            self.V[d] = np.zeros((nnode, self.nx[d]), dtype=float)
            self.V[d][0, :] = 1.0  # root = 1

        # Q map
        self.Q = {int(q): [] for q in self.S}
        self.Q[self.F] = [1]
        for n in self.leafs:
            self.Q[self.Lq(n)].append(n)
        return self

    # ---------- update (deepest -> level 1, NOT root) ----------
    def update_tree(self, *, force_pc: bool = False) -> None:
        """
        Recompute values depth-by-depth, starting from current deepest nodes
        up to (but NOT including) the root (node 1). Each node n uses ONLY
        its parent's current value and the child-mode Pxx.
        """
        if self.tree.number_of_nodes() <= 1:
            return

        # Compute depths from root (topological)
        depth = {1: 0}
        for n in nx.topological_sort(self.tree):
            if n == 1:
                continue
            p = next(self.tree.predecessors(n), None)
            if p is None:
                continue
            depth[n] = depth[p] + 1

        max_depth = max(depth.values())
        if max_depth == 0:
            return

        # Deepest -> ... -> level 1 (skip level 0 root)
        for dlev in range(max_depth, 0, -1):
            level_nodes = [n for n, dv in depth.items() if dv == dlev]
            for n in level_nodes:
                p = next(self.tree.predecessors(n))
                l0 = self._edge_label_0based(p, n)  # 0-based letter
                qn = self.Lq(n)                     # child's DFA mode (1-based)

                for d in range(self.dim):
                    if force_pc or (self.Pxx[qn - 1][d] is None):
                        self.Pxx[qn - 1][d] = self.Pc(self.sysAbs[d], self.pol[qn - 1][d])

                    PC   = self.Pxx[qn - 1][d]                 # (N, N)
                    vp   = self.V[d][p - 1, :]                 # (N,)
                    mask = self.L[d][l0, :].astype(float)      # (N,)
                    self.V[d][n - 1, :] = mask * (vp @ PC.T)

    # ---------- grow ----------
    def grow(self, *args) -> None:
        """
        Expand current leaves. For each leaf n with mode q0, create one child for
        every DFA predecessor of q0; initialize new children with V=0 and set DFA labels.
        Optional: grow('number', k) to expand only top-k leaves by a simple score.
        """
        leafs_old = list(self.leafs)

        if len(args) >= 2 and args[0] == "number":
            k = int(args[1])
            scores = []
            for leaf in self.leafs:
                s = 1.0
                for d in range(self.dim):
                    s *= float(np.max(self.V[d][leaf - 1, :]))
                scores.append(s)
            order = np.argsort(scores)[::-1]
            leafs_old = [self.leafs[i] for i in order[: min(k, len(order))]]

        trans = np.asarray(self.DFA.trans, dtype=int)
        maxnode = self.tree.number_of_nodes()
        new_nodes: List[int] = []

        for n in leafs_old:
            q0 = self.Lq(n)
            S, Lcols = np.where(trans == q0)  # predecessors of q0 (0-based)
            for s0, lab0 in zip(S, Lcols):
                maxnode += 1
                self.tree.add_node(maxnode, q=int(s0 + 1))
                self.tree.add_edge(n, maxnode, l=int(lab0 + 1))
                new_nodes.append(maxnode)

        # update leaf set
        for n in leafs_old:
            if n in self.leafs:
                self.leafs.remove(n)
        self.leafs.extend(new_nodes)

        # init V rows for new nodes & update Q map
        add_cnt = len(new_nodes)
        if add_cnt:
            for d in range(self.dim):
                self.V[d] = np.vstack([self.V[d], np.zeros((add_cnt, self.nx[d]))])
            for nn in new_nodes:
                self.Q[self.Lq(nn)].append(nn)

    # ---------- Q (consistent with update flow) ----------
    def Q_n(self, n: int) -> List[np.ndarray]:
        """List[(N,nu)] using the parent value and the edge label to n."""
        parents = list(self.tree.predecessors(n))
        if not parents:
            return [np.zeros((self.nx[d], self.nu[d])) for d in range(self.dim)]
        p = parents[0]
        l0 = self._edge_label_0based(p, n)
        out: List[np.ndarray] = []
        for d in range(self.dim):
            N, nu = self.nx[d], self.nu[d]
            v_parent = self.V[d][p - 1, :]
            mask = self.L[d][l0, :].astype(float)
            Q = np.zeros((N, nu), dtype=float)
            for u in range(nu):
                Bu = self._block_matrix(d, u)        # (N, N)
                Q[:, u] = mask * (Bu @ v_parent)
            out.append(Q)
        return out

    # ---------- policy utilities ----------
    def _as_dense_policy(self, q: int, d: int) -> np.ndarray:
        N, nu = self.nx[d], self.nu[d]
        P = self.pol[q - 1][d]
        if P is None:
            return np.full((N, nu), 1.0 / nu, dtype=float)
        if issparse(P):
            A = P.toarray()
        else:
            A = np.asarray(P, dtype=float)
        if A.shape != (N, nu):
            raise ValueError(f"policy shape incompatible for q={q}, d={d}: {A.shape} vs {(N,nu)}")
        return A

    def set_policy(self, q: int, d: int, P: ArrayLike) -> None:
        """Set pol[q-1][d] and invalidate Pxx cache."""
        q_idx = q - 1
        N, nu = self.nx[d], self.nu[d]
        if issparse(P):
            P = P.tocsr()
            if P.shape != (N, nu):
                raise ValueError(f"policy shape for q={q}, d={d} must be {(N,nu)}, got {P.shape}")
        else:
            A = np.asarray(P)
            if A.shape != (N, nu):
                raise ValueError(f"policy shape for q={q}, d={d} must be {(N,nu)}, got {A.shape}")
            P = A
        self.pol[q_idx][d] = P
        self.Pxx[q_idx][d] = None

    # ---------- policy improvement ----------
    def maxpolicy(self, rho: List[np.ndarray]) -> List[List[csr_matrix]]:
        """
        For each DFA mode q not in {F, sink}, compute greedy per-dim argmax policy
        using V from parent nodes and masks, then store pol and Pxx for those q.
        """
        nq = len(self.S)
        new_pol: List[List[Optional[csr_matrix]]] = [[None for _ in range(self.dim)] for _ in self.S]

        skip = {self.F}
        if self.sink:
            skip.add(self.sink)

        for q in set(self.S) - skip:
            if not self.Q[q]:  # no nodes in this mode yet
                continue

            Vxa = [np.zeros((self.nx[d], self.nu[d]), dtype=float) for d in range(self.dim)]

            for n in self.Q[q]:
                Qv = self.Q_n(n)  # List[(N,nu)]

                # constants c[d]
                c = np.zeros(self.dim, dtype=float)
                for d in range(self.dim):
                    pol_dense = self._as_dense_policy(q, d)
                    vec = np.sum(Qv[d] * pol_dense, axis=1)  # (N,)
                    c[d] = float(np.asarray(rho[d]).ravel() @ vec)

                prod_c = np.prod(c) if self.dim > 0 else 1.0
                for d in range(self.dim):
                    scale = 1.0 if self.dim == 1 else ((prod_c / c[d]) if c[d] != 0 else 0.0)
                    Vxa[d] += Qv[d] * scale

            # argmax -> one-hot, update caches
            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)
                rows = np.arange(self.nx[d])
                pol_onehot = csr_matrix((np.ones(self.nx[d]), (rows, I)),
                                        shape=(self.nx[d], self.nu[d]))
                new_pol[q - 1][d] = pol_onehot
                self.Pxx[q - 1][d] = self.Pc(self.sysAbs[d], pol_onehot)

        # keep existing pol for untouched modes (e.g., F / sink)
        for q_idx in range(nq):
            for d in range(self.dim):
                if new_pol[q_idx][d] is None:
                    new_pol[q_idx][d] = self.pol[q_idx][d]
        self.pol = new_pol  # type: ignore
        return new_pol  # type: ignore

    # ---------- Pc (MATLAB-faithful) ----------
    @staticmethod
    def Pc(sys_or_P, pol, *, tol=1e-12, enforce_substochastic=True):
        import numpy as np
        from scipy.sparse import issparse

        P_flat = DFATree._get_flat_P(sys_or_P)  # (N, N*nu)
        N, NU = P_flat.shape
        assert NU % N == 0, f"P must be (N, N*nu); got {P_flat.shape}"
        nu = NU // N

        pol_dense = pol.toarray() if issparse(pol) else np.asarray(pol, dtype=float)
        if pol_dense.shape != (N, nu):
            raise ValueError(f"Pc: policy must be (N,{nu}); got {pol_dense.shape}")

        w = pol_dense.flatten(order="F")  # MATLAB vec
        Plarge = P_flat * w[None, :]  # (N, N*nu)

        PC = np.zeros((N, N), dtype=float)
        for u in range(nu):
            PC += Plarge[:, u * N: (u + 1) * N]

        if enforce_substochastic:
            rs = PC.sum(axis=1, keepdims=True)
            over = (rs > 1.0 + tol).ravel()
            if np.any(over):
                PC[over, :] /= rs[over]  # scale those rows back to sum=1
        return PC
