# src/dynprog/dfa_tree_r1_h.py
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Union
from collections import OrderedDict
import re
from scipy.sparse import csr_matrix, issparse


class DFATreeR1:
    """
    Python re-implementation of MATLAB dfa_tree_r1.

    Key points
    ----------
    - Support DFA.S being 0- or 1-based; internally normalized to 1-based
    - Prefer DFA.transitions (then graph; lastly trans); auto-align letter order
    - Q_n returns (nx × nu); Pc supports one-hot sparse / dense randomized policies
    - maxpolicy updates Pxx internally; main does not need to manage Pxx
    - **Label-on-next-state semantics**:
        V_child = (L * V_parent) @ Pxx^T
      This matches the usual Pre operator with post-state predicates.
    """

    def __init__(self,
                 DFA: Any,
                 sysAbs: List[Any],
                 pol: List[List[Union[np.ndarray, csr_matrix]]],
                 nx_list: List[int],
                 L: List[np.ndarray],
                 sink: Optional[int] = None) -> None:

        self.DFA = DFA
        self.sysAbs = list(sysAbs)
        self.pol = pol
        self.nx = list(nx_list)
        self.L = L
        self.dim = len(sysAbs)

        # Normalize to 1-based mapping for states
        S_native = [int(x) for x in DFA.S]
        if S_native == list(range(1, len(S_native) + 1)):
            self._to1 = lambda q: int(q)
            self._from1 = lambda q: int(q)
        elif S_native == list(range(len(S_native))):
            self._to1 = lambda q: int(q) + 1
            self._from1 = lambda q: int(q) - 1
        else:
            raise ValueError("DFA.S must be consecutive integers starting at 0 or 1.")

        def _as_scalar(state, name: str) -> int:
            if isinstance(state, np.ndarray):
                arr = np.asarray(state).ravel()
                if arr.size != 1:
                    raise ValueError(f"{name} must be a single state.")
                return int(arr.item())
            while isinstance(state, (list, tuple)) and len(state) == 1:
                state = state[0]
            if isinstance(state, (np.integer, int)):
                return int(state)
            raise ValueError(f"{name} must be a scalar int; got {state!r}")

        self.S0 = S_native
        self.S1 = [self._to1(q) for q in self.S0]
        self.F1 = self._to1(_as_scalar(DFA.F, "DFA.F"))
        self.sink1 = None
        if hasattr(DFA, "sink") and DFA.sink is not None:
            self.sink1 = self._to1(_as_scalar(DFA.sink, "DFA.sink"))

        # Number of actions per dimension
        self.nu = []
        for d in range(self.dim):
            inu = int(np.prod(np.shape(self.sysAbs[d].inputs)))
            self.nu.append(inu)

        # Letter order
        self.letters = self._collect_letters()
        self.lab2idx = {lab: i for i, lab in enumerate(self.letters)}

        # Controlled transitions, value functions, tree structure, Q mapping
        self.Pxx: List[List[Optional[np.ndarray]]] = [[None for _ in range(self.dim)]
                                                      for _ in self.S1]
        self.V: List[np.ndarray] = [np.zeros((0, self.nx[d])) for d in range(self.dim)]
        self.tree: nx.DiGraph = nx.DiGraph()
        self.leafs: List[int] = []
        self.Q: Dict[int, List[int]] = {q: [] for q in self.S1}

    # ---------- DFA helpers ----------
    @staticmethod
    def _norm_label(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip())

    def _iter_edges(self):
        """Yield (u, v, label_str) where u/v are DFA native indices (same basis as self.S0)."""
        if hasattr(self.DFA, "transitions") and self.DFA.transitions is not None:
            for u, v, data in self.DFA.transitions:
                lab = data.get("label", data.get("condition", "1"))
                yield int(u), int(v), self._norm_label(lab)
            return
        if hasattr(self.DFA, "graph"):
            for u, v, data in self.DFA.graph.edges(data=True):
                lab = data.get("label", data.get("condition", "1"))
                yield int(u), int(v), self._norm_label(lab)
            return
        if hasattr(self.DFA, "trans"):
            trans = np.asarray(self.DFA.trans)
            rows, cols = np.where(trans > -np.inf)
            for u, v in zip(rows, cols):
                yield int(u), int(v), "1"
            return
        raise RuntimeError("DFA has no transitions/graph/trans.")

    def _collect_letters(self) -> List[str]:
        seen = OrderedDict()
        for _, _, lab in self._iter_edges():
            if lab not in seen:
                seen[lab] = True
        if not seen:
            seen["1"] = True
        return list(seen.keys())

    # ---------- Tree construction ----------
    def initiate(self) -> "DFATreeR1":
        self.tree = nx.DiGraph()
        self.tree.add_node(1, q=self.F1)  # Root: accepting state

        F0 = self._from1(self.F1)
        node_id = 2
        self.leafs = []

        # children of root: states with edges to F
        for u, v, lab in self._iter_edges():
            if int(v) != int(F0):
                continue
            l_idx = self.lab2idx[lab]
            self.tree.add_node(node_id, q=self._to1(int(u)))
            self.tree.add_edge(1, node_id, l=l_idx + 1)  # store 1-based label index
            self.leafs.append(node_id)
            node_id += 1

        # Initialize value functions: root is 1 (probability 1)
        for d in range(self.dim):
            self.V[d] = np.zeros((self.tree.number_of_nodes(), self.nx[d]), dtype=float)
            self.V[d][0, :] = 1.0

        self.Q[self.F1] = [1]
        for n in self.leafs:
            self.Q[self.Lq(n)].append(n)
        return self

    def Lq(self, n: int) -> int:
        return int(self.tree.nodes[n]["q"])  # 1-based

    # ---------- Grow ----------
    def grow(self, *args) -> None:
        leafs_old = list(self.leafs)
        if len(args) >= 2 and args[0] == "number":
            k = int(args[1])
            scores = []
            for leaf in self.leafs:
                s = 1.0
                for d in range(self.dim):
                    s *= float(np.max(self.V[d][leaf - 1, :]))
                scores.append(s)
            idx = np.argsort(scores)[::-1]
            leafs_old = [self.leafs[i] for i in idx[: min(k, len(self.leafs))]]

        for n in leafs_old:
            self.growleaf(n)

    def growleaf(self, n: int) -> None:
        if n not in self.leafs:
            raise ValueError("node is not a leaf node")
        q1 = self.Lq(n)
        q0 = self._from1(q1)

        maxnode = self.tree.number_of_nodes()
        new_nodes = []
        for u, v, lab in self._iter_edges():
            if int(v) != int(q0):
                continue
            l_idx = self.lab2idx[lab]
            maxnode += 1
            self.tree.add_node(maxnode, q=self._to1(int(u)))
            self.tree.add_edge(n, maxnode, l=l_idx + 1)
            new_nodes.append(maxnode)

        self.leafs.remove(n)
        self.leafs.extend(new_nodes)
        for d in range(self.dim):
            self.V[d] = np.vstack([self.V[d], np.zeros((len(new_nodes), self.nx[d]))])
        for nn in new_nodes:
            self.Q[self.Lq(nn)].append(nn)

    # ---------- Value update (label-on-next-state Pre) ----------
    def _edge_label_0based(self, p: int, c: int) -> int:
        return int(self.tree.edges[p, c]["l"]) - 1

    def update_node_value(self, n: int) -> None:
        if n == 1:
            return
        parents = list(self.tree.predecessors(n))
        if not parents:
            return
        p = parents[0]
        l0 = self._edge_label_0based(p, n)
        q = self.Lq(n)

        for d in range(self.dim):
            # Lazy compute: if Pxx[q-1][d] is missing, build via current policy
            if self.Pxx[q - 1][d] is None:
                self.Pxx[q - 1][d] = self.Pc(self.sysAbs[d], self.pol[q - 1][d])

            # Correct Pre with post-state mask:
            #   V_child = (L * V_parent) @ Pxx^T
            v_parent = self.V[d][p - 1, :]                  # (N,)
            v_masked = self.L[d][l0, :] * v_parent          # apply label on NEXT state
            pre = v_masked @ self.Pxx[q - 1][d].T           # (N,)
            self.V[d][n - 1, :] = pre

    # ---------- Q_n: consistent with post-state mask ----------
    def _block_matrix(self, d: int, u: int) -> np.ndarray:
        sa = self.sysAbs[d]
        if hasattr(sa, "block"):
            return sa.block(u)
        # Fallback from flat
        P_flat = self._P_flat(d)           # (N, N*nu)
        N = self.nx[d]
        return P_flat[:, u * N:(u + 1) * N]

    def Q_n(self, n: int) -> List[np.ndarray]:
        parents = list(self.tree.predecessors(n))
        if not parents:
            return [np.zeros((self.nx[d], self.nu[d])) for d in range(self.dim)]
        p = parents[0]
        l0 = self._edge_label_0based(p, n)

        out: List[np.ndarray] = []
        for d in range(self.dim):
            N, nu = self.nx[d], self.nu[d]
            v_parent = self.V[d][p - 1, :]                 # (N,)
            v_masked = self.L[d][l0, :] * v_parent         # post-state mask
            Q = np.zeros((N, nu), dtype=float)
            # For each action u: Bu @ (L * v_parent)
            for u in range(nu):
                Bu = self._block_matrix(d, u)              # (N,N)
                Q[:, u] = Bu @ v_masked                    # (N,)
            out.append(Q)
        return out

    def update_tree(self) -> None:
        for n in sorted(self.tree.nodes, reverse=True)[:-1]:
            self.update_node_value(n)

    # ---------- Policy improvement ----------
    def _as_dense_policy(self, q: int, d: int) -> np.ndarray:
        """
        Normalize self.pol[q-1][d] to dense array of shape (nx[d], nu[d]).
        Allowed inputs:
          - None  -> uniform policy
          - csr_matrix sparse one-hot
          - scalar, (nx,), (nu,), (nx,nu), (nu,nx)
        """
        nx_d, nu_d = self.nx[d], self.nu[d]
        pol_qd = None
        if self.pol and self.pol[q - 1][d] is not None:
            pol_qd = self.pol[q - 1][d]

        if pol_qd is None:
            return np.full((nx_d, nu_d), 1.0 / nu_d, dtype=float)

        if issparse(pol_qd):
            arr = pol_qd.toarray()
        else:
            arr = np.asarray(pol_qd, dtype=float)

        # Already in target shape
        if arr.shape == (nx_d, nu_d):
            return arr

        # Scalar -> constant distribution
        if arr.ndim == 0:
            return np.full((nx_d, nu_d), float(arr), dtype=float)

        # 1D cases
        if arr.ndim == 1:
            if arr.size == nu_d:  # same action distribution for every state
                return np.tile(arr.reshape(1, -1), (nx_d, 1))
            if arr.size == nx_d and nu_d == 1:  # single-action case
                return arr.reshape(nx_d, 1)

        # (nu, nx) -> transpose
        if arr.shape == (nu_d, nx_d):
            return arr.T

        raise ValueError(
            f"policy[{q - 1}][{d}] shape {arr.shape} incompatible with (nx,nu)=({nx_d},{nu_d})"
        )

    def maxpolicy(self, rho: List[np.ndarray]) -> List[List[csr_matrix]]:
        """
        Robustly handle None/sparse/dense/various-shaped policies; fall back for abnormal inputs.
        Matches legacy behavior in normal cases.
        """
        nq = len(self.S1)
        new_pol: List[List[csr_matrix]] = [[None for _ in range(self.dim)] for _ in range(nq)]

        skip = {self.F1}
        if self.sink1 is not None:
            skip.add(self.sink1)

        for q in set(self.S1) - skip:
            if not self.Q[q]:
                continue

            # Aggregate Q-values
            Vxa = [np.zeros((self.nx[d], self.nu[d]), dtype=float) for d in range(self.dim)]
            for n in self.Q[q]:
                Qv = self.Q_n(n)  # List[(nx[d], nu[d])]

                # Constant term c[d] per dimension
                c = np.zeros(self.dim, dtype=float)
                for d in range(self.dim):
                    pol_dense = self._as_dense_policy(q, d)  # (nx, nu)
                    rho_d = np.asarray(rho[d], dtype=float).ravel()  # (nx,)
                    contrib = np.sum(Qv[d] * pol_dense, axis=1)  # (nx,)
                    c[d] = float(rho_d @ contrib)  # scalar

                # Accumulate into Vxa: product of constants from other dims
                prod_c = np.prod(c) if self.dim > 0 else 1.0
                for d in range(self.dim):
                    scale = (prod_c / c[d]) if (self.dim > 1 and c[d] != 0) else (1.0 if self.dim == 1 else 0.0)
                    Vxa[d] += Qv[d] * scale

            # Greedy per-dimension action selection -> one-hot; update controlled transitions
            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)
                rows = np.arange(self.nx[d])
                pol_onehot = csr_matrix((np.ones(self.nx[d]), (rows, I)),
                                        shape=(self.nx[d], self.nu[d]))
                new_pol[q - 1][d] = pol_onehot
                self.Pxx[q - 1][d] = self.Pc(self.sysAbs[d], pol_onehot)

        self.pol = new_pol
        # Important: invalidate all cached Pxx so they are recomputed under the new policy
        for i in range(len(self.S1)):
            for d in range(self.dim):
                self.Pxx[i][d] = None

        return new_pol

    # ---------- Controlled transitions & utilities ----------
    def _P_flat(self, d: int) -> np.ndarray:
        sa = self.sysAbs[d]
        if hasattr(sa, "P"):
            return sa.P
        if hasattr(sa, "P_flat"):
            return sa.P_flat
        if hasattr(sa, "block") and hasattr(sa, "inputs"):
            N = sa.block(0).shape[0]
            nu = int(np.prod(np.shape(sa.inputs)))
            return np.hstack([sa.block(u) for u in range(nu)])
        raise AttributeError("MDPModel must have P, P_flat, or (block & inputs).")

    @staticmethod
    def stochasticize_block(Bu, eps=1e-12):
        Bu = np.asarray(Bu, dtype=float).copy()
        Bu[np.isnan(Bu)] = 0.0
        rsum = Bu.sum(axis=1)
        zero = rsum < eps
        if np.any(zero):
            idx = np.where(zero)[0]
            Bu[idx, :] = 0.0
            for i in idx:
                Bu[i, i] = 1.0
        rsum = Bu.sum(axis=1)
        nz = rsum >= eps
        Bu[nz, :] /= rsum[nz, None]
        return Bu

    @staticmethod
    def Pc(sys_or_P: Union[Any, np.ndarray],
           pol: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        # Identify the source
        if hasattr(sys_or_P, "block"):
            B0 = sys_or_P.block(0)
            N = B0.shape[0]
            if hasattr(sys_or_P, "inputs"):
                nu = int(np.prod(np.shape(sys_or_P.inputs)))
            elif hasattr(sys_or_P, "P"):
                nu = sys_or_P.P.shape[1] // N
            else:
                raise AttributeError("Cannot infer nu: need inputs or P")
            get_block = lambda u: sys_or_P.block(u)
        else:
            P_flat = np.asarray(sys_or_P)
            N, NU = P_flat.shape
            assert NU % N == 0, "P_flat must be (N, N*nu)"
            nu = NU // N
            get_block = lambda u: P_flat[:, u * N:(u + 1) * N]

        # Sparse one-hot
        if issparse(pol):
            rows, cols = pol.nonzero()
            a = np.zeros(pol.shape[0], dtype=int)
            a[rows] = cols
            PC = np.zeros((pol.shape[0], N), dtype=float)
            for u in np.unique(a):
                idx = np.where(a == u)[0]
                if idx.size == 0:
                    continue
                Bu = get_block(int(u))
                PC[idx, :] = Bu[idx, :]
            return DFATreeR1.stochasticize_block(PC)

        # Dense randomized
        pol = np.asarray(pol, dtype=float)
        if pol.ndim == 0:
            pol = np.full((N, nu), float(pol))
        elif pol.ndim == 1:
            if nu == 1:
                pol = pol.reshape(-1, 1)
            else:
                raise ValueError(f"policy must have shape (N, nu={nu}); got 1D {pol.shape}")
        elif pol.ndim == 2 and pol.shape[0] == nu and pol.shape[1] == N:
            pol = pol.T

        if pol.shape != (N, nu):
            raise ValueError(f"policy shape mismatch: expected ({N}, {nu}), got {pol.shape}")

        PC = np.zeros((N, N), dtype=float)
        for u in range(nu):
            w = pol[:, u][:, None]
            if np.allclose(w, 0):
                continue
            Bu = get_block(u)
            PC += w * Bu
        return DFATreeR1.stochasticize_block(PC)

    # ---------- Checks only, does not modify values ----------
    def check_values(self, eps: float = 1e-8, include_accepting: bool = False,
                     check_outer: bool = False, verbose: bool = True) -> bool:
        """
        Validate:
          1) For every node and dimension, V[n, :] is within [0, 1] (up to eps)
          2) (Optional) For each DFA state q, take all nodes in that state,
             form outer products across dimensions and sum them;
             verify the result remains within [0, 1] (up to eps)
        Note: by default **exclude the accepting state qf**; set include_accepting=True to include it.
        """
        ok = True
        skip_q = set()
        if not include_accepting:
            skip_q.add(self.F1)  # skip qf
        if getattr(self, "sink1", None) is not None:
            pass  # If you also want to skip sink: skip_q.add(self.sink1)

        # 1) Per-node range checks
        node_issues = []
        for n in sorted(self.tree.nodes):
            q = self.Lq(n)
            if q in skip_q:
                continue
            for d in range(self.dim):
                v = self.V[d][n - 1, :]
                vmin, vmax = float(np.min(v)), float(np.max(v))
                if vmin < -eps or vmax > 1.0 + eps:
                    ok = False
                    if len(node_issues) < 5:
                        node_issues.append(f"node {n}, dim {d}: min={vmin:.6g}, max={vmax:.6g}")

        # 2) Outer-product sum within the same q (diagnostic)
        outer_issues = []
        if check_outer and self.dim >= 2:
            for q in self.S1:
                if q in skip_q:
                    continue
                nodes = self.Q.get(q, [])
                if not nodes:
                    continue

                # Outer products across dimensions, summed over nodes
                agg = None
                for n in nodes:
                    vecs = [self.V[d][n - 1, :] for d in range(self.dim)]
                    out = vecs[0]
                    for d in range(1, self.dim):
                        out = np.multiply.outer(out, vecs[d])
                    agg = out if agg is None else (agg + out)

                vmin, vmax = float(np.min(agg)), float(np.max(agg))
                if vmin < -eps or vmax > 1.0 + eps:
                    ok = False
                    if len(outer_issues) < 5:
                        outer_issues.append(
                            f"q={self._from1(q)}: min={vmin:.6g}, max={vmax:.6g}, "
                            f"shape={agg.shape}, nodes={len(nodes)}"
                        )

        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"[check] {status} (eps={eps})")
            if node_issues:
                print("  Node out-of-range (showing up to 5):")
                for s in node_issues:
                    print("   ", s)
            if outer_issues:
                print("  Outer-product sum out-of-range (showing up to 5):")
                for s in outer_issues:
                    print("   ", s)
        return ok
