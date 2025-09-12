import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union


class DFATree:
    """
    Python translation of MATLAB dfa_tree_r1 class

    A class for handling DFA (Deterministic Finite Automaton) trees with value functions
    for control synthesis and verification.
    """

    def __init__(self, DFA, sysAbs, pol, nx, L):
        """
        Constructor for DFATree

        Args:
            DFA: DFA object with properties S, trans, F, sink, act
            sysAbs: System abstraction (list of objects with P property)
            pol: Policy
            nx: List of state dimensions for each system
            L: Labeling function (list of matrices)
        """
        self.DFA = DFA
        self.sysAbs = sysAbs
        self.L = L
        self.nx = nx
        self.dim = len(sysAbs)
        self.pol = pol

        # Validate DFA.S
        if not np.array_equal(DFA.S, list(range(1, len(DFA.S) + 1))):
            raise ValueError('DFA.S must contain a range starting from 1')

        # Initialize data structures
        self.Q = {q: [] for q in DFA.S}  # Dictionary for DFA modes
        self.Pxx = [[None for _ in range(self.dim)] for _ in range(len(DFA.S))]
        self.V = [None for _ in range(self.dim)]  # Value function

        self.tree = None
        self.leafs = []

    def update_Pxx(self, new_Pxx):
        """Update Pxx dynamically"""
        self.Pxx = new_Pxx
        print('Updated Pxx')

    def initiate(self):
        """Create tree with accepting node and its children"""
        # Find transitions to accepting state
        trans_indices = np.where(self.DFA.trans == self.DFA.F)
        S = trans_indices[0] + 1  # MATLAB uses 1-based indexing
        l = trans_indices[1] + 1

        # Create initial tree structure
        self.tree = nx.DiGraph()

        # Add root node (accepting state)
        self.tree.add_node(1, q=self.DFA.F)

        # Add child nodes and edges
        node_counter = 2
        self.leafs = []

        for i, (source_state, label) in enumerate(zip(S, l)):
            self.tree.add_node(node_counter, q=source_state)
            self.tree.add_edge(1, node_counter, l=label)
            self.leafs.append(node_counter)
            node_counter += 1

        # Initialize value function
        num_nodes = len(self.tree.nodes)
        for d in range(self.dim):
            self.V[d] = np.zeros((num_nodes, self.nx[d]))
            self.V[d][0, :] = 1  # Set accepting node value to 1

        # Update Q mapping
        self.Q[self.DFA.F] = [1]  # Add accepting node
        for n in self.leafs:
            q = self.tree.nodes[n]['q']
            self.Q[q].append(n)

    def Lq(self, n):
        """Get the DFA state associated with tree node n"""
        return self.tree.nodes[n]['q']

    def grow(self, *args):
        """Add children to leafs of graph"""
        leafs_old = self.leafs.copy()

        if len(args) >= 2 and args[0] == 'number':
            # Take n largest possible values
            values = []
            for leaf in self.leafs:
                prod_val = 1
                for d in range(self.dim):
                    prod_val *= np.max(self.V[d][leaf, :])
                values.append(prod_val)

            sorted_indices = np.argsort(values)[::-1]  # Descending order
            nmax = min(args[1], len(leafs_old))
            leafs_old = [self.leafs[i] for i in sorted_indices[:nmax]]

        for n in leafs_old:
            self.growleaf(n)

    def growleaf(self, n):
        """Grow a single leaf node"""
        if n not in self.leafs:
            raise ValueError('node is not a leaf node')

        maxnode = max(self.tree.nodes) if self.tree.nodes else 0
        q = self.Lq(n)

        # Find possible children
        trans_indices = np.where(self.DFA.trans == q)
        S = trans_indices[0] + 1  # Source states
        l = trans_indices[1] + 1  # Labels

        # Add nodes and edges
        new_nodes = []
        for i, (source_state, label) in enumerate(zip(S, l)):
            new_node = maxnode + i + 1
            self.tree.add_node(new_node, q=source_state)
            self.tree.add_edge(n, new_node, l=label)
            new_nodes.append(new_node)

        # Update leafs
        self.leafs.remove(n)
        self.leafs.extend(new_nodes)

        # Extend value function
        for d in range(self.dim):
            new_rows = np.zeros((len(new_nodes), self.nx[d]))
            self.V[d] = np.vstack([self.V[d], new_rows])

        # Update Q mapping
        for new_node in new_nodes:
            q = self.Lq(new_node)
            self.Q[q].append(new_node)

    def update_node_value(self, n):
        """Update value function for a specific node"""
        if n == 1:  # Root node
            return

        # Find parent node
        predecessors = list(self.tree.predecessors(n))
        if not predecessors:
            return

        nparent = predecessors[0]
        edge_data = self.tree.edges[nparent, n]
        l = edge_data['l'] - 1  # Convert to 0-based indexing

        for d in range(self.dim):
            self.V[d][n, :] = (self.L[d][l, :] * self.V[d][nparent, :]) @ self.Pxx[self.Lq(n) - 1][d]

    def Q_n(self, n):
        """Compute Q-values for node n"""
        Vxa = [None for _ in range(self.dim)]

        # Find parent node
        predecessors = list(self.tree.predecessors(n))
        if not predecessors:
            return Vxa

        nparent = predecessors[0]
        edge_data = self.tree.edges[nparent, n]
        l = edge_data['l'] - 1  # Convert to 0-based indexing

        for d in range(self.dim):
            Vxa[d] = (self.L[d][l, :] * self.V[d][nparent, :]) @ self.sysAbs[d].P

        return Vxa

    def update_tree(self):
        """Update value function for all nodes in the tree"""
        nodes = sorted(self.tree.nodes, reverse=True)
        for n in nodes[:-1]:  # Skip root node
            self.update_node_value(n)

    def findSubtree(self, n, nodeIDs=None):
        """Find all nodes in subtree rooted at n"""
        if nodeIDs is None:
            nodeIDs = []

        successors = list(self.tree.successors(n))
        for n_next in successors:
            nodeIDs = self.findSubtree(n_next, nodeIDs)

        nodeIDs.append(n)
        return nodeIDs

    def removeBranch(self, nodes):
        """Remove branch starting from given nodes"""
        nodeIDs = []
        for n in nodes:
            nodeIDs.extend(self.findSubtree(n, []))

        nodeIDs = sorted(set(nodeIDs), reverse=True)  # Remove duplicates and sort descending

        for n in nodeIDs:
            self.tree.remove_node(n)
            if n in self.leafs:
                self.leafs.remove(n)

            # Renumber leafs
            self.leafs = [leaf - (leaf > n) for leaf in self.leafs]

            # Update Q mappings
            for q in self.DFA.S:
                if n in self.Q[q]:
                    self.Q[q].remove(n)
                self.Q[q] = [node - (node > n) for node in self.Q[q]]

            # Remove from value function
            for d in range(self.dim):
                self.V[d] = np.delete(self.V[d], n - 1, axis=0)  # Convert to 0-based indexing

    def maxpolicy(self, rho):
        """Compute maximum policy"""
        pol = [[None for _ in range(self.dim)] for _ in range(len(self.DFA.S))]

        for q in set(self.DFA.S) - {self.DFA.F, self.DFA.sink}:
            Vxa = [0 for _ in range(self.dim)]

            if not self.Q[q]:
                continue

            for n in self.Q[q]:
                Qv = self.Q_n(n)

                # Compute constants for each dimension
                c = np.zeros(self.dim)
                for d in range(self.dim):
                    c[d] = rho[d].T @ np.sum(Qv[d] * self.pol[q - 1][d], axis=1)

                for d in range(self.dim):
                    other_dims = [i for i in range(self.dim) if i != d]
                    Vxa[d] = Vxa[d] + Qv[d] * np.prod(c[other_dims])

            for d in range(self.dim):
                I = np.argmax(Vxa[d], axis=1)
                pol[q - 1][d] = csr_matrix((np.ones(len(I)), (range(len(I)), I)),
                                           shape=(Vxa[d].shape[0], Vxa[d].shape[1]))
                self.Pxx[q - 1][d] = self.Pc(self.sysAbs[d].P, pol[q - 1][d])

        self.pol = pol
        print("Policy updated")
        print(pol)

        return pol

    def plot(self, *args):
        """Plot the tree structure"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.tree)

        # Draw nodes with labels
        node_labels = {n: self.tree.nodes[n]['q'] for n in self.tree.nodes}
        nx.draw_networkx_nodes(self.tree, pos)
        nx.draw_networkx_labels(self.tree, pos, node_labels)
        nx.draw_networkx_edges(self.tree, pos)

        # Draw edge labels
        if len(args) >= 1 and args[0] == 'letters':
            edge_labels = {}
            for u, v, data in self.tree.edges(data=True):
                edge_labels[(u, v)] = self.DFA.act[data['l'] - 1]  # Convert to 0-based
        else:
            edge_labels = {(u, v): data['l'] for u, v, data in self.tree.edges(data=True)}

        nx.draw_networkx_edge_labels(self.tree, pos, edge_labels)
        plt.show()

    def prune(self, tol, *args):
        """Prune nodes with low values"""
        if len(args) >= 1 and args[0] == 'leafs':
            # Only consider leaf nodes
            values = []
            for leaf in self.leafs:
                prod_val = 1
                for d in range(self.dim):
                    prod_val *= np.max(self.V[d][leaf, :])
                values.append(prod_val)

            idx = [i for i, val in enumerate(values) if val < tol]
            nodeids = [self.leafs[i] for i in idx]
        else:
            # Consider all nodes
            nodeids = []
            for n in self.tree.nodes:
                prod_val = 1
                for d in range(self.dim):
                    prod_val *= np.max(self.V[d][n - 1, :])  # Convert to 0-based
                if prod_val < tol:
                    nodeids.append(n)

        if nodeids:
            print(f'Pruning nodeids = {nodeids}')
            self.removeBranch(nodeids)

    @staticmethod
    def Pc(P, pol):
        """Compute policy-controlled transition matrix (needs implementation)"""
        # This function needs to be implemented based on the specific requirements
        # It should return a matrix based on the transition matrix P and policy pol
        raise NotImplementedError("Pc function needs to be implemented")

    @staticmethod
    def num2label(act, nodes_l):
        """Convert numeric labels to AP based labels"""
        letters = []
        for q in range(len(nodes_l)):
            letters.append(act[nodes_l[q] - 1])  # Convert to 0-based indexing
        return letters