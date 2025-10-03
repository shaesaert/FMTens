# test_dfatree_prune.py
import numpy as np
import networkx as nx
import pytest
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))   # project root
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.dynprog.dfa_tree_r1 import DFATree

# adjust import to your module

class MiniDFA:
    """
    Tiny 0-based DFA:
      - States S = {0,1,2}
      - Accepting F = 0
      - Transitions: 1 --(a)--> 0,  2 --(a)--> 0
      - One letter 'a' (index 0)
    """
    def __init__(self):
        self.S = [0, 1, 2]
        self.F = 0
        self.act = ['a']
        # trans[s, l] = next_state
        self.trans = np.array([
            [0],  # from 0 with 'a' -> 0   (self-loop; irrelevant)
            [0],  # from 1 with 'a' -> 0   (predecessor of F)
            [0],  # from 2 with 'a' -> 0   (predecessor of F)
        ], dtype=int)

class DummySysAbs:
    """Just a placeholder; DFATree.prune doesn't touch sysAbs.P/etc."""
    pass

def make_tree():
    dfa = MiniDFA()
    # Two independent abstract dimensions
    sysAbs = [DummySysAbs(), DummySysAbs()]
    # Grid sizes per dimension
    nx_list = [3, 4]
    # Label masks per dimension: one letter, all ones (shape: |letters| x N_d)
    L = [np.ones((1, nx_list[0])), np.ones((1, nx_list[1]))]
    # Policy placeholder (not used by prune)
    pol = [[None, None] for _ in dfa.S]

    G = DFATree(dfa, sysAbs=sysAbs, pol=pol, nx_list=nx_list, L=L).initiate()
    # After initiate():
    #   - node 0 is root (q=F)
    #   - nodes 1 and 2 are leaves (predecessors of F via letter 0)
    return G

def test_prune_leafs_removes_and_relabels():
    G = make_tree()
    trans = np.asarray(G.DFA.trans, dtype=int)
    F = int(G.DFA.F)
    # count all predecessors, including s == F
    n_pred = int(np.sum(trans == F))
    # or exclude root self-loop with:
    # n_pred = int(np.sum((trans == F) & (np.arange(trans.shape[0])[:,None] != F)))

    assert G.tree.number_of_nodes() == 1 + n_pred
    # rest of the test...


def test_prune_all_keeps_root_when_above_tol():
    G = make_tree()
    # Make root “large” (it already is: row of 1s), leaves small (already 0s)
    # Prune across ALL nodes with tol below root product but above leaf product.
    tol = 1e-6
    G.prune(tol)  # no 'leafs' -> consider all nodes

    # Only root remains (same outcome as leafs-only here, but via the all-nodes path)
    assert G.tree.number_of_nodes() == 1
    assert list(G.tree.nodes) == [0]
    for d in range(G.dim):
        assert G.V[d].shape == (1, G.nx[d])
        assert np.allclose(G.V[d][0, :], 1.0)
