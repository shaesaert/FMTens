import numpy as np
from dfa_tree import *
from Translate import *
import spot


def test_setup():
    """Set up test cases - Similar to TestClassSetup in MATLAB"""
    # Create Weak Buchi that we will use as a DFA based on scLTL formula
    formula = '( (!p2 | !p3 ) U p1)'  # p1 = parking, p2 = avoid region
    
    global dfa
    dfa = translate(formula)
    assert dfa is not None

def test_initiate():
    """Test tree initialization"""
    tree = dfa_tree(dfa)
    tree.initiate()
    assert tree.tree is not None
    assert tree.leafs is not None
    
def test_grow():
    """Test growing all leaves of the tree"""
    tree = dfa_tree(dfa)
    tree.initiate()
    tree.grow()
    # Add assertions to verify the growth
    assert len(tree.tree.nodes) > 1
    
def test_growleaf():
    """Test growing a specific leaf"""
    tree = dfa_tree(dfa)
    tree.initiate()
    tree.growleaf(2)
    # Add assertions to verify the specific leaf growth
    assert 2 in tree.tree.nodes
    
def test_plot():
    """Test basic plotting"""
    tree = dfa_tree(dfa)
    tree.initiate()
    # Note: In Python we might want to use matplotlib.pyplot to show the plot
    plot = tree.plot()
    
    
def test_remove_branch():
    """Test branch removal"""
    tree = dfa_tree(dfa)
    tree.initiate()
    initial_node_count = len(tree.tree.nodes)
    tree.remove_branch(2)
    tree.grow()
    # Verify that nodes were removed
    
def test_find_subtree():
    """Test finding subtree nodes"""
    tree = dfa_tree(dfa)
    tree.initiate()
    node_ids = tree.find_subtree(2, [])
    # Verify that we got some nodes back


