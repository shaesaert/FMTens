from unittest import TestCase
from dynprog.dfa_tree import *
from specifications.translate import *


class Testdfa_tree(TestCase):

    def setUp(self):
        """Set up test cases - Similar to TestClassSetup in MATLAB"""
        # Create Weak Buchi that we will use as a DFA based on scLTL formula
        formula = '( (!p2 | !p3 ) U p1)'  # p1 = parking, p2 = avoid region

        global dfa
        dfa = translate(formula)
        assert dfa is not None


    def test_tree_initialization(self):
        """Test tree initialization"""
        tree = dfa_tree(dfa)
        tree.initiate()
        assert tree.tree is not None
        assert tree.leafs is not None



    def test_grow(self):
        """Test growing all leaves of the tree"""
        tree = dfa_tree(dfa)
        tree.initiate()
        tree.grow()
        # Add assertions to verify the growth
        assert len(tree.tree.nodes) > 1


    def test_growleaf(self):
        """Test growing a specific leaf"""
        tree = dfa_tree(dfa)
        tree.initiate()
        tree.growleaf(2)
        # Add assertions to verify the specific leaf growth
        assert 2 in tree.tree.nodes


    def test_plot(self):
        """Test basic plotting"""
        tree = dfa_tree(dfa)
        tree.initiate()
        # Note: In Python we might want to use matplotlib.pyplot to show the plot
        plot = tree.plot()


    def test_remove_branch(self):
        """Test branch removal"""
        tree = dfa_tree(dfa)
        tree.initiate()
        initial_node_count = len(tree.tree.nodes)
        tree.remove_branch(2)
        tree.grow()
        # Verify that nodes were removed


    def test_find_subtree(self):
        """Test finding subtree nodes"""
        tree = dfa_tree(dfa)
        tree.initiate()
        node_ids = tree.find_subtree(2, [])
        # Verify that we got some nodes back



