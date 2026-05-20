from src.specifications.translate import *
import spot
from unittest import TestCase


class Testdtranslate(TestCase):
    def test_until(self):
        """Test until"""
        a = translate('(a U b)');
        assert a.nr_states == 2

    def test_always(self):
        """Test Always"""
        a = spot.translate('(G b)', 'Buchi', 'state-based', 'det');
        b = automaton(a)
        assert b.nr_states == 1

    def test_eventually(self):
        """Test until"""
        a = spot.translate('(F b)', 'Buchi', 'state-based', 'det');
        b = automaton(a)
        assert b.nr_states == 2

    def test_division(self):
        """Test Eventually Always"""
        a = spot.translate('F G b', 'Buchi', 'state-based', 'det');
        b = automaton(a)
        assert b.nr_states == 2