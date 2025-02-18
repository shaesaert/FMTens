from Translate import *
import spot

def test_until():
    """Test until"""
    a = translate('(a U b)'); 
    assert a.nr_states == 2


def test_always():
    """Test Always"""
    a = spot.translate('(G b)', 'Buchi', 'state-based', 'det'); 
    b = automaton(a)
    assert b.nr_states == 1

def test_eventually():
    """Test until"""
    a = spot.translate('(F b)', 'Buchi', 'state-based', 'det'); 
    b = automaton(a)
    assert b.nr_states == 2

def test_division():
    """Test Eventually Always"""
    a = spot.translate('F G b', 'Buchi', 'state-based', 'det'); 
    b = automaton(a)
    assert b.nr_states == 2