from IPython.display import display
import spot
from spot.jupyter import display_inline
from spot import buddy
import subprocess
import copy
import networkx as nx
import math
from functools import partial
from itertools import product

# Job van der Werf authors half of this code.

class automaton:
    def __init__(self,spot_automaton):
        self.spot_automaton = spot_automaton
        self.nr_states = spot_automaton.num_states()
        self.automaton_state = spot_automaton.get_init_state_number()
        self.S = list(range(self.nr_states))
        self.S0 = [self.spot_automaton.get_init_state_number()]
        self.AP = [str(ap) for ap in spot_automaton.ap()]
        self.delta = [{}]
        self.F = self._get_accepting_set()
        self.transitions = []
        for s in self.S:
            for edge in self.spot_automaton.out(s):
                condition = spot.bdd_format_formula(self.spot_automaton.get_dict(), edge.cond)
                cond = condition.split('|')
                for i in range(len(cond)):
                    self.transitions.append((s,edge.dst,{'label':cond[i],'condition':convert(cond[i])}))

        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(self.S)
        self.graph.add_edges_from(self.transitions)

    
    def _get_accepting_set(self):
        accepting_set=[[] for i in range(self.spot_automaton.acc().num_sets())]
        for state in range(0,self.spot_automaton.num_states()):
            edge = _first(self.spot_automaton.out(state))
            #acceptance on state means that every outgoing transition has acceptance
            for i in range(len(accepting_set)):
                if edge.acc.has(i):
                    accepting_set[i].append(state)
        return accepting_set
    
    def draw(self):
        pos = nx.spring_layout(self.graph)
        #nx.draw(self.graph,pos,with_labels=True)
        elabels = nx.get_edge_attributes(self.graph,'label')
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph,pos,connectionstyle="arc3,rad=0.2")
        nx.draw_networkx_edge_labels(self.graph, pos,connectionstyle="arc3,rad=0.2", edge_labels=elabels)
   

def _first(iterable):
    iterator = iter(iterable)
    return next(iterator, None)

def get_possible_labels(aps):
    if len(aps)==1:
        return [{aps[0]:False},{aps[0]:True}]
    else:
        temp=get_possible_labels(aps[1:])
        temp2 = copy.deepcopy(temp)
        for inp in temp:
            inp[aps[0]]=False
        for inp in temp2:
            inp[aps[0]]=True
        return temp+temp2


        
def convert(input):
    result = input.replace('!','not ')
    result = result.replace('&','and')
    result = result.replace('|','or')
    return result


def translate(formula):
    spot_automaton = spot.translate(formula, 'Buchi', 'state-based', 'det'); 
    return automaton(spot_automaton)

if __name__=='__main__':
    spot.setup()
    a = spot.translate('(a U b) & GFb ', 'Buchi', 'state-based', 'det'); 
    print(a.to_str('dot'))
    print(a.get_init_state())
    b = automaton(a)
    b.accepting_set
    b.graph.nodes