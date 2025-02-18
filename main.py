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

class sim_automaton:
    def __init__(self,spot_automaton):
        self.spot_automaton = spot_automaton
        self.nr_states = spot_automaton.num_states()
        self.automaton_state = spot_automaton.get_init_state_number()
        self.accepting_set = self._get_accepting_set()
        self.max_nondeterminism = self.get_max_nondeterminism()
        self.S = list(range(self.nr_states))
        self.S0 = [self.spot_automaton.get_init_state_number()]
        self.Epsilon = [str(ap) for ap in spot_automaton.ap()]
        self.delta = [{}]
        self.F = self._get_accepting_set()
        self.transitions = []
        for s in self.S:
            for edge in self.spot_automaton.out(s):
                condition = spot.bdd_format_formula(self.spot_automaton.get_dict(), edge.cond)
                self.transitions.append((s,edge.dst,{'label':condition,'condition':convert(condition)}))
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.S)
        self.graph.add_edges_from(self.transitions)


    def reset(self):
        self.automaton_state = self.spot_automaton.get_init_state_number()
    
    def labelfunction(self, obs):
        pass

    def step(self,labels,non_deterministic_transition=0):
        possible_next_states = []
        for edge in self.spot_automaton.out(self.automaton_state):
            condition = spot.bdd_format_formula(self.spot_automaton.get_dict(), edge.cond)
            if eval(convert(condition),labels):
                possible_next_states.append(edge.dst)
        if len(possible_next_states)==0:
            return -1 #non-accepting state
        possible_next_states.sort()
        if non_deterministic_transition>=len(possible_next_states):
            return possible_next_states[-1]
        return possible_next_states[non_deterministic_transition]
    
    def get_max_nondeterminism(self):
        max_transitions = 0
        aps = [str(i) for i in self.spot_automaton.ap()]
        possible_labels = get_possible_labels(aps)
        for state in range(0,self.spot_automaton.num_states()):
            for label in possible_labels:
                count = 0
                for edge in self.spot_automaton.out(state):
                    condition = spot.bdd_format_formula(self.spot_automaton.get_dict(), edge.cond)
                    if eval(convert(condition),label):
                        count+=1
                if count>max_transitions:
                    max_transitions = count
        return max_transitions
    
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


if __name__=='__main__':
    spot.setup()
    a = spot.translate('(a U b) & GFb ', 'Buchi', 'state-based', 'det'); 
    print(a.to_str('dot'))
    print(a.get_init_state())
    b = sim_automaton(a)
    b.accepting_set
    b.graph.nodes