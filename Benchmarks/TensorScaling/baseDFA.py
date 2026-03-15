# Auto-generated portable DFA snapshot

DFA_DATA = {'AP': ['p1'], 'S': [0, 1], 'S0': [1], 'F': [[0]], 'nr_states': 2, 'automaton_state': 1, 'delta': [{}], 'transitions': [(0, 0, {'label': '1', 'condition': '1'}), (1, 0, {'label': 'p1', 'condition': 'p1'})], 'graph_data': {'directed': True, 'multigraph': True, 'graph': {}, 'nodes': [{'id': 0}, {'id': 1}], 'links': [{'label': '1', 'condition': '1', 'source': 0, 'target': 0, 'key': 0}, {'label': 'p1', 'condition': 'p1', 'source': 1, 'target': 0, 'key': 0}]}, 'spot_hoa': 'HOA: v1\nStates: 2\nStart: 1\nAP: 1 "p1"\nacc-name: Buchi\nAcceptance: 1 Inf(0)\nproperties: trans-labels explicit-labels state-acc deterministic\nproperties: stutter-invariant terminal very-weak\n--BODY--\nState: 0 {0}\n[t] 0\nState: 1\n[0] 0\n--END--'}

class SimpleDFA:
    def __init__(self, data):
        self.AP = data.get("AP", [])
        self.S = data.get("S", [])
        self.S0 = data.get("S0", [])
        self.F = data.get("F", [])
        self.nr_states = data.get("nr_states", 0)
        self.automaton_state = data.get("automaton_state", 0)
        self.delta = data.get("delta", None)
        self.transitions = data.get("transitions", [])
        self.graph_data = data.get("graph_data", None)
        self.spot_hoa = data.get("spot_hoa", None)  # HOA text (no spot needed)

        # Best-effort networkx graph reconstruction (only if networkx is installed)
        self.graph = None
        if self.graph_data is not None:
            try:
                import networkx as nx
                self.graph = nx.node_link_graph(self.graph_data)
            except Exception:
                self.graph = None

def load_dfa(backend="simple"):
    """backend:
      - 'simple' (default): no dependency on spot or your project code
      - 'original': tries to rebuild src.specifications.translate.automaton (may fail if spot missing)
    """
    data = DFA_DATA

    if backend == "original":
        # This may fail on machines without spot if your package imports spot at import-time.
        from src.specifications.translate import automaton as Automaton

        try:
            dfa = Automaton()
        except Exception:
            dfa = Automaton.__new__(Automaton)

        for k in ["AP","S","S0","F","nr_states","automaton_state","delta","transitions"]:
            if k in data:
                try:
                    setattr(dfa, k, data[k])
                except Exception:
                    pass

        # graph
        if data.get("graph_data") is not None:
            try:
                import networkx as nx
                dfa.graph = nx.node_link_graph(data["graph_data"])
            except Exception:
                pass

        # spot (optional; will be skipped if spot isn't installed)
        if data.get("spot_hoa"):
            try:
                import spot
                dfa.spot_automaton = spot.automaton(data["spot_hoa"])
            except Exception:
                pass

        return dfa

    # default: portable object that always loads
    return SimpleDFA(data)
