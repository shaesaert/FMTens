import networkx as nx
import numpy as np
import pandas as pd
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

class dfa_tree:
    """
    A class for creating and manipulating tree structures based on DFA (Deterministic Finite Automaton)
    """
    
    def __init__(self, dfa):
        """
        Initialize the DFA tree
        
        Args:
            dfa: The DFA object containing transition information
        """
        self.dfa = dfa
        self.tree = None
        self.leafs = None
        
    def initiate(self):
        """
        Create tree with accepting node and its children
        """
        # Find transitions to accepting state
        E = self.dfa.graph.in_edges(nbunch=self.dfa.F[0],data=True)
        s = [] # empty sources
        t = [] # empty targets
        l = [] # empty labels
        q = self.dfa.F[0] #  dfa states, accepting state is first state

        for e in E:
            if e[0] not in self.dfa.F[0]:
                s += [0] # source is always 0 (accepting state)
                l.append(e[2]['label'])
                q.append(e[0])  # add the DFA state to the list of states

        
        # Create target nodes for edges
        t = np.cumsum(np.ones(len(s), dtype=int)) # equal to the number of 
        
        
        # Create edge and node tables using pandas
        edge_table = pd.DataFrame({
            'source': s,
            'target': t,
            'l': l
        })
        node_table = pd.DataFrame({'q': q})
        
        # Create directed graph using networkx
        self.tree = nx.DiGraph()
        
        # Add nodes with their states
        for idx, row in node_table.iterrows():
            self.tree.add_node(idx, q=row['q'])
            
        # Add edges with their labels
        for _, row in edge_table.iterrows():
            self.tree.add_edge(row['source'], row['target'], l=row['l'])
            
        # Store leaf nodes
        self.leafs = list(t)
        
    def Lq(self, n):
        """Get the state of node n"""
        return self.tree.nodes[n]['q']
        
    def grow(self):
        """Add children to all leaf nodes of the graph"""
        leafs_old = self.leafs.copy()
        for n in leafs_old:
            self.growleaf(n)
            
    def find_subtree(self, n, node_ids=None):
        """
        Recursively find all nodes in the subtree starting from node n
        
        Args:
            n: Starting node
            node_ids: List to accumulate node IDs (default: None)
            
        Returns:
            List of node IDs in the subtree
        """
        if node_ids is None:
            node_ids = []
            
        successors = list(self.tree.successors(n))
        if successors:
            for n_next in successors:
                node_ids = self.find_subtree(n_next, node_ids)
                
        node_ids.append(n)
        return node_ids
    
    def remove_branch(self, n):
        """
        Remove branch starting from node n
        
        Args:
            n: Root node of branch to remove
        """
        node_ids = self.find_subtree(n, [])
        
        for node in node_ids:
            self.tree.remove_node(node)
            if node in self.leafs:
                self.leafs.remove(node)
            # Adjust leaf indices for removed nodes
            self.leafs = [x - 1 if x > node else x for x in self.leafs]
            
    def growleaf(self, n):
        """
        Add children to a specific leaf node
        
        Args:
            n: Node ID to grow
        """
        if n not in self.leafs:
            raise ValueError('Node is not a leaf node')
            
        max_node = len(self.tree.nodes)
        q_source = self.Lq(n)
        s = [] # empty sources
        t = [] # empty targets
        l = [] # empty labels
        q = [] # empty q

        # Find possible children
        E = self.dfa.graph.in_edges(nbunch=q_source,data=True)
        q= []
        for e in E:
            if e[0] not in self.dfa.F:
                s += [n] # leaf from which we grow
                l.append(e[2]['label'])
                q.append(e[0])  # add the DFA state to the list of states


        # Add new nodes
        for idx, state in enumerate(q):
            new_node = max_node + idx
            self.tree.add_node(new_node, q=state)
            self.tree.add_edge(n, new_node, l=l[idx])
            
        # Update leaf nodes
        self.leafs.remove(n)
        self.leafs.extend(range(max_node, max_node + len(q)))
        
    def plot(self,ax=None):
        """
        Plot the tree structure      
        Args:
        """
        edge_labels = {(u, v): d['l']  for u, v, d in self.tree.edges(data=True)}  
        node_labels = {n: d['q'] for n, d in self.tree.nodes(data=True)}

         # Works with arc3 and angle3 connectionstyles
        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
        # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]
        G = self.tree
        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos,  ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax)
        labels = {tuple(edge): f"{attrs['l']}" for *edge, attrs in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G,pos,labels,connectionstyle=connectionstyle,label_pos=0.3,font_color="black",bbox={"alpha": 0.5},ax=ax)
  

    @staticmethod
    def num2label(act, node_ls):
        """
        Convert numeric labels to AP based labels
        
        Args:
            act: List of actions
            node_ls: List of node labels
            
        Returns:
            List of converted labels
        """
        return [act[i] for i in node_ls]
