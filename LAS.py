### Live and Safe Marked Graph simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
from MRF import MarkovRandomField, MRF

class LiveAndSafe:
    """
    Directed graph with tokens on edges representing activation points
    For our purposes it will be strictly strongly connected graphs

    Initialize with the following information:
    - V: vertices labeled 0 -> n
    - E: edges in an adjacency list
    - M_in: E -> {0, 1}
    """
    def __init__(self):
        self._vertices = set()   # V
        self._edges = {}         # E (as adjacency list)
        self._tokens = {}        # M dict((u, v) : {0, 1}) where 1 represents a token at edge (u, v)
    
    def add_vertex(self, v: int) -> None:
        """Add a vertex to the set"""
        self._vertices.add(v)
    
    def add_edge(self, u: int, v: int) -> None:
        """Add edge (u, v) to E"""
        if u not in self._edges:
            self._edges[u] = []
            
        self._edges[u].append(v)
        
    def set_vertices(self, vertices: set) -> None:
        """Set the vertices based on input set"""
        self._vertices = vertices
        
    def set_edges(self, edges: set) -> None:
        """
        Set the edges based on the input edges set
        
        Simultaneously sets tokens based on the following:
         - Iterate through all edges (u, v)
         - If u > v set _tokens[(u, v)] = 1
         - Otherwise -> _tokens[(u, v)] = 0
        This sets tokens based on an acyclic orientation
        
        Args:
            edges: set of frozenset pairs of edges (for easy MRF -> LAS conversion)
        """
        for edge in edges:
            # Edge (u, v)
            edge_list = []
            for v in edge:
                edge_list.append(v)
                
            u = edge_list[0]
            v = edge_list[1]
            
            # Add an edge for both direction
            self.add_edge(u, v)
            self.add_edge(v, u)
            
            # Set tokens based on acyclic orientation
            self._tokens[(u, v)] = int(u > v)
            self._tokens[(v, u)] = int(u < v)


    # Example LAS based on 4x3 Neighborhood MRF
# Initialize LAS
LAS = LiveAndSafe()

# Use MRF to add vertices / edges
LAS.set_vertices(MRF.vertices())
LAS.set_edges(MRF.edges())