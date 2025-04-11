### Factored Markov Decision Process simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
from MRF import MarkovRandomField, MRF

class FMDP:
    """
    MDP factored into components.
    Easily converted from LAS and MRF structures.
    
    For the purpose of consistency, any identifier of (u, v) 
    which is meant to be unordered (a set)
    will maintain the assumption that u < v
    
    Initialize an empty FMDP with the following information: 
    - V: Set of actions
    - loc(v): Dictionary (action : list of components it is a member of)
    - S Components: Dictionary (() : list in form [x_u, x_v, (token direction)])
        - Note: the list is meant to be a set, but for the sake of accessibility it will be
                assumed that the index of u = 0, index of v = 1, and index of token direction = 2;
                further, it will be assumed u < v in the list and component
    - Q: queue (FIFO) of enabled actions
    - T: Dictionary (action : tuple(neighbor-count, token-count))
        - We can use these values to easily check if an action should be enabled
          (Action = enabled iff neighbor-count == token-count)
    """
    def __init__(self):
        self._actions = set()         # Actions in FMDP
        self._components = {}         # Map of set{u, v} : [val(u), val(v), direction of token (tuple)]
        self._edges = {}              # Edges dict(action : set(actions))
        self._queue = list()          # Queue of enabled actions
        self._tokens = {}             # Map of actions -> # of neighbors and # of tokens
        self._cpt = {}                # CPT

    def add_action(self, action: int) -> None:
        """Add an action to the set"""
        self._actions.add(action)

    def add_component(self, edge: set, uval: int, vval: int, dir: tuple) -> None:
        """Add a component to the map -> set(u, v) : [uval, vval, (u, v)/(v, u)]
        
        Args:
            edge: a set {u, v} representing the edge
            uval: value of action u
            vval: value of action v
            dir: direction of token as a tuple (u, v) or (v, u)
        """
        self._components[set()]

    def add_loc(self, u: int, v: int) -> None:
        """Add the connection (u, v) to edges
        
        Args:
            u: primary action
            v: connection action to primary action
        """
        if u not in self._edges:
            self._edges[u] = set()
        
        self._edges[u].add(v)

# TODO: joint distribution estimates by fixing strategy of FMDP and sampling sufficiently long paths
        