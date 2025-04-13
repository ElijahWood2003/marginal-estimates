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
        self._components = {}         # Map of frozenset{u, v} : [val(u), val(v), direction of token (tuple)]
        self._edges = {}              # Edges dict(action : list of connected actions])
        self._queue = list()          # Queue of enabled actions
        self._tokens = {}             # Map of actions -> # of neighbors and # of tokens
        self._cpt = {}                # CPT
        self._values = {}             # Dictionary (action : current value)

    def add_action(self, action: int) -> None:
        """Add an action to the set"""
        self._actions.add(action)

    def add_component(self, edge: set, dir: tuple) -> None:
        """Add a component to the map -> set(u, v) : [uval, vval, (u, v)/(v, u)]
        
        Args:
            edge: a set {u, v} representing the edge
            uval: value of action u
            vval: value of action v
            dir: direction of token as a tuple (u, v) or (v, u)
        """
        # As stated above, u < v
        u = dir[int(dir[0] > dir[1])]
        v = dir[int(dir[0] < dir[1])]

        # Ensure the pointer to the values map is stored, not the value itself
        # This means whenever we update the values within the value map each of the component values will change inherently
        self._components[edge] = [self._values[u], self._values[v], dir]

    def add_edge(self, u: int, v: int) -> None:
        """Add the connection (u, v) to edges
        
        Args:
            u: the input edge
            v: the output edge
        """
        if u not in self._edges:
            self._edges[u] = []
        
        self._edges[u].append(v)
    
    def set_value(self, u: int, val: int) -> None:
        """Set the value of action u to the val
        
        Args:
            u: the action we want to set the value of
            val: the value we want to set the action to
        """
        self._values[u] = val


# TODO: joint distribution estimates by fixing strategy of FMDP and sampling sufficiently long paths
        