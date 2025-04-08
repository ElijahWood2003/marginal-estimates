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
    
    For the purpose of clarity, any identifier of (u, v) 
    which is meant to be unordered (a set)
    will maintain the assumption that u < v
    
    Initialize an empty FMDP with the following information: 
    - V: Set of actions
    - P: Set of components as tuples (u, v)
    - loc(v): Dictionary (action : list of components it is a member of)
    - S: Dictionary (Component : list in form [x_u, x_v, (token direction)])
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
        self._components = set()      # Components of FMDP
        self._loc = {}                # Action -> Component map
        self._states = {}             # Component -> Token direction map
        self._queue = list()          # Queue of enabled actions
        self._tokens = {}             # Map of actions -> # of neighbors and # of tokens


# TODO: joint distribution estimates by fixing strategy of FMDP and sampling sufficiently long paths
        