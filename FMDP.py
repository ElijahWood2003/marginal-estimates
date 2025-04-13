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
        self._queue = []              # Queue of enabled actions
        self._cpt = {}                # CPT
        self._domains = {}            # Domains as dict(action : list(values)) for each action

        self._values = {}             # Dictionary (action : value)
        self._tokens = {}             # Dictionary (tuple(u, v) : {0, 1})
        self._token_count = {}        # Map of actions -> [# of neighbors, # of tokens]  

        # Constants
        self.LOC_NEIGHBOR_COUNT = 0
        self.LOC_TOKEN_COUNT = 1

    def add_action(self, action: int, domain = List) -> None:
        """Add an action to the set"""
        self._actions.add(action)
        self._token_count[action] = [0, 0]
        self._domains[action] = domain

    def add_component(self, edge: set, dir: tuple) -> None:
        """Add a component to the map -> set(u, v) : [uval, vval, (u, v)/(v, u)]
        
        Args:
            edge: a set {u, v} representing the edge
            dir: direction of token as a tuple (u, v) or (v, u)
        """
        # As stated in the class notes, for consistency u < v
        u = dir[int(dir[0] > dir[1])]
        v = dir[int(dir[0] < dir[1])]

        # Ensure the pointer to the values map is stored, not the value itself
        # This means whenever we update the values within the value map each of the component values will change inherently
        self._components[frozenset(edge)] = [self._values[u], self._values[v], dir]

    def add_edge(self, u: int, v: int) -> None:
        """
        Add the connection (u, v) to edges
        
        Args:
            u: the input edge
            v: the output edge
        """
        if u not in self._edges:
            self._edges[u] = []
        
        self._edges[u].append(v)

        # Increment the action's number of neighbors by 1
        self._token_count[u][self.LOC_NEIGHBOR_COUNT] += 1
    
    def set_value(self, u: int, val: int) -> None:
        """
        Set the value of action u to the val
        
        Args:
            u: the action we want to set the value of
            val: the value we want to set the action to
        """
        self._values[u] = val
    
    def set_actions(self, actions: set) -> None:
        """Set the actions based on input set"""
        self._actions = actions
    
    def set_edges(self, edges: dict) -> None:
        """
        Set the edges based on the input dictionary

        Args:
            edges: a dictionary of edges assumed to be an adjacency list 
        """
        self._edges = edges
    
    def set_components(self, tokens: dict) -> None:
        """
        Set the components of the FMDP based on 
        the input tokens
        
        Args:
            tokens: dictionary (tuple(u, v) : {0, 1}) 
        """
        for (u, v) in tokens:
            # Check whether a token exists at this location
            if(tokens[(u, v)] == 1):
                self._token_count[v][self.LOC_TOKEN_COUNT] += 1

                # If # of tokens == # of out-neighbors then we add it to the queue of enabled actions
                if(self._token_count[v][self.LOC_TOKEN_COUNT] == self._token_count[v][self.LOC_NEIGHBOR_COUNT]):
                    self._queue.append(v)

                # self.add_component({u, v}, (u, v))
    
    def set_cpt(self, cpt: dict) -> None:
        """Set CPT table based on an input CPT"""
        self._cpt = cpt
            

    def activate_action(self, action: int) -> int:
        """
        Args:
            action: the action to activate
        
        Returns:
            The new value of this action for easy sampling

        Activate an action:
            1. Determine new value of action based on CPT
            2. Set _token_count[action][token_count] = 0
            3. For each out-neighbor, increase their # of tokens by 1 and check for new enabled action 
        """
        # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
        neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
        for a in self._edges[action]:
            neighbor_set.add((a, self._values[a]))

            self._token_count[a][self.LOC_TOKEN_COUNT] += 1
            if(self._token_count[a][self.LOC_TOKEN_COUNT] == self._token_count[a][self.LOC_NEIGHBOR_COUNT]):
                self._queue.append(a)
        
        self._token_count[action][self.LOC_TOKEN_COUNT] = 0

        probabilities = self._cpt[action][frozenset(neighbor_set)]
        value = np.random.choice(probabilities.keys(), p=probabilities.values())
        self._values[action] = value

        return value


        



# TODO: joint distribution estimates by fixing strategy of FMDP and sampling sufficiently long paths
        