### Factored Markov Decision Process simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
from MRF import MarkovRandomField, MRF
import time

class FactoredMarkovDecisionProcess:
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
        self._cpts = {}                # CPT
        self._domains = {}            # Domains as dict(action : list(values)) for each action

        self._values = {}             # Dictionary (action : value)
        self._tokens = {}             # Dictionary (tuple(u, v) : {0, 1})
        self._token_count = {}        # Map of actions -> [# of neighbors, # of tokens]  

        # Constants
        self.NEIGHBOR_COUNT = 0
        self.TOKEN_COUNT = 1

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
        self._token_count[u][self.NEIGHBOR_COUNT] += 1
    
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
        for action in self._actions:
            self._token_count[action] = [0, 0]

    def set_domains(self, domains: dict) -> None:
        """Set the domains based on the input dict"""
        self._domains = domains
    
    def set_edges(self, edges: dict) -> None:
        """
        Set the edges based on the input dictionary

        Args:
            edges: a dictionary of edges assumed to be an adjacency list 
        """
        self._edges = edges
        for action in self._edges.keys():
            self._token_count[action][self.NEIGHBOR_COUNT] = len(self._edges[action])
    
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
                self._token_count[v][self.TOKEN_COUNT] += 1

                # If # of tokens == # of out-neighbors then we add it to the queue of enabled actions
                if(self._token_count[v][self.TOKEN_COUNT] == self._token_count[v][self.NEIGHBOR_COUNT]):
                    self._queue.append(v)

                # self.add_component({u, v}, (u, v))
    
    def set_cpts(self, cpts: dict) -> None:
        """Set CPT table based on an input CPT"""
        self._cpts = cpts
            
    def set_random_values(self) -> None:
        """Sets random initial values for each action"""
        if(self._domains):
            for action, domain in self._domains.items():
                self._values[action] = np.random.choice(domain)

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
            neighbor_set.add((a, int(self._values[a])))

            self._token_count[a][self.TOKEN_COUNT] += 1
            if(self._token_count[a][self.TOKEN_COUNT] == self._token_count[a][self.NEIGHBOR_COUNT]):
                self._queue.append(a)
        
        self._token_count[action][self.TOKEN_COUNT] = 0

        probabilities = self._cpts[action][frozenset(neighbor_set)]
        domains, probs = zip(*probabilities.items())
        value = np.random.choice(domains, p=probs)
        self._values[action] = value

        return value
    
    def sample(self, num_samples: int = 1000, burn_in: int = 100, initial_config: dict[int, int] = None) -> List[Dict[int, int]]:
        """
        Sample the FMDP
        
        Args:
            initial_config: Starting configuration
            num_samples: Number of samples to generate
            burn_in: Number of burn-in iterations
            
        Returns:
            List of sampled configurations
        """
        if(initial_config):
            self._values = initial_config

        samples = []
        current_config = self._values.copy() 
        
        for i in range(burn_in + num_samples):
            action = self._queue.pop(0)
            new_value = self.activate_action(action)
            current_config[action] = new_value

            # Only add to samples if we are past burn in value
            if (i >= burn_in):
                samples.append(current_config.copy())
                
        return samples
    
    def marginal_probability(self, a: int, value: int, num_samples: int = 10000) -> float:
        """
        Estimate marginal probability Pr(X_v = value) of FMDP action
        
        Args:
            a: Action of interest
            value: Value to estimate probability for
            num_samples: Number of samples to use for estimation
            
        Returns:
            Estimated marginal probability
        """
        
        samples = self.sample(num_samples=num_samples)
        count = sum(1 for sample in samples if sample[a] == value)
        return count / num_samples


        # Example FMDP based on 4x3 Neighborhood MRF
from MRF import MRF
from LAS import LAS

# FMDP class
FMDP = FactoredMarkovDecisionProcess()
FMDP.set_actions(MRF.vertices())
FMDP.set_edges(LAS.get_edges())
FMDP.set_components(LAS.get_tokens())
FMDP.set_domains(MRF.get_domains())
FMDP.set_cpts(MRF.get_cpts())
FMDP.set_random_values()

# Testing and tracking time
num_samples = 10000
start = time.perf_counter()
prob = FMDP.marginal_probability(0, 0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"FMDP sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")
        