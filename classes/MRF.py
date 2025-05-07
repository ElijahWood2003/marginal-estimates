### Markov Random Field simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
import time

# TODO: Make nodes anything not just integers
class MarkovRandomField:
    def __init__(self):
        """
        Initialize an empty MRF based on the formal definition:
        - G = (V, A) where V is set of vertices and A is set of undirected edges
        - Each node v ∈ V has random variable X_v with finite domain
        - Neighborhood function N: V → 2^V
        - CPTs for each node: Pr_v(x_i | x_N(i))
        """
        self._vertices = set()   # V
        self._edges = set()      # A (set of frozenset pairs)
        self._neighbors = defaultdict(set)  # N(v)
        self._domains = {}       # 𝒳 for each variable
        self._cpts = {}          # CPT
        
    def add_vertex(self, v: int, domain: List) -> None:
        """Add a vertex (node) with its possible values"""
        if (v in self._vertices):
            raise ValueError(f"Vertex {v} already exists")
            
        self._vertices.add(v)
        self._domains[v] = domain
        self._cpts[v] = {}
        
    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge between vertices u and v"""
        if (u not in self._vertices or v not in self._vertices):
            raise ValueError("Both vertices must exist")
            
        edge = frozenset({u, v})
        if (edge not in self._edges):
            self._edges.add(edge)
            self._neighbors[u].add(v)
            self._neighbors[v].add(u)

    def set_cpt(self, v: int, neighbor_config: Dict[int, int], probabilities: Dict[int, float]) -> None:
        """
        Set conditional probability table for vertex v
        
        Args:
            v: Target vertex
            neighbor_config: Dictionary {neighbor : value} specifying the condition
            probabilities: Dictionary {value : probability} for vertex v
        """
        if (v not in self._vertices):
            raise ValueError(f"Vertex {v} does not exist")
            
        # Validate neighbor configuration
        for neighbor, value in neighbor_config.items():
            if (neighbor not in self._neighbors[v]):
                raise ValueError(f"{neighbor} is not a neighbor of {v}")
            if (value not in self._domains[neighbor]):
                raise ValueError(f"Invalid value {value} for neighbor {neighbor}")
                
        # Ensure the probabilities are a distribution
        if not (np.isclose(sum(probabilities.values()), 1.0, atol=1e-6)):
            raise ValueError("Probabilities must sum to 1")

        # Ensure our values exist within the domain for the random variable
        for value, prob in probabilities.items():
            if (value not in self._domains[v]):
                raise ValueError(f"Invalid value {value} for vertex {v}")
                
        # Use frozenset for hashable neighbor configuration
        config_key = frozenset(neighbor_config.items())
        self._cpts[v][config_key] = probabilities.copy()
        
    def get_conditional_probability(self, v: int, value: int, neighbor_values: Dict[int, int]) -> float:
        """
        Get Pr(X_v = value | X_N(v) = neighbor_values)
        
        Args:
            v: Target vertex
            value: Value of the target vertex
            neighbor_values: Dictionary {neighbor: value}
            
        Returns:
            Conditional probability
        """
        config_key = frozenset(neighbor_values.items())
        return self._cpts[v].get(config_key, {}).get(value, 0.0)
        
    def joint_probability(self, configuration: Dict[int, int]) -> float:
        """
        Compute the joint probability Pr(x) = ∏_i Pr_i(x_i | x_N(i))
        
        Args:
            configuration: Dictionary {vertex: value} representing a global configuration
            
        Returns:
            Joint probability
        """
        prob = 1.0
        for v in self._vertices:
            neighbors = self._neighbors[v]
            neighbor_values = {n: configuration[n] for n in neighbors}
            cond_prob = self.get_conditional_probability(v, configuration[v], neighbor_values)
            prob *= cond_prob
        return prob
        
    def gibbs_sample(self, initial_config: Dict[int, int], num_samples: int = 1000, burn_in: int = 100) -> List[Dict[int, int]]:
        """
        Perform Gibbs sampling to generate samples from the MRF
        
        Args:
            initial_config: Starting configuration
            num_samples: Number of samples to generate
            burn_in: Number of burn-in iterations
            
        Returns:
            List of sampled configurations
        """
        samples = []
        current_config = initial_config.copy()
        
        for i in range(burn_in + num_samples):
            # Visit vertices in random order
            for v in np.random.permutation(list(self._vertices)):
                neighbors = self._neighbors[v]
                neighbor_values = {n: current_config[n] for n in neighbors}
                
                # Get conditional distribution
                config_key = frozenset(neighbor_values.items())
                cond_dist = self._cpts[v].get(config_key, None)
                
                if (cond_dist is None):
                    # If no CPT entry exists, use uniform distribution
                    cond_dist = {val: 1.0/len(self._domains[v]) for val in self._domains[v]}
                
                # Sample new value
                values, probs = zip(*cond_dist.items())
                new_value = np.random.choice(values, p=probs)
                current_config[v] = new_value

            # Only add to samples if we are past burn in value
            if (i >= burn_in):
                samples.append(current_config.copy())
                
        return samples
    
    def marginal_probability(self, v: int, value: int, num_samples: int = 10000) -> float:
        """
        Estimate marginal probability Pr(X_v = value) using Gibbs sampling
        
        Args:
            v: Vertex of interest
            value: Value to estimate probability for
            num_samples: Number of samples to use for estimation
            
        Returns:
            Estimated marginal probability
        """
        # Start with random configuration
        initial_config = {vertex: np.random.choice(self._domains[vertex]) 
                         for vertex in self._vertices}
        
        samples = self.gibbs_sample(initial_config, num_samples=num_samples)
        count = sum(1 for sample in samples if sample[v] == value)
        return count / num_samples

    def auto_propagate_cpt(self) -> None:
        """
        Automatically propagate the conditional probability
        table with random values based on a default potential function

        NOTE: this overwrites all cpts
        """
        cpts = defaultdict(dict)

        for v in self._vertices:
            neighbors = self._neighbors[v]
            domain = self._domains[v]

            # Generate all possible neighbor configurations
            neighbor_domains = [self._domains[len(domain)] for _ in neighbors]
            for config in product(*neighbor_domains):
                neighbor_assign = dict(zip(neighbors, config))

                # Create random distribution with neighbor influence
                if(neighbors):
                    # Base probabilities with some neighbor influence
                    probs = np.ones(len(domain))

                    # For each neighbor, we want to increase the probability of matching its values
                    for n, n_val in neighbor_assign.items():
                        match_idx = domain.index(n_val)
                        probs[match_idx] += 0.7 * np.random.uniform(0.5, 1.5)

                    # Normalize to valid distribution
                    probs /= probs.sum()
                else:
                    # No neighbors - uniform distribution
                    probs = np.ones(len(domain) / len(domain))
                
                # Store with frozenset for hashable dict key
                config_key = frozenset(neighbor_assign.items())
                cpts[v][config_key] = dict(zip(domain, probs))
        
        self._cpts = cpts


    # Additional utility methods
    def vertices(self) -> Set[int]:
        """Get set of all vertices"""
        return self._vertices.copy()
        
    def edges(self) -> Set[FrozenSet[int]]:
        """Get set of all edges"""
        return self._edges.copy()
        
    def neighbors(self, v: Any) -> Set[int]:
        """Get neighbors of vertex v"""
        return self._neighbors[v].copy()
        
    def domain(self, v: Any) -> List[int]:
        """Get possible values for vertex v"""
        return self._domains[v].copy()
    
    def get_domains(self) -> dict:
        """Get all domains across the dictionary"""
        return self._domains
    
    def get_cpts(self) -> dict:
        """Get cpts of MRF"""
        return self._cpts