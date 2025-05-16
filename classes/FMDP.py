### Factored Markov Decision Process simulator class
import numpy as np
from typing import Dict, List, Tuple, Set, FrozenSet, Any
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
        self._cpts = {}               # CPT
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

    # NOTE: Arbitrary method (less efficient than methods below)
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
    
    # NOTE: Arbitrary method
    def sample(self, num_samples: int = 1000, burn_in: int = 100, initial_config: Dict[int, int] = None) -> List[Dict[int, int]]:
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

    def derive_activation(self, initial_action: int) -> List[int]:
        """
        Run BFS to derive depths of all actions
        Then output as a list the desired sequence
        
        Derives the activation sequence to maximize the activations of the initial action
        while minimizing the number of total activations

        Args:
            initial_action: First action (acyclic orientation 
                of the tokens should be pointing towards this action)

        Returns:
            List[int]: The order of actions to take to return to the same initial token state
        """
        depth = {}          # Dictionary to track depths for each action
        depth_list = []     # Depth list to use for activation sequence
        
        depth[initial_action] = 0
        depth_list.append([initial_action])
        
        queue = []
        queue.append(initial_action)
        
        while(queue):
            s = queue.pop(0)
            
            # Iterate through all edges of queued action
            for action in self._edges[s]:
                if action not in depth:
                    queue.append(action)
                    depth[action] = depth[s] + 1

                    # Check if the index already exists
                    if(depth[action] >= len(depth_list)):
                        depth_list.append([action])
                    else:
                        depth_list[depth[action]].append(action)
        
        # Activation sequence (list of actions to return)
        activation_sequence = []

        # Now we have the depth list, iterate through to derive activation sequence
        for i in range(len(depth_list)):
            # Iterate backwards from i
            for k in range(i, -1, -1):
                # For each depth we need to activate each action
                for action in depth_list[k]:
                    activation_sequence.append(action)
        
        # Exclude the last index which will be the target action since we are looping
        return activation_sequence[0: len(activation_sequence) - 1]
    
    def joint_distribution(self, action_samples: int = 1000, burn_in: int = 100, time_limit: int = -1, initial_config: Dict[int, int] = None) -> Dict[tuple, int]:
        """
        Estimate the joint distribution by using gibbs sampling,
        keeping track of the number of times each global state is observed
        Picks an arbitrary fixed activation sequence

        Args:
            action_samples: The # of samples of the target_action (total # of sampled values = action_samples * # of actions)
            burn_in: The number of samples to burn before sampling
            time_limit: If positive, limits the sampling based on the time rather than # of samples (in seconds)
            initial_config: Initial global configuration

        Returns:
            joint distribution as a dictionary:
                values(x1, x2, ... xn) : # of times this global state has been observed

        """
        if(initial_config):
            self._values = initial_config

        activation_order = np.random.permutation(list(self._actions))
        order_length = len(activation_order)

        samples = {} 
        current_config = self._values.copy() 

        # Tuple of all values where each index represents the value for its respective action
        max_key = max(current_config.keys())
        values = tuple(current_config[i] for i in range(max_key + 1))
        
        # If time limit is positive then limit based on time not sample count
        if(time_limit > -1):
            start = time.perf_counter()
            current = start
            sample_count = 0
            
            while(current - start < time_limit):
                action = activation_order[sample_count % order_length]
                
                # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
                neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
                for a in self._edges[action]:
                    neighbor_set.add((a, int(current_config[a])))

                probabilities = self._cpts[action][frozenset(neighbor_set)]
                domains, probs = zip(*probabilities.items())
                value = np.random.choice(domains, p=probs)

                # Convert values to list, mutate, then convert back to tuple
                current_config[action] = value
                value_list = list(values)
                value_list[action] = value
                values = tuple(value_list)

                # Only add to samples if we are past burn in value
                if (sample_count >= burn_in):
                    # Key is values
                    if values in samples:
                        samples[values] += 1
                    else: 
                        samples[values] = 1
                
                sample_count += 1
                        
                # Only update current every 10 samples to save time
                if(sample_count % 10 == 0):
                    current = time.perf_counter()
                
            return samples 

        # Total Number of samples = action_samples * total # of actions
        num_samples = action_samples * len(self._actions)
        
        for i in range(burn_in + num_samples):
            action = activation_order[i % order_length]
            
            # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
            neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
            for a in self._edges[action]:
                neighbor_set.add((a, int(current_config[a])))

            probabilities = self._cpts[action][frozenset(neighbor_set)]
            domains, probs = zip(*probabilities.items())
            value = np.random.choice(domains, p=probs)

            # Convert values to list, mutate, then convert back to tuple
            current_config[action] = value
            value_list = list(values)
            value_list[action] = value
            values = tuple(value_list)

            # Only add to samples if we are past burn in value
            if (i >= burn_in):
                # Key is values
                if values in samples:
                    samples[values] += 1
                else: 
                    samples[values] = 1
                
        return samples

    def joint_distribution_to_marginal_probability(self, joint_distribution: Dict[tuple, int], action: int, value: int) -> float:
        """
        Returns the marginal probability of the action given the joint distribution

        Args:
            joint_distribution: The joint distribution as a Dict[tuple, int]
            action: The action we are marginalizing 
            value: The value of the action we want to know the probability of

        Returns:
            float: The probability the action has the given value
        """
        num_samples = 0
        sum = 0

        # Iterate through every combination of the joint distribution
        for key, v in joint_distribution.items():
            sum += v * (key[action] == value)   # Add to the sum iff the value is what we are looking for
            num_samples += v

        return sum / num_samples 
    
    def gibbs_sampling(self, action: int, value: int, action_samples: int = 1000, burn_in: int = 100, time_limit: int = -1, initial_config: Dict[int, int] = None) -> float:
        """
        Returns the marginal distribution

        Args:
            action (int): The action we are marginalizing 
            value (int): The action we are marginalizing 
            num_samples: The number of samples to use for distribution
            burn_in: The number of samples to burn before sampling
            time_limit: If positive, limits the sampling based on time rather than the number of samples (in seconds)
            initial_config: Initial global configuration

        Returns:
            marginal distribution (float): The estimated probability that the action has the given value
            run time (float): The total running time of this function
        """
        start = time.perf_counter()
        
        joint_distribution = self.joint_distribution(action_samples=action_samples, burn_in=burn_in, time_limit=time_limit, initial_config=initial_config)
        
        end = time.perf_counter()
        run_time = end - start
        
        return self.joint_distribution_to_marginal_probability(joint_distribution=joint_distribution, action=action, value=value), run_time

    def gibbs_sampling_delta(self, target_action: int, target_value: int, delta: float = 0.0001, sample_period: int = 465000, minimum_samples: int = 30000000, initial_config: Dict[int, int] = None) -> float:
        """
        Estimate the marginal distribution by using gibbs sampling,
        keeping track of the number of times each global state is observed
        Picks an arbitrary fixed activation sequence

        This function is used to find the 'ground truth' of a sampling distribution.
        It takes the marginal distribution every sample period # of samples, then
        compares it to the previous sampling period. If the difference between these
        two samples is less than the delta value, it will return this distribution.

        Args:
            target_action: The action we want to marginalize
            target_value: The value of the action we want to marginalize for P(target_action == target_value)
            delta: The difference required between samples to return the distribution
            sample_period: The amount of samples between sampling periods (465000 ~ 5 seconds on Mac)
            minimum_samples: The minimum amount of samples required
            initial_config: Initial global configuration

        Returns:
            estimate of the marginal distribution as a float
            total running time as a float

        """
        start = time.perf_counter()
        
        if(initial_config):
            self._values = initial_config

        activation_order = np.random.permutation(list(self._actions))
        order_length = len(activation_order)

        count = 0
        current_config = self._values.copy()
        
        # Our two distributions we will compare the delta
        d1 = 0.0
        d2 = 1.0

        # Tuple of all values where each index represents the value for its respective action
        max_key = max(current_config.keys())
        values = tuple(current_config[i] for i in range(max_key + 1))

        sample_count = 0
            
        while(abs(d1 - d2) > delta or sample_count < minimum_samples):
            action = activation_order[sample_count % order_length]
            
            # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
            neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
            for a in self._edges[action]:
                neighbor_set.add((a, int(current_config[a])))

            probabilities = self._cpts[action][frozenset(neighbor_set)]
            domains, probs = zip(*probabilities.items())
            value = np.random.choice(domains, p=probs)

            # Convert values to list, mutate, then convert back to tuple
            current_config[action] = value
            value_list = list(values)
            value_list[action] = value
            values = tuple(value_list)

            # Add to count iff target_action == target_value
            count += (current_config[target_action] == target_value) 
            
            sample_count += 1

            # Check if we are at a new sample period
            if(sample_count % sample_period == 0):
                d1 = d2
                d2 = count / sample_count
        
        end = time.perf_counter()
        run_time = end - start
        
        return d2, run_time

    def token_sampling(self, target_action: int, target_value: int, action_samples: int = 1000, burn_in: int = 100, time_limit: int = -1) -> float:
        """
        Returns an estimate of the marginal probability P(action == value) with token sampling
        
        Args:
            target_action: The action we are marginalizing
            target_value: The value of the action we want to know the probability of
            action_samples: The total number of samples of the target_action we want to take
            burn_in: The number of samples to ignore before tracking their values
            time_limit: If positive, limits the sampling based on time rather than the number of samples (in seconds)
        
        Returns:
            marginal distribution (float): Estimated P(action == value)
            run time (float): The total running time of this function
        """
        start = time.perf_counter()
        
        activation_order = self.derive_activation(target_action)
        order_length = len(activation_order)

        current_config = self._values.copy() 

        # Tuple of all values where each index represents the value for its respective action
        count = 0
        
        # If time_limit > -1 then we limit based on time, not # of samples
        if(time_limit > -1):
            start = time.perf_counter()
            current = start
            sample_count = 0
            
            # Perform same sampling as below but limited based on time
            while(current - start < time_limit):
                action = activation_order[sample_count % order_length]
            
                # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
                neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
                for a in self._edges[action]:
                    neighbor_set.add((a, int(current_config[a])))

                probabilities = self._cpts[action][frozenset(neighbor_set)]
                domains, probs = zip(*probabilities.items())
                value = np.random.choice(domains, p=probs)

                # Convert values to list, mutate, then convert back to tuple
                current_config[action] = value
                
                # Only add to samples if we are past burn in value
                if (sample_count >= burn_in):
                    # Add to count iff value of action == value
                    count += (current_config[target_action] == target_value)
                
                sample_count += 1
                
                # Only update current every 10 samples to save time
                if(sample_count % 10 == 0):
                    current = time.perf_counter()
            
            end = time.perf_counter()
            run_time = end - start
            return count / (sample_count - burn_in), run_time
            
        # Sample target will track # of times we sample target_action
        sample_count = 0
        sample_target = 0

        # Limit based on number of samples
        while(sample_target < action_samples):
            action = activation_order[sample_count % order_length]
            sample_target += (action == target_action) 

            # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
            neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
            for a in self._edges[action]:
                neighbor_set.add((a, int(current_config[a])))

            probabilities = self._cpts[action][frozenset(neighbor_set)]
            domains, probs = zip(*probabilities.items())
            value = np.random.choice(domains, p=probs)

            # Convert values to list, mutate, then convert back to tuple
            current_config[action] = value
            
            # Only add to samples if we are past burn in value
            if (sample_count >= burn_in):
                # Add to count iff value of target_action == target_value
                count += (current_config[target_action] == target_value)
            
            sample_count += 1
            
        end = time.perf_counter()
        run_time = end - start
            
        return count / (sample_count - burn_in), run_time
    
    def token_sampling_delta(self, target_action: int, target_value: int, delta: float = 0.0001, sample_period: int = 465000, minimum_samples: int = 30000000, initial_config: Dict[int, int] = None) -> float:
        """
        Estimate the marginal distribution by using token sampling

        This function takes the marginal distribution every sample period # of samples, then
        compares it to the previous sampling period. If the difference between these
        two samples is less than the delta value, it will return this distribution.

        Args:
            target_action: The action we want to marginalize
            target_value: The value of the action we want to marginalize for P(target_action == target_value)
            delta: The difference required between samples to return the distribution
            sample_period: The amount of samples between sampling periods (465000 ~ 5 seconds on Mac)
            minimum_samples: The minimum amount of samples required
            initial_config: Initial global configuration

        Returns:
            marginalized distribution as a float
            total running time as a float

        """
        start = time.perf_counter()
        
        if(initial_config):
            self._values = initial_config

        activation_order = self.derive_activation(target_action)
        order_length = len(activation_order)

        count = 0
        current_config = self._values.copy()
        
        # Our two distributions we will compare the delta
        d1 = 0.0
        d2 = 1.0

        # Tuple of all values where each index represents the value for its respective action
        max_key = max(current_config.keys())
        values = tuple(current_config[i] for i in range(max_key + 1))

        sample_count = 0
            
        while(abs(d1 - d2) > delta or sample_count < minimum_samples):
            action = activation_order[sample_count % order_length]
            
            # CPT: dict(action : dict(fset(tuple(neighbor1, value1), tuple(neighbor2, value2) ...) : dict(action-value : probability))) 
            neighbor_set = set()        # a set of tuples (neighbor1, value1) we will inject into cpt
            for a in self._edges[action]:
                neighbor_set.add((a, int(current_config[a])))

            probabilities = self._cpts[action][frozenset(neighbor_set)]
            domains, probs = zip(*probabilities.items())
            value = np.random.choice(domains, p=probs)

            # Convert values to list, mutate, then convert back to tuple
            current_config[action] = value
            value_list = list(values)
            value_list[action] = value
            values = tuple(value_list)

            # Add to count iff target_action == target_value
            count += (current_config[target_action] == target_value) 
            
            sample_count += 1

            # Check if we are at a new sample period
            if(sample_count % sample_period == 0):
                d1 = d2
                d2 = count / sample_count
        
        end = time.perf_counter()
        run_time = end - start
        
        return d2, run_time