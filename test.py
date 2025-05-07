### Run tests on examples and output results to a csv file for analysis
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import pandas as pd


        # Binary 4x3-Neighborhood MRF Example
    # Initialize MRF
MRF = M.MarkovRandomField()
domain = [0, 1]
height = 3
width = 4

# Create vertices labeled 0 -> 11
for i in range(0, height * width):
    # Add vertex
    MRF.add_vertex(i, domain)

# Create edges
for i in range(0, height):
    for j in range(0, width):
        # Location of current vertex = (i * width + j)
        vloc = i * width + j

        # Add edge to the right of vertex
        if (j != width - 1): MRF.add_edge(vloc, vloc + 1)

        # Add edge below the vertex
        if (i != height - 1): MRF.add_edge(vloc, vloc + width)

# Auto propagate the CPTs with random probabilities
MRF.auto_propagate_cpt()

    # Initialize LAS
LAS = L.LiveAndSafe()
acyclic_pointer = 0

# Use MRF to add vertices / edges
LAS.set_vertices(MRF.vertices())
LAS.set_edges(MRF.edges(), ptr=acyclic_pointer)
tokens = LAS.get_tokens()

    # Initialize FMDP
FMDP = F.FactoredMarkovDecisionProcess()
FMDP.set_actions(MRF.vertices())
FMDP.set_edges(LAS.get_edges())
FMDP.set_components(LAS.get_tokens())
FMDP.set_domains(MRF.get_domains())
FMDP.set_cpts(MRF.get_cpts())
FMDP.set_random_values()

# Estimating gibbs sampling marginal probability that P(x_0 == 0)
num_samples = 100000
start = time.perf_counter()
prob = FMDP.gibbs_sampling(0, 0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")

# Testing marginal distribution and tracking time
num_samples = 100000
start = time.perf_counter()
prob = FMDP.marginal_probability(0, 0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")

print("\n")

# Estimating gibbs sampling marginal probability that P(x_0 == 0)
num_samples = 1000000
start = time.perf_counter()
prob = FMDP.gibbs_sampling(0, 0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")

# Testing marginal distribution and tracking time
num_samples = 1000000
start = time.perf_counter()
prob = FMDP.marginal_probability(0, 0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")


# Testing joint distribution and tracking time
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.marginal_probability(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start


# print(f"Joint distribution of ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")