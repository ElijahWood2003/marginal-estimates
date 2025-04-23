### Main file to run tests on examples
import MRF as M
import LAS as L
import FMDP as F
import time

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

# Testing marginal probability of P(state(0) == 0)
# num_samples = 100000
# start = time.perf_counter()
# prob = MRF.marginal_probability(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")


    # Initialize LAS
LAS = L.LiveAndSafe()
acyclic_pointer = 0

# Use MRF to add vertices / edges
LAS.set_vertices(MRF.vertices())
LAS.set_edges(MRF.edges(), ptr=acyclic_pointer)

# print(LAS._tokens[(0, 4)])      # should be 0 since !(0 > 4)
# print(LAS._tokens[(4, 0)])      # should be 1 since  (4 > 0)


    # Initialize FMDP
FMDP = F.FactoredMarkovDecisionProcess()
FMDP.set_actions(MRF.vertices())
FMDP.set_edges(LAS.get_edges())
FMDP.set_components(LAS.get_tokens())
FMDP.set_domains(MRF.get_domains())
FMDP.set_cpts(MRF.get_cpts())
FMDP.set_random_values()

# Testing joint distribution and tracking time
num_samples = 100000
start = time.perf_counter()
joint_dist = FMDP.joint_distribution(0, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

prob = FMDP.marginal_probability(joint_dist, 0, 0)

print(f"Joint distribution of ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 0): {prob} \n")

# Testing marginal distribution and tracking time
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.marginal_probability(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"FMDP sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")