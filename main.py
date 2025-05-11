### Main file to run tests on examples
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import test

# TODO: Create test.py file with test functions
# TODO: Output test data to csv file
# TODO: Run tests on range of samples and compare times
# TODO: Compare accuracies of different # of samples
# TODO: Create simple graph showing differences in speeds / accuracies

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
target_action = 0

# Use MRF to add vertices / edges
LAS.set_vertices(MRF.vertices())
LAS.set_edges(MRF.edges(), ptr=target_action)
tokens = LAS.get_tokens()

    # Initialize FMDP
FMDP = F.FactoredMarkovDecisionProcess()
FMDP.set_actions(MRF.vertices())
FMDP.set_edges(LAS.get_edges())
FMDP.set_components(LAS.get_tokens())
FMDP.set_domains(MRF.get_domains())
FMDP.set_cpts(MRF.get_cpts())
FMDP.set_random_values()

num_cycles = 1
tests_per_cycle = 1
num_samples_list = [10000, 100000, 500000]
time_trials = [1, 10, 30]
target_value = 0


# Run tests on data
test.run_tests(num_cycles=num_cycles, tests_per_cycle=tests_per_cycle, num_samples_list=num_samples_list,
            time_trials=time_trials, target_action=target_action, target_value=target_value, MRF=MRF, LAS=LAS, FMDP=FMDP)

# Graph data
# test.graph_data()