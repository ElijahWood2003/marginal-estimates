### Run tests on examples and output results to a csv file for analysis
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import pandas as pd


def run_tests(num_cycles: int, tests_per_cycle: int, num_samples_list: list[int], time_trials: list[int], target_action: int, target_value: int, MRF: M.MarkovRandomField, LAS: L.LiveAndSafe, FMDP: F.FactoredMarkovDecisionProcess) -> None:
    """
    Run tests to track the accuracy and speed of estimating marginal distributions for
    gibbs sampling versus token sampling

    Args:
        num_cycles: The number of cycles of testing to perform (on each cycle the CPT table gets re-randomized)
        tests_per_cycle: The number of tests to run for each cycle
        num_samples: A list determining the number of samples to test for accuracy
        time_trials: A list determining the time durations to test for speed
        target_action: Target action to marginalize
        target_value: The value we want to find the marginal distribution of for the target_action
        MRF: The initialized MRF
        LAS: The initialized LAS
        FMDP: The initialized FMDP
    
    Return:
        None
    """
    # DF for test data
    acc_df = pd.read_csv("data/accurace_test_data.csv")
    speed_df = pd.read_csv("data/speed_test_data.csv")

    # Track number of cycles
    cycles = 0

    while(cycles < num_cycles):
        # Randomize CPT table for each new cycle
        MRF.auto_propagate_cpt()
        FMDP.set_cpts(MRF.get_cpts())
        
        # Track number of tests
        tests = 0

        while(tests < tests_per_cycle):
            # TEST FOR SPEED / ACCURACY FOR EACH VALUE IN num_samples
            for num_samples in num_samples_list:
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                gibbs_prob = FMDP.gibbs_sampling(action=target_action, value=target_value, num_samples=num_samples)
                end = time.perf_counter()
                gibbs_time_elapsed = end - start

                # Testing token sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                token_prob = FMDP.marginal_probability(initial_action=0, target_value=0, num_samples=num_samples)
                end = time.perf_counter()
                token_time_elapsed = end - start

                # Place data into dataframe at lowest location
                # accuracy_test_data shape = [sample_type,num_samples,time_elapsed,estimated_distribution]
                acc_df.loc[len(acc_df)] = ["Gibbs", f'{num_samples}', f'{gibbs_time_elapsed}', f'{gibbs_prob}']
                acc_df.loc[len(acc_df)] = ["Token", f'{num_samples}', f'{token_time_elapsed}', f'{token_prob}']


            # TEST FOR ACCURACY FOR EACH VALUE IN time_test
            for time_trial in time_trials:


                # speed_test_data shape = [sample_type,set_time,estimated_distribution]
                speed_df.loc[len(speed_df)] = ["Gibbs", f'{num_samples}', f'{gibbs_time_elapsed}', f'{gibbs_prob}']
                speed_df.loc[len(speed_df)] = ["Token", f'{num_samples}', f'{token_time_elapsed}', f'{token_prob}']

            tests += 1

    cycles += 1

    acc_df.to_csv("data/accuracy_test_data.csv", index=False, header=True)
    speed_df.to_csv("data/speed_test_data.csv", index=False, header=True)

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

# Estimating gibbs sampling marginal probability that P(x_5 == 1)
num_samples = 100000
start = time.perf_counter()
prob = FMDP.gibbs_sampling(action=5, value=1, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_5 == 1): {prob} \n")

# Testing marginal distribution and tracking time
num_samples = 100000
start = time.perf_counter()
prob = FMDP.marginal_probability(initial_action=5, target_value=1, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_5 == 1): {prob} \n")

print("\n")

# Estimating gibbs sampling marginal probability that P(x_0 == 1)
num_samples = 100000
start = time.perf_counter()
prob = FMDP.gibbs_sampling(action=0, value=1, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 1): {prob} \n")

# Testing marginal distribution and tracking time
num_samples = 100000
start = time.perf_counter()
prob = FMDP.marginal_probability(initial_action=0, target_value=1, num_samples=num_samples)
end = time.perf_counter()
time_elapsed = end - start

print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
print(f"Estimated P(x_0 == 1): {prob} \n")

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