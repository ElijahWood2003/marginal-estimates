### Run tests on examples and output results to a csv file for analysis
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    acc_df = pd.read_csv("data/accuracy_test_data.csv")
    speed_df = pd.read_csv("data/speed_test_data.csv")
    gibbs_samping = "Gibbs"
    token_sampling = "Token"

    # Track number of cycles
    cycles = 0

    while(cycles < num_cycles):
        # Randomize CPT table for each new cycle
        MRF.auto_propagate_cpt()
        FMDP.set_cpts(MRF.get_cpts())
        
        # Track number of tests
        tests = 0
        
        print(f"Running cycle: {cycles}")

        while(tests < tests_per_cycle):
            # Test for speed / accuracy for each value in num_samples
            for num_samples in num_samples_list:
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                gibbs_prob = FMDP.gibbs_sampling(action=target_action, value=target_value, num_samples=num_samples)
                end = time.perf_counter()
                gibbs_time_elapsed = end - start

                # Testing token sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                token_prob = FMDP.token_sampling(initial_action=target_action, target_value=target_value, num_samples=num_samples)
                end = time.perf_counter()
                token_time_elapsed = end - start

                # Place data into dataframe at lowest location : accuracy_test_data shape = [sample_type,num_samples,time_elapsed,estimated_distribution]
                acc_df.loc[len(acc_df)] = [f'{gibbs_samping}', f'{num_samples}', f'{gibbs_time_elapsed}', f'{gibbs_prob}']
                acc_df.loc[len(acc_df)] = [f'{token_sampling}', f'{num_samples}', f'{token_time_elapsed}', f'{token_prob}']

            # Test for accuracy for each value in time_test
            for time_trial in time_trials:
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                gibbs_prob = FMDP.gibbs_sampling(action=target_action, value=target_value, time_limit=time_trial)

                # Testing token sampling marginal probability that P(target_action == target_value)
                token_prob = FMDP.token_sampling(initial_action=target_action, target_value=target_value, time_limit=time_trial)

                # Place data into dataframe at lowest location : speed_test_data shape = [sample_type,set_time,estimated_distribution]
                speed_df.loc[len(speed_df)] = [f'{gibbs_samping}', f'{time_trial}', f'{gibbs_prob}']
                speed_df.loc[len(speed_df)] = [f'{token_sampling}', f'{time_trial}', f'{token_prob}']

            tests += 1
            print(f"Finished test {tests}")
        
        print("\n")
        cycles += 1

    acc_df.to_csv("data/accuracy_test_data.csv", index=False, header=True)
    speed_df.to_csv("data/speed_test_data.csv", index=False, header=True)

def graph_data():
    """
    Graph the data from the csv files
    
    

    """
    # Get DF
    acc_df = pd.read_csv('data/accuracy_test_data.csv')
    speed_df = pd.read_csv('data/speed_test_data.csv')

    # Calculate the "true" distribution for each cycle (average of 30s estimates)
    true_dist = speed_df[speed_df['set_time'] == 30].groupby(['cycle', 'sample_type'])['estimated_distribution'].mean()
    true_dist = true_dist.reset_index().rename(columns={'estimated_distribution': 'true_distribution'})

    # Merge this ground truth back with the original data
    merged = pd.merge(speed_df, true_dist, on=['cycle', 'sample_type'])

    # Calculate absolute error from ground truth
    merged['abs_error'] = np.abs(merged['estimated_distribution'] - merged['true_distribution'])

    # Filter out the 30s estimates (since they're our ground truth)
    comparison_data = merged[merged['set_time'].isin([1, 10])]

    # Create the visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create a boxplot showing error distribution by time and sample type
    ax = sns.boxplot(
        x='set_time', 
        y='abs_error', 
        hue='sample_type', 
        data=comparison_data,
        palette={'Gibbs': 'skyblue', 'Token': 'salmon'},
        width=0.6
    )

    # Add mean markers
    means = comparison_data.groupby(['set_time', 'sample_type'])['abs_error'].mean().reset_index()
    sns.stripplot(
        x='set_time', 
        y='abs_error', 
        hue='sample_type', 
        data=means,
        palette={'Gibbs': 'blue', 'Token': 'red'},
        size=10, 
        marker='D',
        jitter=False,
        ax=ax,
        legend=False
    )

    # Customize the plot
    plt.title('Accuracy of Short vs Medium Estimates Compared to 30s Ground Truth', pad=20)
    plt.xlabel('Estimation Time (seconds)')
    plt.ylabel('Absolute Error from Ground Truth')
    plt.legend(title='Sampling Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Annotate the means
    # for i, row in means.iterrows():
    #     ax.text(
    #         row['set_time'] - (0.15 if row['sample_type'] == 'Gibbs' else 0.15),
    #         row['abs_error'] + 0.001,
    #         f"μ={row['abs_error']:.4f}",
    #         color='blue' if row['sample_type'] == 'Gibbs' else 'red',
    #         fontsize=9
    #     )

    plt.tight_layout()
    plt.show()


#         # Binary 4x3-Neighborhood MRF Example
#     # Initialize MRF
# MRF = M.MarkovRandomField()
# domain = [0, 1]
# height = 3
# width = 4

# # Create vertices labeled 0 -> 11
# for i in range(0, height * width):
#     # Add vertex
#     MRF.add_vertex(i, domain)

# # Create edges
# for i in range(0, height):
#     for j in range(0, width):
#         # Location of current vertex = (i * width + j)
#         vloc = i * width + j

#         # Add edge to the right of vertex
#         if (j != width - 1): MRF.add_edge(vloc, vloc + 1)

#         # Add edge below the vertex
#         if (i != height - 1): MRF.add_edge(vloc, vloc + width)

# # Auto propagate the CPTs with random probabilities
# MRF.auto_propagate_cpt()

#     # Initialize LAS
# LAS = L.LiveAndSafe()
# acyclic_pointer = 0

# # Use MRF to add vertices / edges
# LAS.set_vertices(MRF.vertices())
# LAS.set_edges(MRF.edges(), ptr=acyclic_pointer)
# tokens = LAS.get_tokens()

#     # Initialize FMDP
# FMDP = F.FactoredMarkovDecisionProcess()
# FMDP.set_actions(MRF.vertices())
# FMDP.set_edges(LAS.get_edges())
# FMDP.set_components(LAS.get_tokens())
# FMDP.set_domains(MRF.get_domains())
# FMDP.set_cpts(MRF.get_cpts())
# FMDP.set_random_values()

# # Estimating gibbs sampling marginal probability that P(x_5 == 1)
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.gibbs_sampling(action=5, value=1, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_5 == 1): {prob} \n")

# # Testing marginal distribution and tracking time
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.token_sampling(initial_action=5, target_value=1, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_5 == 1): {prob} \n")

# print("\n")

# # Estimating gibbs sampling marginal probability that P(x_0 == 1)
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.gibbs_sampling(action=0, value=1, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 1): {prob} \n")

# # Testing marginal distribution and tracking time
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.token_sampling(initial_action=0, target_value=1, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 1): {prob} \n")

# # Estimating gibbs sampling marginal probability that P(x_0 == 0)
# num_samples = 1000000
# start = time.perf_counter()
# prob = FMDP.gibbs_sampling(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Gibbs sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")

# # Testing marginal distribution and tracking time
# num_samples = 1000000
# start = time.perf_counter()
# prob = FMDP.token_sampling(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start

# print(f"Token sampling ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")


# Testing joint distribution and tracking time
# num_samples = 100000
# start = time.perf_counter()
# prob = FMDP.token_sampling(0, 0, num_samples=num_samples)
# end = time.perf_counter()
# time_elapsed = end - start


# print(f"Joint distribution of ({num_samples} samples) took {time_elapsed:.6f} seconds")
# print(f"Estimated P(x_0 == 0): {prob} \n")