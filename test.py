### Run tests on examples and output results to data for analysis
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
GROUND_PATH = "data/4x3_neighborhood/ground_truth_data.csv"
TIME_PATH = "data/4x3_neighborhood/set_time_data.csv"
SAMPLES_PATH = "data/4x3_neighborhood/set_samples_data.csv"
PARAM_PATH = "data/param_neighborhood/param_data.csv"

def run_4x3_tests(num_cycles: int, tests_per_cycle: int, num_samples_list: list[int], time_trials: list[int], target_action: int, target_value: int, delta: int, MRF: M.MarkovRandomField, LAS: L.LiveAndSafe, FMDP: F.FactoredMarkovDecisionProcess, minimum_samples: int = 30000000) -> None:
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
    samples_df = pd.read_csv(SAMPLES_PATH)
    time_df = pd.read_csv(TIME_PATH)
    ground_df = pd.read_csv(GROUND_PATH)
    
    gibbs_samping = "Gibbs"
    token_sampling = "Token"

    # Track number of cycles
    cycles = 0
    meta_cycle = 0
    if(len(samples_df) > 0):
        meta_cycle = samples_df['cycle'].iloc[-1] + 1
        
    # Track total test running time
    total_start_time = time.perf_counter()

    while(cycles < num_cycles):
        print(f"Running cycle: {cycles + 1}/{num_cycles}")
        
        # Randomize CPT table for each new cycle
        MRF.auto_propagate_cpt()
        FMDP.set_cpts(MRF.get_cpts())
        
        # Finding the ground truth (1 ground truth per cycle)
        ground_truth_prob, ground_truth_time = FMDP.gibbs_sampling_delta(target_action=target_action, target_value=target_value, delta=delta, minimum_samples=minimum_samples)
        
        # Place ground truth in DF
        ground_df.loc[len(ground_df)] = [f'{meta_cycle}', f'{ground_truth_time}', f'{delta}', f'{ground_truth_prob}']
        print(f"Cycle {cycles} ground truth found in {ground_truth_time}")

        # Track number of tests
        tests = 0

        while(tests < tests_per_cycle):
            print(f"Running test: {tests}/{tests_per_cycle}")
            
            # Test for speed / accuracy for each value in num_samples
            for action_samples in num_samples_list:  
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                gibbs_prob, gibbs_time_elapsed = FMDP.gibbs_sampling(action=target_action, value=target_value, action_samples=action_samples)

                # Testing token sampling marginal probability that P(target_action == target_value)
                token_prob, token_time_elapsed = FMDP.token_sampling(target_action=target_action, target_value=target_value, action_samples=action_samples)

                # Place data into dataframe at lowest location : accuracy_test_data shape = [sample_type,num_samples,time_elapsed,estimated_distribution]
                samples_df.loc[len(samples_df)] = [f'{meta_cycle}', f'{gibbs_samping}', f'{action_samples}', f'{gibbs_time_elapsed}', f'{gibbs_prob}']
                samples_df.loc[len(samples_df)] = [f'{meta_cycle}', f'{token_sampling}', f'{action_samples}', f'{token_time_elapsed}', f'{token_prob}']

            # Test for accuracy for each value in time_test
            for time_trial in time_trials:
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                gibbs_prob, gibbs_time_elapsed = FMDP.gibbs_sampling(action=target_action, value=target_value, time_limit=time_trial)

                # Testing token sampling marginal probability that P(target_action == target_value)
                token_prob, token_time_elapsed = FMDP.token_sampling(target_action=target_action, target_value=target_value, time_limit=time_trial)

                # Place data into dataframe at lowest location : speed_test_data shape = [sample_type,set_time,estimated_distribution]
                time_df.loc[len(time_df)] = [f'{meta_cycle}', f'{gibbs_samping}', f'{time_trial}', f'{gibbs_prob}']
                time_df.loc[len(time_df)] = [f'{meta_cycle}', f'{token_sampling}', f'{time_trial}', f'{token_prob}']

            tests += 1
        
        print("\n")
        cycles += 1
        meta_cycle += 1

    samples_df.to_csv(SAMPLES_PATH, index=False, header=True)
    time_df.to_csv(TIME_PATH, index=False, header=True)
    ground_df.to_csv(GROUND_PATH, index=False, header=True)
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"Finished running tests. Total run time: {total_time} \n")

def run_param_tests(num_cycles: int, tests_per_cycle: int, param_list: list[tuple], domain: list[int], target_action: int, target_value: int, delta: float, sample_period_list: list[int], minimum_samples: int) -> None:
    """
    Parameterized MRF tests which can take any # of (width, height) tuples
    and build width * height neighborhood MRF's to run tests on.
    
    Tests the length of time each MRF takes to converge to a distribution within the given delta
    for both Gibbs sampling and token sampling. Then outputs this data into a CSV file for data processing.
    
    Args:
        num_cycles: The number of cycles of testing to perform (on each cycle the CPT table gets re-randomized)
        tests_per_cycle: The number of tests to run for each cycle
        param_list: A list of the differently sized Neighborhood MRF's as tuples with form (height, width)
        domain: The discrete domain the random variables can take
        target_action: Target action to marginalize
        target_value: The value we want to find the marginal distribution of for the target_action
        delta: The delta we want the distributions to converge between
        sample_period_list: The number of samples taken at each interval before checking the delta (per parameter)
        minimum_samples: The minimum number of samples taken before the delta is checked

    """
    mrf_list = [None] * len(param_list)
    las_list = [None] * len(param_list)
    fmdp_list = [None] * len(param_list)
    
    # Build MRF, LAS, and FMDP for each tuple in the parameter list
    for n in range(len(param_list)):
        width = param_list[n][0]
        height = param_list[n][1]
        
        mrf_list[n] = M.MarkovRandomField()
        
        # Create vertices labeled 0 -> height * width - 1
        for i in range(0, height * width):
            mrf_list[n].add_vertex(i, domain)
        
        # Create edges
        for i in range(0, height):
            for j in range(0, width):
                # Location of current vertex = [i * width + j]
                vloc = i * width + j
                
                if (j != width - 1): mrf_list[n].add_edge(vloc, vloc + 1)
                if (i != height - 1): mrf_list[n].add_edge(vloc, vloc + width)
        
        # Auto propogate the CPT with random probabilities
        mrf_list[n].auto_propagate_cpt()

        # Use MRF to add vertices / edges to LAS
        las_list[n] = L.LiveAndSafe()
        las_list[n].set_vertices(mrf_list[n].vertices())
        las_list[n].set_edges(mrf_list[n].edges(), ptr=target_action)
        
        # Build FMDP
        fmdp_list[n] = F.FactoredMarkovDecisionProcess()
        fmdp_list[n].set_actions(mrf_list[n].vertices())
        fmdp_list[n].set_edges(las_list[n].get_edges())
        fmdp_list[n].set_components(las_list[n].get_tokens())
        fmdp_list[n].set_domains(mrf_list[n].get_domains())
        fmdp_list[n].set_cpts(mrf_list[n].get_cpts())
        fmdp_list[n].set_random_values()
        
    # DF for test data
    param_df = pd.read_csv(PARAM_PATH)
    
    gibbs_samping = "Gibbs"
    token_sampling = "Token"

    # Track number of cycles
    cycles = 0
    meta_cycle = 0
    if(len(param_df) > 0):
        meta_cycle = param_df['cycle'].iloc[-1] + 1
        
    # Track total test running time
    total_start_time = time.perf_counter()

    while(cycles < num_cycles):
        print(f"Running cycle: {cycles + 1}/{num_cycles}")
        
        # Re-randomize CPTs for each MRF
        for mrf, fmdp in zip(mrf_list, fmdp_list):
            mrf.auto_propagate_cpt()
            fmdp.set_cpts(mrf.get_cpts())

        # Track number of tests
        tests = 0

        while(tests < tests_per_cycle):
            print(f"Running test: {tests}/{tests_per_cycle}")
            
            # Test for speed / accuracy for each value in num_samples
            for fmdp, param, sample_period in zip(fmdp_list, param_list, sample_period_list):
                parameter = f"{param[0]}x{param[1]}"
                
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                gibbs_prob, gibbs_time_elapsed = fmdp.gibbs_sampling_delta(target_action=target_action, target_value=target_value, delta=delta, sample_period=sample_period, minimum_samples=sample_period)

                # Testing token sampling marginal probability that P(target_action == target_value)
                token_prob, token_time_elapsed = fmdp.token_sampling_delta(target_action=target_action, target_value=target_value, delta=delta, sample_period=sample_period, minimum_samples=sample_period)

                # Place data into dataframe at lowest location : param_data shape = [cycle,sample_type,time_elapsed,parameter,delta,sample_period,estimated_distribution]
                param_df.loc[len(param_df)] = [f'{meta_cycle}', f'{gibbs_samping}', f'{gibbs_time_elapsed}', f'{parameter}', f'{delta}', f'{sample_period}', f'{gibbs_prob}']
                param_df.loc[len(param_df)] = [f'{meta_cycle}', f'{token_sampling}', f'{token_time_elapsed}', f'{parameter}', f'{delta}', f'{sample_period}', f'{token_prob}']

            tests += 1
        
        print("\n")
        cycles += 1
        meta_cycle += 1

    param_df.to_csv(PARAM_PATH, index=False, header=True)
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"Finished running tests. Total run time: {total_time} \n")

def graph_samples_data(samples_list: list[int]):
    # Load the data with explicit numeric conversion
    ground_truth = pd.read_csv(GROUND_PATH, dtype={
        'cycle': int,
        'time_elapsed': float,
        'delta': float,
        'estimated_distribution': float
    })

    set_samples_data = pd.read_csv(SAMPLES_PATH, dtype={
        'cycle': int,
        'sample_type': str,
        'num_samples': int,
        'time_elapsed': float,
        'estimated_distribution': float
    })

    # Merge with set samples data
    merged = pd.merge(set_samples_data, ground_truth[['estimated_distribution']], 
                    left_on='cycle', right_index=True,
                    suffixes=('', '_ground_truth'))

    # Calculate absolute difference from ground truth
    merged['abs_diff'] = np.abs(merged['estimated_distribution'] - 
                        merged['estimated_distribution_ground_truth'])

    # Group by sample_type and num_samples to get mean differences
    plot_data = merged.groupby(['sample_type', 'num_samples'])['abs_diff'].mean().reset_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Bar positions
    bar_width = 0.35
    x_pos = np.arange(len(samples_list))  # x positions for time settings

    # Plot Gibbs and Token side by side
    for i, sample_type in enumerate(['Gibbs', 'Token']):
        subset = plot_data[plot_data['sample_type'] == sample_type]
        plt.bar(x_pos + (i * bar_width), 
                subset['abs_diff'],
                width=bar_width,
                label=sample_type,
                alpha=0.8)

    # Customize plot
    plt.title('Average Absolute Difference from Ground Truth\nby Number of Samples and Sampling Method', pad=20)
    plt.xlabel('Number of Samples', labelpad=10)
    plt.ylabel('Average Absolute Difference', labelpad=10)
    plt.xticks(x_pos + bar_width/2, samples_list)
    plt.legend(title='Sampling Method')

    # Add value labels on top of bars
    Y_CONST = 0.0001
    for i, sample_type in enumerate(['Gibbs', 'Token']):
        subset = plot_data[plot_data['sample_type'] == sample_type]
        for j, val in enumerate(subset['abs_diff']):
            plt.text(x_pos[j] + (i * bar_width), val + Y_CONST, 
                    f'{val:.4f}', 
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(fname="images/set_samples.png", dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()

def graph_time_data(time_trials: list[int]):
    # Load the data with explicit numeric conversion
    ground_truth = pd.read_csv(GROUND_PATH, dtype={
        'cycle': int,
        'time_elapsed': float,
        'delta': float,
        'estimated_distribution': float
    })

    set_time_data = pd.read_csv(TIME_PATH, dtype={
        'cycle': int,
        'sample_type': str,
        'set_time': int,
        'estimated_distribution': float
    })

    # Merge with set time data
    merged = pd.merge(set_time_data, ground_truth[['estimated_distribution']], 
                    left_on='cycle', right_index=True,
                    suffixes=('', '_ground_truth'))

    # Calculate absolute difference from ground truth
    merged['abs_diff'] = np.abs(merged['estimated_distribution'] - 
                        merged['estimated_distribution_ground_truth'])

    # Group by sample_type and set_time to get mean differences
    plot_data = merged.groupby(['sample_type', 'set_time'])['abs_diff'].mean().reset_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Bar positions
    bar_width = 0.35
    x_pos = np.arange(len(time_trials))  # x positions for time settings

    # Plot Gibbs and Token side by side
    for i, sample_type in enumerate(['Gibbs', 'Token']):
        subset = plot_data[plot_data['sample_type'] == sample_type]
        plt.bar(x_pos + (i * bar_width), 
                subset['abs_diff'],
                width=bar_width,
                label=sample_type,
                alpha=0.8)

    # Customize plot
    plt.title('Average Absolute Difference from Ground Truth\nby Time Setting and Sampling Method', pad=20)
    plt.xlabel('Time Setting (seconds)', labelpad=10)
    plt.ylabel('Average Absolute Difference', labelpad=10)
    plt.xticks(x_pos + bar_width/2, time_trials)
    plt.legend(title='Sampling Method')

    # Add value labels on top of bars
    Y_CONST = .0001
    for i, sample_type in enumerate(['Gibbs', 'Token']):
        subset = plot_data[plot_data['sample_type'] == sample_type]
        for j, val in enumerate(subset['abs_diff']):
            plt.text(x_pos[j] + (i * bar_width), val + Y_CONST, 
                    f'{val:.4f}', 
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(fname="images/set_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()
    
def graph_param_data(cycles=None):
    # Read and process the data
    df = pd.read_csv(PARAM_PATH)
    if cycles is not None:
        df = df[df['cycle'].isin(cycles)]
    averaged_df = df.groupby(['parameter', 'sample_type'])['time_elapsed'].mean().reset_index()

    # Pivot the data for grouped bar plotting
    pivot_df = averaged_df.pivot(index='parameter', columns='sample_type', values='time_elapsed')

    # Plotting
    ax = pivot_df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'], width=0.8)
    
    # Customize the plot
    plt.title('Average Time Elapsed by Parameter and Sample Type', fontsize=14)
    plt.xlabel('Parameter (Grid Size)', fontsize=12)
    plt.ylabel('Average Time Elapsed (seconds)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Sample Type')
    
    # Display values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(fname="images/param_test.png", dpi=300, bbox_inches='tight')
    plt.close()