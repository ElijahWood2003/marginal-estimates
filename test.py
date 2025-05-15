### Run tests on examples and output results to a csv file for analysis
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_tests(num_cycles: int, tests_per_cycle: int, num_samples_list: list[int], time_trials: list[int], target_action: int, target_value: int, delta: int, MRF: M.MarkovRandomField, LAS: L.LiveAndSafe, FMDP: F.FactoredMarkovDecisionProcess) -> None:
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
    samples_df = pd.read_csv("data/set_samples_data.csv")
    time_df = pd.read_csv("data/set_time_data.csv")
    ground_df = pd.read_csv("data/ground_truth_data.csv")
    
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
        print(f"Running cycle: {cycles}")
        
        # Randomize CPT table for each new cycle
        MRF.auto_propagate_cpt()
        FMDP.set_cpts(MRF.get_cpts())
        
        # Finding the ground truth (1 ground truth per cycle)
        start = time.perf_counter()
        ground_truth_prob = FMDP.marginal_distribution_delta(target_action=target_action, target_value=target_value, delta=delta)
        end = time.perf_counter()
        ground_truth_time = end - start
        
        # Place ground truth in DF
        ground_df.loc[len(ground_df)] = [f'{meta_cycle}', f'{ground_truth_time}', f'{delta}', f'{ground_truth_prob}']
        print(f"Cycle {cycles} ground truth found in {ground_truth_time}")

        # Track number of tests
        tests = 0

        while(tests < tests_per_cycle):
            # Test for speed / accuracy for each value in num_samples
            for action_samples in num_samples_list:  
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                gibbs_prob = FMDP.gibbs_sampling(action=target_action, value=target_value, action_samples=action_samples)
                end = time.perf_counter()
                gibbs_time_elapsed = end - start

                # Testing token sampling marginal probability that P(target_action == target_value)
                start = time.perf_counter()
                token_prob = FMDP.token_sampling(target_action=target_action, target_value=target_value, action_samples=action_samples)
                end = time.perf_counter()
                token_time_elapsed = end - start

                # Place data into dataframe at lowest location : accuracy_test_data shape = [sample_type,num_samples,time_elapsed,estimated_distribution]
                samples_df.loc[len(samples_df)] = [f'{meta_cycle}', f'{gibbs_samping}', f'{action_samples}', f'{gibbs_time_elapsed}', f'{gibbs_prob}']
                samples_df.loc[len(samples_df)] = [f'{meta_cycle}', f'{token_sampling}', f'{action_samples}', f'{token_time_elapsed}', f'{token_prob}']

            # Test for accuracy for each value in time_test
            for time_trial in time_trials:
                # Estimating gibbs sampling marginal probability that P(target_action == target_value)
                gibbs_prob = FMDP.gibbs_sampling(action=target_action, value=target_value, time_limit=time_trial)

                # Testing token sampling marginal probability that P(target_action == target_value)
                token_prob = FMDP.token_sampling(target_action=target_action, target_value=target_value, time_limit=time_trial)

                # Place data into dataframe at lowest location : speed_test_data shape = [sample_type,set_time,estimated_distribution]
                time_df.loc[len(time_df)] = [f'{meta_cycle}', f'{gibbs_samping}', f'{time_trial}', f'{gibbs_prob}']
                time_df.loc[len(time_df)] = [f'{meta_cycle}', f'{token_sampling}', f'{time_trial}', f'{token_prob}']

            tests += 1
            print(f"Finished test {tests}")
        
        print("\n")
        cycles += 1
        meta_cycle += 1

    samples_df.to_csv("data/set_samples_data.csv", index=False, header=True)
    time_df.to_csv("data/set_time_data.csv", index=False, header=True)
    ground_df.to_csv("data/ground_truth_data.csv", index=False, header=True)
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"Finished running tests. Total run time: {total_time} \n")

def graph_samples_data(samples_list: list[int]):
    # Load the data with explicit numeric conversion
    ground_truth = pd.read_csv('data/ground_truth_data.csv', dtype={
        'cycle': int,
        'time_elapsed': float,
        'delta': float,
        'estimated_distribution': float
    })

    set_samples_data = pd.read_csv('data/set_samples_data.csv', dtype={
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
    plt.show()


def graph_time_data(time_trials: list[int]):
    # Load the data with explicit numeric conversion
    ground_truth = pd.read_csv('data/ground_truth_data.csv', dtype={
        'cycle': int,
        'time_elapsed': float,
        'delta': float,
        'estimated_distribution': float
    })

    set_time_data = pd.read_csv('data/set_time_data.csv', dtype={
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
    plt.show()