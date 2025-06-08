### Main file to run tests on examples
import classes.MRF as M
import classes.LAS as L
import classes.FMDP as F
import test

# TODO: Add sample period to param data csv
# TODO: Decrease GT delta for 10x10
# TODO: Take deltas based on sample period and compare that to the last X samples are all within the delta range (determine true limit)

def test_4x3():
    """
    
    Binary 4x3-Neighborhood MRF Example.
    Output goes to data files.
    Includes graphing functions at bottom.
    """
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

    # Variables for testing
    num_cycles = 8
    tests_per_cycle = 5
    num_samples_list = [25000, 100000, 500000]
    time_trials = [5, 30, 60]
    target_value = 0
    long_delta = 0.0000001
    short_delta = 0.000001

    # Run tests on data
    # test.run_4x3_tests(num_cycles=num_cycles, tests_per_cycle=tests_per_cycle, num_samples_list=num_samples_list, time_trials=time_trials, target_action=target_action, target_value=target_value, delta=short_delta, MRF=MRF, LAS=LAS, FMDP=FMDP)

    # Graph data
    test.graph_time_data(time_trials)
    test.graph_samples_data(num_samples_list)
    
# test_4x3()

def test_param():
    """
    Parameterized neighborhood MRF example.
    Dimensions given as a list of tuples in param_list.
    Output goes to data files.
    """
    # Varibles for testing
    num_cycles = 1
    tests_per_cycle = 5
    param_list = [(10, 10), (20, 20), (50, 50)]
    domain = [0, 1]
    target_action = 0
    target_value = 0
    delta_list = [0.00001, 0.0001, 0.001]
    gt_delta = [0.0001, 0.0001, 0.0001]
    gt_min_samples = [50000000, 50000000, 60000000]    # 50m, 75m, 100m
    gt_sample_period = [1000000, 1500000, 2500000]
    # sample_period_list = [750000, 1500000, 2000000]
    # minimum_samples = 2500000
    test.run_param_tests_ground_truth(num_cycles=num_cycles, tests_per_cycle=tests_per_cycle, param_list=param_list, domain=domain, target_action=target_action, target_value=target_value, delta_list=delta_list, gt_delta=gt_delta, gt_min_samples=gt_min_samples, gt_sample_period=gt_sample_period)

test_param()
# test.graph_param_data()