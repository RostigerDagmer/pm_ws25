import pandas as pd
import pm4py
from pm4py.algo.simulation.montecarlo.utils import replay
from pm4py.algo.simulation.montecarlo import algorithm as monte_carlo_simulator
from pm4py.objects import random_variables

def execute_script():
    # =====================================================
    # Step 1: Load the event log and discover the Petri net
    # =====================================================

    # Read the event log from an XES file.
    log = pm4py.read_xes("../tests/input_data/running-example.xes")

    # Discover a Petri net using the inductive miner. Note that discovering the net on the entire log
    # might lead to underfitting. In some cases, focusing on a subset of variants can yield a more precise model.
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)

    # =====================================================
    # Step 2: Create and customize the stochastic map
    # =====================================================

    # Generate the stochastic map from the log and the discovered Petri net.
    # The map assigns a duration model (random variable) to each transition.
    stochastic_map = replay.get_map_from_log_and_net(log, net, initial_marking, final_marking)
    print("Original stochastic map (derived from log):")
    print(stochastic_map)

    # To customize activity durations, you can manually override the random variables
    # for each transition. Here are three examples of random variable distributions:

    for trans in list(stochastic_map.keys()):
        # --- Normal Distribution ---
        # Parameters:
        #   mu (mean): expected duration value (here set to 5 time units)
        #   sigma (standard deviation): variability around the mean (set to 3 time units)
        rv_normal = random_variables.normal.random_variable.Normal(mu=5, sigma=3)
        print(f"Normal distribution for transition '{trans}': {rv_normal}")

        # --- Exponential Distribution ---
        # Parameters:
        #   loc: a shift parameter (minimum duration, here set to 1)
        #   scale: the mean waiting time between events (set to 1)
        rv_exponential = random_variables.exponential.random_variable.Exponential(loc=1, scale=1)
        print(f"Exponential distribution for transition '{trans}': {rv_exponential}")

        # --- Uniform Distribution ---
        # Parameters:
        #   loc: lower bound of the duration (here set to 0)
        #   scale: range of the duration (so durations are equally likely between 0 and 1)
        rv_uniform = random_variables.uniform.random_variable.Uniform(loc=0, scale=1)
        print(f"Uniform distribution for transition '{trans}': {rv_uniform}")

        # For this example, we choose to override with the normal distribution.
        stochastic_map[trans] = rv_normal

    print("Customized stochastic map (after manual override):")
    print(stochastic_map)

    # =====================================================
    # Step 3: Configure simulation parameters and run simulation
    # =====================================================

    # Set up simulation parameters:
    sim_parameters = {
        # Use the customized stochastic map for transition durations.
        "provided_stochastic_map": stochastic_map,
        # Number of simulation runs to perform.
        "num_simulations": 2,
        # Case arrival ratio in seconds.
        # (For example, 86400 seconds corresponds to a new case starting every day.)
        "case_arrival_ratio": 86400
    }

    # Run the Monte Carlo simulation using the Petri net, the log, and the parameters.
    simulated_log, simulation_results = monte_carlo_simulator.apply(log, net, initial_marking, final_marking,
                                                                    parameters=sim_parameters)

    # Convert the simulated log into a pandas DataFrame for easier manipulation.
    simulated_log_df = pm4py.convert_to_dataframe(simulated_log)
    print("Simulated timestamps before start time adjustment:")
    print(simulated_log_df["time:timestamp"])

    # =====================================================
    # Step 4: Adjust simulated timestamps to add a start time
    # =====================================================

    # To align the simulation results with a specific real-world start time,
    # we can apply an offset to all timestamps.
    # For example, to set the start time to January 1, 2024:
    desired_start_time = pd.Timestamp("2024-01-01")
    unix_epoch = pd.Timestamp("1970-01-01")
    # Calculate the offset between the desired start time and the Unix epoch.
    offset = desired_start_time - unix_epoch

    # Adjust each timestamp in the simulated log by adding the offset.
    simulated_log_df["time:timestamp"] = simulated_log_df["time:timestamp"] + offset
    print("Simulated timestamps after start time adjustment:")
    print(simulated_log_df["time:timestamp"])

    # =====================================================
    # End of Script
    # =====================================================

    # The above script:
    # - Discovers a process model from an event log.
    # - Creates a stochastic map for transition durations.
    # - Demonstrates how to manually override durations using various distributions.
    # - Configures and runs the Monte Carlo simulation.
    # - Adjusts the simulation's timestamps to align with a desired start time.


if __name__ == "__main__":
    execute_script()
