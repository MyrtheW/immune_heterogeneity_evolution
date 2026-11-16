import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import yaml
from analyze_and_plotting import *
import model.objects as objects
import model.dynamics as dynamics
import model.tools as tools

def simulate_and_plot(params, seed=42, t_iteration=1000, folder=None, t_max=None, t_step_max=None, t_burn_in=None, t_exponential=True, control=None):
    """
    Simulate the SIR model with multiple strains and plot the results.

    Parameters:
    - params: Parameters object for the simulation.
    - seed (int): Random seed for reproducibility.
    - t_iteration (float): Average time between strain introductions.
    - folder (str): Directory to save results and plots.
    - t_max (float): Maximum simulation time (optional, mutually exclusive with iterations).
    - t_step_max (float): Maximum time step for ODE solver.
    - t_burn_in (float): Time to consider as burn-in before statistics are calculated.

    Returns:
    - dict: Dictionary with simulation statistics and plot results.
    """
    random.seed(seed)
    np.random.seed(seed)

    def seperate_strains(solutions, strain_ids_xt, g='I', max_steps=100, region=None):
        """
        Extract compartment values for each strain from simulation solutions.

        Parameters:
        - solutions (list): List of ODE solution objects for each iteration.
        - strain_ids_xt (list): List of lists with strain IDs present at each iteration.
        - g (str): Compartment to extract ('I', 'S', etc.).
        - max_steps (int): Maximum number of time steps to sample per solution.

        Returns:
        - list: For each strain, a tuple (times, values) with time series and compartment values.
        """
        total_strains = max(strain_id for sublist in strain_ids_xt for strain_id in sublist) + 1
        Xs = [[np.array([]), np.array([])] for _ in range(total_strains)]  # For each strain: [times, values of g summed over regions]
        iterations = len(solutions)

        for i in range(iterations):
            sol = solutions[i]
            for a in range(len(strain_ids_xt[i])):  # For each strain present at this time point
                strain_id = strain_ids_xt[i][a] # Get the strain ID
                ts = sol.sol.t if len(sol.sol.t) < max_steps else np.linspace(sol.sol.t[0], sol.sol.t[-1], max_steps)
                if region == None: xs = np.sum(sol[ts, :, a, g], axis=0)/sol.parameters.M  # Sum over regions for this strain
                else: xs = sol[ts, region, a, g]
                Xs[strain_id][0] = np.append(Xs[strain_id][0], ts) # time point
                Xs[strain_id][1] = np.append(Xs[strain_id][1], xs) # compartment values
        return Xs

    def combine_solutions(solutions):
        """
        Combine multiple ODE solutions into a single solution object. 
        If the number of strains differs at different time points, the solution will be padded with zeros:
             Find the solution with the maximum number of rows (max_shape).
             For each solution, if its .shape[0] is less than max_shape, determine the integer factor by which it is smaller.
             Then, for each group of 4 rows (corresponding to each state: I, S, R, C), interleave (factor-1) groups of 4 rows of zeros (of length .shape[1]) after each existing group.
        This ensures all solutions are padded to the same shape and can be concatenated.

        Returns:
        - sol_combined: Combined solution object with concatenated time and state arrays. 
        """
        ts = np.concatenate([sol.sol.t for sol in solutions])
        max_N = max(sol.parameters.N for sol in solutions)
        adjusted_solutions = []

        for sol in solutions:
            if sol.parameters.N < max_N:
                adjusted_y = []
                for i in range(sol.sol.y.shape[0]):
                    adjusted_y.append(sol.sol.y[i])
                    if (i + 1) % (sol.parameters.N*4) == 0: # every 4th row, for each state, I, S, R, C
                        for _ in range(max_N - sol.parameters.N):
                            for _ in range(4): adjusted_y.append(np.zeros(sol.sol.y.shape[1], dtype=float))
                adjusted_solutions.append(np.array(adjusted_y))
            else:
                adjusted_solutions.append(sol.sol.y)

        xs = np.concatenate(adjusted_solutions, axis=1)
        sol_combined = deepcopy(solutions[-1]) # Create a new solution object by copying one of the existing solutions. 
        sol_combined.parameters.N = max_N # Make sure that the self.params.N are in line with the maximum N
        sol_combined.sol.t = ts
        sol_combined.sol.y = xs
        assert sol_combined.sol.y.shape[0] == sol_combined.parameters.M * sol_combined.parameters.N * 4, "Shape mismatch"
        sol_combined.tspan = (ts[0], ts[-1])
        return sol_combined

    if folder is not None and not os.path.exists(folder):
        os.makedirs(folder)
    
    # Calculate equilibrium time and infection times
    itimes = np.arange(0, t_max, t_iteration)
    itimes = [0]
    while itimes[-1] < t_max:
        if t_exponential: itimes.append(itimes[-1] + np.random.exponential(t_iteration))
        else: itimes.append(itimes[-1] + t_iteration)
    itimes = np.array(itimes)

    # Create regions and initial state
    K_ = np.full((params.N, params.N), 0.0) 
    np.fill_diagonal(K_, 1.0) # set diagonal to 1, representing a full protection to the same strain 
    initial_susceptibility = 0.5 # Otherwise, 0.5; 0.5 equals delta/alpha for the default parameters.
    Ss = [initial_susceptibility for r in range(1, params.M + 1)] # 0.5 equals delta/alpha for the default parameters.
    normal_regions = [objects.Region(viruses = [objects.Virus(Ss[i] - params.I0, params.I0, 0, 1-Ss[i], strain=objects.Strain(a=0, strain_id=0))], K= deepcopy(K_)) 
                    for i in range(params.M)] # initialize with one strain with few infections.  

    initial_state = objects.SIRState(parameters=params, regions=normal_regions)

    def run_simulation_iterations(itimes, current_state):
        """
        Run simulation for each strain introduction and update state.

        Parameters:
        - itimes (array): Array of time intervals for each iteration.
        - current_state (SIRState): Current simulation state.

        Returns:
        - solutions (list): List of ODE solutions for each iteration.
        - current_state (SIRState): Final simulation state.
        - growth_rates_xt (list): Growth rates for each introduced strain.
        - iteration_time (float): Average time per iteration.
        - strain_ids_xt (list): Strain IDs present at each iteration.
        """
        solutions = []
        growth_rates_xt, strain_ids_xt = [], []
        iteration_time = 0

        for i in range(len(itimes) - 1):
            print(f"Iteration {i} / {len(itimes)}")
            tspan = (itimes[i], itimes[i + 1])
            a_variant, _ = current_state.add_strain(t=itimes[i])
            growth_rates_xt.append(tools.initial_growth_rate(X=current_state, virus=a_variant))
            start_time = time.time()
            sol = dynamics.simulate(current_state, tspan, t_step_max)
            iteration_time += time.time() - start_time
            solutions.append(sol)
            strain_ids_xt.append(list(current_state.circulating_strains))
            sol(tspan[1], current_state)  # Update current state

        iteration_time /= len(itimes) - 1
        return solutions, current_state, growth_rates_xt, iteration_time, strain_ids_xt

    print(f"\n seed {seed} \n")

    solutions, current_state, growth_rates_xt, iteration_time, strain_ids_xt = run_simulation_iterations(itimes, initial_state)
    I_xt = seperate_strains(solutions, strain_ids_xt, g='I')  # (t, x) for each strain, sorted by strain ID
    I_xt_M = [seperate_strains(solutions, strain_ids_xt, g='I', region=i) for i in range(current_state.parameters.M)]  # (t, x) for each strain, sorted by strain ID
    sol_combined = combine_solutions(solutions) # Combine seperate solution objects 
    del solutions # delete solutions
    ss_idx = np.searchsorted(itimes, t_burn_in, side='left')


    if seed == 1: 
        plot = True; summary_only=False 
    else: plot = False; summary_only=True

    analyze_results = analyze_and_plot_multi_strains(
        current_state, sol_combined, itimes, I_xt, strain_ids_xt, folder=folder, show=False,
        plot=plot, tspan=(itimes[ss_idx], itimes[-1]), plot_idx=min(0, len(itimes)-100), growth_rates_xt=growth_rates_xt,
        summary_only=summary_only, I_xt_M=I_xt_M
    )
    # Store scalar outcome averages for this seed.
    if folder is not None:
        pd.DataFrame([{**analyze_results, 'iteration_time': iteration_time}]).to_csv(
            os.path.join(folder, 'simulation_results.csv'), index=False
        )
    del sol_combined, current_state, I_xt, I_xt_M, strain_ids_xt, growth_rates_xt

    return {**analyze_results, 'iteration_time': iteration_time}


def load_results_from_csv(folder):
    """
    Load simulation results from a CSV file.

    Parameters:
    - folder (str): Path to the folder containing the simulation_results.csv file.

    Returns:
    - dict: Dictionary containing the results.
    """
    csv_path = os.path.join(folder, 'simulation_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file '{csv_path}' does not exist.")
    
    return pd.read_csv(csv_path).iloc[0].to_dict()
    
def load_results(folder):
    """
    Load simulation results from a pickle file.

    Parameters:
    - folder (str): Path to the folder containing the simulation_results.pkl file.

    Returns:
    - tuple: (sol_combined, itimes, final_state, growth_rates_xt, strain_ids_xt, I_xt)
    """
    with open(os.path.join(folder, 'simulation_results.pkl'), 'rb') as f:
        data = pickle.load(f)
        sol_combined, itimes, final_state, growth_rates_xt, strain_ids_xt, I_xt = data['sol_combined'], data['itimes'], data['final_state'], data['growth_rates_xt'], data['strain_ids_xt'], data['I_xt']
    return sol_combined, itimes, final_state, growth_rates_xt, strain_ids_xt, I_xt


def load_and_plot_simulation_results(folder):
    """
    Load simulation results from a folder and plot using analyze_and_plot_multi_strains.

    Parameters:
    - folder (str): Path to the folder containing the simulation results.

    Returns:
    - dict: Results of the analyze_and_plot_multi_strains function.
    """
    sol_combined, itimes, final_state, growth_rates_xt, strain_ids_xt, I_xt = load_results(folder)
    start_steady_state = round(len(itimes)/2) #t_step_max 

    # Plot the results
    plot_results = analyze_and_plot_multi_strains(
        final_state, sol_combined, itimes, I_xt, strain_ids_xt, folder=folder, plot=True, show=True,
        tspan=(itimes[start_steady_state], itimes[-1]), growth_rates_xt=growth_rates_xt
    )
    if folder is not None:
        with open(os.path.join(folder, 'simulation_results.csv'), 'w') as f:
            keys = list({**plot_results, 'iteration_time': 0}.keys())
            values = list({**plot_results, 'iteration_time': 0}.values())
            f.write(','.join(keys) + '\n')  # Write the keys as the header row, each key in a separate column
            f.write(','.join(map(str, values)) + '\n')  # Write the values as the second row, each value in a separate column

    return {**plot_results, 'iteration_time': np.nan} 


def summarize_simulation_runs(run_results, quantiles=(0.1, 0.9)):
    """Summarize run-level outcomes using a median and configurable quantiles."""
    quantiles = tuple(quantiles)
    if len(quantiles) != 2 or any(q < 0 or q > 1 for q in quantiles):
        raise ValueError("quantiles must contain two values between 0 and 1")
    if quantiles[0] > quantiles[1]:
        raise ValueError("quantiles must be in ascending order")

    summary = {}
    outcome_names = [
        name for name in run_results.columns
        if name not in {"seed", "iteration_time", "config_signature"}
    ]
    for name in outcome_names:
        values = pd.to_numeric(run_results[name], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            summary[f"{name}-median"] = np.nan
            for quantile in quantiles:
                summary[f"{name}-q{quantile:g}"] = np.nan
            continue
        summary[f"{name}-median"] = float(np.median(values))
        for quantile in quantiles:
            summary[f"{name}-q{quantile:g}"] = float(np.quantile(values, quantile))
    return summary


def run_replicates(params, simulation_config, folder, quantiles=(0.1, 0.9)):
    """Run or load the requested seeds, then save run and summary CSV files."""
    params = deepcopy(params)
    simulation_config = dict(simulation_config)
    n_simulations = int(simulation_config.pop("n_simulations", 1))
    base_seed = int(simulation_config.pop("seed", 42))
    simulation_config.pop("output_folder", None)
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    os.makedirs(folder, exist_ok=True)
    run_rows = []
    runs_path = os.path.join(folder, "simulation_runs.csv")

    for run_index in range(n_simulations):
        seed = base_seed + run_index
        run_folder = os.path.join(folder, f"simulation_{run_index:03d}")
        result_path = os.path.join(run_folder, "simulation_results.csv")

        # Reuse an existing replicate result; never overwrite it by rerunning.
        if os.path.exists(result_path):
            outcomes = load_results_from_csv(run_folder)
            run_rows.append({"seed": seed, **outcomes})
            continue

        os.makedirs(run_folder, exist_ok=True)
        outcomes = simulate_and_plot(
            params=deepcopy(params),
            seed=seed,
            folder=run_folder,
            **simulation_config,
        )
        run_rows.append({"seed": seed, **outcomes})

    run_results = pd.DataFrame(run_rows)
    summary = summarize_simulation_runs(run_results, quantiles=quantiles)
    # Keep existing aggregate files untouched.
    if not os.path.exists(runs_path):
        run_results.to_csv(runs_path, index=False)
    summary_path = os.path.join(folder, "simulation_results.csv")
    if not os.path.exists(summary_path):
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary


def run_simulation_and_collect_results(config, param_ranges, folder=None, reevaluate_simulations=False, replace_files=False, prep_array_job=False):
    """
    Run systematic simulations, varying up to two parameters, and collect outcomes in a DataFrame.
    Optionally prepares simulation directories and configuration files for cluster array jobs.

    Parameters:
    - config (dict): Path to base YAML configuration file.
    - param_ranges (dict): Dictionary of parameters to vary and their ranges.
    - folder (str): Base folder to save simulation results.
    - reevaluate_simulations (bool): Whether to rerun simulations for existing parameters.
    - replace_files (bool): Whether to replace existing files if they exist.
    - prep_array_job (bool): Whether to prepare simulation directories for array jobs.

    Returns:
    - pd.DataFrame: DataFrame containing the results of the simulations.
    """
    results = []
    param_names = list(param_ranges.keys())

    with open(config, 'r') as file:
        base_config = yaml.safe_load(file)


    # Prepare directories and configuration files for array jobs if requested
    if prep_array_job:
        array_job_dirs = []

    # Iterate over all combinations of parameter values
    for param1_value in param_ranges[param_names[0]]:
        for param2_value in param_ranges[param_names[1]] if len(param_names) > 1 else [None]:
            print(f"{param_names[0]}: {param1_value}, {param_names[1]}: {param2_value}" if param2_value is not None else f"{param_names[0]}: {param1_value}")
            # Create subfolder for this parameter combination
            subfolder_name = f"{param_names[0]}_{param1_value}"
            if param2_value is not None:
                subfolder_name += f"_{param_names[1]}_{param2_value}"
            subfolder_path = os.path.join(folder, subfolder_name)

            config = deepcopy(base_config)  # Create a copy of the base config for each iteration
            # Update parameters in config
            if param_names[0] in config['parameters']:
                config['parameters'][param_names[0]] = param1_value
            else:
                config['simulation'][param_names[0]] = param1_value
            if param2_value is not None:
                if param_names[1] in config['parameters']:
                    config['parameters'][param_names[1]] = param2_value
                else:
                    config['simulation'][param_names[1]] = param2_value

            # Prepare array job directories if requested
            if prep_array_job:
                if os.path.exists(os.path.join(subfolder_path, 'simulation_results.csv')) or os.path.exists(os.path.join(subfolder_path, 'simulation_results.pkl')):
                    print(f"{subfolder_path}/simulation_results.csv' or {subfolder_path}/simulation_results.pkl' does exist for {param1_value}, {param2_value}, This combination is not included in the job array.")
                    continue
                os.makedirs(subfolder_path, exist_ok=True)
                config_path = os.path.join(subfolder_path, 'parameters.yaml')
                with open(config_path, 'w') as config_file:
                    yaml.dump(config, config_file) # some characterst seem odly formated, but this is no issue when reopening them. 
                array_job_dirs.append(subfolder_path)
                continue  # Skip running simulations if preparing array jobs

            if (replace_files or reevaluate_simulations) and os.path.exists(subfolder_path):
                shutil.rmtree(subfolder_path)
            params = objects.Parameters(**config['parameters'])
            outcomes = run_replicates(
                params=params,
                simulation_config=config['simulation'],
                folder=subfolder_path,
            )
            result = {param_names[0]: param1_value}
            if param2_value is not None:
                result[param_names[1]] = param2_value
            result.update(outcomes)  # Merge outcomes into result dictionary
            results.append(result)

    # Save array job directories to a file if requested
    if prep_array_job:
        array_job_file = os.path.join(folder, "slurm_array_dir_file.txt")
        with open(array_job_file, 'w') as f:
            f.write("\n".join(array_job_dirs))
        print(f"Prepared {len(array_job_dirs)} simulation directories.")
        return None

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Preserve distinct replicate rows when combining with previous sweep data.
    output_file = os.path.join(folder, "aggregated_results.csv")
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        merge_columns = param_names or list(df.columns[:1])
        merged_df = pd.concat([existing_df, df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=merge_columns, keep='last')
        merged_df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)

    return df

def run_simulation(config_file, folder=None):
    """
    Run a single simulation using parameters from a YAML file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.
    - folder (str): Optional folder to save the results.

    Returns:
    - dict: Results of the simulation.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Repeated runs share the configured parameters and use consecutive seeds.
    params = objects.Parameters(**config['parameters'])
    return run_replicates(
        params=params,
        simulation_config=config['simulation'],
        folder=folder or config['simulation'].get('output_folder', "simulation_results"),
    )

if __name__ == "__main__":
    """
    Main entry point for running systematic simulations via command line or as a script.

    This block handles different scenarios based on the existence of the output folder and command-line arguments:
    - OPTION 1: If a new folder is specified, run new simulations and save results in the new folder.
    - OPTION 2: If reevaluate_simulations is True, load existing results and plot them, but run new simulations for new parameters (using the existing YAML config file).
    - OPTION 3: If an existing folder is specified, load the existing DataFrame and append new results to it. Replace old results if they already exist (using the existing YAML config file).
    - OPTION 4: If prep_array_job=True, prepare the folders and configuration files for array jobs, without running simulations.
    - OPTION 5: If the folder exists but aggregated_results.csv does not, load existing results by opening the individual CSV files in subfolders.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run systematic simulations.")
    parser.add_argument("--output_folder", type=str, required=True, default="output", help="Output folder for simulation results.")
    parser.add_argument("--param_ranges", type=str, required=True, default="{}", help="Parameter ranges as a YAML string.")
    parser.add_argument("--config", type=str, required=False, default="parameters/ageing.yaml", help="Path to the configuration file (optional if output_folder already exists).")    
    parser.add_argument("--replace_files", action="store_true", required=False, default=False, help="Replace existing files if they exist.")
    parser.add_argument("--reevaluate_simulations", action="store_true", required=False, default=False, help="Reevaluate simulations for existing parameters.")
    parser.add_argument("--prep_array_job", action="store_true", required=False, default=False, help="Prepare folders for array jobs.")
    
    # Check if the script is run without command-line arguments
    if len(sys.argv) < 2:
        print("No command-line arguments provided. Running with default parameters.")
        config = "parameters/figS4_introduction.yaml" #'parameters/figS3_introduction.yaml'
        with open("parameters/figS4_introduction_ranges.yaml", 'r') as file:
            param_ranges = yaml.safe_load(file)
        replace_files, reevaluate_simulations, prep_array_job = False, False, False
    else:
        args = parser.parse_args()
        with open(args.param_ranges, 'r') as file:
            param_ranges = yaml.safe_load(file)
        output_folder, config, replace_files, reevaluate_simulations, prep_array_job = args.output_folder, args.config, args.replace_files, args.reevaluate_simulations, args.prep_array_job         # Extract arguments
    with open(config, 'r') as file: # NEW 
        config_content = yaml.safe_load(file)
        output_folder = config_content['simulation']['output_folder']
        
    assert len(param_ranges.keys()) <= 2, "Error: A maximum of two parameters can be varied at a time." # Assertion to ensure no more than two parameters are varied

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        if not config:
            raise ValueError("A configuration file must be provided when creating a new output folder.")
        # Copy the config file to the output folder
        config_copy_path = os.path.join(output_folder, 'parameters.yaml')
        with open(config, 'r') as src, open(config_copy_path, 'w') as dst:
            dst.write(src.read())
    else:
        config = os.path.join(output_folder, "parameters.yaml")
        if not os.path.exists(config):
            raise FileNotFoundError(f"The configuration file '{config}' does not exist in {output_folder}.")
    with open(config, 'r') as file:
        config_content = yaml.safe_load(file)

    df = run_simulation_and_collect_results(
        config=config,
        param_ranges=param_ranges,
        folder=output_folder,
        replace_files=replace_files,
        prep_array_job=prep_array_job,
        reevaluate_simulations=reevaluate_simulations
    )
    
    if not prep_array_job:
        # Load the aggregated results DataFrame
        df = pd.read_csv(os.path.join(output_folder, "aggregated_results.csv"))

        # Fill missing combinations of parameter values with NA
        param_keys = list(param_ranges.keys())
        if len(param_keys) == 2:
            idx = pd.MultiIndex.from_product([param_ranges[param_keys[0]], param_ranges[param_keys[1]]], names=[param_keys[0], param_keys[1]])
            df = df.set_index([param_keys[0], param_keys[1]]).reindex(idx).reset_index()
        elif len(param_keys) == 1:
            idx = pd.Index(param_ranges[param_keys[0]], name=param_keys[0])
            df = df.set_index(param_keys[0]).reindex(idx).reset_index()

        if 'σ' in param_keys :
            sigma_values = df['σ'].astype('string')
            df['control'] = sigma_values.str.contains('control_', na=False).astype(int)
            df['σ'] = sigma_values.str.replace(r'control_', '', regex=True)
            param_keys.append('control')
            param_ranges = {'σ': [0, 0.001, 0.01, 0.1], 'control':[0,1]}


        # Define parameter and outcome names
        param_names = {
            'c': "Mixing",
            'σ': "Stochastic immune\nvariation",
            'd': "Cross-immunity \n breadth $d$",
            't_iteration': "Time between \n strain introduction (δ)",
            'N': "Maximum number of strains $N_{max}$",
            'γ': "Ageing rate $γ$",
            "M": "Number of age groups $M$",
            "ageing": "Ageing",
            "α": "Transmission rate \n $\\alpha$",
            "δ": "Recovery rate \n $\\delta$",
            "control" : "Negative control"
        }


        # Define outcomes of interest for plotting
        outcomes_of_interest = {
            'exposure_other': 'Between-\nsubpopulation\nexposure',
            'diversity_infected_cos': 'Strain \ncirculation \ndifferences',
            'diversity_susceptibles_cos': 'Immune\ndifferences',
            'entropy': 'Strain\ncoexistence',
            'mrca_times_pairwise': 'Phylogenetic\ndiversity',
        }
        
        plot_simulation_results(
            df_original=df, # NOTE: when median is used to plot, it will be very low for situations where the variant was extinct for a long time
            param_ranges=param_ranges,
            outcomes_of_interest=outcomes_of_interest,
            param_names=param_names,
            folder=output_folder, log_scale=False,
            quantiles= (0.1, 0.9),
        )
