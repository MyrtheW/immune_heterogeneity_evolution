import time
import argparse
import matplotlib.pyplot as plt
import yaml
from systematic_simulations import load_and_plot_simulation_results, run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single mixing evolution simulation with a YAML config file."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="parameters/fig1d.yaml",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        nargs="?",
        default="",

    )
    parser.add_argument(
        "--load_existing",
        action="store_true",
        help="Load existing results"
    )
    args = parser.parse_args()

    config_file = args.config
    folder = args.output_folder
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        folder = config['simulation']['output_folder']

    if args.load_existing:
        load_and_plot_simulation_results(output_folder=args.output_folder)
    else:
        start_time = time.time()
        default_results = run_simulation(config_file=config_file, folder=folder)
        simulation_time = time.time() - start_time

        print(f"Simulation completed in {simulation_time:.2f} seconds")
        plt.show()

