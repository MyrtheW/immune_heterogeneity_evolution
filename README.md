# mixing_evolution

This repository contains code for running and analyzing mixing evolution simulations.

## Overview

The main scripts in this repository are:

- **single_run.py**: Run a single simulation with a YAML config file and optional output folder. It calls 'systematic_simulations.py'. See usage below.
- **systematic_simulations.py**: Run a batch of simulations by varying parameters, using a YAML config file and a YAML file specifying parameter ranges. See usage below. 


Other scripts:
- **model/dynamics.py**: Contains the ODE's in the function 'SIR_'
- **model/objects.py**: Python classes are defined here. Important functions here are the adding and removal of variants. 
- **model/tools.py**: contains functions used for outcome calculations and other helper functions, such as the calculation of the cross-immunity. 
- **analyze_and_plotting.py**: contains various functions to aggregate and plot outcomes. 


## Requirements
The following third-party Python packages are required:

- `pandas` 2.3.2
- `NumPy` 2.2.3
- `SciPy` 1.15.2
- `PyYAML` 6.0.2

```bash
pip install pandas numpy pyyaml scipy pickle 
```
<!-- 1. Clone this repository.
1. Install required Python packages (if any). For example:
   ```bash
   pip install -r requirements.txt # these are the requirments: matplotlib=3.10.1 numpy=2.2.3 pandas=2.2.3 scipy=1.15.2 pyyaml networkx (for plotting)
   ```
2. (Optional) Set up your SLURM environment if running on a cluster. -->

## Usage

### Run a Single Simulation
Runs a single simulation for one parameter set. For long simulation times (`t_max`), simulations and plotting may take considerable time. Note that, when `t_max` - `t_burn_in` is long (i.e. more than 100 variants), metrics of variant diversity (t_MRCA, cross-immunity and mutational distance) and the phylogeny will not be plotted by default, as this is computationally expensive and not optimized. 
```bash
python single_run.py <config_file> [output_folder]
```
- `<config_file>`: Path to a YAML configuration file (required). See `parameters/ageing.yaml` for an example, and the meaning of the parameters.
- `[output_folder]`: Optional output folder for results. A figure with outcomes over time will be produced, and a sepearate plot with the tree. 

To see help:
```bash
python single_run.py --help
```

Alternatively, parameter and output paths can be defined in the script itself. 

### Run Systematic Simulations
Runs multiple simulations, with up to two parameters varied, and plots the cumulative outcomes. The baseline parameters and the parameters to be varied are to be defined in two seperate yaml files. See `parameters/ageing.yaml` and `ranges.yaml` for examples. 
For each simulation, a seperate folder will be made within the output folder, containing the results. 

```bash
python systematic_simulations.py --output_folder <folder> --param_ranges <ranges_yaml> --config <config_yaml> [--replace_files] [--reevaluate_simulations] [--prep_array_job]
```
- `--output_folder`: Output folder for simulation results (required).
- `--param_ranges`: Path to a YAML file specifying parameter ranges to sweep (required).
- `--config`: Path to the base YAML configuration file (required if creating a new output folder).
- `--replace_files`: Replace existing files for single simulation if they exist (optional flag).
- `--reevaluate_simulations`: Reevaluate simulations for existing parameters (optional flag).
- `--prep_array_job`: Prepare directories for SLURM array jobs (optional flag).

Alternatively, parameter and output paths can be defined in the script itself. 


### Notes on the model implementation 
The main model equations are presented in the file **model/dynamics.py** in the function 'SIR_'.

Note that the $S$ and $I$ compartments do not represent the entire population (i.e., $I_i^a + S_i^a \neq 1$). To account for the remaining population, we also track $R_i^a$, individuals recovered from strain $a$ in compartment $i$, and $R_i^{\prime a}$, individuals infected or recovered with a strain other than $a$ who thereby acquired cross-immunity to strain $a$. Their dynamics are:

$$
\frac{dR_i^a}{dt} = \delta I_i^a - \gamma R_i^a + \gamma\left(R_{i-1}^a - R_i^a\right)
$$

$$
\frac{dR_i^{\prime a}}{dt} = \alpha S_i^a \sum_{j=1}^{M} C_{ij} \sum_{b \neq a}^{N} K_i^{ba} I_j^b - \gamma R_i^{\prime a}
$$



 Numpy was utilized to accelerate the calculation of the ODEs through vectorized operations

- $\mathbf{K}^a \in \mathbb{R}^N$ is the cross-immunity vector for strain $a$ to the other $N$ strains.
- $\mathbf{I}(t) \in \mathbb{R}^{M \times N}$ is the matrix of infections across all subpopulations and strains.
- $\mathbf{I}^a(t) \in \mathbb{R}^M$ is the vector of infections across all subpopulations for strain $a$.
- $\mathbf{S}^a(t) \in \mathbb{R}^M$ is the vector of susceptibilities across all subpopulations for strain $a$.
- $\mathbf{C} \in \mathbb{R}^{M \times M}$ is the contact matrix.
- $\mathbf{1}_N \in \mathbb{R}^N$ is a vector of ones used to sum across strains (columns).
- $\mathbf{F} \in \mathbb{R}^{M \times M}$ is the flux matrix describing aging between age groups.
- $\mathbf{e}_1 \in \mathbb{R}^M$ is the first standard basis vector, representing births in the first age group.

Here, $M$ is the number of subpopulations and $N$ is the number of strains. The symbol $\odot$ denotes element-wise multiplication.

**Equations without aging**
$$
\dot{\mathbf{I}}^a(t) = \alpha \mathbf{S}^a(t) \odot \bigl(\mathbf{C}\mathbf{I}^a(t)\bigr) - (\delta + \gamma)\mathbf{I}^a(t)
$$

$$
\dot{\mathbf{S}}^a(t) = -\alpha \mathbf{S}^a(t) \odot \bigl((\mathbf{C}\mathbf{I}(t)) \odot \mathbf{K}^a\bigr)\mathbf{1}_N + (1-\gamma)\mathbf{S}^a(t)
$$

**Equations with aging**
$$
\dot{\mathbf{I}}^a(t) = \alpha \mathbf{S}^a(t) \odot \bigl(\mathbf{C}\mathbf{I}^a(t) - \delta\mathbf{I}^a(t) + \gamma\mathbf{F}\mathbf{I}^a(t)\bigr)
$$

$$
\dot{\mathbf{S}}^a(t) = -\alpha \mathbf{S}^a(t) \odot \bigl((\mathbf{C}\mathbf{I}(t)) \odot \mathbf{K}^a\bigr)\mathbf{1}_N + \gamma\bigl(\mathbf{F}\mathbf{S}^a(t) + \mathbf{e}_1\bigr)
$$

The aging flux matrix is

$$
\mathbf{F} = \frac{1}{M}
\begin{bmatrix}
-1 & 0 & \cdots & 0 \\
1 & -1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & -1
\end{bmatrix}.
$$