import time
from matplotlib import pyplot as plt
from model.tools import * 
import numpy as np
import os
import networkx as nx
import sys
import random
from scipy.optimize import curve_fit
import copy
import matplotlib as mpl
import pandas as pd

sys.path.insert(0, "/Users/mwille26/Library/Python/3.13/lib/python/site-packages")
    

####################
# Analyze results 
####################

def exposure_other_groups(sol, tvals=None, virus=None, g="I"):
    """
    Calculate the exposure of each group to other groups.
    """
    C = sol.parameters.C
    I_xM_xt = group_prevalence_xt(sol, tvals=tvals, virus=virus, g="I")
    # set off diagonal elements to 0
    exposure_within_group = C* np.identity(len(C)) @ I_xM_xt  # multiply with the second axis of I_xM_xt
    # select only the off-diagonal elements of C
    force_of_infection =  (C @ I_xM_xt) 
    return exposure_within_group / force_of_infection


def distance_xt_2(frequencies_xt, final_state, metric='mrca'):
    average_pairwise_distances = []
    for t in range (frequencies_xt.shape[1]):
        non_zero_indices = np.nonzero(frequencies_xt[:, t])[0]
        if len(non_zero_indices) > 2:
            if metric == 'mrca':
                numerator = np.sum([frequencies_xt[i, t] * frequencies_xt[j, t] * final_state.tmrca_all[i, j] for i in non_zero_indices for j in non_zero_indices if i != j], axis=0)
            elif metric == 'mut_distance':
                numerator = np.sum([frequencies_xt[i, t] * frequencies_xt[j, t] * final_state.distances_all[i, j] for i in non_zero_indices for j in non_zero_indices if i != j], axis=0)
            denominator = np.sum(frequencies_xt[:, t] * (1 - frequencies_xt[:, t]), axis=0)  # sum of f_a(t) * f_b(t) for all pairs a != b, is equavalent to np.sum([frequencies_xt[i] * frequencies_xt[j] for i in range(N) for j in range(N) if i != j], axis=0)
            average_pairwise_distances.append(numerator / denominator) 
        else:
            average_pairwise_distances.append(0)  
    return average_pairwise_distances


def tmrca_xt(time_points, sol, strains_ids_xt, final_state, tvals=None, t_steps=4): # This function is an additional metric, used to plot and isnpect results over time. 
    """
    Calculate the time to the most recent common ancestor (tMRCA) for strains at different time points.
    """
    ts = []
    mrca_times = []
    for idx, (strain_ids, ti) in enumerate(zip(strains_ids_xt, time_points)):
            if idx == 0:
                continue  # Skip the first iteration as there is no previous time point
            reduced_tmrca_matrix = final_state.tmrca_all[np.ix_(strain_ids,strain_ids)] # extract the reduced tmrca matrix for the current strains
            t_oldest = np.min(reduced_tmrca_matrix)
            relevant_tvals = tvals[(tvals >= time_points[idx - 1]) & (tvals <= time_points[idx])]
            if len(relevant_tvals) > t_steps:
                relevant_tvals = np.linspace(relevant_tvals[0], relevant_tvals[-1], t_steps)
            for t in relevant_tvals:
                ts.append(t)
                mrca_times.append(t - t_oldest)
    return mrca_times, ts

def group_prevalence_xt(sol, tvals=None, virus=None, g="I"): # This function is an additional metric, used to plot and isnpect results over time. 
    """
    Calculate the prevalence of a specific group (infected or susceptible) over time for each region.

    Parameters:
    - sol: The solution object containing simulation results.
    - tvals: Time values for plotting. If None, defaults to 1000 evenly spaced points in the solution's time span.
    - virus: Specific virus index to plot. If None, plots all viruses.

    - per_region: Whether to plot prevalence for each region separately.
    - g: Group to plot ('I' for infected, 'S' for susceptible).
    """
    # Calculate average prevalence of Ia of Sa*fa.
    if g == "I" :
        prevalences = prevalence(virus=virus, sol=sol, tvals=tvals, g=g, region=None, mean_region=False, mean_virus=False)  #/ sol.parameters.M
        prevalences = np.vstack([np.sum(prevalences[range(i, sol.parameters.N * sol.parameters.M, sol.parameters.M),:], axis=0) for i in range(sol.parameters.M)])
    if g == "S" :  # Normalize by the number of viruses for susceptibles.
        prevalences = prevalence(virus=virus, sol=sol, tvals=tvals, region=None, g=g, mean_region=False, mean_virus=False) # first by region, virus is more seperated 
        frequencies = frequency(X=None, virus=range(sol.parameters.N), region=None, sol=sol, tvals=tvals)
        prevalences = np.vstack([
            np.sum(prevalences[range(i, sol.parameters.N * sol.parameters.M, sol.parameters.M),:] * frequencies, axis=0)
            for i in range(sol.parameters.M)
        ]) # at this point prevalences is a matrix of size (M, tvals)
    return prevalences

def sweep_info_frequency(descendant_frequency,all_time_points, axs=None, threshold=1e-6):
    """
    # for each variant, extract info on duration of sweep 
    # Does it reach 1? 
    """ 
    sweep_durations = [] # length of #variants 
    peak_infections = []
    monotonicity = [] 
    fixated = []
    for variant in range(len(descendant_frequency)):
        I = descendant_frequency[variant]
        above_threshold = I > threshold
        if np.any(above_threshold): # collect these metrics, duration and peak infection in a list
            times_above = all_time_points[above_threshold]
            start_time = times_above[0]
            end_time = times_above[-1]
            sweep_durations.append(end_time - start_time)
            peak_infections.append(np.max(I))
            idx_max = np.argmax(I[above_threshold])
            monotonicity.append(np.all(np.diff(I[above_threshold][:idx_max]) >= 0)) # for as long as it is increasing
            fixated.append(np.any(I >= 0.99))
        else:
            sweep_durations.append(np.nan)
            peak_infections.append(np.nan)
            monotonicity.append(np.nan)
            fixated.append(np.nan)
    # percentage of True in monotonicity
    print(f"Monotonicity: {len([m for m in monotonicity if m is np.True_]) / len([m for m in monotonicity if m is not np.nan]) * 100:.2f}%")
    # print percentage of fixated variants
    print(f"Fixated: {len([f for f in fixated if f is np.True_]) / len([f for f in fixated if f is not np.nan]) * 100:.2f}%")
    print(f"Median sweep duration: {np.nanmedian(sweep_durations):.2f}")
    if axs is not None:
        axs[3,1].hist(sweep_durations, bins=20, range=(0, np.nanmax(sweep_durations) if np.nanmax(sweep_durations) > 0 else 1))
        axs[3,1].set_xlabel("Sweep Duration")
        axs[3,1].set_ylabel("Frequency \n sweep duration")
        axs[4,1].hist(peak_infections, bins=20, range=(0,  np.nanmax(peak_infections) if np.nanmax(peak_infections) > 0 else 1))
        axs[4,1].set_ylabel("Frequency \n peak height")
        axs[4,1].set_xlabel("Max frequency ")
    return sweep_durations, peak_infections, monotonicity, fixated

def descendents_frequency_xt(infected_by_strain, sol, tvals=None, virus=None, x_min=0, root_strain=None):
    """
    Calculate the frequency of a strain including its descendents over time.
    """
    descendent_frequencies = copy.deepcopy(infected_by_strain)
    parent_frequencies = np.zeros_like(descendent_frequencies) 
    def traverse_tree_post_order(strain, descendent_frequencies, parent_frequencies, infected_by_strain): 
        strain_id = strain.id
        for i, child in enumerate(strain.children):
            child_name = child.id
            if child_name < len(descendent_frequencies) and strain_id < len(descendent_frequencies):
                parent_frequencies[child_name] = parent_frequencies[strain_id] + infected_by_strain[strain_id]
            descendent_frequencies, parent_frequencies = traverse_tree_post_order(child, descendent_frequencies,  parent_frequencies, infected_by_strain)
            if strain.t0 > x_min:  # Only consider strains that appeared after x_min
                if child_name < len(descendent_frequencies) and strain_id < len(descendent_frequencies):
                    descendent_frequencies[strain_id] += descendent_frequencies[child_name]
            else: 
                descendent_frequencies[strain_id] = descendent_frequencies[strain_id] * 0
        return descendent_frequencies, parent_frequencies
            # add the child's time points and I values to the parent's
    descendent_frequencies, parent_frequencies = traverse_tree_post_order(root_strain, descendent_frequencies, parent_frequencies, infected_by_strain)
    return descendent_frequencies, parent_frequencies

def get_common_time_grid(I_xt_region, x_min):
    """
    Get a common time grid for all strains in a region, starting from x_min.
    """
    valid_strains = [
        (times, values)
        for times, values in I_xt_region
        if len(times) > 0 and times[0] > x_min
    ]
    if not valid_strains:
        return np.array([]), np.zeros((len(I_xt_region), 0))

    common_time_grid = np.unique(np.concatenate([times for times, _ in valid_strains]))
    infected_by_strain = np.zeros((len(I_xt_region), len(common_time_grid)))
    for strain_index, (times, values) in enumerate(I_xt_region):
        if len(times) > 0 and times[0] > x_min:
            infected_by_strain[strain_index] = np.interp(
                common_time_grid, times, values, left=0, right=0
            )
    return common_time_grid, infected_by_strain


def predictability_of_evolution(descendent_frequencies, time_points, frequency_of_interest=0.5): # Only needed for supplementary figure 
    """
    normalize all frequencies to the frequency of interest (e.g. 50%) and see how many variants continue from there onwards that frequency
    """
    normalized_grid = []
    common_time_grid = np.arange(-300, 1001, step=1)

    for strain_idx in range(len(descendent_frequencies)):
        # first, find the time point where descendent frequency crosses frequency_of_interest
        t_idx = np.where(descendent_frequencies[strain_idx] >= frequency_of_interest)[0]
        if len(t_idx) > 0:
            normalized_grid.append(np.interp(common_time_grid, time_points - time_points[t_idx[0]], descendent_frequencies[strain_idx], left=0, right=0))
        else: normalized_grid.append(np.full_like(common_time_grid, np.nan))
    return common_time_grid, normalized_grid


####################
# Plotting functions
####################

def set_ylims_based_on_x(ax, x, y):
    """Set y-limits based on the y-data within the current x-limits."""
    xlim = ax.get_xlim()
    mask = (x >= xlim[0]) & (x <= xlim[1])
    if np.any(mask):
        y_visible = np.array(y)[mask]
        if np.any(np.isfinite(y_visible)):
            ymin = np.nanmin(y_visible)
            ymax = np.nanmax(y_visible)
            if ymin == ymax:
                ymin -= 0.05 * abs(ymin) if ymin != 0 else -0.05
                ymax += 0.05 * abs(ymax) if ymax != 0 else 0.05
            ax.set_ylim(ymin, ymax)

def plot_entropy_xt(variant_entropy_xt, sol, tvals=None, lim=False, region=None, ax=None):
    """
    Plot the entropy of the strain frequencies over time.
    """
    if tvals is None:
        tvals = np.linspace(sol.tspan[0], sol.tspan[1], 1000)
    if ax is None:
        ax = plt.gca()
    for region in range(sol.parameters.M):
        entropy_xt_region = entropy_xt(sol, tvals, region=region)
        ax.plot(tvals, entropy_xt_region, color=f"C{region}", label=f"Age {region + 1}")
    ax.plot(tvals, variant_entropy_xt,  color='black', label="Entropy of circulating strains") #, linestyle='dashdot'
    ax.set_ylabel("Entropy of \n circulating strains")
    ax.set_xlabel("Time")
    if lim: ax.set_ylim(0, 1)

def plot_g_diversity_xt(diversity_matrix, tvals=None,  sol=None, virus=None, ax=None, g="I", per_region=True, metric='euclidean', lim=True, title="Diversity of \n infections \n"):
    """
    Plot the diversity of infections over time.
    """
    if tvals is None:
        tvals = np.linspace(sol.tspan[0], sol.tspan[1], 1000)
    if ax is None:
        ax = plt.gca()
    if diversity_matrix.ndim == 1:
        average_diversities = diversity_matrix.flatten()  
    elif diversity_matrix.ndim > 1:
        for i in range(diversity_matrix.shape[0]):
            ax.plot(tvals, diversity_matrix[i,:].flatten(), label=f" {i+1}", color=f"C{i}", alpha=0.5, linewidth=0.7)
        average_diversities = np.mean(diversity_matrix, axis=0)
    ax.plot(tvals, average_diversities, label="Average", color='black')
    set_ylims_based_on_x(ax, np.array(tvals), np.array(average_diversities))
    if g == "I":
        ax.set_ylabel("Diversity of Infections" + f"({metric})")
        if lim: ax.set_ylim(0, 0.1)
    elif g == "S":
        ax.set_ylabel("Diversity of Susceptibles" + f"({metric})")
        if lim: ax.set_ylim(0, 0.5)
    ax.set_xlabel("Time")


def plot_group_prevalence_xt(sol, prevalences, tvals=None, ax=None, per_region=True, g="I", continous_colors=False, fluctuation_period=None):
    """
    Plot the prevalence of a specific group (infected or susceptible) over time for each region.
    """

    if tvals is None:
        tvals = np.linspace(sol.tspan[0], sol.tspan[1], 1000)  # Default time values if not provided.
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axis if none is provided.

    ax.plot(tvals, np.mean(prevalences, axis=0), color='black', alpha=0.5)  # Plot the average prevalence.
    if per_region: 
        for i in range(sol.parameters.M):  # Loop through each region.
            colors = [plt.cm.bwr(i / (sol.parameters.M - 1)) if continous_colors else f"C{i}" for i in range(sol.parameters.M) ]
            ax.plot(tvals, prevalences[i,:], label=f"{i+1}", color=colors[i], alpha=0.5, linewidth=0.7)
    if fluctuation_period:
        ax.plot([tvals[-1] - fluctuation_period, tvals[-1]], [1e-3, 1e-3], color='gray', linestyle='-', linewidth=0.5) 
    if g == 'I':  # Infected group.
        ax.set_yscale('log')  # Logarithmic scale for infected group.
        ax.set_ylim(1e-7, 1)  # Set y-axis limits.
        ax.set_ylabel("Infections \n by age group $I_i(t)$")
    elif g == 'S':  # Susceptible group.
        ax.set_ylabel("Susceptible \n by age group $S_i(t)$")
        try:
            delta_alpha = (sol.parameters.δ + sol.parameters.γ) / sol.parameters.α  # Calculate δ/α ratio.
        except:
            delta_alpha = None
        ax.axhline(y=delta_alpha, color='gray', linestyle='--', linewidth=0.5)  # Add a horizontal line for δ/α.
        ax.axhline(y= sol.parameters.δ / sol.parameters.α, color='gray', linestyle='--', linewidth=0.5)  # Add a horizontal line for δ/α.
    ax.set_xlabel("Time")


def plot_group_distribution_prevalence(sol, mean_prevalences, tvals=None, lim=False, g="I", ax=None, continous_colors=False):
    """
    Plot the distribution of mean prevalences across age groups.
    """
    if tvals is None:
        tvals = np.linspace(sol.tspan[0], sol.tspan[1], 1000)
    if ax is None:
        ax = plt.gca()
    # colors = [plt.cm.bwr(i / (sol.parameters.M - 1)) if continous_colors else f"C{i}" for i in range(sol.parameters.M) ]
    ax.bar([f"{i+1}" for i in range(sol.parameters.M)], mean_prevalences, color="lightgrey", edgecolor='None')
    ax.set_xlabel("Age group")
    if lim: ax.set_ylim(0, 0.5)
    if g == "I":
        ax.set_ylabel("Infection \nprevalence")
        ymax = np.nanmax(mean_prevalences)
    elif g == "S":
        ax.axhline(y= (sol.parameters.δ + sol.parameters.γ) / sol.parameters.α, color='gray', linestyle='--', linewidth=0.5)  # Add a horizontal line for δ/α.
        ax.axhline(y= sol.parameters.δ / sol.parameters.α, color='gray', linestyle='--', linewidth=0.5)  # Add a horizontal line for δ/α.
        ax.set_ylabel("Susceptibles")
        ymax = 1
    ax.set_ylim(0, ymax)
    return mean_prevalences


def plot_tree(root_strain, name_attribute=None, color_attribute=None, scale_by='t0', color_dict=None, ax=None, node_sizes=None, node_heights=None, xmin=0, xmax=None):
    """
    Plot the phylogenetic tree, giving nodes a color based on one of the attributes and scaling branches.
    """
    G = nx.DiGraph()  # Create a directed graph to represent the tree
    pos = {}  # Dictionary to store node positions
    labels = {}  # Dictionary to store node labels
    colors = {} # List to store node colors
    node_sizes2 = {}

    def add_edges(strain, pos_x=0, pos_y=0, depth=0):
        """
        Recursively add edges and nodes to the graph.

        Parameters:
        - strain: The current strain (node) being processed.
        - pos_x, pos_y: Current position of the node.
        - depth: Depth of the node in the tree.
        """
        # Determine node name and color based on attributes
        node_name = getattr(strain, name_attribute) if name_attribute else str(strain.id)
        node_color = getattr(strain, color_attribute) if color_attribute else str(strain.id)
        if strain.t0 > xmin - 100:
            # Set node position and label
            pos[node_name] = (float(strain.t0), -depth * 3) if strain.t0 else (0, -depth * 3)
            labels[node_name] = f"{node_name}\n{node_color}"
            colors[node_name] = node_color  # Store color for the node
            node_sizes2[node_name] = node_sizes[strain.id] if node_sizes is not None else 100  # Default size if not provided

        # Add edges for child nodes
        for i, child in enumerate(strain.children):
            child_name = getattr(child, name_attribute) if name_attribute else str(child.id)
            if scale_by == 'distance':
                branch_length = child.distance  # Scale by branch distance
            elif scale_by == 't0':
                branch_length = child.t0 - strain.t0  # Scale by time difference
            else:
                raise ValueError("scale_by must be 'distance' or 't0'")

            if strain.t0 > xmin - 100:
                G.add_edge(node_name, child_name, color= node_name)  # Add edge to the graph

            # Recursively add child nodes
            if node_heights is not None:
                add_edges(child, pos_x + i - len(strain.children) / 2, pos_y - branch_length, -node_heights[child.id])
            else:
                add_edges(child, pos_x + i - len(strain.children) / 2, pos_y - branch_length, depth + i - len(strain.children) / 2 + np.random.rand())

    add_edges(root_strain)  # Start adding edges from the root strain

    # Scale node sizes if provided
    # enumerate nodes in G, iterate over them
    node_names = list(G.nodes())
    node_sizes = np.array([node_sizes2[node] for node in node_names]) if node_sizes is not None else None
    if node_sizes is not None and len(node_sizes) > 0:
        min_size = 1e-5
        mask = node_sizes < min_size
        node_sizes[mask] = min_size
        node_sizes = np.log(node_sizes + 1e-9)  # Log-scale node sizes
        min_size, max_size = np.min(node_sizes), np.max(node_sizes)
        # Improved indices part: use numpy for masking and indexing
        node_sizes_arr = np.array([0 + 50 * (node_size - min_size) / (max_size - min_size) for node_size in node_sizes])
        node_sizes = node_sizes_arr
    else:
        node_sizes = np.ones(len(colors)) * 100  # Default node size

    # Map colors using the provided color dictionary or a default mapping
    colors = [color_dict[int(colors[node_name])] for node_name in node_names] if color_dict else map_colors_discrete([colors[node_name] for node_name in node_names]).values()

    if ax is None:
        ax = plt.gca()  # Use the current axis if none is provided
    # iterate over the nodes in G and find the y min and ymax
    ymin, ymax = float('inf'), float('-inf')
    for (x, y) in pos.values():
        if y < ymin and x > xmin and x < xmax: ymin = y
        if y > ymax and x > xmin and x < xmax: ymax = y

    # Draw the graph
    edge_colour_ids = list(nx.get_edge_attributes(G,'color').values())
    edge_colors = [color_dict[int(color)] for color in edge_colour_ids] if color_dict else map_colors_discrete(edge_colour_ids).values()
    nx.draw(G, pos, labels=labels, with_labels=False, node_color=colors, node_size=node_sizes, edge_color=edge_colors, font_size=8, ax=ax, width=0.5, arrows=False)

    # Set axis properties
    ax.set_ylabel("")
    ax.set_xlabel("Time")
    ax.set_yticks([])
    plt.xticks(rotation=45)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

    # Extract and return sorted time points
    time_points = sorted(pos[0] for pos in pos.values())
    return time_points, ymin, ymax



def plot_y_xt(time_points, y_values, ax=None, label=None):
    ax.plot(time_points, y_values, color='black', alpha=0.4, linestyle='-') 
    ax.axhline(y=np.mean(y_values), color='gray', linestyle='--', linewidth=0.5, label=f"Average ({np.mean(y_values):.2f})")
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylabel(label)
    ax.set_xlabel("Time")

def plot_initial_growth_rates(time_points, growth_rates_xt, color_dict, ax=None):
    """
    Plot the initial growth rates for each strain over time.
    """
    ax.set_ylabel("Initial \n growth rate")
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    for i in range((len(growth_rates_xt) - len(time_points)), len(growth_rates_xt)):
        ax.scatter(time_points[i - (len(growth_rates_xt) - len(time_points))], growth_rates_xt[i], label=f"Strain {i}", color=  list(color_dict[i]))


def map_colors_discrete(max_number_colors):
    """
    Map discrete colors for a given number of colors.
    """
    color_dict = {i: plt.cm.tab20b(i / max_number_colors) for i in range(max_number_colors)}
    values = list(color_dict.values())
    random.shuffle(values)
    return {i: values[i] for i in range(max_number_colors)}


def plot_infected_fractions(I_xt, color_dict, ax=None, label="Infections \n by strain $I^a(t)$", log_scale=True, xmin=None, xmax=None):
    """
    Plot the infected fractions over time.
    """
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1)
        ax.axhline(y=1e-6, color='gray', linestyle='--', linewidth=0.5)
    else: 
        ax.set_ylim(0, 1)
    ax.set_ylabel(label)
    ax.set_xlabel("Time")
    # mask the I_xt values outside the x limits

    for i in range(len(I_xt)): # I for each strain, with the (0) time points, and (1) I values and (2) strain ID. 
        # mask values outside the x limits
        if xmin is not None and xmax is not None:
            mask = (I_xt[i][0] >= xmin) & (I_xt[i][0] <= xmax)
            if np.any(mask):
                ax.plot(I_xt[i][0][mask], I_xt[i][1][mask], label=f"Strain {i}", color=list(color_dict[i]), alpha=1)
        else:
            ax.plot(I_xt[i][0], I_xt[i][1], label=f"Strain {i}", color=list(color_dict[i]), alpha=1)



def plot_infected_stack_subplot(I_xt, j, region_idx, color_dict, axs_j, x_min, x_max, y_max=0, normalize=True, common_time_grid_all=None, infected_by_strain_all=None):
    """
    Plot a stackplot of infected fractions for a specific region.
    """
    if region_idx == "all_regions":
        common_time_grid, infected_by_strain = common_time_grid_all, infected_by_strain_all
        colors = list(color_dict.values())[:len(I_xt[0])] # this is a fallback in case the previous line doesn't work, but it may not be correct if the order of strains in I_xt[region_idx] is different from the order of colors in color_dict.
    else: 
        if region_idx >= len(I_xt):
            return
        common_time_grid, infected_by_strain = get_common_time_grid(I_xt[region_idx], -1) #max(-1,x_min-1000))
        colors = list(color_dict.values())[:len(I_xt[0])] # this is a fallback in case the previous line doesn't work, but it may not be correct if the order of strains in I_xt[region_idx] is different from the order of colors in color_dict.
      
    # remove all elements with x values outside of x_min and x_max, in common_time_grid and infected_by_strain
    if x_min is not None and x_max is not None:
        mask = (common_time_grid >= x_min) & (common_time_grid <= x_max)
        common_time_grid = common_time_grid[mask]
        infected_by_strain = np.array(infected_by_strain)[:, mask]

    if normalize and len(infected_by_strain) > 0:
        infected_by_strain = np.array(infected_by_strain)
        total = np.sum(infected_by_strain, axis=0)
        axs2 = axs_j.twinx() 
        if y_max == 0:
            y_max = np.max(total)*2.5
        axs2.set_ylim(0, y_max)
        axs2.plot(common_time_grid, total, color='black')
        axs2.set_ylabel('Infection \n prevalence')

        # Avoid division by zero
        total[total == 0] = 1
        infected_by_strain = infected_by_strain / total


    axs_j.stackplot(common_time_grid, infected_by_strain, colors=colors, alpha=0.7)

    if region_idx == "all_regions":
        label = "Total population"
    else:
        label = f"Subpopulation {region_idx + 1}"
    axs_j.text(0.02, 0.95, label, transform=axs_j.transAxes, ha='left', va='top')
    if normalize: axs_j.set_ylim(0, 1)
    if x_min is not None and x_max is not None:
        axs_j.set_xlim(x_min, x_max)
        # rename axis labels, starting at 0 instead of x_min
        axs_j.set_xticks(np.arange(x_min, x_max + 1, step=round((x_max - x_min) / 5)))
        axs_j.set_xticklabels([f"{int(tick)-x_min}" for tick in np.arange(x_min, x_max + 1, step=round((x_max - x_min) / 5))], rotation=45, ha='right')

    return y_max

def plot_infected_stackplot(I_xt, color_dict, ax=None, regions_to_plot=[0, 1, 9], x_min=None, 
                            x_max=None, normalize=True, folder=None, infected_by_strain_all=None, common_time_grid_all=None):
    """
    Plot the infected fractions over time as a stackplot for each region, normalized to 100% at each time point.
    """
    fig, axs = plt.subplots(len(regions_to_plot)+1, 1, figsize=(5, 1*(len(regions_to_plot)+1)), sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0.3}) # also share secondary y-axis
 
    y_max = plot_infected_stack_subplot(I_xt, 0, "all_regions", color_dict, axs[-1], x_min, x_max, 0, normalize, common_time_grid_all, infected_by_strain_all)
    for j, region_idx in enumerate(regions_to_plot):
        # Find a common time grid for all strains in this region
        axs_j = axs[j]
        y_max = plot_infected_stack_subplot(I_xt, j, region_idx, color_dict, axs_j, x_min, x_max, y_max, normalize)    
    axs[-1].set_xlabel('Time')


    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.0)
    if folder is not None:
        fig.savefig(f"{folder}/stackplot_{normalize}.pdf", dpi=50)
    #return common_time_grid, infected_by_strain
    #plt.show()


def autocorrelation(x_xt):
    """
    Calculate the autocorrelation of a time series.
    """
    n = len(x_xt)
    mean = np.mean(x_xt)
    var = np.var(x_xt)
    result = np.correlate(x_xt - mean, x_xt - mean, mode='full') / (var) / np.correlate(np.ones(n), np.ones(n), mode='full')
    return result[result.size // 2:]

def autocorrelation_plot(autocorr, tvals, label=None, ax=None, fit=True):
    """
    Plot the autocorrelation of a time series.
    """
    if ax is None:
        ax = plt.gca()
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.plot(tvals - tvals[0], autocorr, label=label, alpha=0.4)
    ax.set_xlabel("Lag ($\\delta$)")
    ax.annotate(f"starting time t = {int(tvals[0])}", xy=(0, 1), xytext=(10, 10),
            textcoords='offset points', ha='center', va='bottom', fontsize=10, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
    ax.set_ylabel("Autocorrelation \n" + f"({label})")

    if fit:
        # Fit an exponential decay to the maximum value in each window
        def exp_func(x, a, b):
            x = np.array(x, dtype=np.float64)  # Ensure x is a numpy array
            return 1 * np.exp(-b * x)

        try:
            popt, _ = curve_fit(exp_func, tvals - tvals[0], autocorr, p0=(1, 1e-8))
            fit_curve = exp_func(tvals - tvals[0], *popt)

            print(1/popt[1])
            ax.plot(tvals  - tvals[0], fit_curve, linestyle='--', color=ax.lines[-1].get_color(), label=f"Exp fit: $a e^{{-b x}}$, 1/b={1/popt[1]:.0f}")
        except (RuntimeError, RuntimeWarning):
            None #ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes, color='red', fontsize=10, ha='center')

    ax.legend(loc='upper right', frameon=False)
    plt.show()


    
def burn_in_time(data, threshold=0.01, window_size=10):
    """
    Determine the burn-in time for the simulation to reach steady state.
    """
    data = np.asarray(data)  # Ensure data is a numpy array
    if data.ndim == 2:
        data = np.var(data, axis=0)  # Variance across compartments along the time axis
    elif data.ndim > 2:
        raise ValueError("Data must be either 1D or 2D.")

    if window_size > len(data):
        print("Window size is larger than the data length. Please choose a smaller window size.")
        window_size = round(len(data) / 10)
    moving_avg = np.array([np.mean(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])
    overall_mean = np.mean(data[len(data)//2:])
    steady_state_idx = np.where(abs(moving_avg - overall_mean ) < threshold)[0]
    # burn-in time = when |moving average – s.s. average|/s.s. average < threshold * (1 + variance / window_size)
    steady_state_variance = np.var(data[len(data)//2:])  # Calculate variance in the steady-state period
    adjusted_threshold = threshold * (1 + steady_state_variance / window_size)  # Adjust threshold based on variance and window size
    steady_state_idx = np.where(abs(moving_avg - overall_mean) / overall_mean < adjusted_threshold)[0]

    if len(steady_state_idx) > 0: 
        return steady_state_idx[0]  # Return the index of the burn-in time
    else:
        raise ValueError("Steady state not reached within the data.")

def _time_average(values, times=None):
    """
    Return the finite time average for one simulation outcome.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.nan
    if times is None:
        return float(np.mean(values[valid]))
    times = np.asarray(times, dtype=float)
    valid &= np.isfinite(times)
    if not np.any(valid):
        return np.nan
    if np.count_nonzero(valid) < 2 or times[valid][-1] == times[valid][0]:
        return float(np.mean(values[valid]))
    return float(np.trapezoid(values[valid], x=times[valid]) / (times[valid][-1] - times[valid][0]))


def _weighted_time_average(values, weights, times=None):
    """Return a finite, prevalence-weighted time average."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    if times is not None:
        times = np.asarray(times, dtype=float)
        valid &= np.isfinite(times)
        if not np.any(valid):
            return np.nan
        if np.count_nonzero(valid) > 1 and times[valid][-1] != times[valid][0]:
            weighted_integral = np.trapezoid(values[valid] * weights[valid], x=times[valid])
            weight_integral = np.trapezoid(weights[valid], x=times[valid])
            if weight_integral > 0:
                return float(weighted_integral / weight_integral)
    return float(np.average(values[valid], weights=weights[valid]))


def _mean_over_groups(values):
    """
    Average group values while leaving all-missing time points as NaN.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0)
    totals = np.nansum(values, axis=0)
    return np.divide(totals, counts, out=np.full(totals.shape, np.nan), where=counts > 0)


def analyze_simulation_outcomes(
    exposure_xM_xt,
    diversity_infected_cos_xM_xt,
    diversity_susceptibles_cos_xM_xt,
    diversity_susceptibles_cos_xM_xt_weighted,
    variant_entropy_xt,
    mrca_times_pairwise_xt,
    total_prevalence_xt,
    prevalences_I_xM_xt,
    outcome_times,
    mrca_times,
):
    """
    Reduce the configured outcomes to one time average for a simulation.
    """
    infection_prevalence = _mean_over_groups(prevalences_I_xM_xt)
    susceptible_diversity_weighted = _mean_over_groups(
        diversity_susceptibles_cos_xM_xt_weighted
    )
    return {
        "exposure_other": _time_average(
            _mean_over_groups(1 - exposure_xM_xt), outcome_times
        ),
        "diversity_infected_cos": _time_average(
            _mean_over_groups(diversity_infected_cos_xM_xt), outcome_times
        ),
        "diversity_susceptibles_cos": _time_average(
            _mean_over_groups(diversity_susceptibles_cos_xM_xt), outcome_times
        ),
        "temporary_diversity_susceptibles_cos_weighted": _weighted_time_average(
            susceptible_diversity_weighted, infection_prevalence, outcome_times
        ),
        "entropy": _time_average(variant_entropy_xt, outcome_times),
        "mrca_times_pairwise": _time_average(mrca_times_pairwise_xt, mrca_times),
        "mrca_times_average_weighted": _weighted_time_average(
            mrca_times_pairwise_xt, total_prevalence_xt, mrca_times
        ),
    }


# Plotting functions render the analyzed simulation output.
def analyze_and_plot_multi_strains(final_state, sol_combined, itimes, I_xt, strain_ids_xt, folder=None, plot=True, show=False, tspan=None, growth_rates_xt=None, plot_autocorr=False, plot_variant_distances=False, I_xt_M=None, plot_idx=None, summary_only=False):
    """
    Plot the infected compartments, strain diversity, and phylogenetic tree for multiple strains over time.
    This function generates a comprehensive visualization of the dynamics of multiple strains in a simulation. 
    It includes plots for infected compartments, strain diversity, phylogenetic tree, and group distribution prevalence.
    Parameters:
    -----------
    sol_combined : callable
        A function that provides the solution of the simulation over time.
    itimes : array-like
        Time points for the simulation.
    I_xt_M : list of tuples
        A list of tuples where each tuple contains time points and corresponding infected fractions for each strain, in the order of the strain ids `strain_ids_xt`, for each region. (?)
    I_xt : list of tuples
        A list of tuples where each tuple contains time points and corresponding infected fractions for each strain, in the order of the strain ids `strain_ids_xt`. (?)
    strain_ids_xt : list
        A list of strain IDs used to map colors from `color_dict`.
    folder : str, optional
        The folder where the resulting plots will be saved. Default is 'results'.
    show : bool, optional
        If True, displays the plots. If False, closes the plots after saving. Default is False.
    tspan : tuple, optional
        A tuple specifying the time span (start, end) for the simulation. If provided, overrides `itimes`. It starts after the burn in
    plot_idx: int, optional
        If provided, only plots the iterations from itimes[plot_idx].
    Returns:
    --------
    mean_diversity_infected : float
        The mean diversity of infected compartments over time.
    mean_diversity_susceptibles : float
        The mean diversity of susceptible compartments over time.
    mrca_times : list
        Times to the most recent common ancestor (MRCA) for the strains.
    pairwise_mut_distances_overtime : list
        Pairwise mutation distances between strains over time.
    mrca_times_pairwise : list
        Pairwise MRCA times for the strains.
    time_points : array-like
        Time points corresponding to the phylogenetic tree and diversity calculations.
    Notes:
    ------
    - The function generates a multi-panel figure with six subplots, each visualizing different aspects of the simulation.
    - The phylogenetic tree is generated using the Newick format and visualized with strain sizes and colors.
    - The function saves the resulting figure as 'results_xt.png' in the specified folder.
    Plot the infected compartments for multiple strains.
    """
    if plot_idx is None:
        plot_idx = 0
    number_of_strains = final_state.total_strains if final_state else len(itimes)
    color_dict = map_colors_discrete(number_of_strains) # this takes only the unique values from strain_ids_I_xt or strain_ids_xt
    time_points = itimes[:-1]

    if tspan: # reduce the time span of the plot to the tspan
        sol_combined = sol_combined.subset(tspan)
        #states = [states[i] for i in np.where(itimes >= tspan[0])[0].tolist() if i < len(states)]
        itimes = itimes[(itimes >= tspan[0]) & (itimes <= tspan[1])]
        number_evaluations = len(itimes) * 10
        tvals = np.linspace(tspan[0], tspan[1], number_evaluations)
        t_variant_introduction = time_points[(len(time_points) - (len(itimes)-1)):] 
        strain_ids_xt = strain_ids_xt[-len(t_variant_introduction):]  # Select the last that match the time points.

    else:
        number_evaluations = len(itimes) * 10 # make sure it is more frequent than the introductions time points, but sampled at constant time intervals. 
        tvals = np.linspace(sol_combined.tspan[0], sol_combined.tspan[1], number_evaluations)
        t_variant_introduction = time_points

    # CALCULATING VALUES 
    all_time_points, infected_by_strain = get_common_time_grid(I_xt, x_min = -1) # if you were instead to start from tspan[0] here, the colours and other strain indices would not match!.
    frequencies_xt = infected_by_strain / np.sum(infected_by_strain, axis=0, keepdims=True)  # normalize to get frequencies

    # calculate the time needed for calculating each of the metrics, and then print them
    start_time = time.time()   
    #pmut_distances_xt = distance_xt_2(frequencies_xt, final_state, metric='mut_distance')
    mrca_times_pairwise_xt = distance_xt_2(frequencies_xt, final_state, metric='mrca')
    all_time_points_mask = (all_time_points >= tspan[0]) & (all_time_points <= tspan[1])
    mrca_times_pairwise_xt = np.array(mrca_times_pairwise_xt)[all_time_points_mask] # only use the points after burnin time 
    #pmut_distances_xt = np.array(pmut_distances_xt)[all_time_points_mask]
    pmut_distances_xt = np.zeros_like(mrca_times_pairwise_xt)  # Placeholder for pairwise mutation distances
    ts = all_time_points[all_time_points_mask]
    mrca_times_pairwise_xt = ts - mrca_times_pairwise_xt  # Adjust MRCA times to be relative to the current time (note that in the equation, this is done within the summation, but one can take it outside)
    print(f"Calculated pairwise MRCA times in {time.time() - start_time:.2f} seconds.")


    if not summary_only:
        mrca_times, t_mrca_times = tmrca_xt(t_variant_introduction, sol_combined, strain_ids_xt, final_state, tvals=tvals)
        diversity_susceptibles_xM_xt = diversity_g(sol=sol_combined, tvals=tvals, metric='euclidean', region=range(sol_combined.parameters.M), g="S")
        diversity_infected_xM_xt = diversity_g(sol=sol_combined, tvals=tvals, metric='euclidean', region=range(sol_combined.parameters.M), g="I")
        prevalences_S_xM_xt = group_prevalence_xt(sol_combined, tvals=tvals, g="S")
 
    diversity_susceptibles_cos_xM_xt = diversity_g(sol=sol_combined, tvals=tvals, metric='cosine', region=range(sol_combined.parameters.M), g="S")
    diversity_susceptibles_cos_xM_xt_weighted = diversity_g(sol=sol_combined, tvals=tvals, metric='cosine', region=range(sol_combined.parameters.M), g="S", weighted=True)
    diversity_infected_cos_xM_xt = diversity_g(sol=sol_combined, tvals=tvals, metric='cosine', region=range(sol_combined.parameters.M), g="I")
    variant_entropy_xt = entropy_xt(sol_combined, tvals) 
    prevalences_I_xM_xt = group_prevalence_xt(sol_combined, tvals=tvals, g="I")

    start_time = time.time()
    exposure_xM_xt = exposure_other_groups(sol_combined, tvals=tvals) 
    print(f"Calculated exposure to other groups in {time.time() - start_time:.2f} seconds.")
    print("Finished calculating metrics.")

    if summary_only:
        return analyze_simulation_outcomes(
            exposure_xM_xt=exposure_xM_xt,
            diversity_infected_cos_xM_xt=diversity_infected_cos_xM_xt,
            diversity_susceptibles_cos_xM_xt=diversity_susceptibles_cos_xM_xt,
            diversity_susceptibles_cos_xM_xt_weighted=diversity_susceptibles_cos_xM_xt_weighted,
            variant_entropy_xt=variant_entropy_xt,
            mrca_times_pairwise_xt=mrca_times_pairwise_xt,
            total_prevalence_xt=np.sum(infected_by_strain, axis=0)[all_time_points_mask],
            prevalences_I_xM_xt=prevalences_I_xM_xt,
            outcome_times=tvals,
            mrca_times=ts,
        )

    start_time = time.time()

    ####################
    # PLOTTING
    ####################
    # Plotting the infected compartments and phylogenetic tree
    if plot:
        if len(itimes[plot_idx:]) < 500:

            # plot stackplot 
            x_min = 2000 #itimes[plot_idx] #itimes[0] if len(itimes) <= 101 else itimes[-100]
            x_max=min(itimes[-1],3000) #itimes[-1]
            plot_infected_stackplot(I_xt_M, color_dict, x_min=x_min, x_max=x_max, normalize=True, folder=folder, infected_by_strain_all=infected_by_strain, common_time_grid_all=all_time_points)
            plot_infected_stackplot(I_xt_M, color_dict, x_min=x_min, x_max=x_max, normalize=False, folder=folder, infected_by_strain_all=infected_by_strain, common_time_grid_all=all_time_points)

            # Seperate plot with the tree and prevalences. 
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 4), gridspec_kw={'height_ratios': [2, 1, 1]})
            # Instead of plotting on axs[1], plot infected fractions on a secondary y-axis of axs[0]
            root_strain = get_root_strain(final_state.regions[0].viruses[0].strain)
            time_points, ymin, ymax = plot_tree(root_strain, color_attribute=None, scale_by='t0', color_dict=color_dict, node_sizes=np.array([sum(I[1]/ sol_combined.parameters.M) for I in I_xt]), ax=axs[0], xmin=x_min, xmax=itimes[-1])
            axs[0].set_xlim(x_min, itimes[-1])
            # remove x-axes labels from axs[0]
            for i in (0,1):
                axs[i].set_xticklabels([])
                axs[i].set_xticks([])

            axs[0].set_ylim(ymin, ymax)
            axs[2].set_xlabel("Time")
            for t in itimes:
                if t > x_min: axs[0].axvline(x=t, color='grey', linestyle='-', linewidth=0.1, alpha=0.7)        # Add vertical thin grey lines at each variant introduction time
            descendent_prevalences, parent_prevalences = descendents_frequency_xt(infected_by_strain, sol_combined, tvals=tvals, x_min=itimes[plot_idx]-100, root_strain=root_strain)
            infections_total =  np.array(infected_by_strain).sum(axis=0)
            descendent_frequencies = [descendent_prevalences[i]/ infections_total for i in range(len(descendent_prevalences))]

            plot_infected_stack_subplot(I_xt_M, 0, "all_regions", color_dict, axs[1], x_min, itimes[-1], normalize=True, common_time_grid_all=all_time_points, infected_by_strain_all=infected_by_strain)    


            plot_infected_fractions([(all_time_points, descendent_prevalences[i]/ infections_total) for i in 
                                     range(len(descendent_prevalences))], color_dict={key - (len(I_xt)- len(descendent_prevalences)): value for key, value in color_dict.items()}, ax=axs[2], 
                                     label="Descendant\nfrequency", log_scale=False, xmin=x_min, xmax=itimes[-1])
            axs[1].set_xticks([])                # remove x axis labels
            axs[1].set_ylabel("Strain\nfrequency")
            axs[2].set_ylabel("Descendant\nfrequency")
            axs[2].set_ylim(1e-7, 1.05)
            axs[1].set_xlim(x_min, itimes[-1])
            axs[2].set_xlim(x_min, itimes[-1])
            axs[2].set_xlim(x_min, x_max)
            # rename axis labels, starting at 0 instead of x_min
            axs[2].set_xticks(np.arange(x_min, x_max + 1, step=round((x_max - x_min) / 5)))
            axs[2].set_xticklabels([f"{int(tick)-x_min}" for tick in np.arange(x_min, x_max + 1, step=round((x_max - x_min) / 5))], rotation=45, ha='right')
 
            fig.savefig(f"{folder}/tree.pdf", dpi=100)


        # Age distribution 
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 5.5))
        plt.subplots_adjust(right=0.9, left=0.1)
        plot_group_distribution_prevalence(sol_combined, np.mean(prevalences_I_xM_xt, axis=1), tvals=tvals, ax=axs[0], g="I")
        plot_group_distribution_prevalence(sol_combined, np.mean(prevalences_S_xM_xt, axis=1), tvals=tvals, ax=axs[1], g="S")
        
        # save as pdf
        fig.savefig(f"{folder}/FigS3_age_distributions.pdf", dpi=300)

        ####
        # BIG PLOT WITH ALL THE METRICS
        ####
        fig, axs = plt.subplots(nrows=13, ncols=2, figsize=(15, 20), gridspec_kw={'width_ratios': [6, 1]})  # Adjusted column widths
        sweep_durations, peak_infections, monotonicity,  fixated = sweep_info_frequency(descendent_frequencies,all_time_points, axs)
        plot_group_distribution_prevalence(sol_combined, np.mean(prevalences_S_xM_xt, axis=1), tvals=tvals, ax=axs[8, 1], g="S")
        plot_g_diversity_xt(exposure_xM_xt, ax=axs[0, 0], tvals=tvals, lim=False, g="I", title="Ratio exposure \nfrom other groups" ,metric='Ratio exposure \nfrom other groups') 
        plot_infected_fractions([(I[0] , I[1]/ sol_combined.parameters.M) for I in I_xt], color_dict, ax=axs[1, 0])
        plot_entropy_xt(variant_entropy_xt, sol_combined, tvals=tvals, ax=axs[2, 0]) # alternatively use variant_entropy_xt*np.mean(prevalences_I_xM_xt, axis=0)/ np.mean(prevalences_I_xM_xt)
        plot_initial_growth_rates(t_variant_introduction, growth_rates_xt, color_dict, ax=axs[3, 0])
        plot_y_xt(ts, mrca_times_pairwise_xt, ax=axs[4, 0], label='$T_{MRCA} pairwise$') #Average pairwise 
        plot_y_xt(t_mrca_times, mrca_times, ax=axs[5, 0], label='$T_{MRCA}$') #Average pairwise 
        plot_y_xt(ts, pmut_distances_xt, ax=axs[6, 0], label='Mutation \n distance')
        plot_g_diversity_xt(diversity_susceptibles_cos_xM_xt,ax=axs[6, 0], tvals=tvals, lim=False, g="I", metric='cosine') 
        plot_group_prevalence_xt(sol_combined, prevalences_I_xM_xt, tvals=tvals, ax=axs[7, 0], per_region=True, g="I") 
        plot_group_prevalence_xt(sol_combined, prevalences_S_xM_xt, tvals=tvals, ax=axs[8, 0], per_region=True, g="S")
        plot_group_distribution_prevalence(sol_combined, np.mean(prevalences_I_xM_xt, axis=1), tvals=tvals, ax=axs[7, 1], g="I")
        plot_group_distribution_prevalence(sol_combined, np.mean(prevalences_S_xM_xt, axis=1), tvals=tvals, ax=axs[8, 1], g="S")
        plot_g_diversity_xt(diversity_infected_xM_xt, ax=axs[9, 0], tvals=tvals, lim=False, g="I", metric='euclidean') 
        plot_g_diversity_xt(diversity_susceptibles_xM_xt, ax=axs[10, 0], tvals=tvals, lim=False, g="S", metric='euclidean')
        plot_g_diversity_xt(diversity_infected_cos_xM_xt, ax=axs[11, 0], tvals=tvals, lim=False, g="I", metric='cosine') 
        plot_g_diversity_xt(diversity_susceptibles_cos_xM_xt, ax=axs[12, 0], tvals=tvals, lim=False, g="S", metric='cosine')

        # Add parameter description in the last row (row 13, column 0)
        param_text = "\n".join([f"{k}: {v}" for k, v in vars(final_state.parameters).items() if not k.startswith("__") and k != "C"])
        axs[0, 1].text(0.01, 0.99, param_text, va='top', ha='left', fontsize=8, wrap=True)

        # Update the formatting of the plots
        for i in list(range(3)) + list(range(9, 13)):  #Remove borders for column 2  Iterate over all rows
            axs[i, 1].spines['top'].set_visible(False)
            axs[i, 1].spines['right'].set_visible(False)
            axs[i, 1].spines['left'].set_visible(False)
            axs[i, 1].spines['bottom'].set_visible(False)
            axs[i, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for i in range(12): 
            for j in range(2):
                if not ((j == 1 and (i == 8 or i==4 or i==3))): 
                    axs[i, j].set_xlabel("")
                if not ((j == 1 and ( i==4 or i==3))): 
                    axs[i, j].set_xticklabels([])
                    axs[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
                axs[i, j].patch.set_alpha(0)  # Make the plot area transparent
        for i in range(13): axs[i,0].set_xlim(itimes[plot_idx], itimes[-1])
        for i in range(13): 
            axs[i,0].set_ylabel(axs[i,0].get_ylabel(), rotation=0, labelpad=40, ha='right', va='center')
            axs[i,1].set_ylabel("")

        plt.gcf().set_size_inches(10, 15)
        if folder is not None: 
            try:
                plt.savefig(os.path.join(folder, "results_xt.png"), bbox_inches='tight', dpi=300)
                # make sure to continue if saving does not work: 
            except Exception as e:
                print(f"Warning: Could not save figure to {folder}/results_xt.png. Error: {e}")
        if not show: plt.close()

        # nan in exposure_xM_xt with np.inf
        exposure_xM_xt = np.nan_to_num(exposure_xM_xt, nan=0, posinf=0, neginf=0)

    print(f"Finished plotting in {time.time() - start_time:.2f} seconds.")
    metrics = analyze_simulation_outcomes(
        exposure_xM_xt=exposure_xM_xt,
        diversity_infected_cos_xM_xt=diversity_infected_cos_xM_xt,
        diversity_susceptibles_cos_xM_xt=diversity_susceptibles_cos_xM_xt,
        diversity_susceptibles_cos_xM_xt_weighted=diversity_susceptibles_cos_xM_xt_weighted,
        variant_entropy_xt=variant_entropy_xt,
        mrca_times_pairwise_xt=mrca_times_pairwise_xt,
        total_prevalence_xt=np.sum(infected_by_strain, axis=0)[all_time_points_mask],
        prevalences_I_xM_xt=prevalences_I_xM_xt,
        outcome_times=tvals,
        mrca_times=ts,
    )
    return metrics




def plot_simulation_results(df_original, param_ranges, outcomes_of_interest, log_scale=None, folder=None, param_names=None, numerical_values=True, y_log=True, quantiles=(0.10, 0.90)):
    """
    Plots the outcomes of interest against the parameter(s) varied.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the simulation results.
    - param_ranges (dict): Dictionary of parameters varied and their ranges.
    - outcomes_of_interest (dict): Dictionary of outcome variables to plot (key: column name, value: plot title).
    - log_scale (bool): Whether to use a logarithmic scale for the x-axis. Defaults to None: it will be determined by the data.
    - filename (str): Optional filename to save the plot.

    Returns:
    - None
    """
    params_plot = list(param_ranges.keys())
    for i in range(2):
        if i==1 and len(params_plot) == 1:
            continue 
        # Extract the parameter names
        x_param = params_plot[i]  # Primary x-axis parameter
        x_param2 = params_plot[1-i] if len(params_plot) > 1 else None  # Optional grouping parameter

        fig, axs = plt.subplots(len(outcomes_of_interest), 1, figsize=(3.7, 1.4 * len(outcomes_of_interest)))
        df = df_original.copy()  # Create a copy of the DataFrame to avoid modifying the original
        try:         # Convert x_param values to float where possible
            df[x_param] = df[x_param].apply(lambda x: float(x) if isinstance(x, (int, float, np.integer, np.floating, str)) and str(x).replace('.', '', 1).replace('-', '', 1).isdigit() else x)
        except Exception:
            pass
        df = df[df[x_param].notna()]
        x_values = df[x_param].drop_duplicates().tolist()
        try:
            x_values = sorted(x_values)
        except TypeError:
            x_values = sorted(x_values, key=str)
        x_positions_by_value = {value: position for position, value in enumerate(x_values)}
        x_labels = [str(value) for value in x_values]

        if len(outcomes_of_interest) == 1:
            axs = [axs]  # Ensure axs is iterable for a single plot

        def plot_data(ax, subset, x_param, x_param2, outcome, group_value, param_names, color, title, jitter=False):
            if x_param2 and group_value is not None: 
                label = f"{param_names[x_param2]}={group_value}" if group_value == param_names[x_param2][0] else f"{group_value}"
            else: 
                label = ""
            x_vals = subset[x_param]
            x_positions = np.asarray([x_positions_by_value[value] for value in x_vals])
            if jitter:
                x_positions = x_positions - 0.08
                            
            median_column = f"{outcome}-median"
            lower_column = f"{outcome}-q{quantiles[0]:g}"
            upper_column = f"{outcome}-q{quantiles[1]:g}"
            if median_column in subset.columns:
                median_values = pd.to_numeric(subset[median_column], errors="coerce").to_numpy()
                if lower_column in subset.columns and upper_column in subset.columns:
                    lower_values = pd.to_numeric(subset[lower_column], errors="coerce").to_numpy()
                    upper_values = pd.to_numeric(subset[upper_column], errors="coerce").to_numpy()
                    ax.vlines(x_positions, lower_values, upper_values, color=color, linewidth=2)
                ax.plot(
                    x_positions,
                    median_values,
                    marker="o",
                    linestyle="None",
                    label=label,
                    color=color,
                    markersize=4,
                )
            else:
                outcome_values = subset[outcome]                 # Plot directly if it's a single value
                ax.plot(x_positions, outcome_values, marker='o', linestyle='-', label=label, color=color)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # make the y-scale logarithmic 
            if y_log: ax.set_yscale('log') 
            ax.set_ylabel(title)
            if len(x_labels):
                ax.set_xlim(-0.5, len(x_labels) - 0.5)

        for ax, (outcome, title) in zip(axs, outcomes_of_interest.items()):
            if x_param2:
                param2_values = df[x_param2].unique()
                numerical_values = (all(isinstance(val, (int, float)) for val in param2_values) or numerical_values)
                if len(param2_values) > 1:
                    colors = [ plt.cm.tab20b(i / 2) for i in range(len(param2_values))] # red and blue
                    # colors = [ ] # [plt.cm.Blues(0.2 + 0.8 * i / (len(param2_values) - 1)) for i in range(len(param2_values))]
                else: 
                    colors = [plt.cm.tab20b(1)]
                for group_value, color in zip((param2_values), colors):
                    subset = df[df[x_param2] == group_value]
                    try:
                        subset = subset.sort_values(by=x_param)
                    except Exception:
                        pass
                    if group_value == param2_values[0]:
                        jitter = True
                    else: 
                        jitter = False
                    plot_data(
                        ax, subset, x_param, x_param2, outcome, group_value,
                        param_names, color, title, jitter=jitter,
                    )
            else:
                plot_data(
                    ax, df, x_param, x_param2, outcome, None, param_names,
                    color=plt.cm.tab20b(1), title=title,
                )

        
        # Customize the plots remove clutter from x-axis. 
        for i_ax in range(len(outcomes_of_interest)-1):
            axs[i_ax].set_xticklabels([])
            axs[i_ax].tick_params(axis='x', which='both', bottom=False, top=False)
        for i_ax in range(len(outcomes_of_interest)):
            ymax= None; ymin = None
            # if list(outcomes_of_interest.keys())[i_ax] == 'diversity_infected_cos': ymax = 0.08
            if list(outcomes_of_interest.keys())[i_ax] == 'diversity_susceptibles_cos': ymin= 1e-16 #ymax = 0.00015
            if list(outcomes_of_interest.keys())[i_ax] == 'entropy': ymax = 1.5; ymin  = 0.6#2.1 # for fig2 and fig 3: ymin=, for fig S3/s6: ymin = 0.8, ymax=1.8, for fig 4 ymin = 0.6  
            if list(outcomes_of_interest.keys())[i_ax] == 'mrca_times_pairwise': ymax = 1500; ymin = 100 # for fig2 and fig 3: ymin=300, for fig 4 : ymin=100 S3/s6: ymin = 100, ymax=NA; for fig ? ymax = 1000; ymin = 100 
            axs[i_ax].set_ylim(bottom=ymin, top=ymax)

            axs[i_ax].yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))             # make sure that 4 labels are plotted for each y-axis (including 0)
            def scientific_formatter(value, position):
                if abs(value) < 0.001:
                    exponent = int(np.floor(np.log10(abs(value))))
                    mantissa = value / (10 ** exponent)
                    return rf'${mantissa:g}\times10^{{{exponent}}}$'
                return f'{value:g}'
            axs[i_ax].yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(scientific_formatter)
            )
            if list(outcomes_of_interest.keys())[i_ax] == 'entropy':
                axs[i_ax].set_yticks([0.8,0.9,1,1.5,2])
        axs[len(outcomes_of_interest) - 1].set_xlabel(param_names[x_param])
        plt.tight_layout()
        plt.subplots_adjust(left=0.5, bottom=0.25, hspace=0.18)
        # Save plot if filename is provided
        if folder:
            plt.savefig(os.path.join(folder, f"aggregated_results{i}.pdf"), dpi=300, bbox_inches='tight')

        plt.show()