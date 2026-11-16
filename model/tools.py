import numpy as np
from scipy.spatial.distance import cosine

def symb_to_compartment(g):
    """
    Convert a compartment symbol to its corresponding index.

    Parameters:
    g (str): Compartment symbol ('S', 'I', 'C', or 'R').

    Returns:
    int: Index of the compartment.

    Raises:
    ValueError: If the symbol is not recognized.
    """
    mapping = {'S': 0, 'I': 1, 'C': 2, 'R': 3}
    if g in mapping:
        return mapping[g]
    else:
        raise ValueError(f"Unknown compartment symbol: {g}")  

def sir_index(i, a, g, N, vsize = 4):
    """
    Calculate the index/indices for a strain in a specific compartment and region.

    The indexing structure is: Region (most apart) --> Virus --> SIR-states (most together).

    Parameters:
    i (int or list of ints): Region index or indices.
    a (int or list of ints): Virus (strain) index or indices.
    g (str or list of strings): Compartment symbol(s) ('S', 'I', 'R', 'C') or indices.
    N (int): Number of strains.
    vsize (int): Number of compartments per strain (default 4: S, I, R, C).

    Returns:
    np.ndarray: Array of calculated indices.

    Raises:
    AssertionError: If any strain index is out of range.
    """
    if isinstance(a, int): a = [a]
    if isinstance(i, int): i = [i]
    if isinstance(g, str): g = [g]
    a, i, g = np.array(a), np.array(i), np.array(g)
    
    assert all(0 <= x <= N for x in a), f"Error accessing strain {a} in SIR model with {N} strains"  # Python is 0-indexed
    a = np.array(a)
    g = np.array([symb_to_compartment(x) for x in g])
    rsize = vsize * N  # number of compartments times number of viruses
    region_offset = i * rsize  # offset for the region based on the region index
    virus_offset = a * vsize  # offset for the virus within the region [x * vsize for x in a]
    return np.array([region_offset + vo + comp for vo in virus_offset for comp in g]).flatten()  # The final index is region_offset + virus_offset + compartment index is returned as a flattened array

def precompute_indices(M, N):
    """
    Precompute indices for SIR model dynamics.

    This function generates a dictionary of precomputed indices for each 
    compartment ('S', 'I', 'C', 'R') in the SIR model for a given number of 
    regions (M) and strains (N). The indices are used to efficiently 
    access and manipulate the state of each individual in the model.

    Parameters:
    M (int): Number of regions.
    N (int): Number of strains.

    Returns:
    dict: A dictionary where each key is a strain index (a) and the value is 
          another dictionary containing the following keys:
          - 'idx_S': Indices for susceptible regions in strain a.
          - 'idx_I': Indices for all regions in strain a.
          - 'idx_C': Indices for carrier regions in strain a.
          - 'idx_R': Indices for recovered regions in strain a.
          - 'idx_Ib': Reshaped indices for infections across all regions across all strains.
          - 'not_a': Indices of all strains except strain a.
    """
    indices_dict = {}
    for a in range(N):
        indices_dict[a] = {
            'idx_S': sir_index(range(M), a, 'S', N),
            'idx_I': sir_index(range(M), a, 'I', N),
            'idx_C': sir_index(range(M), a, 'C', N),
            'idx_R': sir_index(range(M), a, 'R', N),
            'idx_Ib': np.reshape(sir_index(range(M), range(N), 'I', N), (N, M)).T,
            'not_a': np.delete(np.arange(N), a)
        }
    return indices_dict

def flux_matrix(M):
    """
    Define the flux matrix F for M regions, used to model ageing in an efficient way.

    Parameters:
    M (int): Number of regions.

    Returns:
    np.ndarray: The flux matrix of shape (M, M).
    """
    F = np.zeros((M, M))
    np.fill_diagonal(F, -1)
    np.fill_diagonal(F[1:], 1)
    return F

def grow_cross_immunity(K, a, symmetric_cross_immunity):
    """
    Expand the cross-immunity matrix by adding a new virus.

    Parameters:
    K (np.ndarray): Existing cross-immunity matrix.
    a (int or None): Index for the new virus. If None, append at the end.
    symmetric_cross_immunity (float or np.ndarray): Cross-immunity values for the new virus.

    Returns:
    np.ndarray: Updated cross-immunity matrix.

    Raises:
    ValueError: If input dimensions are inconsistent.
    """
    if a is None or a >= K.shape[0]:
        if a is None: a = K.shape[0]
        K = np.pad(K, ((0, 1), (0, 1)), mode='constant', constant_values=0.0)    # Pad matrix to add new virus
    K[a, :] = symmetric_cross_immunity
    K[:, a] = symmetric_cross_immunity   
    return K 

def distances_to_cross_immunity(distances, r=None, max_escape=0.8, σ=0, breadth=1, alpha = 1, X=None, a_idx_circulating=None, a_idx_all=None): 
    """
    Map genetic or antigenic distances to cross-immunity values using an exponential function.

    Parameters:
    distances (np.ndarray): Array of distances between strains.
    k (float): Steepness of the logistic function.
    r, max_escape, σ, breadth, alpha: Parameters for cross-immunity mapping.
    X: Optional state object for feedback.
    a_idx_circulating, a_idx_all: Indices for feedback.

    Returns:
    np.ndarray: Cross-immunity values.

    Raises:
    AssertionError: If cross-immunity values at zero distance are not 1.
    """
    def exponential(x, alpha=1, breadth=10): 
        x = np.asarray(x, dtype=float)
        return alpha * np.exp(-x / breadth)

    distances = np.where(distances is None, np.inf, distances).astype(float)
    cross_immunity_values =  exponential(distances, alpha=alpha, breadth=breadth) # gamma_breadth(distances, alpha, breadth, σ, X, r, a_idx_circulating, a_idx_all) # In the current code, by default, simply the exponential function is used. 
    assert np.allclose(cross_immunity_values[distances == 0], 1), "Cross-immunity values at zero distance must be 1" # Assert that where distances is 0, cross_immunity_values = 1. there are errors later on, when cross_immunity_values are non 0 for the diagonal.
    # cap = 0.1 # TEMPORARY
    # cross_immunity_values = np.maximum(cross_immunity_values, cap)
    
    return cross_immunity_values


#######################################
# Functions to evaluate outcomes. 
######################################

def initial_growth_rate(X, virus, sol=None, t=None, region=None):
    """
    $\\alpha \lambda^a - \delta$; $\lambda^a$ the dominant eigen value of \textbf{$S^aC$ }
    """
    if X is not None:
        if region is not None:
            S = X.regions[region].viruses[virus].S
        else:
            S = [X.regions[r].viruses[virus].S for r in range(len(X.regions))]
        return initial_growth_rate_(X.parameters, S)
    elif sol is not None and t is not None:
        if region is not None:
            S = sol(t, region, virus, "S")
        else:
            S = sol(t, range(sol.parameters.M), virus, "S")
        return initial_growth_rate_(sol.parameters, S)

def eigenvalues(K):
    """
    Compute the eigenvalues of the matrix K.
    """
    return np.linalg.eigvals(K)

def initial_growth_rate_(params, S):
    """
    $\\alpha \lambda^a - \delta$; $\lambda^a$ the dominant eigen value of \textbf{$S^aC$ }
    """
    λ = np.max(np.abs(eigenvalues(np.array(S) * params.C))) # dominant eigenvalue
    return params.α * λ - params.δ

def prevalence(sol, tvals=None, region=None, virus=None, g='I', mean_region=True, mean_virus=True):
    """
    Calculate the mean prevalence of infections in a region.

    Parameters:
    sol (Solution): Solution object.
    tvals (list): List of time values.
    region (int): Region index.
    virus (int): Virus index.
    g (str): Compartment to evaluate.

    Returns:
    float: Mean prevalence of infections.
    """
    if tvals is None: tvals = np.linspace(sol.tspan[0], sol.tspan[1], 100)
    if region is not None: 
        if virus is not None:
            I = sol[tvals, region, virus, g]
        else:
            I = sol[tvals, region, range(sol.parameters.N), g]
            if mean_virus: I = sum(I)
    else:
        if virus is not None:
            I = sol[tvals, range(sol.parameters.M), virus, g]
            if mean_virus: I = np.sum(I, axis=0)
        else:
            I = sol[tvals, range(sol.parameters.M), range(sol.parameters.N), g]
            if mean_virus: I = np.sum(I, axis=0)
    return np.mean(I) if mean_region else I

def diversity_g(sol, tvals=None, virus=None, region=None, metric='euclidean', g='S', weighted=False):
    """
    Calculate the similarity of the susceptibles (g=S) or infections (g=I) values across regions.

    Parameters:
    sol (Solution): Solution object.
    tvals (list): List of time values. If not provided, use all tvals.
    virus (int): Virus index. If not provided, use all viruses.
    r (int): r index. If provided, calculate the similarity of this region to others.
    metric (str): Metric to use for similarity calculation. Options are 'euclidean' and 'cosine'.

    Returns:
    np.ndarray: Similarity values. (euclidean: 0 is similar, high values are dissimilar), for cosine: 0 is similar, 1 is dissimilar.
    """
    if tvals is None:
        tvals = np.linspace(sol.tspan[0], sol.tspan[1], 1000)
    def calculate_similarity(region, g_values, tvals, sol, metric, g, prevalences):
        g_region = g_values[:, region, :]
        similarities = np.zeros(len(tvals))
        for i in range(sol.parameters.M):
            if i != region:
                if g == "S" and metric == 'euclidean'  and weighted:
                    frequencies = frequency(X=None, virus=range(sol.parameters.N), region=i, sol=sol, tvals=tvals)
                    similarities += np.sqrt(np.sum(np.square((g_region - g_values[:, i, :]) * frequencies.T), axis=1))
                elif metric == 'euclidean':
                    similarities += np.sqrt(np.sum(np.square((g_region - g_values[:, i, :])), axis=1)) / np.sum(prevalences, axis=1)
                elif metric == 'cosine': 
                    similarities = np.array([cosine(g_region[t, :], g_values[t, i, :]) for t in range(len(tvals))])
                elif g == "S" and metric == 'cosine' and weighted:
                    frequencies = frequency(X=None, virus=range(sol.parameters.N), region=i, sol=sol, tvals=tvals)
                    similarities += np.array([cosine(g_region[t, :] * frequencies[t, :], g_values[t, i, :] * frequencies[t, :]) for t in range(len(tvals))])
        similarities /= (sol.parameters.M - 1) 
        return similarities
    
    g_values = sol[tvals, range(sol.parameters.M), range(sol.parameters.N), g].reshape(sol.parameters.N, sol.parameters.M, len(tvals)).T 
    prevalences = np.sum(g_values, axis=1) #/ sol.parameters.M # sum over all regions 

    if virus is not None: g_values = g_values[:, :, virus] 
    if region is not None:  # Specific region provided
        if isinstance(region, (range, slice)):
            similarities = np.zeros(((sol.parameters.M), len(tvals)))
            for i in region: #, reg in enumerate(region_indices):
                similarities[i, :] = calculate_similarity(i,  g_values, tvals, sol, metric, g, prevalences)
            return similarities
        else:
            similarities = calculate_similarity(region, g_values, tvals, sol, metric, g, prevalences)
    else:  # No specific region provided, average across all pairwise distances. 
        similarities = np.zeros(len(tvals))
        count = 0
        for i in range(sol.parameters.M):
            for j in range(i + 1, sol.parameters.M):
                if g == "S" and  metric == 'euclidean' : 
                    frequencies = frequency(X=None, virus=range(sol.parameters.N), region=i, sol=sol, tvals=tvals)
                    similarities += np.sqrt(np.sum(np.square((g_values[:, i, :] - g_values[:, j, :])*frequencies.T), axis=1)) # instead of prevelences multiply with frequency 
                elif metric == 'euclidean': #similarities +=  np.sqrt(np.sum(np.square((g_values[:, i, :] - g_values[:, j, :])/np.where(prevalences != 0, prevalences, 1)), axis=1)) # 1/sol.parameters.N *
                    similarities += np.sqrt(np.sum(np.square((g_values[:, i, :] - g_values[:, j, :])), axis=1)) / np.sum(prevalences, axis=1)
                elif metric == 'cosine': similarities += np.array([cosine(g_values[t, i, :], g_values[t, j, :]) for t in range(len(tvals))])
                count += 1
        similarities /= count
    return similarities

def average_cross_immunity(state, a, b): 
    """
    Calculate the average cross-immunity between two strains across all regions. NOTE: this function is not used in the current code, but it can be useful for analysis.
    """
    assert state is not None, "State cannot be None"
    values = [state.regions[r].K_all[a, b] for r in range(state.parameters.M)]
    if any(v is None for v in values):
        print(f"Warning: None value found in cross-immunity values for strains {a}, {b}")
    return np.mean(values)  


def entropy(frequencies):
    """
    Calculate the entropy of a vector of frequency values.
    """
    frequencies = np.array(frequencies)
    frequencies = frequencies[frequencies > 0]  # Remove zero frequencies to avoid log(0)
    if len(frequencies) < 2:
        return 0
    return -np.sum(frequencies * np.log(frequencies)) #Do not divide by the length: /len(frequencies)

def entropy_xt(sol, tvals=None, lim=False, region=None):
    """
    Plot the entropy of the strain frequencies over time.
    """
    frequency_matrix = np.zeros((sol.parameters.N, len(tvals)))
    for a in range(sol.parameters.N): 
        frequency_matrix[a, :] = frequency(X=None, virus=a, region=region, sol=sol, tvals=tvals)
    
    return [entropy(frequency_matrix[:, t]) for t in range(len(tvals))]

def frequency(X=None, region=None, virus=None, sol=None, tvals=None):
    """
    Calculate the frequency of a viral variant infections.

    Parameters:
    X (SIRState): The state of the system.
    region (int): Region index.
    virus (int): Virus index.
    sol (Solution): Solution object.
    tvals (list): List of time values.

    Returns:
    float or np.ndarray: Frequency of infections.
    """
    if sol is not None and tvals is not None:
        if region is not None:
            Itot = sol[tvals, region, range(sol.parameters.N), 'I'].sum(axis=0) 
            Ia = sol[tvals, region, virus, 'I'] 
        else:
            Itot = sum(sol[tvals, i, range(sol.parameters.N), 'I'].sum(axis=0) for i in range(sol.parameters.M))
            Ia = sum(sol[tvals, i, virus, 'I'] for i in range(sol.parameters.M))
        return Ia / np.where(Itot == 0, 1, Itot)

    elif X is not None and region is not None and virus is not None:
        return _frequency(X, region, virus)
    elif X is not None and virus is not None:
        return np.mean([frequency(X, i, virus) for i in range(len(X.regions))])

def _frequency(X, i, a):
    """
    Helper function to calculate the frequency of infections in a region.

    Parameters:
    X (SIRState): The state of the system.
    i (int): Region index.
    a (int): Virus index.

    Returns:
    float: Frequency of infections.
    """
    return X.regions[i].viruses[a].I / sum([X.regions[i].viruses[a].I for a in range(X.parameters.N)])

def get_root_strain(strain):
    """
    Return the root strain of the tree given any strain.

    Parameters:
    strain (Strain): The strain from which to start the search.

    Returns:
    Strain: The root strain of the tree.
    """
    current_strain = strain
    while current_strain.parent is not None:
        current_strain = current_strain.parent
    return current_strain
