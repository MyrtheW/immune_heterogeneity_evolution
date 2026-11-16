import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy
import time
import model.objects as objects
from model.tools import sir_index


class Solution:
    """
    A class to represent the solution of the SIR model over time.

    Attributes:
        sol: The ODE solution object from scipy.integrate.solve_ivp.
        tspan: Tuple of (start, end) time for the simulation.
        parameters: Parameters object used for the simulation.
        K: Cross-immunity matrix.
        simulation_time: Time taken to simulate the solution.
    """
    def __init__(self, sol, tspan, parameters, K, simulation_time=None):
        self.sol = sol  # ODE solution object
        self.tspan = tspan  # time span (start, end)
        self.parameters = parameters  # parameters object
        self.K = K  # cross-immunity matrix
        self.simulation_time = simulation_time  # time it took to simulate the solution

    def __call__(self, t, sirstate=None):
        """
        Evaluate the solution at time t.

        Parameters:
            t (float): Time at which to evaluate the solution.
            sirstate (SIRState, optional): If provided, updates this state in-place.

        Returns:
            SIRState or None: Returns a new SIRState if sirstate is not provided, otherwise updates sirstate in-place.

        Raises:
            AssertionError: If t is outside the solution time span.
        """
        assert t <= self.tspan[1], f"Time span of solution {self.tspan} - got {t}"
        t_index = np.searchsorted(self.sol.t, t)
        if t_index >= len(self.sol.t): 
            t_index = len(self.sol.t) - 1
            print("Warning: Time index is out of bounds")  # Warn if t is beyond last time point
        if sirstate:
            sirstate.update_from_vector(dat=self.sol.y[:, t_index])
            sirstate.t = t 
        else:
            return objects.SIRState.from_vector(self.sol.y[:, t_index], self.K, self.parameters)

    def vec(self, t=None):
        """
        Get the solution vector at time t. 
        
        Usage: sol.vec(t)
        """
        if t is not None: 
            t_index = np.searchsorted(self.sol.t, t)
            if t_index >= len(self.sol.t): 
                t_index = len(self.sol.t) - 1
                print("Warning: Time index is out of bounds")
            return self.sol.y[:, t_index]
        else:
            return self.sol.y

    def __getitem__(self, key):
        """
        Get the solution at specific times, regions, and compartments.

        Usage: sol[tvals, i, a, g]

        Parameters:
            key (tuple): (tvals, i, a, g) where
                tvals: array-like of time points,
                i: region index or slice,
                a: strain index or slice,
                g: compartment symbol or index.

        Returns:
            np.ndarray: Array of solution values at the specified indices.

        Raises:
            AssertionError: If strain index 'a' is not provided.
        """
        tvals, i, a, g = key
        if isinstance(i, slice):  # Check if i is a slice object
            i = range(self.parameters.M)
        if isinstance(a, slice):  # Check if a is a slice object
            a = range(self.parameters.N)
        assert a is not None, "Strain index 'a' or slice must be provided"
        idx = sir_index(i, a, g, self.parameters.N)
        t_indexes = np.atleast_1d(np.clip(np.searchsorted(self.sol.t, tvals), 0, len(self.sol.t) - 1))
        #make sure that t_indexes
        return self.sol.y[np.ix_(idx, t_indexes)] # Find the indices of tvals in the solution time array

    def subset(self, tspan_subset):
        """
        Create a new Solution object with a subset of the time span.

        Parameters:
        tspan_subset (tuple): The new time span (start, end) within the original tspan.

        Returns:
        Solution: A new Solution object with the subset of the time span.

        Raises:
        AssertionError: If tspan_subset is not within the original tspan.
        """
        t_start, t_end = tspan_subset
        assert self.tspan[0] <= t_start <= t_end <= self.tspan[1], \
            f"Subset tspan {tspan_subset} must be within the original tspan {self.tspan}"

        t_indexes = np.where((self.sol.t >= t_start) & (self.sol.t <= t_end))[0]
        new_sol = deepcopy(self.sol)
        new_sol.t = new_sol.t[t_indexes]
        new_sol.y = new_sol.y[:, t_indexes]

        return Solution(new_sol, (t_start, t_end), self.parameters, self.K, self.simulation_time)

def simulate(X, tspan, t_step_max=None):
    """
    Simulate the SIR model over the given time span.

    Parameters:
    X (SIRState): The initial state of the system.
    tspan (tuple): The time span for the simulation (start, end).
    t_step_max (float, optional): Maximum time step for the ODE solver.

    Returns:
    Solution: The solution object containing the simulation results.

    Raises:
    AssertionError: If population sizes are not conserved.
    """
    u0 = X.vec()  # Initial state vector

    p = X.parameter_dict()
    start_time = time.time()
    
    population_sizes = np.array(_conserved(u0, p['params'])).flatten()   
    assert np.all(np.isclose(np.abs(population_sizes), np.ones(p['params'].M*p['params'].N), atol= 1e-8)), "Some elements in u do not sum to the population size" # Check population size

    t_step_max = min(t_step_max, (tspan[1] - tspan[0])/10) if t_step_max is not None else None
    sol = solve_ivp(
        SIR_, tspan, y0=u0, args=tuple([p]), method='RK45', vectorized=False,
        max_step=t_step_max, first_step=t_step_max/10, atol=1e-10, rtol=1e-10
    ) # My function is currently not suited for vectorized y[n,k] calls.
    return Solution(sol=sol, tspan=tspan, parameters=deepcopy(X.parameters), K=p['K'], simulation_time=time.time() - start_time)

def SIR_(t, u, p): 
    """
    Compute derivatives for a multi-strain SIR model with cross-immunity and spatial dynamics.

    Parameters:
        t (float): Current time.
        u (numpy.ndarray): State vector of compartments for all strains and regions.
        p (dict): Model parameters and indices, including:
            - 'params': Contains M (regions), N (strains), α (transmission), γ (recovery), 
              δ (waning immunity), and C (contact matrix).
            - 'Kas' (numpy.ndarray): Cross-immunity matrices by variant a.
            - 'idxs' (dict): Indices for compartments in `u`.
        a (int, optional): Strain index to compute derivatives for. Defaults to all strains.

    Returns:
        numpy.ndarray: Derivatives of the state variables.

    Raises:
        AssertionError: If derivatives contain NaN, population sizes are not conserved, 
        or conservation laws are violated.

    Notes:
        - Cross-immunity is modeled using `K`. This fast version only works if K=1 on the diagonals. 
        - Spatial interactions use the contact matrix `C`.
        - Assertions ensure numerical stability and conservation.
    """
    N, α, δ, C, γF, γe1, idxs = p['params'].N, p['params'].α, p['params'].δ, p['params'].C, p['params'].γ*p['F'], p['γe1'], p['idxs']
    du = np.zeros_like(u)

    # For each strain a, compute the derivatives for all regions and compartments
    for a in range(N): 
        idx_S, idx_I, idx_C, idx_R, idx_Ib, not_a = (idxs[a][key] for key in ['idx_S', 'idx_I', 'idx_C', 'idx_R', 'idx_Ib', 'not_a'])
        S, I, RC, R, Ib = u[idx_S], u[idx_I], u[idx_C], u[idx_R], u[idx_Ib]
        Ka = p['Kas'][a] 
        du[idx_I] += α * S * (C @ I) - δ * I +  (γF @ I)  
        du[idx_S] += - α * S * np.sum(((C @ Ib)) * Ka, axis=1)  + (γF @ S) + γe1   
        du[idx_R] += δ * I +  (γF @ R)  
        du[idx_C] += α * S * np.sum(((C @ Ib[:,not_a])) * Ka[:,not_a], axis=1) +  (γF @ RC) 

    validate_solution(u, du, p)
    return du

def validate_solution(u, du, p, atol=1e-6):
    """
    Validate the solution of the SIR model.

    Parameters:
        u (np.ndarray): State vector.
        du (np.ndarray): Derivative vector.
        p (dict): Parameter dictionary.
        atol (float): Absolute tolerance for conservation checks.

    Raises:
        AssertionError: If conservation or positivity checks fail.

    Notes:
        If the t_step_max is too large, the solution may not be conserved.
    """
    population_sizes = np.array(_conserved(u, p['params'])).flatten()   
    try:
        assert not np.isnan(np.min(du)), "There are None items in the derivatives array"
        assert np.all(np.isclose(np.abs(population_sizes), np.ones(p['params'].M*p['params'].N), atol=atol)), "Some elements in u do not sum to the population size"
        assert np.all(np.abs(np.array(_conserved(du, p['params'])).flatten()) < atol), "Some elements in du are not conserved"
        assert np.all(u + du >= -atol), "Some elements in u + du are below the threshold"
        assert np.all(u >= -atol), "Some elements in u are negative"
    except AssertionError as e:
        print("Assertion failed:", e)
        print("u (state vector):", u)
        print("du (derivatives):", du)
        print("population_sizes:", population_sizes)
        print("p['params']:", p.get('params', None))
        for a, Kas_a in enumerate(p['Kas']):
            print(f"p['Kas'][{a}]:", Kas_a)
        print("atol:", atol)
        raise
    
def _conserved(du, parameters):
    """
    Check the conservation of the SIR model.

    Parameters:
        du (array): The derivatives of the state du or the state u.
        parameters (Parameters): The parameters of the system.

    Returns:
        list: The conserved quantities for each region and strain.
    """
    return [[du[sir_index(i, a, 'S', parameters.N)] +
             du[sir_index(i, a, 'I', parameters.N)] +
             du[sir_index(i, a, 'C', parameters.N)] +
             du[sir_index(i, a, 'R', parameters.N)]
             for a in range(parameters.N)]
            for i in range(parameters.M)]

def gradient_(X):
    """
    Compute the gradient (time derivative) of the SIR model state.

    Parameters:
        X (SIRState): The state of the system.

    Returns:
        du: The gradient of the state as a new SIRState object.
    """
    u = X.vec()
    p = X.parameter_dict()
    du = SIR_(0, u, p)  # Compute derivatives in-place
    return du

def growth_rates(X):
    """
    Compute the growth rates of the SIR model state.

    Parameters:
        X (SIRState): The state of the system.

    Returns:
        growth_rates: The growth rates for each variant in N 
    """
    du = gradient_(X)
    N, M = X.parameters.N, X.parameters.M
    growth_rates = [np.sum([du[sir_index(i, a, 'I', N)] for i in range(M)]) for a in range(N)]
    return growth_rates