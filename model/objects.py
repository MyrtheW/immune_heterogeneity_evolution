import numpy as np
import copy
import model.tools as tools
from model.tools import distances_to_cross_immunity, grow_cross_immunity, precompute_indices, flux_matrix, sir_index
from model.dynamics import growth_rates
class Parameters:
    """
    A class to represent the parameters of the SIR model.
    """
    def __init__(self, N=2, α=10.0, γ=0.01, δ=1.0, M=1, c=0.1, σ=0, d=10, cross_immunity_efficacy=1, I0=1e-6, Ie=1e-7, ageing=True, strain_introduction = "uniform_excluding_zeros", **kwargs):
        self.N = 1  # Number of circulating viruses
        self.Nmax = N # Maximum number of viruses
        self.α = α # Infection rate
        self.γ = γ # Ageing rate . Note that in contrast to the overleaf, gamma is the ageing rate. In the code, the population rate is gamma / M. 
        self.δ = δ # Recovery rate
        self.M = M  # Number of geographic regions
        self.c = c  # Strength of connection between regions, in [0,1]
        self.σ = σ # standard deviation of cross-immunity across populations 
        self.d = d # breadth of cross-immunity
        self.cross_immunity_efficacy = cross_immunity_efficacy # strength of cross-immunity
        self.I0 = I0  # Initial infected proportion
        self.Ie = Ie  # Extinction threshold for sum over infected proportions
        self.mixing_matrix()
        self.ageing = ageing
        self.strain_introduction = strain_introduction

    def copy(self, update=None):
        """
        Create a deep copy of the current Parameters instance, optionally updating attributes.

        Args:
            update (dict, optional): A dictionary of attribute names and values to update in the copied instance.
                If provided, the specified attributes will be updated in the new Parameters object.

        Returns:
            Parameters: A new Parameters instance with the same attributes as the original, 
            optionally updated with values from the `update` dictionary. Also calls `mixing_matrix()` 
            if updates are applied.
        """
        new_params = Parameters(**self.__dict__)
        if update:
            for key, value in update.items():
                setattr(new_params, key, value)
            new_params.mixing_matrix()
        return new_params
    
    def mixing_matrix(self, mode="default", variance=0, **kwargs):
        """
        Generate mixing matrix C with different options.
        mode: "default", "symmetric_row_boost", "symmetric_decay", "asymmetric_row_boost", "asymmetric_decay"
        kwargs: parameters for the chosen mode.

        Create the mixing matrix based on the mixing parameter c.
        c = Cij/Cii, and  M*Cij + Cii = 1, such that:
        Cii = 1/(1+Mc) and Cij = c/(1+Mc)

        For c=1/M, the matrix is fully connected, with identical values in each cell.
        For c<1/M, the matrix is diagnoal terms are higher than off-diagonal terms.
        """
        if mode == "default":
            self.C = self.c / (1 + (self.M - 1) * self.c) * np.ones((self.M, self.M)) + np.diag(np.ones(self.M) * ((1-self.c)/(1 + (self.M - 1) * self.c)))
        else:
            raise ValueError("Other mixing modes are deprecated")
        if variance:
            noise = np.random.normal(0, variance/2, self.C.shape)
            noise = np.maximum(noise + noise.T, -self.C)
            self.C += noise  # Ensure the matrix stays symmetric
            self.C = self.C / self.C.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
        if np.any(self.C < 0) or np.any(self.C > 1):
            print("Warning: Some values in the mixing matrix C are outside the range [0, 1].")

class Virus:
    """
    A class to represent a virus with compartments S, I, C, R.
    """
    def __init__(self, S, I, C=0, R=None, strain=None):
        self.S = S # S is the proption of individuals in region i susceptible to strain a
        self.I = I # I is the proption of individuals in region i infected with strain a
        self.C = C # C is the proption of individuals in region i infected with a strain other than a through which they will obtain cross-immunity to strain a. 
        self.R = R if R is not None else 1 - S - I - C # R is recovered 
        self.strain = strain # virus is the Strain object 
        assert strain is not None
        assert np.isclose(self.population(), 1, rtol=1e-12), f"Virus creation: Size of host population is not one; got {self.population()}"

    def population(self):
        """
        Calculate the total host population.
        """
        return self.S + self.I + self.C + self.R

    def correct_population(self):
        """
        Modifies the C compartment, to correct the population size of the virus compartments, setting it to 1. 
        """
        self.C = 1 - self.S - self.I - self.R
        if self.C < 0:
            self.C = 0
            self.S = 1 - self.R - self.I
        assert self.C >= 0 and self.S >= 0, f"Virus correction: C is negative or S is negative; got {self.C}, {self.S}"
        #np.isclose(self.population(), 1, rtol=1e-12), f"Virus correction: Size of host population is not one; got {self.population()}"

    def vec(self):
        """
        Convert the virus compartments to a vector by concatenation
        """
        return [self.S, self.I, self.C, self.R]
    
    def getattr(self, key):
        """
        Get the steady state solution for a specific region, virus and compartment
        """
        return self.__getattribute__(key)

class Strain:
    """
    A class to represent a unique virus strain and its genealogy.
    """
    def __init__(self, parent=None, children=None, distance=1, t0=0, te=None, a=None, dummy=False, strain_id = None): # Initialize a strain as a dummy strain when it is not a real strain
        self.parent = parent  # parent strain
        self.children = children if children is not None else []  
        self.t0 = t0  # time of introduction 
        self.te = te  # time of extinction
        self.a = a # index of the strain in the list of circulating strains
        self.distance = distance  #  By default the distance to the parent strain is 1, alternatively, one could sample this, e.g. self.distance = abs(np.random.normal(0, 1.25))
        self.id = strain_id if strain_id is not None else id(self) if not dummy else None 

    def add_child(self, child):
        """
        Add a child strain to the current strain.
        """
        self.children.append(child)
        child.parent = self
    
    def initialize(self, t, id):
        self.id = id
        self.t0 = t

    def set_extinct(self, t):
        self.te = t # set time of extinction

class Region:
    """
    A class to represent a region containing multiple viruses and a cross-immunity matrix.
    """
    def __init__(self, viruses, K):
        assert len(viruses) == K.shape[0] == K.shape[1], f"Inconsistency between number of viruses {len(viruses)} and cross-immunity matrix {K.shape}"
        assert all(np.isclose(v.population(), 1, rtol=1e-12) for v in viruses), "Host pop. is not one for some viruses."
        self.viruses = viruses
        self.K = K
        # self.K_all = copy.deepcopy(K) 

    def vec(self):
        """
        Convert the region's viruses to a vector.
        """
        return np.concatenate([v.vec() for v in self.viruses])

class SIRState:
    """
    A class to represent the state of the SIR model.
    """
    def __init__(self, parameters, regions, root=None, t=0): #Possibly one could write a function that initilizes all with a single or multiple strains
        assert regions, "Must have at least one region"
        assert len(regions) == parameters.M
        for i, r in enumerate(regions):
            assert len(r.viruses) == parameters.N, f"Region {i} has size {len(r.viruses)} instead of N={parameters.N}"
        self.parameters = parameters
        self.regions = regions
        self.root = root  # Root strain
        self.total_strains = 1  # Initially, only the root strain exists. The total strains keeps track of the number of strains in the entire simulation 
        self.distances_all = np.zeros((parameters.N, parameters.N))  # Matrix of distances of strains, initialized as a 2D numpy array with a single entry 0. Note, this could be improved to a sparse matrix, for only co-circulating strains
        self.tmrca_all = np.zeros((parameters.N, parameters.N))  # Start at t0=0 Matrix of times at which the  most recent common ancestor of strains appearded, initialized as a 2D numpy array with a single entry 0. Note, this could be improved to a sparse matrix, for only co-circulating strains
        self.strains_all = [root] if root else []  # List of all strains ever circulated, initialized with the root strain. This list maps virus index 'a' to the strain 'id'
        self.circulating_strains = np.zeros(parameters.N, dtype=int)  # List of IDs of the circulating strains, initialized with the root strain. This list maps virus index 'a' to the strain 'id'
        self.t = t  # Current time

    def update_tmrca(self, id_parent, new_strain_t0):
        """
        Append a new strain to the tmrca matrix, copying the tmrca from the parent strain and adding 'new_strain_t0', the time at which the new strain appears.
        """
        assert len(self.tmrca_all) == len(self.tmrca_all[0]), "tmrca_all must be a square matrix"  # Ensure tmrca_all is initialized as a square matrix
        # Create a new row and column for the new strain. Note: this can become a sparse matrix, with values only for co-circulating strains
        tmrca_new_strain = np.array([self.tmrca_all[id_parent][j] for j in range(self.total_strains)] + [new_strain_t0])
        if len(tmrca_new_strain) <= self.tmrca_all.shape[0]:
            tmrca_new_strain = np.pad(tmrca_new_strain, (0, self.tmrca_all.shape[0] - len(tmrca_new_strain)), mode='constant', constant_values=0)
        if self.total_strains + 1 >= self.tmrca_all.shape[0]:
            self.tmrca_all = np.pad(self.tmrca_all, ((0, 1), (0, 1)), mode='constant', constant_values=0)    
        self.tmrca_all[self.total_strains, :] = tmrca_new_strain
        self.tmrca_all[:, self.total_strains] = tmrca_new_strain

    def strain_distances(self, strain, id_parent):
        """
        Append a new strain to the distance matrix, copying the distances from the parent strain and adding 1.
        """
        assert len(self.distances_all) == len(self.distances_all[0]), "distances_all must be a square matrix"  # Ensure distances_all is initialized as a square matrix
        # Note, this can become a sparse matrix or a new matrix, with values only for co-circulating strains, instead of all strains. 
        distances_new_strain = np.array([self.distances_all[id_parent][j] + strain.distance for j in range(self.total_strains)] + [0])
        if len(distances_new_strain) <= self.distances_all.shape[0]:
            distances_new_strain = np.pad(distances_new_strain, (0, self.distances_all.shape[0] - len(distances_new_strain)), mode='constant', constant_values=0)
        if self.total_strains + 1 >= self.distances_all.shape[0]: 
            self.distances_all = np.pad(self.distances_all, ((0, 1), (0, 1)), mode='constant', constant_values=0)    
        self.distances_all[self.total_strains, :] = distances_new_strain
        self.distances_all[:, self.total_strains] = distances_new_strain       
        return distances_new_strain  # Return the distances of the new strain

    @classmethod
    def from_vector(cls, dat, Ks, parameters, strains=None):
        """
        Create an SIRState from a vector and an array of cross-immunity matrices.
        """
        print("A new state is created from a solution vector. Note that information may be lost")
        assert len(dat) == parameters.M * parameters.N * 4
        regions = []
        if not strains:
            strains = [Strain(a=a, dummy=True) for a in range(parameters.N)]
        for i in range(parameters.M):
            viruses = [Virus(S=dat[*sir_index(i, a, 'S', parameters.N)],
                             I=dat[*sir_index(i, a, 'I', parameters.N)],
                             C=dat[*sir_index(i, a, 'C', parameters.N)],
                             R=dat[*sir_index(i, a, 'R', parameters.N)],
                             strain=strains[a]) for a in range(parameters.N)]
            regions.append(Region(viruses, Ks[i]))
        return cls(parameters, regions)
    
    def update_from_vector(self, dat):
        """
        Update the state from a vector. 
        """
        assert len(dat) == self.parameters.M * self.parameters.N * 4
        assert not np.isnan(np.min(dat)), "There are None items in the array"
        if np.any(dat < 0):
            print("Warning: Negative values found in state vector; setting to 0.")
        if np.any(dat > 1):
            print("Warning: Values greater than 1 found in state vector; setting to 1.")
        dat = np.where(dat < 0, 0, dat) # If any values are negative, set them to 0 
        dat = np.where(dat > 1, 1, dat) # If any values are greater than 1, set them to 1 

        for i in range(self.parameters.M):
            for a in range(self.parameters.N):
                for g in ['S', 'I', 'R']:
                    setattr(self.regions[i].viruses[a], g, dat[*sir_index(i, a, g, self.parameters.N)])
                self.regions[i].viruses[a].correct_population() 
                    
    def __getitem__(self, key):
        """
        Get the value of a specific compartment for a given region and virus.
        """
        i, a, g = key
        return getattr(self.regions[i].viruses[a], g)

    def vec(self):
        """
        Convert the state to a vector.
        """
        return np.concatenate([r.vec() for r in self.regions])
    
    def parameter_dict(self):
        Ks = [r.K for r in self.regions]
        ageing = self.parameters.ageing
        p = {
            'params': self.parameters,
            'K': Ks,
            'Kas': [np.array([K_row[a, :] for K_row in Ks]) for a in range(self.parameters.N)], # Cross immunity matrix, but formulated by strain x region. Check if these are forward or backward immunities. if you use matrx multiplication, you need to transpose the matrix. Now, instead, I use scalrar multiplication and sum over the rows
            'idxs': precompute_indices(self.parameters.M, self.parameters.N), # pre-calculate the indices for SIR model dynamics
            'F': flux_matrix(self.parameters.M) if ageing else (np.zeros((self.parameters.M, self.parameters.M)) - np.eye(self.parameters.M))/self.parameters.M,  # pre-calculate the flux matrices with -1 on the diagonal
            'γe1': np.array([self.parameters.γ if i == 0 else 0 for i in range(self.parameters.M)]) if ageing else np.array([self.parameters.γ/self.parameters.M for _ in range(self.parameters.M)])  # First standard basis vector γ*e1

        }
        return p
    
    def delete_strain(X, a, t, remove_rows=False):
        """
        Delete a virus strain from the model with the given index.

        Parameters:
        X (SIRState): The  state of the system.
        a (int): The index of the strain to remove. 
        t: The current time point 
        """
        if a < len(X.regions[0].viruses): # If a is None, or a is larger than the number of viruses, we are dealing with a scenario where a new strain is added at position a. .
            X.regions[0].viruses[a].strain.set_extinct(t) # set the time of extinction for the strain
        for r in X.regions: 
            r.viruses[a].R += r.viruses[a].I # Set the number of recovered individuals to the number of infected individuals
            r.viruses[a].I = 0 # Set the number of infected individuals to zero 
            assert np.isclose(r.viruses[a].population(), 1, atol=1e-10), "Population should be one after deletion" # Check that the population is still one
        # Dynamic datastructure:  
        if remove_rows:
            for b in range(X.parameters.N): # Loop over all viruses in the region
                if X.regions[0].viruses[b].strain.a > a:
                    X.regions[0].viruses[b].strain.a -= 1 # Decrease the index of the strain by one
            X.circulating_strains = np.delete(X.circulating_strains, a)  # Update X.circulating_strains: remove a from the circulating strains
            X.parameters.N -= 1 # Decrease the number of viruses
            circulating_strains = [X.regions[0].viruses[b].strain for b in range(X.parameters.N)]
            for strain in circulating_strains:
                if strain.a > a:
                    strain.a -= 1 # This is not very important, because strain.a is not used in the model.
            for r in X.regions: 
                del r.viruses[a]  # Update r.viruses: remove a from the circulating strains
                r.K = np.delete(r.K, a, axis=0)  # Remove row 'a' from the cross-immunity matrix
                r.K = np.delete(r.K, a, axis=1)  # Remove column 'a' from the cross-immunity matrix
        print(X.parameters.N, " circulating strains")

    def select_parent_strain(self):
        """
        Pick a region index and parent strain index combination for the new strain to be added to the model based on the highest number of infections.

        Parameters:
        self (SIRState): The state of the system.

        Returns:
        tuple: The region index and the parent strain index.
        """
        max_infections, a_parent, r_parent = -1, -1, -1
        for r_idx, region in enumerate(self.regions):
            for i in range(self.parameters.N):
                if region.viruses[i].I >= max_infections:
                    max_infections = region.viruses[i].I
                    a_parent = i
                    r_parent = r_idx
        if max_infections == 0:
            print("Warning: no infections found in the system")
        if a_parent == -1 or r_parent == -1:
            print("Warning: no parent strain found")
        return r_parent, a_parent
   
    def select_parent_strain_probabilistic(self, probability="prevalence"):
        """
        Pick a region index and parent strain index combination for the new strain to be added to the model 
        based on a probability proportional to the number of infections.

        Parameters:
        self (SIRState): The state of the system.

        Returns:
        tuple: The region index and the parent strain index.
        """
        infection_counts, indices = [], []

        # Collect all region-virus combinations and their infection counts
        for r_idx, region in enumerate(self.regions):
            for i in range(self.parameters.N):
                infection_counts.append(region.viruses[i].I)
                indices.append((r_idx, i))
        # Convert infection counts to probabilities
        total_infections = sum(infection_counts)
        if total_infections == 0:
            print("Warning: no infections found in the system")
            return -1, -1
        if probability == "prevalence":
            probabilities = [count / total_infections if count > 0 else 0 for count in infection_counts]
        elif probability == "uniform":
            probabilities = [1 / len(indices)] * len(indices)
        elif probability == "uniform_excluding_zeros":
            # where infection counts is zero, set probability to zero. Normalize the remainging probabilities to sum to one.
            non_zero_indices = [idx for idx, count in enumerate(infection_counts) if count > 0]
            probabilities = [1 / len(non_zero_indices)if count > 0 else 0 for count in infection_counts]
        # Randomly select a region-virus combination based on the probabilities
        selected_index = np.random.choice(len(indices), p=probabilities/np.sum(probabilities))
        return indices[selected_index]

    def select_low_frequency_strains(self, I_threshold=1e-7):
        """
        Select all strains that are below a certain infection threshold in all regions.

        Parameters:
        self (SIRState): The state of the system.
        I_threshold (float): The threshold for considering a strain low frequency.

        Returns:
        list: A list of indices of low frequency strains.
        """
        low_frequency_strains = []
        for i in range(self.parameters.N):
            total_infections = sum(region.viruses[i].I for region in self.regions)
            if total_infections < I_threshold:
                low_frequency_strains.append(i)
        return low_frequency_strains

    def select_lowest_growth_rate(self, a_parent=None):
        """
        Find the index of the strain that has the lowest number of infections when summed over all regions,
        excluding the parent strain (a_parent).

        Parameters:
            a_parent (int, optional): Index of the parent strain to exclude from extinction.

        Returns:
            int: The index of the strain with the lowest number of infections, or -1 if not found.
        """
        a_extinct = -1
        growth_rates_ = growth_rates(self)
        selected_variants = [i for i in range(self.parameters.N) if growth_rates_[i] < 0]
        if not selected_variants:  # if it is empty, select all variants
            print("all growth rates are positive")
            selected_variants = range(self.parameters.N)
        min_infections = float('inf')
        for i in selected_variants:
            total_infections = sum(region.viruses[i].I for region in self.regions)
            if total_infections <= min_infections and i != a_parent:
                min_infections = total_infections
                a_extinct = i
        if a_extinct == -1:
            print("Warning: No strain found for extinction (all are parent or have no infections).")
        return a_extinct

    def select_lowest_frequency_strain(self, a_parent=None):
        """
        Find the index of the strain that has the lowest number of infections when summed over all regions,
        excluding the parent strain (a_parent).

        Parameters:
            a_parent (int, optional): Index of the parent strain to exclude from extinction.

        Returns:
            int: The index of the strain with the lowest number of infections, or -1 if not found.
        """
        min_infections = float('inf')
        a_extinct = -1
        for i in range(self.parameters.N):
            total_infections = sum(region.viruses[i].I for region in self.regions)
            if total_infections <= min_infections and i != a_parent:
                min_infections = total_infections
                a_extinct = i
        if a_extinct == -1:
            print("Warning: No strain found for extinction (all are parent or have no infections).")
        return a_extinct
    
    def add_strain(X, t=None, remove_rows=True ):
        """
        Add a new virus strain to the model, replacing low frequency strains if needed.

        Parameters:
            t (float, optional): Time of strain introduction. If None, uses current time.
            remove_rows (bool): Whether to remove extinct strains from data structures.

        Returns:
            tuple: (a, new_strain.id) where a is the index of the new strain and new_strain.id is its unique identifier.

        Raises:
            AssertionError: If the parent strain is selected for extinction.
        """
        if t is None: 
            t = X.t # If no time is given, use the current time

        M, I0, Ie, Nmax, d, σ, cross_immunity_efficacy = \
            (getattr(X.parameters, k) for k in ['M', 'I0', 'Ie', 'Nmax', 'd', 'σ', 'cross_immunity_efficacy']) # Unpack parameters for clarity 
        r_parent, a_parent = X.select_parent_strain_probabilistic(probability=X.parameters.strain_introduction) 
        parent_strain = X.regions[0].viruses[a_parent].strain # get parent strain, before it may be set extinct.

        # Check if strains can be set extinct
        low_frequency_strains = sorted(X.select_low_frequency_strains(I_threshold=Ie), reverse=True) # Select all strains that are below a certain infection threshold in all regions.
        for strain_a in low_frequency_strains:
            if strain_a != a_parent: # Never set the parent strain extinct (although one could do this)
                X.delete_strain(strain_a, t, remove_rows=remove_rows) # Set the strain extinct.
                if strain_a < a_parent and remove_rows: 
                    a_parent -= 1  # Adjust parent index if needed
        if X.parameters.N < Nmax: # One can add a strain without having to set strains extinct.
            a = X.parameters.N  # The new strain is added at the end of the list of circulating strains.
            X.parameters.N += 1 # Increase the number of viruses 
            for r in range(M):
                X.regions[r].viruses.append(Virus(S=1, I=0, C=0, R=0, strain=Strain(a=a, dummy=True))) # Add a compartment to the solution.
        else:
            if True: # alternative deletion
                a = X.select_lowest_growth_rate(a_parent) # The strain with the lowest growth rate will be replaced by the new strain.
            else:
                a = X.select_lowest_frequency_strain(a_parent) # The strain with the lowest number of infections when summed over all regions will be replaced by the new strain.
            assert a_parent != a, "The strain selected to go extinct cannot be the parent strain." 
            X.delete_strain(a, t, remove_rows=False) # Set the strain extinct, but don't remove rows. Instead, these rows can be used to host values for the new strain.  

        new_strain = Strain(parent=parent_strain, distance=1, t0=t, a=a, strain_id=X.total_strains) # create new strain object
        parent_strain.add_child(new_strain) # add child to parent strain.

        X.update_tmrca(id_parent = parent_strain.id, new_strain_t0 = new_strain.t0) # update the tmrca matrix with the new strain
        distances = X.strain_distances(strain=new_strain, id_parent = parent_strain.id)   # distances to all other strains 
        X.strains_all.append(new_strain) # add the new strain to the list of all strains ever circulated.

        if a >= len(X.circulating_strains): 
            X.circulating_strains = np.append(X.circulating_strains, X.total_strains)
        else: 
            X.circulating_strains[a] = int(X.total_strains)  # replace the new strain in the circulating strains.

        # Approximate susceptibilities to the new strain 
        for r in range(M): # It is important to first complete the cross-immunity     
            symmetric_cross_immunity_circulating_strains = distances_to_cross_immunity(distances[X.circulating_strains], X=X, r=r, σ=σ, breadth=d, alpha=cross_immunity_efficacy, a_idx_all=-1, a_idx_circulating=a)
            X.regions[r].K = grow_cross_immunity(X.regions[r].K, a, symmetric_cross_immunity_circulating_strains) 

        # if sigma is a string and starts with the string control. then split up the string at _ and then set sigma_control = true
        if isinstance(σ, str) and σ.startswith("control"):
            sigma_control = True
            σ = float(σ.split("_")[1])
            S_total = 0
            S_total_non_random = 0
        else:
            sigma_control = False
        for r in range(M):
            I = I0 if r == r_parent else 0 # set only the initial infections in the region where it arises.
            K_ap = symmetric_cross_immunity_circulating_strains[a_parent] # get the cross-immunity of the new strain to the parent strain.
            S_p = X.regions[r].viruses[a_parent].S # Susceptibility of the parent strain in region r
            S = (S_p)**K_ap 
            if σ != 0: 
                mu = (S_p)**K_ap

                alpha = mu/σ
                beta = (1 - mu)/σ
                if alpha > 0 and beta > 0:
                    S = np.random.beta(alpha, beta)
                else:
                    print("Warning: Invalid alpha or beta for Beta distribution; using mean value.")
                    S = mu
             
                if 1 - S - I < 0: 
                    S = mu

            if sigma_control:
                S_total += S
                S_total_non_random += mu
            else:
                X.regions[r].viruses[a] = Virus(S, I, C= 1 - S - I, R=0, strain=new_strain)
            X.regions[r].viruses[a] = Virus(S, I, C= 1 - S - I, R=0, strain=new_strain) # TEMPORARY

        if sigma_control:        
            S_total_new = 0
            scaler =  S_total / S_total_non_random 
            for r in range(M):
                I = I0 if r == r_parent else 0
                K_ap = symmetric_cross_immunity_circulating_strains[a_parent]
                S = (S_p)**K_ap * scaler
                S_total_new += S
                if 1 - S - I < 0: 
                    S = (S_p)**K_ap
                X.regions[r].viruses[a] = Virus(S, I, C= 1 - S - I, R=0, strain=new_strain)
    
        X.total_strains += 1 # Update the counter of the total strains 
        
        return a, new_strain.id
