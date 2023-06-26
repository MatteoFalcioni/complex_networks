import numpy as np
import math
import random
from tqdm import tqdm                   #just to show progress bars
import matplotlib.pyplot as plt
from joblib import Parallel, delayed    #to compute things in parallel
import multiprocessing

class IsingModel():
    
    def __init__(self, graph, J=1.0, iterations=10000, initial_state=1, temperature_range=np.arange(0,10,0.1)):
        
        self.name = "IsingModel"
        self.N = graph.number_of_nodes()
        self.graph = graph
        self.J = J
        self.iterations = iterations
        self.initial_state = initial_state
        self.temperature_range = temperature_range
        self.arr_of_data = None
        self.list_of_neigh = {}
        for node in self.graph.nodes():
            self.list_of_neigh[node] = list(self.graph.neighbors(node))
        
    def initialize(self, initial_state):
        
        self.state = np.random.choice([-1,1], self.N, p=[1-initial_state, initial_state])

    def set_J(self, J):
        """Set the value of J

        Parameter(s)
        ----------
        J : int
            Interaction coefficient.
        """
        self.J = J

    def set_iterations(self, iterations):
        """Set your desired number of iterations per temperature value

        Parameter(s)
        ----------
        iterations: int
            Iterations per temperature value.
        """
        self.iterations = iterations

    def set_initial_state(self, initial_state):
        """Set initial state

        Parameter(s):
        initial_state: int [0,1]
            Initial state of all nodes of the system.
        """
        if np.abs(initial_state) > 1:
            raise Exception("initial_state should be between 0 and 1")
        
        self.initial_state = initial_state
    
    def __netmag(self):
        
        return np.sum(self.state)
    
    def __netenergy(self):
        en = 0.
        for i in range(self.N):
            ss = np.sum(self.state[self.list_of_neigh[i]])
            en += self.state[i] * ss
        return -0.5 * self.J * en
    
    def __montecarlo(self, temperature):
        beta = 1/temperature
        rsnode = np.random.randint(0, self.N)            # pick a random source node
        s = self.state[rsnode]                              # get the spin of this node
        ss = np.sum(self.state[self.list_of_neigh[rsnode]]) # sum of all neighbouring spins        
        delE = 2.0 * self.J * ss * s                        # transition energy
        if delE < 0:
            s = - s
        else:
            prob = math.exp(-delE * beta)                       # calculate transition probability
            if random.random() < prob:                          # conditionally accept the transition
                s = -s
        self.state[rsnode] = s
        
    def simulate(self, temperature, iterations=None):
        """Simulate the model at temperature T using a Metropolis algorithm.
        
        Parameters
        ----------
        temperature: float
            Temperature of the simulation.
        iterations: int
            Number of iteration of the simulation. If not specified, the value set on construction is used.
        
        Returns
        ----------
        data: dictionary
            Value of all the quantities calculated.

        """

        if iterations == None:
            iterations = self.iterations

        data = {}   #empty dictionary
        M = np.zeros(iterations)
        m = np.zeros(iterations)
        E = np.zeros(iterations)
    
        self.initialize(self.initial_state) # initialize spin vector    
    
        for _ in range(int(iterations/2)):
            self.__montecarlo(temperature)
        for i in range(int(iterations/2)):
            self.__montecarlo(temperature)
            M[i] = self.__netmag()
            m[i] = self.__netmag()/self.N
            E[i] = self.__netenergy()

        data['magnetization_per_spin'] = self.__netmag()/self.N
        data['energy'] = self.__netenergy()
    
        m4 = m**4
        m2 = m**2
        data['binder_cumulant'] = 1- m4.mean()/(3*(m2.mean()**2))
    
        M2 = M**2           #using K_B = 1
        data['susceptibility_per_spin'] = self.N/(1*temperature) * (M2.mean() - M.mean()**2)
    
        E2 = E**2
        data['specific_heat_per_spin'] = self.N/((1*temperature)**2) * (E2.mean() - E.mean()**2)

        return dict(data)

    # def simulate_decay(self, temperature):
        
    #     self.initialize(self.initial_state)
        
    #     for i in range(self.iterations):
    #         self.__montecarlo(temperature)
    #         mag = self.__netmag()
    #         if mag <= (0.75*self.orig_mag):
    #             return i
    
    def iterate(self, temperature_range=None, iterations=None, simulations=1, parallel=True, verbose=0):
        """Run a simulation and calculate quantities wrt a temperature range.

        Parameters
        ----------
        temperature_range: array_like
            Temperature range over which the model shall be simulated.
        iterations: int
            Number of iteration of each simulation. If not specified, the value set on construction is used.
        parallel: bool
            Whether or not to simulate using python parallelization
        verbose: int, optional
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        Returns
        ----------
        self.arr_of_data: np.array of dictionaries
            All data for each value of temperature. Each dictionary of the array contains the data of a specific temperature.
        """

        if temperature_range is None:
            temperature_range = self.temperature_range
        else:
            self.temperature_range = temperature_range
        
        if iterations is None:
            iterations = self.iterations
        else:
            self.iterations = iterations

        def calculate_means(temperature):
            toBeMeaned = np.zeros(simulations, dtype=dict)
            
            for j in tqdm(range(simulations), leave=False):
                toBeMeaned[j] = self.simulate(temperature_range[temperature], iterations)
            
            keys = ['magnetization_per_spin','energy','binder_cumulant',
                    'susceptibility_per_spin', 'specific_heat_per_spin']
            means = {}
            for key in keys:
                values = [d[key] for d in toBeMeaned]
                means[key] = np.mean(values)
            return means
        
        if parallel:
            num_cores = multiprocessing.cpu_count()
            self.arr_of_data = Parallel(n_jobs=num_cores, verbose=verbose)(delayed(calculate_means)(i) for i in range(len(temperature_range)))
        else:
            self.arr_of_data = np.zeros(len(temperature_range), dtype=dict)    
            
            #for each temperature
            for i in tqdm(range(len(temperature_range))):
                self.arr_of_data[i] = calculate_means(i)

        return self.arr_of_data
    
    def get_data(self, quantity):
        """Return the data of given quantity (or quantities)
        
        Parameter:
        quantity: str
            The quantity that need to be returned.
        
        Return:
        data: np.array
            Array containing the quantity wrt self.temperature_range
        """

        if self.arr_of_data is None:
            self.iterate()

        data = np.array([item[quantity] for item in self.arr_of_data])
        return data
    
    def save(self, filename):
        """Save simulation data in a file"""

        np.save(filename, self.arr_of_data)

    def load(self, filename):
        """Load a previous simulation from a file"""

        self.arr_of_data = np.load(filename)
    
    def get_temperature_range(self):
        return self.temperature_range

    
    def plot(self, quantities=None, ylabels=None):
        """Plot the given quantity (or quantities) using matplotlib
        
        Parameters:
        ----------
        quantities: str, list of str
            The quantity(s) that needs to be plotted. Both string and list of strings are accepted types.
            By default, all the quantities are plotted
        ylabel(s): str, list of str
            The ylabel of each graph. By default, the name of the quantities is used

        """
        if quantities is None:
            quantities = ['magnetization_per_spin','energy','binder_cumulant',
                          'susceptibility_per_spin', 'specific_heat_per_spin']
        elif isinstance(quantities, str):
            # Convert the input to a list if it's a single string
            quantities = [quantities]

        if ylabels is None:
            ylabels = quantities
        elif isinstance(ylabels, str):
            ylabels = [ylabels]

        for quantity, ylabel in zip(quantities, ylabels):
            data = np.array([item[quantity] for item in self.arr_of_data])
            plt.figure()
            plt.plot(self.temperature_range, data)
            plt.xlabel('Temperature')
            plt.ylabel(ylabel)
            plt.title(ylabel + ' vs Temperature')

    # def viz_decay(self, N, temperature, ensemble=5):
    #     """A simulation which returns number of steps required for the magnetization in the network to decay to 0.75 of original value.

    #     Parameters
    #     ----------
    #     N: int
    #         Size of the network

    #     temperature: array_like
    #         Temperature range over which the model is simulated

    #     ensemble: int
    #         Number of samples from which the median is considered

    #     Returns
    #     -------
    #     decay_time: ND array
    #         Decay time array

    #     """
    #     self.orig_mag = N
    #     tau = np.zeros(len(temperature))
    #     tau_2 = np.zeros(ensemble)
    #     for i in tqdm(range(len(temperature))):
    #         for j in range(ensemble):
    #             tau_2[j] = self.simulate_decay(temperature[i])
    #         tau[i] = np.median(tau_2)
            
    #     plt.figure()
    #     plt.style.use('seaborn-whitegrid')
    #     plt.plot(temperature, tau, 'x')
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Tau')
    #     plt.xscale('log')
    #     plt.yscale('log')
        
    #     return tau

    # def curie_temp(self, end, ensemble, start=0.1, threshold=0.1):
    #     """Determines the Curie temperature / critical temperature

    #     Parameters
    #     ----------

    #     end: float
    #         upper limit of the temperature range

    #     ensemble: int
    #         number of samples from which the median is evaluated

    #     start: float
    #         lower limit of the temperature range

    #     threshold: float
    #         value to which the original magnetization must decay at the Curie temperature
    #     """
        
    #     orig_mag = self.N
        
    #     step_size = (end - start)/30
        
    #     res = np.zeros(ensemble)
        
    #     temperature = np.arange(start, end, step_size)
    #     for t in tqdm(range(ensemble)):
    #         for i in tqdm(range(len(temperature))):
    #             # print(" Temp : " + str(temperature[i]))
    #             mag = self.simulate(temperature[i], energy = False)
    #             if mag < threshold:
    #                 res[t] = temperature[i]
    #                 break                    
                    
    #     return np.median(res)

    # def mean_degree(self, directed=False):
	        
    #     n_edges = self.graph.number_of_edges()
    #     if directed == True:
    #         return n_edges/self.N
    #     return (2*n_edges/self.N)
    