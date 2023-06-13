import numpy as np
import math
import random
from tqdm import tqdm                   #just to show progress bars
import matplotlib.pyplot as plt
from joblib import Parallel, delayed    #to compute things in parallel
import multiprocessing

class IsingModel():
    
    def __init__(self, graph, J=1.0, iterations=10000, initial_state=1):
        
        self.name = "IsingModel"
        self.N = graph.number_of_nodes()
        self.graph = graph
        self.J = J
        self.iterations = iterations
        self.initial_state = initial_state
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
        prob = math.exp(-delE * beta)                       # calculate transition probability
        if prob > random.random():                          # conditionally accept the transition
            s = -s
        self.state[rsnode] = s
        
    def simulate(self, temperature, iterations=None,
                 magnetization_per_spin=True, 
                 energy=True,
                 binder_cumulant=False,
                 susceptibility_per_spin=False,
                 specific_heat_per_spin=False):
        """Simulate the model at temperature T using a Metropolis algorithm.
        
        Parameters
        ----------
        temperature: float
            Temperature of the simulation.
        iterations: int
            Number of iteration of the simulation. If not specified, the value set on construction is used.
        magnetization_per_spin: bool
            Whether or not to simulate the magnetization per spin
        energy: bool
            Whether or not to simulate the energy
        binder_cumulant: bool
            Whether or not to simulate the binder cumulant
        susceptibility_per_spin: bool
            Whether or not to simulate the susceptibility per spin
        specific_heat_per_spin: bool
            Whether or not to simulate the specific heat per spin
        
        Returns
        ----------
        data: tuple
            Value of everything set up to true at the end of the simulation

        """

        if iterations == None:
            iterations = self.iterations

        data = {}   #empty dictionary
        M = np.zeros(iterations)
        m = np.zeros(iterations)
        E = np.zeros(iterations)
    
        self.initialize(self.initial_state) # initialize spin vector    
    
        for i in range(iterations):
            self.__montecarlo(temperature)
            M[i] = self.__netmag()
            m[i] = self.__netmag()/self.N
            E[i] = self.__netenergy()

        if magnetization_per_spin:
            data['magnetization_per_spin'] = self.__netmag()/self.N
        if energy:
            data['energy'] = self.__netenergy()
        if binder_cumulant:
            m4 = m**4
            m2 = m**2
            data['binder_cumulant'] = 1- m4.mean()/(3*(m2.mean()**2))
        if susceptibility_per_spin:
            M2 = M**2
            #K_B = 1
            data['susceptibility_per_spin'] = self.N/(1*temperature) * (M2.mean() - M.mean()**2)
        if specific_heat_per_spin:
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
    
    def viz(self, temperature, iterations=None,
            magnetization_per_spin=True, 
            energy=True,
            binder_cumulant=False,
            susceptibility_per_spin=False,
            specific_heat_per_spin=False):
        """Simulate and visualise the energy and magnetization wrt a temperature range.
        
        Parameters
        ----------
        temperature: array_like
            Temperature range over which the model shall be simulated.
            Parameters
        iterations: int
            Number of iteration of each simulation. If not specified, the value set on construction is used.
        magnetization_per_spin: bool
            Whether or not to simulate and graph the magnetization per spin
        energy: bool
            Whether or not to simulate and graph the energy
        binder_cumulant: bool
            Whether or not to simulate and graph the binder cumulant
        susceptibility_per_spin: bool
            Whether or not to simulate and graph the susceptibility per spin
        specific_heat_per_spin: bool
            Whether or not to simulate and graph the specific heat per spin
        
        Returns
        ----------
        arr_of_data: np.array of tuples
            Everything set up to true. Each element is an array of the same lenght of temperature
        """

        if iterations == None:
            iterations = self.iterations

        # np.array where each element is a tuple. Each tuple is a set of data (e.g. mag, energy and binder_cum)
        # and we have a set of them, one for each temperature.
        arr_of_data = np.zeros(len(temperature), dtype=dict)
        
        for i in tqdm(range(len(temperature))):
            # print(" Temp : " + str(temperature[i]))
            arr_of_data[i] = self.simulate(temperature[i], iterations,
                                    magnetization_per_spin, energy, binder_cumulant, 
                                    susceptibility_per_spin, specific_heat_per_spin)

        if (magnetization_per_spin):
            #extract the element 'magnetization_per_spin' for each element of arr_of_data -> get m over temperature
            m = np.array([item['magnetization_per_spin'] for item in arr_of_data])

            plt.figure()
            plt.plot(temperature, m)
            plt.xlabel('Temperature')
            plt.ylabel('Magnetization per spin')
            plt.title('Magnetization vs Temperature')
        
        if (energy):
            E = np.array([item['energy'] for item in arr_of_data])
            plt.figure()
            plt.plot(temperature, E)
            plt.xlabel('Temperature')
            plt.ylabel('Energy')
            plt.title('Energy vs Temperature')

        if (binder_cumulant):
            b = np.array([item['binder_cumulant'] for item in arr_of_data])
            plt.figure()
            plt.plot(temperature, b)
            plt.xlabel('Temperature')
            plt.ylabel('Binder cumulant')
            plt.title('Binder cumulant vs Temperature')

        if (susceptibility_per_spin):
            s = np.array([item['susceptibility_per_spin'] for item in arr_of_data])
            plt.figure()
            plt.plot(temperature, s)
            plt.xlabel('Temperature')
            plt.ylabel('Scusceptibility per spin')
            plt.title('Scusceptibility per spin vs Temperature')

        if (specific_heat_per_spin):
            s = np.array([item['specific_heat_per_spin'] for item in arr_of_data])
            plt.figure()
            plt.plot(temperature, s)
            plt.xlabel('Temperature')
            plt.ylabel('Specific heat per spin')
            plt.title('Specific heat per spin vs Temperature')

        return arr_of_data

    def viz_parallel(self, temperature):
        """Simulate and visualise the energy and magnetization wrt a temperature range with python parallelization.
        
        Parameters
        ----------
        temperature: array_like
            Temperature range over which the model shall be simulated.
        
        Returns
        ----------
        Magnetization and Energy
        """
        mag = []
        ene = []
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.simulate)(i) for i in temperature)
    
        for cr in results:
            mag.append(cr[0])
            ene.append(cr[1])

        plt.figure()
        plt.plot(temperature, mag)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs Temperature')

        plt.figure()
        plt.plot(temperature, ene)
        plt.xlabel('Temperature')
        plt.ylabel('Energy')
        plt.title('Energy vs Temperature')

        return mag, ene

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
    