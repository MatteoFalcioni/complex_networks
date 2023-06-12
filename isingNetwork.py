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
        self.size = graph.number_of_nodes()
        self.graph = graph
        self.J = J
        self.iterations = iterations
        self.initial_state = initial_state
        self.list_of_neigh = {}
        for node in self.graph.nodes():
            self.list_of_neigh[node] = list(self.graph.neighbors(node))
        
    def initialize(self, initial_state):
        
        self.state = np.random.choice([-1,1], self.size, p=[1-initial_state, initial_state])

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
        for i in range(self.size):
            ss = np.sum(self.state[self.list_of_neigh[i]])
            en += self.state[i] * ss
        return -0.5 * self.J * en
    
    def __montecarlo(self):
        beta = 1/self.temperature
        rsnode = np.random.randint(0, self.size)            # pick a random source node
        s = self.state[rsnode]                              # get the spin of this node
        ss = np.sum(self.state[self.list_of_neigh[rsnode]]) # sum of all neighbouring spins        
        delE = 2.0 * self.J * ss * s                        # transition energy
        prob = math.exp(-delE * beta)                       # calculate transition probability
        if prob > random.random():                          # conditionally accept the transition
            s = -s
        self.state[rsnode] = s
        
    def simulate(self, temperature):
        
        self.temperature = temperature
    
        self.initialize(self.initial_state) # initialize spin vector    
    
        for i in range(self.iterations):
            self.__montecarlo()
            mag = self.__netmag()
            ene = self.__netenergy()
        return np.abs(mag)/float(self.size), ene

    def simulate_decay(self, temperature):
        
        self.temperature = temperature
        
        self.initialize(self.initial_state)
        
        for i in range(self.iterations):
            self.__montecarlo()
            mag = self.__netmag()
            if mag <= (0.75*self.orig_mag):
                return i
    
    def viz(self, temperature):
        """Simulate and visualise the energy and magnetization wrt a temperature range.
        
        Parameters
        ----------
        temperature: array_like
            Temperature range over which the model shall be simulated.

        """
        mag = np.zeros(len(temperature))
        ene = np.zeros(len(temperature))
        
        for i in tqdm(range(len(temperature))):
            # print(" Temp : " + str(temperature[i]))
            mag[i], ene[i] = self.simulate(temperature[i])
        
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

    def viz_parallel(self, temperature):
        """Simulate and visualise the energy and magnetization wrt a temperature range with python parallelization.
        
        Parameters
        ----------
        temperature: array_like
            Temperature range over which the model shall be simulated.

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

    def viz_decay(self, N, temperature, ensemble=5):
        """A simulation which returns number of steps required for the magnetization in the network to decay to 0.75 of original value.

        Parameters
        ----------
        N: int
            Size of the network

        temperature: array_like
            Temperature range over which the model is simulated

        ensemble: int
            Number of samples from which the median is considered

        Returns
        -------
        decay_time: ND array
            Decay time array

        """
        self.orig_mag = N
        tau = np.zeros(len(temperature))
        tau_2 = np.zeros(ensemble)
        for i in tqdm(range(len(temperature))):
            for j in range(ensemble):
                tau_2[j] = self.simulate_decay(temperature[i])
            tau[i] = np.median(tau_2)
            
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.plot(temperature, tau, 'x')
        plt.xlabel('Temperature')
        plt.ylabel('Tau')
        plt.xscale('log')
        plt.yscale('log')
        
        return tau

    def curie_temp(self, end, ensemble, start=0.1, threshold=0.1):
        """Determines the Curie temperature / critical temperature

        Parameters
        ----------

        end: float
            upper limit of the temperature range

        ensemble: int
            number of samples from which the median is evaluated

        start: float
            lower limit of the temperature range

        threshold: float
            value to which the original magnetization must decay at the Curie temperature
        """
        
        orig_mag = self.size
        
        step_size = (end - start)/30
        
        res = np.zeros(ensemble)
        
        temperature = np.arange(start, end, step_size)
        for t in tqdm(range(ensemble)):
            for i in tqdm(range(len(temperature))):
                # print(" Temp : " + str(temperature[i]))
                mag = self.simulate(temperature[i], energy = False)
                if mag < threshold:
                    res[t] = temperature[i]
                    break                    
                    
        return np.median(res)

    def mean_degree(self, directed=False):
	        
        n_edges = self.graph.number_of_edges()
        if directed == True:
            return n_edges/self.size
        return (2*n_edges/self.size)
    