# Calculate the critical temperature wrt delta, by
# seeing # when (at which temperature) the graphs U(T) 
# (binder_cumulant wrt temperature) with N=150 and U(T) 
# with N=200 diverge.

from isingNetwork import IsingModel
from euclidianNetwork import EuclidianNetwork
import numpy as np
from joblib import Parallel, delayed    #to compute things in parallel
import multiprocessing
import matplotlib.pyplot as plt

epsilon = 0.2
delta_range = np.arange(0.6,1.8,0.2)

num_cores = multiprocessing.cpu_count()

def simulateDelta(delta):
    g = EuclidianNetwork(150, delta)
    model150 = IsingModel(g.get_graph())

    g = EuclidianNetwork(200, delta)
    model200 = IsingModel(g.get_graph())

    def getBinder150(temperature):
        data150 = model150.simulate(temperature)
        b150 = data150['binder_cumulant']
        return b150
    
    def getBinder200(temperature):
        data200 = model200.simulate(temperature)
        b200 = data200['binder_cumulant']
        return b200
    

    simulations = 10
    temperature = 1.2

    b150 = 0
    b200 = 0

    while((b200 - b150) < epsilon):
        temperature += 0.05
        print('temperature = ' + str(temperature))

        binder150 = np.zeros(simulations)
        binder200 = np.zeros(simulations)

        num_cores = multiprocessing.cpu_count()
        binder150 = Parallel(n_jobs=num_cores)(delayed(getBinder150)(temperature) for _ in range(simulations))
        binder200 = Parallel(n_jobs=num_cores)(delayed(getBinder200)(temperature) for _ in range(simulations))

        b150 = np.mean(binder150)
        b200 = np.mean(binder200)

        print(b200 - b150)

    return temperature

temp_range = Parallel(n_jobs=num_cores)(delayed(simulateDelta)(i) for i in delta_range)

plt.figure()
plt.scatter(delta_range, temp_range)
plt.show()