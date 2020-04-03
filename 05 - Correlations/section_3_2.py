from typing import Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

import neurons
import synapses

Number = Union[float, int]
NumberN = Union[float, int, None]
NumberC = Union[float, int, complex]

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)


sim_time = 10
dt = 0.0001

N_pop = 10
target_rate = 10

c_list = np.array([0.1])

source = neurons.PoissonNeuron(sim_time, dt)
source.generate_spikes(target_rate)

for c in c_list:
    p = np.sqrt(c)
    noise_rate = (1-p) * target_rate

    population = np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_pop)])
    for neuron in population:
        neuron.copy_spikes(source, p, mode='instantaneous')

    neurons.plot_spikes(population, title='Spike train of population with c = ' + str(c))


neurons.plot_cross_correlogram(population[0], population[1], max_lag=200e-3, bin_width=10e-3)









