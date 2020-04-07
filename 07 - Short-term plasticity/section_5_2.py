import numpy as np
import matplotlib.pyplot as plt

import neurons
import synapses

sim_time = 5
dt = 0.1e-3

# Neuron parameters
input_rate = 2

# Synaptic parameters
U = 0.15
max_weight = 2.5
tau_f = 750e-3
tau_d = 50e-3


source = neurons.PoissonNeuron(sim_time, dt)
source.generate_spikes(input_rate)

STPSynapse = synapses.STPSynapse(sim_time, dt, init_weight=0, max_weight=max_weight, \
                                 U=U, tau_f=tau_f, tau_d=tau_d)


#%%
LIF = neurons.LIFNeuron(sim_time, dt)
LIF.connect_neurons([source], [STPSynapse])

LIF.simulate()

#%%
STPSynapse.plot_STP_dynamics()
synapses.plot_synaptic_weights([STPSynapse])


LIF.plot_V()














