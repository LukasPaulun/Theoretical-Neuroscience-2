#%%
import neurons
import synapses

sim_time = 5
dt = 0.1e-3

# Neuron parameters
input_rate = 10

# Synaptic parameters
U = 0.4
max_weight = 1
tau_f = 50e-3
tau_d = 750e-3

#%%
# Create source neuron with Poisson spikes
source = neurons.PoissonNeuron(sim_time, dt)
source.generate_spikes(input_rate)

# Generate STP Synapse with facilitation but without depression
STPSynapse = synapses.STPSynapse(sim_time, dt, init_weight=0, U=U, \
                                  max_weight=max_weight, tau_f=tau_f, tau_d=tau_d, STD=False)

# Create LIF neuron and run simulation
LIF = neurons.LIFNeuron(sim_time, dt, tau_m=20e-3)
LIF.connect_neurons([source], [STPSynapse])

LIF.simulate()

#%%
LIF.plot_V()
STPSynapse.plot_STP_dynamics()
synapses.plot_synaptic_weights([STPSynapse])







