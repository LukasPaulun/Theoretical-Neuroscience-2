#%%
import neurons
import synapses

sim_time = 5
dt = 0.1e-3

# Neuron parameters
input_rate = 20

# Synaptic parameters
U = 0.45
max_weight = 2.5
tau_f = 50e-3
tau_d = 750e-3

#%%
# Generate source neuron with regular spikes
source = neurons.RegularNeuron(sim_time, dt)
source.generate_spikes(input_rate)

# Create STP Synapse with STF and STD
STPSynapse = synapses.STPSynapse(sim_time, dt, init_weight=0, max_weight=max_weight, \
                                 U=U, tau_f=tau_f, tau_d=tau_d)


# Create LIF neuron and run simulation
LIF = neurons.LIFNeuron(sim_time, dt)
LIF.connect_neurons([source], [STPSynapse])

LIF.simulate()

#%%
#LIF.plot_V()
STPSynapse.plot_STP_dynamics()
synapses.plot_synaptic_weights([STPSynapse])















