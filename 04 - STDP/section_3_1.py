import neurons

sim_time = 15
dt = 0.0001

# Parameters for input neurons
rate_1 = 5
rate_2 = 8

# Synapse parameters
init_weight = 1
max_weight = 6

A_P = 0.05
tau_P = 17e-3
A_D = -0.025
tau_D = 34e-3

# Create two Poisson neurons as input
input_1 = neurons.PoissonNeuron(sim_time, dt)
input_2 = neurons.PoissonNeuron(sim_time, dt)

input_1.generate_spikes(rate_1)
input_2.generate_spikes(rate_2)

# Create two STDP synapses with given parameters
STDPSynapses = [neurons.STDPSynapse(sim_time, dt, 'exc', init_weight=init_weight, \
                                A_P=A_P, tau_P=tau_P, A_D=A_D, tau_D=tau_D) for _ in range(2)]

# Create LIF neuron and connect the two Poisson neurons via the STDP synapses
LIF = neurons.LIFNeuron(sim_time, dt, w_SRA=0)
LIF.connect_neurons([input_1, input_2], STDPSynapses)

# Run simulations
LIF.simulate()

# Create plots
neurons.plot_spikes([input_1, input_2], title='Spike trains of input Poisson neurons')

LIF.plot_V(title='Membrane trace of LIF neuron')
neurons.plot_firing_rates([LIF], title='Firing rate of LIF neuron')

title = 'Evolution of synaptic weights\nRate of input neuron 1: ' + str(rate_1) + 'Hz\nRate of input neuron 2: ' + str(rate_2) + 'Hz'
neurons.plot_synaptic_weights(STDPSynapses, title=title)










