import neurons

sim_time =20
dt = 0.0001

# Parameters for input neurons
rate_1 = 5
rate_2 = 5

# Synapse parameters
init_weight = 1
max_weight = 6

A_P = 0.05
tau_P = 17e-3
A_D = -0.025
tau_D = 34e-3


input_1 = neurons.PoissonNeuron(sim_time, dt)
input_2 = neurons.PoissonNeuron(sim_time, dt)

STDPSynapses = [neurons.STDPSynapse(sim_time, dt, 'exc', init_weight=init_weight, \
                                A_P=A_P, tau_P=tau_P, A_D=A_D, tau_D=tau_D) for _ in range(2)]

input_1.generate_spikes(rate_1)
input_2.generate_spikes(rate_2)

LIF = neurons.LIFNeuron(sim_time, dt, w_SRA=0)

LIF.connect_neurons([input_1, input_2], STDPSynapses)

LIF.simulate()

neurons.plot_synaptic_weights(STDPSynapses)








