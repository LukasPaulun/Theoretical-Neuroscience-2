import numpy as np
import neurons
import synapses
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sim_time = 200
dt = 1e-3

# Parameters for the input populations
N_exc_pop = np.array([25, 25])
exc_target_rate = np.array([10, 10])    # [Hz]

N_inh_pop = 30
inh_target_rate = 10    # [Hz]

# Synaptic parameters
exc_init_weight = np.array([0.1, 0.1])
inh_init_weight = 0.1
A_P = 0.006
A_D = -0.5 * A_P

# Correlation parameters
c = np.array([0.1, 0.2])

# Parameter for synaptic normalisation
W_tot = 5
nu_SN = 0.2
step_SN = 1      # when to normalise in [s]

exc_population = []
exc_synapses = []
# Loop through all populations
for ii in range(N_exc_pop.size):
    # Create source neuron for population and generate source spike train
    source = neurons.PoissonNeuron(sim_time, dt)
    source.generate_spikes(exc_target_rate[ii])

    # Create excitatory population of Poisson neurons and their STDP synapses
    exc_population.append( np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_exc_pop[ii])]) )
    exc_synapses.append( np.array([synapses.STDPSynapse(sim_time, dt, \
                                                        init_weight=exc_init_weight[ii], \
                                                        normalize=True, \
                                                        A_P=A_P, A_D=A_D) \
                                   for _ in range(N_exc_pop[ii])]) )

    p = np.sqrt(c[ii])
    noise_rate = (1-p) * exc_target_rate[ii]
    # Generate independent Poisson spike trains and copy spikes from source neuron
    for neuron in exc_population[ii]:
        neuron.generate_spikes(noise_rate)
        neuron.copy_spikes(source, p, mode='exp', tau_c=20e-3)


# Create inhibitory population and synapses (no STDP)
inh_population = np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_inh_pop)])
inh_synapses = np.array([synapses.Synapse(sim_time, dt, typ='inh', \
                                          init_weight=inh_init_weight, normalize=False) \
                         for _ in range(N_inh_pop)])
for neuron in inh_population:
    neuron.generate_spikes(inh_target_rate)

# Create LIF neuron and connect input populations
LIF = neurons.LIFNeuron(sim_time, dt, tau_m=20e-3, W_tot=W_tot, nu_SN=nu_SN, step_SN=step_SN)
LIF.connect_neurons(exc_population[0], exc_synapses[0])
LIF.connect_neurons(exc_population[1], exc_synapses[1])
LIF.connect_neurons(inh_population, inh_synapses)

LIF.simulate()

# Make plots
neurons.plot_firing_rates([LIF])


colors = N_exc_pop[0]*['blue'] + N_exc_pop[1]*['orange']
synapses.plot_synaptic_weights(np.append(exc_synapses[0], exc_synapses[1]), \
                              color_list=colors)

blue_line = mlines.Line2D([], [], color='blue', \
                          label='Firing rate ' + str(exc_target_rate[0]) + 'Hz, correlation ' +\
                                 str(c[0]) + ', initial weight ' + str(exc_init_weight[0]))
orange_line = mlines.Line2D([], [], color='orange', \
                          label='Firing rate ' + str(exc_target_rate[1]) + 'Hz, correlation ' +\
                                 str(c[1]) + ', initial weight ' + str(exc_init_weight[1]))
plt.legend(handles=[blue_line, orange_line], loc='upper left')

weights1 = [syn.weight[-1] for syn in exc_synapses[0]]
weights2 = [syn.weight[-1] for syn in exc_synapses[1]]

plt.figure()
plt.hist(weights1)
plt.hist(weights2)








