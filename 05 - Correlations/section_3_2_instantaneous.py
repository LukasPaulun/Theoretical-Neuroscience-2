import numpy as np
import neurons

sim_time = 15
dt = 0.0001

# Parameters for population of correlated neurons
N_pop = 10
target_rate = 10

# Correlation parameters
c = np.array([0.2, 0.5])

# Create populations of Poisson neurons
population = [np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_pop)])
                for _ in range(c.size)]

for ii in range(c.size):
    # Compute copy probability
    p = np.sqrt(c[ii])
    noise_rate = (1-p) * target_rate

    # Create source neuron
    source = neurons.PoissonNeuron(sim_time, dt)
    source.generate_spikes(target_rate)

    # Generate independent spikes with noise_rate and copy spikes from source with probability p
    for neuron in population[ii]:
        neuron.generate_spikes(noise_rate)
        neuron.copy_spikes(source, p, mode='exp', tau_c=10e-3)

    # Plot spike trains and correlograms
    neurons.plot_spikes(population[ii], title='Spike trains of population with c = ' + str(c[ii]))

    lags, cor = neurons.cross_correlogram(population[ii][0], population[ii][1], \
                                   max_lag=100e-3, bin_width=5e-3, \
                                   title='Cross-correlogram of first two neurons from population with c = ' + str(c[ii]))

# Plot correlogram of two neurons from different populations
if c.size > 1:
    neurons.cross_correlogram(population[0][0], population[1][0], max_lag=100e-3, bin_width=5e-3, \
        title='Cross correlogram for two neurons from different populations')


