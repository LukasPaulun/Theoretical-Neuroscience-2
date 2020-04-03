import numpy as np
import neurons
import matplotlib.pyplot as plt

sim_time = 15
dt = 0.0001

# Parameters for population of correlated neurons
N_pop = 10
target_rate = 10

c = 0.4
p = np.sqrt(c)
noise_rate = (1-p) * target_rate

# Plot parameters
max_lag = 100e-3
bin_width = 5e-3

source = neurons.PoissonNeuron(sim_time, dt)
source.generate_spikes(target_rate)

population = np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_pop)])

for neuron in population:
    neuron.generate_spikes(noise_rate)
    neuron.copy_spikes(source, p, mode='exp', tau_c=20e-3)


lags = np.arange(-max_lag, max_lag, bin_width)
cor = np.zeros_like(lags)
for i in range(N_pop-1):
    for j in range(i+1, N_pop-1):
        _, cor_temp = neurons.cross_correlogram(population[i], population[j], \
                                                    max_lag, bin_width, plot=False)

        cor += cor_temp

neurons.cross_correlogram(population[0], population[1], max_lag, bin_width)

plt.figure()
plt.plot(lags, cor)




