import numpy as np
import neurons
import matplotlib.pyplot as plt

sim_time = 15
dt = 0.0001

# Parameters for population of correlated neurons
N_pop = 10
target_rate = 10

# Correlation parameters
c = 0.2
tau_c = 20e-3       # width of exponential jitter
p = np.sqrt(c)
noise_rate = (1-p) * target_rate

# Plot parameters
max_lag = 100e-3
bin_width = 5e-3

# Generate source neuron
source = neurons.PoissonNeuron(sim_time, dt)
source.generate_spikes(target_rate)

# Generate population of Poisson neurons
population = np.array([neurons.PoissonNeuron(sim_time, dt) for _ in range(N_pop)])

# Generate independent spike trains and copy spikes from rousce
for neuron in population:
    neuron.generate_spikes(noise_rate)
    neuron.copy_spikes(source, p, mode='exp', tau_c=tau_c)


lags = np.arange(-max_lag, max_lag, bin_width)
cor = np.zeros_like(lags)
for i in range(N_pop-1):
    for j in range(i+1, N_pop-1):
        if i == 0 and j == 1:
            lags, cor = neurons.cross_correlogram(population[i], population[j], \
                                                  max_lag, bin_width, plot=True, \
                                                  title='Cross-correlogram for two individual neurons')
        else:
            _, cor_temp = neurons.cross_correlogram(population[i], population[j], \
                                                    max_lag, bin_width, plot=False)
            cor += cor_temp

fig, ax = plt.subplots(1,1, figsize=(14,7))

pairings = int(N_pop * (N_pop-1) / 2)
ax.plot(1000*lags, cor/pairings)

ax.set_xlabel('Cross-correlation lag [ms]')
ax.set_ylabel('Correlation []')
ax.set_title('Population of ' + str(N_pop) + ' neurons\n' + \
             r'Mean correlations computed from all $\frac{N (N-1)}{2} = $' + str(pairings) + \
                 ' possible pairings')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])





