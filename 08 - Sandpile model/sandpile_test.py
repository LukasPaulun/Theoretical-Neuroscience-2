import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sandpiles
import powerlaw

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

np.seterr(divide='ignore', invalid='ignore')

#%%
# gridsize
Nx = 50
Ny = Nx

Niter = 1e5

sandpile = sandpiles.Sandpile(Nx, Ny, 4)
mean_heights = sandpile.simulate(Niter=Niter, plot=False, save=False)


#%%
#parameter_range = {'alpha': [0, 0.9]}

energies = sandpile.get_avalanche_energies()
times = sandpile.get_avalanche_times()
linear_sizes = sandpile.get_avalanche_linear_sizes()

energyfit = powerlaw.Fit(energies, xmin=1, xmax=sandpile.N, discrete=True)
timefit = powerlaw.Fit(times, xmin=1, xmax=100, discrete=True)
sizefit = powerlaw.Fit(linear_sizes, xmin=1, discrete=True)

# Create plots for avalanche sizes and durations
fig, ax = plt.subplots(1, 3, figsize=(18,7))

# Plot distribution of avalanche sizes
powerlaw.plot_pdf(energies, ax=ax[0], color='black', label='Empirical pdf')
energyfit.power_law.plot_pdf(ax=ax[0], color='blue', linestyle='--', \
                            label=r'Power law, $\alpha = $' + f'{energyfit.power_law.alpha:.2f}')
energyfit.truncated_power_law.plot_pdf(ax=ax[0], color='red', linestyle='--', \
                                      label=r'Truncated power law, $\alpha = $' + \
                                          f'{energyfit.truncated_power_law.alpha:.2f}' + \
                                          r', $\lambda = $' + f'{energyfit.truncated_power_law.parameter2:.2e}')
ax[0].set_xlabel('Avalanche energy')
ax[0].set_ylabel(r'$p(X)$')
ax[0].legend(loc='lower left', fontsize=14)

# Plot distribution of avalanche durations
powerlaw.plot_pdf(times, ax=ax[1], color='black', label='Empirical pdf')
timefit.power_law.plot_pdf(ax=ax[1], color='blue', linestyle='--', \
                            label=r'Power law, $\alpha = $' + f'{timefit.power_law.alpha:.2f}')
timefit.truncated_power_law.plot_pdf(ax=ax[1], color='red', linestyle='--', \
                                      label=r'Truncated power law, $\alpha = $' + \
                                          f'{timefit.truncated_power_law.alpha:.2f}' + \
                                          r', $\lambda = $' + f'{timefit.truncated_power_law.parameter2:.2e}')
ax[1].set_xlabel('Avalanche duration')
ax[1].legend(loc='lower left', fontsize=14)

powerlaw.plot_pdf(linear_sizes, ax=ax[2], color='black', label='Empirical pdf')
sizefit.power_law.plot_pdf(ax=ax[2], color='blue', linestyle='--', \
                            label=r'Power law, $\alpha = $' + f'{sizefit.power_law.alpha:.2f}')
sizefit.truncated_power_law.plot_pdf(ax=ax[2], color='red', linestyle='--', \
                                      label=r'Truncated power law, $\alpha = $' + \
                                          f'{sizefit.truncated_power_law.alpha:.2f}' + \
                                          r', $\lambda = $' + f'{sizefit.truncated_power_law.parameter2:.2e}')
ax[2].set_xlabel('Linear avalanche size')
ax[2].legend(loc='lower left', fontsize=14)

# #plt.tight_layout()