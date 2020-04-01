"""
Example script for how to use the module neurons.py
"""
import neurons
import numpy as np

# Global parameters, must be the same for all created neurons
sim_time = 5
dt = 0.001

# Create an array of 100 excitatory Poisson neurons
exc_input = np.array([neurons.PoissonNeuron(sim_time, dt, 'exc') for _ in range(10)])

# Create spike trains for the excitatory Poisson neurons with mean rate 10 Hz
for neuron in exc_input:
    neuron.generate_spikes(rate=10)

# Create an array of 100 inhibitory Poisson neurons
inh_input = np.array([neurons.PoissonNeuron(sim_time, dt, 'inh') for _ in range(10)])

# Create spike trains for the inhibitory Poisson neurons with mean rate 10 Hz
# starting at 1s and ending at 4s
for neuron in inh_input:
    neuron.generate_spikes(rate=10, start=1, stop=4)

# Plot spike trains of the excitatory and the inhibitory population
neurons.plot_spikes(exc_input, 'Spike trains of excitatory Poisson neurons')
neurons.plot_spikes(inh_input, 'Spike trains of inhibitory Poisson neurons')

# Create an LIF neuron
LIF = neurons.LIFNeuron(sim_time, dt)

# Connect the Poisson neurons to the LIF neuron
LIF.connect(exc_input)
LIF.connect(inh_input)

# Run the simulation of the LIF neuron
LIF.simulate()

# Plot the trace of the membrane potential and ISI statistics
LIF.plot_V('Membrane trace of the LIF neuron')
LIF.plot_ISI()





