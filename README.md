## Theoretical-Neuroscience-2
Repository for the course Theoretical Neuroscience 2 by Jochen Triesch.

### Modules in lib/
  - neurons.py
  - synapses.py
  - devices.py

Example of features:
  - create populations of neurons with regular or Poisson spike behavior
  - create current generators
  - create STDP synapses that show LTP and LTD behavior
  - connect such neurons or current generators to LIF neurons and use different synapses for the connection
  - simulate the membrance trace of the LIF neuron
  - extract statistics, e.g. about the ISIs
  - create various plots, e.g. spike trains, membrane potential traces, ISI histograms
  
See the file example.py for an example of how to create Poisson neurons, generate spike trains, connect them to an LIF neuron and simulate its membrane potential.

### The numbered directories contain solutions to the assignments.

### Connect with colab
First, the user will have to clone this github repository into her own google drive. Then, colab notebooks can be created from the template colab_template.ipynb, which automatically import the latest version of the module neurons.py.
