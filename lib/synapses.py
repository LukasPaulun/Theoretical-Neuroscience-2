from typing import Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import neurons

Number = Union[float, int]
NumberN = Union[float, int, None]
NumberC = Union[float, int, complex]

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

class Synapse:
    """
    All synapses inherit from this class.

    Parameters
    ----------
    sim_time : Number, optional
        Time of simulation in [s]. The default is 0.
    dt : Number, optional
        Time resolution in []. The default is 0.
    typ : str, optional
        Excitatory \'exc\' or inhibitory \'inh\' neuron. The default is 'exc'.
    init_weight : Number, optional
        Initial weights of the synapse. The default is 1.

    Attributes
    ----------
    sim_time : Number
        where sim_time is stored
    dt : Number
        where dt is stored
    typ : str
        where typ is stored
    N : int
        total number of time steps in the simulation
    time : np.ndarray
        array of all timesteps of the simulation, shape (N)
    weight : np.ndarray
        array with synaptic weights at each timestep, initialized to init_weight, shape (N)
    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,

                 typ: str = 'exc',
                 init_weight: Number = 1):

        assert dt > 0, 'dt must be greater than zero'
        assert sim_time >= 0, 'sim_time must be greater than or equal to zero'
        assert typ=='exc' or typ=='inh', 'Connection-type must be \'exc\' or \'inh\''

        self.sim_time = sim_time
        self.dt = dt

        if sim_time == 0:
            self.N = 0
            self.time = np.array([])
            self.weight = np.array([])
        else:
            self.N = round(self.sim_time / self.dt)
            self.time = np.arange(0, self.sim_time, self.dt)
            self.weight = init_weight * np.ones(self.N)

        self.typ = typ

    def update_weights(self, *args, **kwargs):
        pass


class STDPSynapse(Synapse):
    """
    STDP Synapse
    ----------
    max_weight : Number, optional
        Maximum allowed weight of the synapse. The default is 6.
    A_P : Number, optional
        Amplitude for LTP. The default is 0.05.
    tau_P : Number, optional
        Time constant for LTP. The default is 17e-3.
    A_D : Number, optional
        Amplitude for LTD. The default is -0.025.
    tau_D : Number, optional
        Time constant for LTD. The default is 34e-3.
    mode : str, optional
        Pairing mode for STDP. The default is 'narrow_nearest_neighbor'.
        Explanation of modes:
            'narrow_nearest_neighbor': See Morrison, Diesmann, Gerstner (2008), figure 7c
    """

    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,
                 typ: str = 'exc',
                 init_weight: Number = 0.5,

                 max_weight: Number = 6,

                 A_P: Number = 0.05,
                 tau_P: Number = 17e-3,

                 A_D: Number = -0.025,
                 tau_D: Number = 34e-3,

                 mode: str = 'narrow_nearest_neighbor'):
        Synapse.__init__(self, sim_time, dt, typ, init_weight)

        self.max_weight = max_weight

        self.A_P = A_P
        self.tau_P = tau_P

        self.A_D = A_D
        self.tau_D = tau_D

        self.mode = mode

    def update_weights(self,
                       t : Number,
                       pre_neuron: neurons.Neuron,
                       post_neuron: neurons.Neuron
                       ):
        """
        Update the weights according to the STDP rule.

        Parameters
        ----------
        t : Number
            Current time step.
        pre_neuron : TYPE
            Presynaptic neuron.
        post_neuron : TYPE
            Postsynaptic neuron.
        """
        assert type(pre_neuron).__bases__[0] == neurons.Neuron, 'pre_neuron is not from parent neurons.Neuron'
        assert type(post_neuron).__bases__[0] == neurons.Neuron, 'postert a string or n_neuron is not from parent neurons.Neuron'


        if self.mode == 'narrow_nearest_neighbor':
            # LTP: Postsynaptic neuron spikes, presynaptic neuron does not spike
            # but there were presynaptic spikes in the past
            if post_neuron.spikes[t] > 0 and pre_neuron.spikes[t] == 0 and np.any(pre_neuron.spikes[:t]):       # perform LTP
                last_pre_spike_index = np.where(pre_neuron.spikes[:t] > 0)[0][-1]
                if np.any(post_neuron.spikes[:t]):
                    last_post_spike_index = np.where(post_neuron.spikes[:t] > 0)[0][-1]
                else:
                    last_post_spike_index = np.nan

                # Narrow nearest neighbour implementation of STDP
                if np.isnan(last_post_spike_index) or last_pre_spike_index > last_post_spike_index:
                    delta_t = (last_pre_spike_index - t) * self.dt

                    new_weight = self.weight[t] + self.A_P * np.exp(delta_t / self.tau_P)
                    if new_weight <= self.max_weight:
                        self.weight[t+1:] = new_weight

            # LTD: Presynaptic neuron spikes, postsynaptic neuron does not spike
            # but there were postsynaptic spikes in the past
            elif post_neuron.spikes[t] == 0 and pre_neuron.spikes[t] > 0 and np.any(post_neuron.spikes[:t]):     # perform LTD
                if np.any(pre_neuron.spikes[:t]):
                    last_pre_spike_index = np.where(pre_neuron.spikes[:t] > 0)[0][-1]
                else:
                    last_pre_spike_index = np.nan
                last_post_spike_index = np.where(post_neuron.spikes[:t] > 0)[0][-1]

                # Narrow nearest neighbour implementation of STDP as above
                if np.isnan(last_pre_spike_index) or last_post_spike_index > last_pre_spike_index:
                    delta_t = (t - last_post_spike_index) * self.dt

                    new_weight = self.weight[t] + self.A_D * np.exp(-delta_t / self.tau_D)
                    if new_weight <= self.max_weight:
                        self.weight[t+1:] = new_weight


def plot_synaptic_weights(synapse_list: Iterable,
                 title: str = None):
    """
    Plot weights of synapses for a given array of synapses

    Parameter
    ----------
    synapse_list : Iterable
        list or array of synapses to plot the weights from
    title : str, optional
        Title for the plot. Default is None.
    """
    assert iter(synapse_list), 'neuron_list must be of type Iterable'

    fig, ax = plt.subplots(1,1, figsize=(14,7))

    for ii, synapse in enumerate(synapse_list):
        ax.plot(synapse.time, synapse.weight, label = 'Synapse ' + str(ii+1))

    sim_time = max([synapse.sim_time for synapse in synapse_list])
    ax.set_xlim([0, sim_time])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Synaptic weights')
    plt.legend(loc='upper left')
    if title == None:
        ax.set_title('Evolution of synaptic weights')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])


