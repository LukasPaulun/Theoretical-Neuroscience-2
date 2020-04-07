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
    normalize : bool, optional
        Whether the synapse should be subject to synaptic normalization. Default is False.

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
    normalize : bool
        Where normalize is stored
    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,

                 typ: str = 'exc',
                 init_weight: Number = 1,

                 normalize: bool = False
                 ):

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

        self.normalize = normalize

    def update_weights(self,
                       t: Number,
                       pre_neuron,
                       post_neuron,
                       *args, **kwargs):
        """
        Generic function to update the weights of the synapse. Can perform
        synaptic normalization if self.normalize is True.

        Parameters
        ----------
        t : Number
            Time index where to perform the weight update.
        pre_neuron : Neuron
            Presynaptic neuron.
        post_neuron : Neuron
            Postsynaptic neuron.
        *args :
        **kwargs :
            If self.normalize is True this should contain:
                W_tot: Total weight for synaptic normalization.
                nu_SN: Normalization rate.
                step_SN: Steps when to perform synaptic normalization
        """

        assert type(pre_neuron).__bases__[0] == neurons.Neuron, 'pre_neuron is not from parent neurons.Neuron'
        assert type(post_neuron).__bases__[0] == neurons.Neuron, 'post_neuron is not from parent neurons.Neuron'

        if self.normalize:
            assert 'W_tot' in kwargs.keys(), 'Synaptic normalization requires the parameter \'W_tot\''
            assert 'nu_SN' in kwargs.keys(), 'Synaptic normalization requires the parameter \'nu_SN\''
            assert 'step_SN' in kwargs.keys(), 'Synaptic normalization requiers the parameter \'step_SN\''

            if t % (kwargs['step_SN'] / pre_neuron.dt) == 0:
                cur_sum = 0
                for synapse in post_neuron.synapses:
                    if synapse.normalize:
                        cur_sum += synapse.weight[t]
                norm_factor = 1 + kwargs['nu_SN'] * (kwargs['W_tot']/cur_sum - 1)
                self.weight[t+1:] = self.weight[t] * norm_factor

        else:
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
                 normalize: bool = False,

                 max_weight: Number = 6,

                 A_P: Number = 0.05,
                 tau_P: Number = 17e-3,

                 A_D: Number = -0.025,
                 tau_D: Number = 34e-3,

                 mode: str = 'narrow_nearest_neighbor'):
        Synapse.__init__(self, sim_time, dt, typ, init_weight, normalize)

        self.max_weight = max_weight

        self.A_P = A_P
        self.tau_P = tau_P

        self.A_D = A_D
        self.tau_D = tau_D

        self.mode = mode

    def update_weights(self,
                       t : Number,
                       pre_neuron,
                       post_neuron,
                       *args, **kwargs
                       ):
        """
        Update the weights according to the STDP rule.

        Parameters
        ----------
        t : Number
            Current time step.
        pre_neuron : Neuron
            Presynaptic neuron.
        post_neuron : Neuron
            Postsynaptic neuron.
        *args :
        **kwargs :
            If self.normalize is True this should contain:
                W_tot: Total weight for synaptic normalization.
                nu_SN: Normalization rate.
                step_SN: Steps when to perform synaptic normalization
        """
        Synapse.update_weights(self, t, pre_neuron, post_neuron, *args, **kwargs)

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

class STPSynapse(Synapse):
    """
    STP Synapse
    ----------

    """

    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,
                 typ: str = 'exc',
                 init_weight: Number = 0.5,
                 normalize: bool = False,

                 max_weight: Number = 1,
                 U: Number = 0.2,

                 STF: bool = True,
                 tau_f = 50e-3,

                 STD: bool = True,
                 tau_d = 750e-3):

        Synapse.__init__(self, sim_time, dt, typ, init_weight, normalize)

        self.max_weight = max_weight
        self.U = U
        self.STF = STF
        self.tau_f = tau_f
        self.STD = STD
        self.tau_d = tau_d

        if self.N == 0:
            self.u = np.array([])
            self.x = np.array([])
        else:
            self.u = np.zeros(self.N)
            self.x = np.ones(self.N)

    def update_weights(self,
                       t : Number,
                       pre_neuron,
                       post_neuron,
                       *args, **kwargs
                       ):
        Synapse.update_weights(self, t, pre_neuron, post_neuron, *args, **kwargs)

        if self.STF:
            self.u[t+1:] = self.u[t] + self.dt * (-self.u[t] / self.tau_f)
        if self.STD:
            self.x[t+1:] = self.x[t] + self.dt * ((1-self.x[t]) / self.tau_d)

        if pre_neuron.spikes[t] > 0:
            #self.weight[t] = self.max_weight * self.u[t+1] * self.x[t]

            if self.STF:
                self.u[t+1:] += self.U * (1-self.u[t])
            if self.STD:
                self.x[t+1:] -= self.u[t+1]*self.x[t]

            self.weight[t+1:] = self.max_weight * self.u[t+1] * self.x[t-1]

    def plot_STP_dynamics(self, title: str = None):
        """
        Plot evolution of short-term plasticity variables u and x

        Parameter
        ----------
        title : str, optional
            Title for the plot. Default is None.
        """

        fig, ax = plt.subplots(1,1, figsize=(14,7))

        ax.plot(self.time, self.u, label=r'Neurotransmitter release probability $u$')
        ax.plot(self.time, self.x, label=r'Fraction of available neurotransmitter $x$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[]')
        ax.legend(loc='upper right')
        if title == None:
            ax.set_title(r'Short-term plasticity dynamics')
        else:
            ax.set_title(str(title))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])


def plot_synaptic_weights(synapse_list: Iterable,
                 title: str = None,
                 color_list = [],
                 labels = []):
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
        if len(color_list) > 0:
            ax.plot(synapse.time, synapse.weight, color=color_list[ii])
        else:
            ax.plot(synapse.time, synapse.weight)

    sim_time = max([synapse.sim_time for synapse in synapse_list])
    ax.set_xlim([0, sim_time])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Synaptic weights')
    if len(labels) > 0:
        plt.legend(loc='upper left')
    if title == None:
        ax.set_title('Evolution of synaptic weights')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])



