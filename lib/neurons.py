from typing import Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time

import synapses
import devices

Number = Union[float, int]
NumberN = Union[float, int, None]
NumberC = Union[float, int, complex]

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

class Neuron:
    """
    All neurons inherit from this class.

    Parameters
    ----------
    sim_time : Number, optional
        Time of simulation in [s]. The default is 0.
    dt : Number, optional
        Time resolution in []. The default is 0.

    Attributes
    ----------
    sim_time : Number
        where sim_time is stored
    dt : Number
        where dt is stored
    N : int
        total number of time steps in the simulation
    time : np.ndarray
        array of all timesteps of the simulation, shape (N)
    spikes : np.ndarray
        array with number of spikes at each timestep, initialized to zero, shape (N)
    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0):

        assert dt > 0, 'dt must be greater than zero'
        assert sim_time >= 0, 'sim_time must be greater than or equal to zero'

        self.sim_time = sim_time
        self.dt = dt

        if sim_time == 0:
            self.N = 0
            self.time = np.array([])
            self.spikes = np.array([])
        else:
            self.N = round(self.sim_time / self.dt)
            self.time = np.arange(0, self.sim_time, self.dt)
            self.spikes = np.zeros(self.N)

    def copy_spikes(self,
                    source,
                    p: Number = 0,
                    mode: str = 'inst',
                    tau_c: Number = 5e-3):
        """
        Copies spikes from a source neuron with probability p into the neuron.
        Can be used to generate correlated populations of neurons

        Parameters
        ----------
        source : Neuron
            Source neuron where the spikes are copied from.
        p : Number, optional
            Copy probability. The default is 0.
        mode : str, optional
            Copy mode. The default is 'inst'.
                'inst': Instantaneous copying to the same time step in the target neuron.
                'exp': Copying with exponential jitter
        tau_c : Number, optional
            Optional parameter for mode='exp' Determines the width of the exponential jitter. The default is 20e-3.

        """
        assert type(source).__base__ == Neuron, 'Source is not a neuron'
        assert mode in ['inst', 'exp']

        for spike_index in np.argwhere(source.spikes > 0):
            for spike in range(int(source.spikes[spike_index[0]])):
                if np.random.rand() < p:
                    if mode == 'inst':
                        # Copy spike to the same time
                        target_index = spike_index[0]
                    elif mode == 'exp':
                        # Draw a shift from an exponential distribution with parameter tau_c
                        shift = np.random.exponential(tau_c)
                        shift = shift * np.random.choice((-1,1))

                        target_index = spike_index[0] + int(round(shift / source.dt))
                        if target_index < 0:
                            target_index = 0
                        elif target_index >= source.N:
                            target_index = source.N-1

                    self.spikes[target_index] += 1



    def bin_spikes(self,
                   bin_width: Number = 5e-3) -> Iterable[np.ndarray]:
        """
        Bin the spikes of the neuron into bins of width bin_width.

        Parameters
        ----------
        bin_width : Number, optional
            Width of bins in [s]. The default is 5e-3.

        Returns
        -------
        binned_time : np.ndarray
            Array of binned times.
        binned_spikes : TYPE
            Array of spike number per time bin.
        """
        binned_time = np.arange(0, self.sim_time, bin_width)
        binned_spikes = np.array([])

        bin_width_ii = int(round(bin_width / self.dt))

        for ii in range(0, self.N, bin_width_ii):
            binned_spikes = np.append(binned_spikes, \
                                        np.sum(self.spikes[ii : ii+bin_width_ii]))

        return binned_time, binned_spikes


    def get_spike_times(self) -> np.ndarray:
        """
        Converts the spike array of shape (N) to an array of flexible length
        that stores the times of spikes

        Returns
        -------
        spike_times : np.ndarray
            Array with the spike times in [s].

        """
        spike_times = np.array([])
        for event_index in np.nonzero(self.spikes)[0]:
            for event in range(int(self.spikes[event_index])):
                spike_times = np.append(spike_times, event_index*self.dt)

        return spike_times

    def get_ISI_statistics(self) -> [np.ndarray, float, float, float]:
        """
        Compute ISIs, their mean, standard deviation and CV

        Returns
        -------
        ISI : np.ndarray
            Array of ISIs in ms
        mean : float
            mean ISI
        std : float
            standard deviation of ISIs
        CV : float
            Coefficient of variation of ISIs

        """
        spike_times = self.get_spike_times()
        ISI = 1000*np.diff(spike_times)

        mean = np.mean(ISI)
        std = np.std(ISI)
        cv = std / mean

        return ISI, mean, std, cv

    def delete_spikes(self):
        """
        Resets the spikes to a zero array of shape (N)
        """
        self.spikes = np.zeros(self.N)

    def plot_ISI(self,
                 binwidth: int = 10):
        """
        Plots a histogram of the ISIs together with their mean, std and CV

        Parameter
        ---------
        binwidth : int, optional
            Width of bins in [ms]. Default is 10.
        """
        fig, ax = plt.subplots(1,1, figsize=(14,7))

        ISI, mean, std, cv = self.get_ISI_statistics()

        ax.hist(ISI, align='mid', bins=range(0, int(max(ISI)) + binwidth, binwidth))

        ax.set_xlabel(r'Interspike Intervals [ms]')
        ax.set_ylabel('Counts')
        ax.set_title('Mean ISI: {:.3f} ms \nStandard deviation: {:.3f} ms \
                     \nCV: {:.3f}'.format(mean, std, cv))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])


class RegularNeuron(Neuron):
    """
    Creates a neuron that can spike in regular intervals.

    Parameters
    ----------
    sim_time : Number, optional
        Time of simulation in [s]. The default is 0.
    dt : Number, optional
        Time resolution in []. The default is 0.
    typ : str, optional
        Excitatory \'exc\' or inhibitory \'inh\' neuron. The default is 'exc'.
    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0):

        Neuron.__init__(self, sim_time, dt)

    def generate_spikes(self,
                        rate: Number = 0,
                        start: Number = 0,
                        stop: Number = None):
        """
        Generates spikes in regular intervals

        Parameters
        ----------
        rate : Number, optional
            Firing rate in [Hz]. The default is 0.
        start : Number, optional
            Start time of firing in [s]. The default is 0.
        stop : Number, optional
            Stop time of firing in [s]. The default is the end of the simulation.
        """

        if stop == None:
            stop = self.sim_time
        assert rate >= 0, 'rate must be greater than or equal to zero'
        assert rate <= 1/self.dt, 'rate is too large for resolution dt'
        assert start >= 0, 'start has to be greater than or equal to zero'
        assert stop <= self.sim_time, 'stop has to be smaller than or equal to sim_time'

        self.rate = rate
        if self.rate > 0:
            start_index = round(start / self.dt)
            stop_index = round(stop / self.dt)
            self.spikes[start_index:stop_index:round(1/(self.dt*self.rate))] = 1

class PoissonNeuron(Neuron):
    """
    Creates a neuron that can spike like a Poisson process with a given rate.

    Parameters
    ----------
    sim_time : Number, optional
        Time of simulation in [s]. The default is 0.
    dt : Number, optional
        Time resolution in []. The default is 0.
    typ : str, optional
        Excitatory \'exc\' or inhibitory \'inh\' neuron. The default is 'exc'
    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0):

        Neuron.__init__(self, sim_time, dt)

    def generate_spikes(self,
                        rate: Number = 0,
                        start: Number = 0,
                        stop: Number = None):
        """
        Generates spikes according to Poisson statistics

        Parameters
        ----------
        rate : Number, optional
            Firing rate in [Hz]. The respective parameter for the Poisson
            distribution is 1/rate. The default is 0.
        start : Number, optional
            Start time of firing in [s]. The default is 0.
        stop : Number, optional
            Stop time of firing in [s]. The default is the end of the simulation.

        Returns
        -------
        None.

        """

        if stop == None:
            stop = self.sim_time
        assert rate >= 0, 'rate must be greater than or equal to zero'
        assert start >= 0, 'start has to be greater than or equal to zero'
        assert stop <= self.sim_time, 'stop has to be smaller than or equal to sim_time'

        self.rate = rate
        if self.rate > 0:
            beta = 1 / self.rate

            spike_time = start + np.random.exponential(beta)
            while spike_time < stop:
                spike_index = round(spike_time / self.dt)
                if spike_index < self.N:
                    self.spikes[spike_index] += 1
                spike_time += np.random.exponential(beta)

class LIFNeuron(Neuron):
    """
    Creates a LIF neuron that can receive input and simulate the resulting
    membrane potential trace and spike behavior.

    Parameters
    ----------
    sim_time : Number, optional
        Time of simulation in [s]. The default is 0.
    dt : Number, optional
        Time resolution in []. The default is 0.

    V_init : Number, optional
        Initial membrane potential. The default is -60e-3.
    tau_m : Number, optional
        Membrane time constant. The default is 10e-3.
    E_l : Number, optional
        Resting potential. The default is -60e-3.
    V_thresh : Number, optional
        Threshold potential. The default is -50e-3.
    V_spike : Number, optional
        Potential at a spike. The default is 0.
    V_reset : Number, optional
        Reset potential. The default is -70e-3.

    R_m : Number, optional
        Membrane resistance. The default is 10e6.
    E_k: Number, optional
        Potassium reversal potential for spike rate adaptation (SRA). Default is -70e-3.
    tau_SRA: Number, optional
        Time constant for SRA. Default is 100e-3.
    w_SRA: Number, optional
        Weight difference for SRA. Default is 0 (no SRA).

    W_tot: Number, optional
        Total weight of incoming synapses for synaptic normalization. Default is 5.
    nu_SN: Number, optional
        Normalization rate for synaptic normalization. Default is 0.2
    step_SN: Number, optional
        When to normalize synaptic weights in [s]. Default is 1.

    E_e : Number, optional
        Reversal potential for exc input. The default is 0.
    tau_e : Number, optional
        EPSP time constant. The default is 3e-3.
    w_e : Number, optional
        Weight of exc synapse. The default is 0.5.

    E_i : Number, optional
        Reversal potential for inh input. The default is -80e-3.
    tau_i : Number, optional
        IPSP time constant. The default is 5e-3.
    w_i : Number, optional
        Weight of inh synapse. The default is 0.5.
    """

    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,

                 V_init: Number = -60e-3,
                 tau_m: Number = 10e-3,
                 E_l: Number = -60e-3,
                 V_thresh: Number = -50e-3,
                 V_spike: Number = 0,
                 V_reset: Number = -70e-3,
                 R_m: Number = 10e6,

                 E_k: Number = -70e-3,
                 tau_SRA: Number = 100e-3,
                 w_SRA: Number = 0,

                 W_tot: Number = 5,
                 nu_SN: Number = 0.2,
                 step_SN: Number = 1,

                 E_e: Number = 0,
                 tau_e: Number = 3e-3,

                 E_i: Number = -80e-3,
                 tau_i: Number = 5e-3):

        Neuron.__init__(self, sim_time, dt)

        self.V = V_init * np.ones(self.N)
        self.tau_m = tau_m
        self.E_l = E_l
        self.V_thresh = V_thresh
        self.V_spike = V_spike
        self.V_reset = V_reset
        self.R_m = R_m

        self.g_SRA = np.zeros(self.N)
        self.E_k = E_k
        self.tau_SRA = tau_SRA
        self.w_SRA = w_SRA

        self.neuron_input = np.array([])
        self.synapses = np.array([])

        self.W_tot = W_tot
        self.nu_SN = nu_SN
        self.step_SN = step_SN

        self.g_e = np.zeros(self.N)
        self.E_e = E_e
        self.tau_e = tau_e

        self.g_i = np.zeros(self.N)
        self.E_i = E_i
        self.tau_i = tau_i

        self.generator_input = np.array([])

    def clear_history(self):
        """
        Clears all values of V, g_e, g_i and all spike events
        """
        self.V = self.V[0] * np.ones(self.N)
        self.g_e = np.zeros(self.N)
        self.g_i = np.zeros(self.N)
        self.delete_spikes()

    def connect_neurons(self,
                neuron_list: Iterable = np.array([]),
                synapse_list: Iterable = np.array([])):
        """
        Connects one or multiple input neurons to the LIF neuron.

        Parameters
        ----------
        neuron_list : Iterable, optional
            Array of input neurons. The default is np.array([]).
        synapse_list : Iterable, optional
            Array of synapses. Must have the same shape as neuron_list
        """
        assert sum(1 for _ in neuron_list) == sum(1 for _ in synapse_list), 'neuron_list and synapse_list must have the same shape'

        for ii, neuron in enumerate(neuron_list):
            self.neuron_input = np.append(self.neuron_input, neuron)
            self.synapses = np.append(self.synapses, synapse_list[ii])

    def connect_generator(self, generator = None):
        """
        Connects a current generator to the LIF neuron.

        Parameter
        ---------
        current : CurrentGenerator, optional
            current generator to connect. Default is None.
        """
        assert type(generator) == devices.CurrentGenerator, 'Input must be a current generator'
        self.generator_input = np.append(self.generator_input, generator)

    def delete_connections(self):
        """
        Deletes all existing input neurons and generators
        """
        self.neuron_input = np.array([])
        self.synapses = np.array([])
        #self.exc_input = np.array([])
        #self.inh_input= np.array([])
        self.generator_input = np.array([])

    def simulate(self):
        """
        Runs the simulation and approximates the LIF equation with Euler\'s method.
        """
        for t in np.arange(self.N)[:-1]:
            # Check whether spike threshold was reached
            if self.V[t] >= self.V_thresh:
                self.V[t] = self.V_spike
                self.spikes[t] += 1
                self.V[t+1] = self.V_reset
            else:
                self.V[t+1] = self.V[t]
                self.V[t+1] += self.dt * 1/self.tau_m * (self.E_l-self.V[t])  # decay to resting potential
                self.V[t+1] += self.dt * 1/self.tau_m * self.g_e[t]*(self.E_e-self.V[t]) # exc synaptic input
                self.V[t+1] += self.dt * 1/self.tau_m * self.g_i[t]*(self.E_i-self.V[t]) # inh synaptic input
                self.V[t+1] += self.dt * 1/self.tau_m * self.R_m*sum([gen.current[t] for gen in self.generator_input]) # external current input
                self.V[t+1] += self.dt * 1/self.tau_m * self.g_SRA[t]*(self.E_k-self.V[t]) # SRA (spike rate adaptation)

            # Exponential decay of conductances
            self.g_e[t+1] = self.g_e[t] + self.dt * (-self.g_e[t]/self.tau_e)
            self.g_i[t+1] = self.g_i[t] + self.dt * ( -self.g_i[t]/self.tau_i)

            # Go through every input neuron and check whether there is a presynaptic spike
            for jj, neuron in enumerate(self.neuron_input):
                synapse = self.synapses[jj]
                typ = synapse.typ

                # Update conductances stepwise for each input spike
                if neuron.spikes[t] > 0:
                    if typ == 'exc':
                        self.g_e[t+1] += synapse.weight[t]*neuron.spikes[t]
                    if typ == 'inh':
                        self.g_i[t+1] += synapse.weight[t]*neuron.spikes[t]

                # Update synaptic weights
                synapse.update_weights(t, neuron, self, W_tot=self.W_tot, nu_SN=self.nu_SN, step_SN=self.step_SN)

            # Compute conductance of SRA (spike rate adaptation)
            if self.w_SRA > 0:
                self.g_SRA[t+1] = self.g_SRA[t] + \
                    self.dt * (-self.g_SRA[t]/self.tau_SRA) + \
                    self.w_SRA*self.spikes[t]

    def plot_V(self,
               start: Number = 0,
               stop: Number = None,
               title: str = None):
        """
        Plots the membrane potential trace.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        """
        if stop == None:
            stop = self.sim_time

        assert start >= 0, 'start must be >= 0'
        assert start < self.sim_time, 'start must be < sim_time'
        assert stop > 0, 'stop must be > 0'
        assert stop <= self.sim_time, 'stop must be <= sim_time'
        assert start < stop, 'start must be < stop'

        start_idx = int(round(start / self.dt))
        stop_idx = int(round(stop / self.dt))-1

        fig, ax = plt.subplots(1, 1, figsize=(14,7))

        # plot everything in mV
        ax.plot(self.time[start_idx:stop_idx], 1000*self.V[start_idx:stop_idx], color='blue', label=r'Membrane potential $V_m$')
        ax.hlines(1000*self.V_thresh, self.time[start_idx], self.time[stop_idx], linestyle='dashed', color='black', label=r'Threshold $V_\theta$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Membrane potential $V_m$ [mV]')
        ax.legend(loc='upper right')
        if title == None:
            ax.set_title('Membrane trace of LIF neuron')
        else:
            ax.set_title(title)

        plt.show()

    def plot_g(self, title: str = None):
        """
        Plot the conductance trace.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14,7))

        axes[0].plot(self.time, self.g_e, color='blue', label=r'Excitatory conductance $g_e$')
        axes[0].plot(self.time, -self.g_i, color='red', label=r'Inhibitory conductance $g_i$')

        axes[0].set_xlabel('Time [s]')
        axes[0].set_ylabel('Conductance []')
        axes[0].legend(loc='best')
        axes[0].set_title('Synaptic conductances')

        axes[1].plot(self.time, self.g_SRA, color='black', label=r'SRA conductance $g_{SRA}$')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Conductance []')
        axes[1].legend(loc='best')
        axes[1].set_title('SRA conductance')

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


def plot_spikes(neuron_list: Iterable,
                title: str = None):
    """
    Plot spike trains for a given array of neurons

    Parameter
    ----------
    neuron_list : Iterable
        list or array of neurons to plot the spike trains from
    title : str, optional
        Title for the plot. Default is None.
    """

    assert iter(neuron_list), 'neuron_list must be of type Iterable'

    fig, ax = plt.subplots(1,1, figsize=(14,7))

    for ii, neuron in enumerate(neuron_list):
        spike_times = neuron.get_spike_times()
        ax.scatter(spike_times, (ii+1)*np.ones_like(spike_times), \
                   color='black', marker='.')

    sim_time = max([neuron.sim_time for neuron in neuron_list])
    ax.set_xlim([0, sim_time])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Neuron')

    ax.set_yticks(np.arange(0, sum(1 for _ in neuron_list)+1, 5))
    if title == None:
        ax.set_title('Spike trains')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

def plot_firing_rates(neuron_list: Iterable,
                      bin_width: Number = 1,
                      title: str = None):
    """
    Plot firing rates for a given array of neurons

    Parameter
    ----------
    neuron_list : Iterable
        list or array of neurons to plot the firing rates from
    bin_width : Number, optional
        Bin-width of where to evaluate firing rates in [s]. Default is 1.
    title : str, optional
        Title for the plot. Default is None.
    """
    assert iter(neuron_list), 'neuron_list must be of type Iterable'

    fig, ax = plt.subplots(1,1, figsize=(14,7))

    for ii, neuron in enumerate(neuron_list):
        spike_times = neuron.get_spike_times()

        x = np.arange(0, neuron.sim_time, bin_width)
        y = np.histogram(spike_times, bins=np.append(x, neuron.sim_time))[0] / bin_width

        ax.plot(x, y, label='Neuron ' + str(ii+1))

    sim_time = max([neuron.sim_time for neuron in neuron_list])
    ax.set_xlim([0, sim_time])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Firing rate [Hz]')
    plt.legend(loc='upper left')
    if title == None:
        ax.set_title('Firing rates')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])


def cross_correlogram(neuron_1,
                      neuron_2,
                      max_lag: Number = 100e-3,
                      bin_width: Number = 5e-3,
                      title: str = None) -> Iterable[np.ndarray]:
    """
    Compute (and plot) a cross-correlogram of two given neurons

    Parameters
    ----------
    neuron_1 : Neuron
        First neuron.
    neuron_2 : Neuron
        Second neuron.
    max_lag : Number, optional
        The plot will go from -max_lag to +max_lag in [s]. The default is 100e-3.
    bin_width : Number, optional
        Width of bins in [s]. The default is 5e-3.
    title : str, optional
        Title of figure. The default is None.

    Return
    -------
    lags : np.ndarray
        Array with the lag times from -max_lag to +mag_lax in [s]
    cor : np.ndarray
        Array with cross-correlations at lags in lags.
        To plot the result, simply call plt.plot(lags, cor)
    """
    assert type(neuron_1).__base__ == Neuron, 'neuron_1 is not a neuron'
    assert type(neuron_2).__base__ == Neuron, 'neuron_2 is not a neuron'

    _, binned_spikes_1 = neuron_1.bin_spikes(bin_width)
    _, binned_spikes_2 = neuron_2.bin_spikes(bin_width)

    fig, ax = plt.subplots(1,1, figsize=(14,7))

    lags, cor, _, _ = ax.xcorr(binned_spikes_1, binned_spikes_2, \
             usevlines=False, maxlags=int(round(max_lag / bin_width)), normed=True, lw=2, linestyle = '-', marker = None)

    xtickpos = np.linspace(-max_lag, max_lag, 11) / bin_width # x-tick-positions
    xticklabels = np.linspace(-max_lag, max_lag, 11) * 1000 # x-tick-labels in [ms]

    ax.set_xlabel('Cross-correlation lag [ms]')
    ax.set_xticks(xtickpos.astype(int))
    ax.set_xticklabels(xticklabels.astype(int))
    ax.set_ylabel('Correlation []')
    if title == None:
        ax.set_title('Cross-correlogram')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return lags, cor
