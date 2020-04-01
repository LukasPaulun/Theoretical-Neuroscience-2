# Add a line locally
# Add a second line in spyder

from typing import Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets

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
    typ : str, optional
        Excitatory \'exc\' or inhibitory \'inh\' neuron. The default is 'exc'.

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
    spikes : np.ndarray
        array with number of spikes at each timestep, initialized to zero, shape (N)

    """
    def __init__(self,
                 sim_time: Number = 0,
                 dt: Number = 0,
                 typ: str = 'exc'):

        assert dt > 0, 'dt must be greater than zero'
        assert sim_time >= 0, 'sim_time must be greater than or equal to zero'
        assert typ=='exc' or typ=='inh', 'Connection-type must be \'exc\' or \'inh\''

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

        self.typ = typ

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
                 dt: Number = 0,
                 typ: str = 'exc'):

        Neuron.__init__(self, sim_time, dt, typ)

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
                 dt: Number = 0,
                 typ: str = 'exc'):

        Neuron.__init__(self, sim_time, dt, typ)

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
    typ : str, optional
        Excitatory \'exc\' or inhibitory \'inh\' neuron. The default is 'exc'
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
                 typ: str = 'exc',

                 V_init: Number = -60e-3,
                 tau_m: Number = 10e-3,
                 E_l: Number = -60e-3,
                 V_thresh: Number = -50e-3,
                 V_spike: Number = 0,
                 V_reset: Number = -70e-3,
                 R_m: Number = 10e6,

                 E_k: Number = -70e-3,
                 tau_SRA: Number = 100e-3,
                 w_SRA: Number = 0.06,

                 E_e: Number = 0,
                 tau_e: Number = 3e-3,
                 w_e: Number = 0.5,

                 E_i: Number = -80e-3,
                 tau_i: Number = 5e-3,
                 w_i: Number = 0.5):

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

        self.exc_input = np.array([])
        self.g_e = np.zeros(self.N)
        self.E_e = E_e
        self.tau_e = tau_e
        self.w_e = w_e

        self.inh_input = np.array([])
        self.g_i = np.zeros(self.N)
        self.E_i = E_i
        self.tau_i = tau_i
        self.w_i = w_i

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
                neuron_list: np.ndarray = np.array([])):
        """
        Connects one or multiple input neurons to the LIF neuron.

        Parameters
        ----------
        neuron_list : np.ndarray, optional
            Array of input neurons. The default is np.array([]).
        """

        for neuron in np.array(neuron_list):
            if neuron.typ == 'exc':
                self.exc_input = np.append(self.exc_input, neuron)
            elif neuron.typ == 'inh':
                    self.inh_input = np.append(self.inh_input, neuron)

    def connect_generator(self, generator = None):
        """
        Connects a current generator to the LIF neuron.

        Parameter
        ---------
        current : CurrentGenerator, optional
            current generator to connect. Default is None.
        """
        assert type(generator) == CurrentGenerator, 'Input must be a current generator'
        self.generator_input = np.append(self.generator_input, generator)

    def delete_connections(self):
        """
        Deletes all existing input neurons and generators
        """
        self.exc_input = np.array([])
        self.inh_input= np.array([])
        self.generator_input = np.array([])

    def simulate(self):
        """
        Runs the simulation and approximates the LIF equation with Euler\'s method.
        """
        for ii in np.arange(self.N)[:-1]:
            # Check whether spike threshold was reached
            if self.V[ii] >= self.V_thresh:
                self.V[ii] = self.V_spike
                self.spikes[ii] += 1
                self.V[ii+1] = self.V_reset
            else:
                self.V[ii+1] = self.V[ii]
                self.V[ii+1] += self.dt * 1/self.tau_m * (self.E_l-self.V[ii])  # decay to resting potential
                self.V[ii+1] += self.dt * 1/self.tau_m * self.g_e[ii]*(self.E_e-self.V[ii]) # exc synaptic input
                self.V[ii+1] += self.dt * 1/self.tau_m * self.g_i[ii]*(self.E_i-self.V[ii]) # inh synaptic input
                self.V[ii+1] += self.dt * 1/self.tau_m * self.R_m*sum([gen.current[ii] for gen in self.generator_input]) # external current input
                self.V[ii+1] += self.dt * 1/self.tau_m * self.g_SRA[ii]*(self.E_k-self.V[ii]) # SRA (spike rate adaptation)
                    # self.dt * 1/self.tau_m * \
                    # (self.E_l-self.V[ii] + \
                    #   self.g_e[ii]*(self.E_e-self.V[ii]) + \
                    #   self.g_i[ii]*(self.E_i-self.V[ii]) + \
                    #   self.R_m*sum([gen.current[ii] for gen in self.generator_input]) + \
                    #   self.g_SRA*(self.E_k-self.V[ii])
                    #   )

            # Compute synaptic conductance only if input is present
            if self.exc_input.size > 0:
                exc_input_spikes = sum([exc_input_neuron.spikes[ii] for exc_input_neuron in self.exc_input])
                self.g_e[ii+1] = self.g_e[ii] + \
                    self.dt * (-self.g_e[ii]/self.tau_e) + \
                    self.w_e*exc_input_spikes
            if self.inh_input.size > 0:
                inh_input_spikes = sum([inh_input_neuron.spikes[ii] for inh_input_neuron in self.inh_input])
                self.g_i[ii+1] = self.g_i[ii] + \
                    self.dt * ( -self.g_i[ii]/self.tau_i) + \
                    self.w_i*inh_input_spikes

            # Compute conductance of SRA (spike rate adaptation)
            if self.w_SRA > 0:
                self.g_SRA[ii+1] = self.g_SRA[ii] + \
                    self.dt * (-self.g_SRA[ii]/self.tau_SRA) + \
                    self.w_SRA*self.spikes[ii]

    def plot_V(self, title: str = None):
        """
        Plots the membrane potential trace.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14,7))

        # plot everything in mV
        ax.plot(self.time, 1000*self.V, color='blue', label=r'Membrane potential $V_m$')
        ax.hlines(1000*self.V_thresh, 0, self.time[-1], linestyle='dashed', color='black', label=r'Threshold $V_\theta$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Membrane potential $V_m$ [mV]')
        ax.legend(loc='upper right')
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

class CurrentGenerator:
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
            self.current = np.array([])
        else:
            self.N = round(self.sim_time / self.dt)
            self.time = np.arange(0, self.sim_time, self.dt)
            self.current = np.zeros(self.N)


    def generate_current(self,
                        current: Number = 0,
                        start: Number = 0,
                        stop: Number = None):
        """
        Generate a current

        Parameters
        ----------
        current : Number, optional
            Current in [A]. The default is 0.
        start : Number, optional
            Start time of firing in [s]. The default is 0.
        stop : Number, optional
            Stop time of firing in [s]. The default is the end of the simulation.
        """

        if stop == None:
            stop = self.sim_time
        assert start >= 0, 'start has to be greater than or equal to zero'
        assert stop <= self.sim_time, 'stop has to be smaller than or equal to sim_time'

        start_index = round(start / self.dt)
        stop_index = round(stop / self.dt)
        self.current[start_index:stop_index] = current


def plot_spikes(neuron_list, title=None):
    """
    """

    assert type(neuron_list) == np.ndarray, 'neuron_list must be of type np.ndarray'

    fig, ax = plt.subplots(1,1, figsize=(14,7))

    for ii, neuron in enumerate(neuron_list):
        for spike_time in neuron.time[neuron.spikes > 0]:
            ax.plot([spike_time, spike_time], [ii+0.95, ii+1.05], color='black')

    sim_time = max([neuron.sim_time for neuron in neuron_list])
    ax.set_xlim([0, sim_time])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Neuron')
    if title == None:
        ax.set_title('Spike train')
    else:
        ax.set_title(str(title))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])








