from typing import Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Number = Union[float, int]
NumberN = Union[float, int, None]
NumberC = Union[float, int, complex]

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

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

    def plot_current(self,
                     title: str = None):
        """
        Plots the generated current

        Parameter
        ----------
        title : str, optional
            Title for the plot
        """
        fig, ax = plt.subplots(1,1, figsize=(14,7))

        ax.plot(self.time, 1e9*self.current)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Current [nA]')
        if title == None:
            ax.set_title('Generated current')
        else:
            ax.set_title(title)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])



