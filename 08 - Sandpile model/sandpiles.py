from typing import Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

class Sandpile():
    def __init__(self,
                 Nx: int,
                 Ny: int,
                 topple_height: int = 4,

                 init_mode: str = 'random'):

        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx * self.Ny

        self.topple_height = topple_height

        if init_mode == 'random':
            self.grid = np.random.randint(low=0, high=2*self.topple_height, size=(self.Nx, self.Ny))
            while self.get_unstable_idx().shape[0] > 0:
                self.topple()

        self.avalanches = np.array([])

        # Define colors for plotting
        colors = np.zeros((self.topple_height+1, 3))

        for ii in range(colors.shape[0]):
            if ii < colors.shape[0]-1:
                colors[ii, :] = 1 - ii/self.topple_height
            else:
                colors[ii, :] = np.array([1, 0, 0])
        self.cmap = matplotlib.colors.ListedColormap(colors, name='sandpile')

    def get_unstable_idx(self) -> np.ndarray:
        return np.argwhere(self.grid >= self.topple_height)

    def get_mean_height(self) -> float:
        return np.mean(self.grid)

    def add_grain(self,
                  x=np.nan,
                  y=np.nan):

        assert (np.isnan(x) and np.isnan(y)) or (type(x)==int and type(y)==int), \
            'x and y must be integers or left unspecified'

        if np.isnan(x) and np.isnan(y):
            x = np.random.randint(0, self.Nx)
            y = np.random.randint(0, self.Ny)
        else:
            assert x>=0 and x<self.Nx, 'x must be between 0 and Nx'
            assert y>=0 and y<self.Ny, 'y must be between 0 and Ny'

        self.grid[y, x] += 1

    def topple(self):
        unstable_idx = self.get_unstable_idx()
        for topple_idx in unstable_idx:
            self.grid[topple_idx[0], topple_idx[1]] -= 4

            if topple_idx[0]-1 >= 0:
                self.grid[topple_idx[0]-1, topple_idx[1]] += 1
            if topple_idx[0]+1 < self.Ny:
                self.grid[topple_idx[0]+1, topple_idx[1]] += 1
            if topple_idx[1]-1 >= 0:
                self.grid[topple_idx[0], topple_idx[1]-1] += 1
            if topple_idx[1]+1 < self.Nx:
                self.grid[topple_idx[0], topple_idx[1]+1] += 1

    def init_plot(self) -> Iterable:
        plt.ioff()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        image = ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=self.topple_height+1, animated=True)

        xtick_pos = np.arange(4, self.Nx, 5)
        xtick_labels = np.arange(5, self.Nx+1, 5)
        ytick_pos = np.arange(4, self.Ny, 5)
        ytick_labels = np.arange(5, self.Ny+1, 5)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels)
        ax.xaxis.tick_top()
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels)

        cbar = plt.colorbar(image, ax=ax, ticks=np.arange(0.5, self.topple_height+1.5, 1))
        labels = []
        for ii in range(self.topple_height+1):
            if ii == 0:
                labels.append('0 grains')
            elif ii == 1:
                labels.append('1 grain')
            elif ii > 1 and ii < self.topple_height:
                labels.append(str(ii) + ' grains')
            else:
                labels.append('Toppling')
        cbar.ax.set_yticklabels(labels)

        return fig, ax, image

    def simulate(self,
                 Niter: int = 100,
                 burn_in: int = 1000,
                 plot: bool = True,
                 interval: int = 200,
                 save: bool = False,
                 filename: str = 'animation.mp4') -> Iterable[list]:

        mean_heights = np.zeros(int(Niter))

        avalanche_on = False


        if plot or save:
            fig, ax, image = self.init_plot()
            ims = [[image]]

        for ii in range(int(Niter)):
            if ii % 1000 == 0:
                print(ii)
            #mean_heights[ii] = self.get_mean_height()

            if not avalanche_on:
                self.add_grain()
                if plot or save:
                    ims.append([ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=self.topple_height+1, animated=True)])

                unstable_idxs = self.get_unstable_idx()

                if unstable_idxs.shape[0] > 0:
                    avalanche_on = True

                    avalanche = Avalanche(self, ii, unstable_idxs)

                    self.topple()
                    if plot or save:
                        ims.append([ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=self.topple_height+1, animated=True)])
            else:
                unstable_idxs = self.get_unstable_idx()
                if unstable_idxs.shape[0] == 0:
                    avalanche_on = False

                    avalanche.compute_all_parameters()
                    self.avalanches = np.append(self.avalanches, avalanche)
                else:
                    avalanche.add_time(ii)
                    avalanche.add_idxs(unstable_idxs)

                    self.topple()
                    if plot:
                        ims.append([ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=self.topple_height+1, animated=True)])

        if plot:
            anim = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat=False)
            plt.show()
        if save:
            anim = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat=False)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15)
            anim.save(filename, writer=writer)

        return mean_heights

    def get_avalanche_times(self):
        return np.array([av.total_time for av in self.avalanches])

    def get_avalanche_energies(self):
        return np.array([av.energy for av in self.avalanches])

    def get_avalanche_linear_sizes(self):
        return np.array([av.linear_size for av in self.avalanches])


    def one_avalanche(self):
        pass

    def plot_cluster(self):
        pass

class Avalanche():
    def __init__(self,
                 parent_sandpile: Sandpile,
                 timestep: int,
                 unstable_idxs: np.ndarray):

        self.parent_sandpile = parent_sandpile

        self.times = np.array([timestep])
        self.toppled_cells = unstable_idxs

    def add_time(self,
                 timestep: int):
        self.times = np.append(self.times, timestep)

    def add_idxs(self,
                 unstable_idxs: np.ndarray):
        self.toppled_cells = np.append(self.toppled_cells, unstable_idxs, axis=0)

    def compute_all_parameters(self):
        self._get_total_time()
        self._get_energy()
        self._get_region()
        self._get_linear_size()

    def _get_total_time(self):
        self.total_time = self.times.shape[0]

    def _get_energy(self):
        self.energy = self.toppled_cells.shape[0]

    def _get_region(self):
        self.region = np.unique(self.toppled_cells, axis=0)
        for idx in self.region:
            if idx[0]-1 >= 0:
                self.region = np.append(self.region, [[idx[0]-1, idx[1]]], axis=0)
            if idx[0]+1 < self.parent_sandpile.Ny:
                self.region = np.append(self.region, [[idx[0]+1, idx[1]]], axis=0)
            if idx[1]-1 >= 0:
                self.region = np.append(self.region, [[idx[0], idx[1]-1]], axis=0)
            if idx[1]+1 < self.parent_sandpile.Nx:
                self.region = np.append(self.region, [[idx[0], idx[1]+1]], axis=0)

        self.region = np.unique(self.region, axis=0)

        self.region_size = self.region.shape[0]
        self.center_of_mass = 1/self.region_size * np.sum(self.region, axis=0)

    def _get_linear_size(self):
        self.linear_size = 1/self.region_size * \
            np.sum(np.linalg.norm(self.region - self.center_of_mass, axis=1))



