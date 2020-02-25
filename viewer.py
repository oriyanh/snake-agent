# Author: David Neuhof

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import namedtuple

class GameViewer:
    def __init__(self, board, colorbar=True):
        CellType = namedtuple('CellType', ['val', 'label', 'color'])
        self.cell_types = [CellType(-1, 'empty', 'white'),
                           CellType(0, 'player 1', 'blue'),
                           CellType(1, 'player 2', 'magenta'),
                           CellType(2, 'player 3', 'green'),
                           CellType(3, 'player 4', 'cyan'),
                           CellType(4, 'player 5', 'purple'),
                           CellType(5, 'obstacle', 'black'),
                           CellType(6, 'fruit 2', 'orange'),
                           CellType(7, 'fruit 5', 'red'),
                           CellType(8, 'fruit -1', 'gray')
                           ]
        self.cmap = mpl.colors.ListedColormap([t.color for t in self.cell_types])
        bounds = list([t.val for t in self.cell_types]) + [9]
        norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)
        self.h, self.w = board.shape
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.w / 5, 1+self.h / 5), tight_layout=True)

        self.img = self.ax.imshow(board, interpolation='none',
                                  extent=[0, self.w, 0, self.h],
                                  cmap=self.cmap, norm=norm)
        self.ax.set_axis_off()

        for x in range(self.w + 1):
            self.ax.axvline(x, lw=0.4, color='gray')
        for y in range(self.h + 1):
            self.ax.axhline(y, lw=0.4, color='gray')

        if colorbar:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            cb = plt.colorbar(self.img, cax, cmap=self.cmap, norm=norm, boundaries=bounds)
            cb.set_ticks(np.asarray(bounds[:-1]) + 0.5)
            cb.set_ticklabels([f"{t.val}: {t.label}" for t in self.cell_types])

        self.fig.canvas.set_window_title('Snake')
        self.fig.show()


    def update(self, board, title=""):
        self.img.set_data(board)
        self.ax.set_title(title)
        self.fig.canvas.draw()
        plt.pause(0.1)
