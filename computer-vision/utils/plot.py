import collections
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline # type: ignore

from torch import nn as nn

from utils.hparams import HyperParameters

class ProgressBoard(HyperParameters):
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        backend_inline.set_matplotlib_formats('svg')
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        axes = self.axes if self.axes else plt.gca()
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(axes.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class PlotModule(nn.Module):
    def __init__(self, plot_train_per_epoch=4, plot_valid_per_epoch=1):
        super().__init__()
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        self.loss_board = ProgressBoard(fig=fig, axes=axs[0])
        self.accuracy_board = ProgressBoard(fig=fig, axes=axs[1], ylim=[0,1])

    def plot(self, key, value, epoch, i, dl_len, train):
        """Plot a point in animation."""
        if train:
            x = epoch + i / dl_len
            n = dl_len / self.plot_train_per_epoch
        else:
            x = epoch + 1
            n = dl_len / self.plot_valid_per_epoch
        if key == 'loss':
            board = self.loss_board
        elif key == 'accuracy':
            board = self.accuracy_board
        else:
            raise ValueError("Unknown key")
        board.xlabel = 'epoch'
        board.draw(x, value, ('train_' if train else 'val_') + key, 
                   every_n=int(n))