#!/usr/bin/env python
from os.path import join

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from gpmap.settings import PLOTS_FORMAT, PLOTS_DIR


# Functions
def init_fig(nrow=1, ncol=1, figsize=None, style='ticks',
             colsize=3, rowsize=3):
    sns.set_style(style)
    if figsize is None:
        figsize = (colsize * ncol, rowsize * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    return(fig, axes)


def init_single_fig(figsize=None, style='ticks',
             colsize=3, rowsize=3, is_3d=False):
    sns.set_style(style)
    if figsize is None:
        figsize = (colsize, rowsize)
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(1, 1, 1, projection='3d' if is_3d else None)
    return(fig, axes)


def savefig(fig, fpath, tight=True, fmt=PLOTS_FORMAT):
    fpath = '{}.{}'.format(fpath, fmt)
    if tight:
        fig.tight_layout()
    fig.savefig(fpath, format=fmt, dpi=240)
    plt.close()


def plot_comp_line(axes, x1, x2, y, size, lw=1):
    axes.plot((x1, x2), (y, y), lw=lw, c='black')
    axes.plot((x1, x1), (y-size, y), lw=lw, c='black')
    axes.plot((x2, x2), (y-size, y), lw=lw, c='black')


def empty_axes(axes):
    sns.despine(ax=axes, left=True, bottom=True)
    axes.set_xticks([])
    axes.set_yticks([])


def create_patches_legend(axes, colors_dict, loc=1, **kwargs):
    axes.legend(handles=[mpatches.Patch(color=color, label=label)
                         for label, color in colors_dict.items()],
                loc=loc, **kwargs)


def set_boxplot_colorlines(axes, color):
    # From
    # https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot
    for i, artist in enumerate(axes.artists):
        artist.set_edgecolor(color)
        artist.set_facecolor('None')

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for line in axes.lines:
            line.set_color(color)
            line.set_mfc(color)
            line.set_mec(color)


def arrange_plot(axes, xlims=None, ylims=None, zlims=None,
                 xlabel=None, ylabel=None, zlabel=None,
                 showlegend=False, legend_loc=None, hline=None, vline=None,
                 rotate_xlabels=False, cols_legend=1, rotation=90,
                 legend_frame=True, title=None, ticklabels_size=None,
                 yticklines=False, despine=False, legend_fontsize=10):
    if xlims is not None:
        axes.set_xlim(xlims)
    if ylims is not None:
        axes.set_ylim(ylims)
    if zlims is not None:
        axes.set_zlim(zlims)
        
    if title is not None:
        axes.set_title(title)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if zlabel is not None:
        axes.set_zlabel(zlabel)

    if showlegend:
        axes.legend(loc=legend_loc, ncol=cols_legend,
                    frameon=legend_frame, fancybox=legend_frame,
                    fontsize=legend_fontsize)
    elif axes.legend_ is not None:
        axes.legend_.set_visible(False)

    if hline is not None:
        xlims = axes.get_xlim()
        axes.plot(xlims, (hline, hline), linewidth=1, color='grey',
                  linestyle='--')
        axes.set_xlim(xlims)

    if vline is not None:
        ylims = axes.get_ylim()
        axes.plot((vline, vline), ylims, linewidth=1, color='grey',
                  linestyle='--')
        axes.set_ylim(ylims)

    if rotate_xlabels:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation,
                             ha='right')
    if ticklabels_size is not None:
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabels_size)
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabels_size)
    if yticklines:
        xlims = axes.get_xlim()
        for y in axes.get_yticks():
            axes.plot(xlims, (y, y), lw=0.2, alpha=0.1, c='lightgrey')

    if despine:
        sns.despine(ax=axes)


def plot_post_pred_ax(x, q, axes, color):
    for i in range(int((q.shape[0] - 1) / 2)):
        axes.fill_between(x, q[i, :], q[-(i + 1), :], facecolor=color,
                          interpolate=True, alpha=0.1)
    axes.plot(x, q[int(q.shape[0] / 2), :], color=color, linewidth=2)


def add_panel_label(axes, label, fontsize=20, yfactor=0.015, xfactor=0.25):
    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    x = xlims[0] - (xlims[1] - xlims[0]) * xfactor
    y = ylims[1] + (ylims[1] - ylims[0]) * yfactor
    axes.text(x, y, label, fontsize=fontsize)


def add_grey_area(axes, between=(-0.2, 0.2), ylims=None, add_vline=True):
    if ylims is None:
        ylims = axes.get_ylim()
    axes.fill_between(between, (ylims[0], ylims[0]), (ylims[1], ylims[1]),
                      facecolor='grey', interpolate=True, alpha=0.2)
    if add_vline:
        axes.plot((0, 0), ylims, linestyle='--', c='grey', lw=0.5)


def add_image(axes, fpath):
    fmt = fpath.split('.')[-1]
    arr_image = plt.imread(fpath, format=fmt)    
    axes.imshow(arr_image)
    axes.axis('off')


class FigGrid(object):
    def __init__(self, figsize=(11, 9.5), xsize=100, ysize=100):
        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(xsize, ysize, wspace=1, hspace=1)
        self.xsize = 100
        self.ysize = 100

    def new_axes(self, xstart=0, xend=None, ystart=0, yend=None):
        if xend is None:
            xend = self.xsize
        if yend is None:
            yend = self.ysize
        return(self.fig.add_subplot(self.gs[ystart:yend, xstart:xend]))

    def savefig(self, fname):
        savefig(self.fig, fname, tight=False)
        
        
def plot_eigenvalues(axes, df_visual, n_components):
    x = range(1, n_components)
    print(df_visual['eigenvalue'].values[1:])
    y = 1 / abs(df_visual['eigenvalue'].values[1:])
    axes.scatter(x, y, color='blue', s=15)
    axes.plot(x, y, color='blue')
    axes.set_xlabel('k', fontsize=14)
    axes.set_ylabel('1 / |Eigenvalue|', fontsize=14)
