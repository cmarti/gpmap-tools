#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from gpmap.settings import PLOTS_FORMAT, PROT_AMBIGUOUS_VALUES, AMBIGUOUS_VALUES
from itertools import product
from gpmap.utils import translante_seqs
from gpmap.base import extend_ambigous_seq
from gpmap.plot_utils import init_fig


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
    

def save_plotly(fig, fpath=None):
    if fpath is None:
        fig.show()
    else:
        fpath = '{}.html'.format(fpath)
        fig.write_html(fpath)


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


def plot_decay_rates(decay_df, axes=None, fpath=None):
    if axes is None and fpath is None:
        msg = 'Either axes or fpath argument must be provided'
        raise ValueError(msg)
    
    fig = None
    if axes is None:
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3)
    
    axes.plot(decay_df['k'], decay_df['decay_rates'],
              linewidth=1, color='purple')
    axes.scatter(decay_df['k'], decay_df['decay_rates'],
                 s=15, c='purple')
    axes.set(yscale='log')
    axes.set_xlabel(r'Eigenvalue order $k$')
    axes.set_ylabel(r'Decay rate $\frac{-1}{\lambda_{k}}$')
    
    if fig is not None:
        savefig(fig, fpath)


def get_edges_coords(nodes_df, edges_df, x='1', y='2', z=None, avoid_dups=False):
    if avoid_dups:
        s = np.where(edges_df['j'] > edges_df['i'])[0]
        edges_df = edges_df.iloc[s, :]

    colnames = [x, y]
    if z is not None:
        colnames.append(z)
        
    nodes_coords = nodes_df[colnames].values
    edges_coords = np.stack([nodes_coords[edges_df['i']],
                             nodes_coords[edges_df['j']]], axis=2).transpose((0, 2, 1))
    return(edges_coords)
    

def plot_edges(axes, nodes_df, edges_df, x='1', y='2', z=None,
               color='grey', width=0.5, cmap='binary',
               alpha=0.1, zorder=1, avoid_dups=False,
               max_width=1, min_width=0.1):
    # TODO: get colors and width as either fixed values or from edges_df
    edges_coords = get_edges_coords(nodes_df, edges_df, x=x, y=y, z=z,
                                    avoid_dups=avoid_dups)

    if color in edges_df.columns:
        cmap = cm.get_cmap(cmap)
        color = cmap(edges_df[color])
        
    if width in edges_df.columns:
        w = edges_df[width]
        width = min_width + w * (max_width - min_width) / (w.max() - w.min())
        
    if z is None:
        ln_coll = LineCollection(edges_coords, colors=color, linewidths=width,
                                 alpha=alpha, zorder=zorder)
    else:
        ln_coll = Line3DCollection(edges_coords, color=color, linewidths=width,
                                   alpha=alpha, zorder=zorder)
    axes.add_collection(ln_coll)
    return(ln_coll)


def minimize_nodes_distance(nodes_df1, nodes_df2, axis):
    d = np.inf
    sel_coords = None
    
    coords1 = nodes_df1[axis]
    coords2 = nodes_df2[axis]
    
    for scalars in product([1, -1], repeat=len(axis)):
        c = np.vstack([v * s for v, s in zip(coords1.T, scalars)]).T
        distance = np.sqrt(np.sum((c - coords2) ** 2, 1)).mean(0)
        if distance < d:
            d = distance
            sel_coords = c
    
    nodes_df1[axis] = sel_coords
    return(nodes_df1)


def plot_nodes(axes, nodes_df, x='1', y='2', z=None,
               color='f', size=2.5, cmap='viridis', palette='Set1',
               alpha=1, zorder=2, max_size=40, min_size=1,
               edgecolor='black', lw=0,
               label=None, clabel='Function',
               sort=True, sort_by=None, ascending=False, 
               vmax=None, vmin=None, fontsize=12, legendloc=0,
               subset=None):
    if subset is not None:
        nodes_df = nodes_df.loc[subset, :]
    
    add_cbar, add_legend = False, False
    if color in nodes_df.columns:
        
        # Categorical color map
        if nodes_df[color].dtype == object:
            if isinstance(palette, str):
                labels = np.unique(nodes_df[color])
                n_colors = labels.shape[0]
                colors = sns.color_palette(palette, n_colors)
                palette = dict(zip(labels, colors))
            elif not isinstance(palette, dict):
                raise ValueError('palette must be a str or dict')
            color = np.array([palette[label] for label in nodes_df[color]])
            add_legend = True
            
        # Continuous color map
        elif nodes_df[color].dtype in (float, int):
            if sort:
                if sort_by is None:
                    sort_by = color
                nodes_df = nodes_df.sort_values(sort_by, ascending=ascending)
            cmap = cm.get_cmap(cmap)
            color = nodes_df[color]
            add_cbar = True
        else:
            msg = 'color dtype is not compatible: {}'.format(nodes_df[color].dtype)
            raise ValueError(msg)
    
    if size in nodes_df.columns and not isinstance(size, int):
        s = np.power(nodes_df[size], 2)
        size = min_size + s * (max_size - min_size) / (s.max() - s.min())

    if z is not None:
        sc = axes.scatter(nodes_df[x], nodes_df[y], zs=nodes_df[z], c=color,
                          linewidth=lw, s=size, zorder=zorder, alpha=alpha,
                          edgecolor=edgecolor, cmap=cmap, label=label,
                          vmax=vmax, vmin=vmin)
        axes.set_zlabel('Diffusion axis {}'.format(z), fontsize=fontsize)
    
    else:
        sc = axes.scatter(nodes_df[x], nodes_df[y], c=color,
                          linewidth=lw, s=size, zorder=zorder, alpha=alpha,
                          edgecolor=edgecolor, cmap=cmap, label=label,
                          vmax=vmax, vmin=vmin)
    
    if add_cbar:
        plt.colorbar(sc, ax=axes).set_label(label=clabel, fontsize=fontsize)
    if add_legend:
        create_patches_legend(axes, palette, loc=legendloc, fontsize=fontsize)
        
    axes.set_xlabel('Diffusion axis {}'.format(x), fontsize=fontsize)
    axes.set_ylabel('Diffusion axis {}'.format(y), fontsize=fontsize)


def get_nodes_df_highlight(nodes_df, genotype_groups, is_prot=False,
                           alphabet_type='dna', codon_table='Standard'):
    # TODO: force protein to be in the table if we want to do highlight
    # protein subsequences to decouple the genetic code from plotting as 
    # it is key to visualization and should not be changed afterwards
    groups_dict = {}
    if is_prot:
        if 'protein' not in nodes_df.columns:
            nodes_df['protein'] = translante_seqs(nodes_df.index,
                                                  codon_table=codon_table)
        
        for group in genotype_groups:
            for seq in extend_ambigous_seq(group, PROT_AMBIGUOUS_VALUES):
                groups_dict[seq] = group
        nodes_df['group'] = [groups_dict.get(x, None) for x in nodes_df['protein']]
    else:
        nodes_df['group'] = np.nan
        for group in genotype_groups:
            genotype_labels = extend_ambigous_seq(group, AMBIGUOUS_VALUES[alphabet_type])
            nodes_df.loc[genotype_labels, 'group'] = group
    nodes_df = nodes_df.dropna()
    return(nodes_df)
    
def highlight_genotype_groups(axes, nodes_df, genotype_groups,
                              x='1', y='2', z=None, size=50, edgecolor='black',
                              lw=1, palette='colorblind', legendloc=1,
                              fontsize=12, is_prot=False,
                              alphabet_type='dna', codon_table='Standard'):
    nodes_df = get_nodes_df_highlight(nodes_df, genotype_groups,
                                      alphabet_type=alphabet_type,
                                      codon_table=codon_table,
                                      is_prot=is_prot)
    
    plot_nodes(axes, nodes_df, x=x, y=y, z=z, color='group', size=size,
               palette=palette, edgecolor=edgecolor, lw=lw,
               fontsize=fontsize, legendloc=legendloc)


def plot_visualization(axes, nodes_df, edges_df=None, x='1', y='2', z=None,
                       nodes_color='f', nodes_size=2.5, nodes_cmap='viridis', nodes_alpha=1,
                       nodes_min_size=1, nodes_max_size=40,
                       nodes_edgecolor='black', nodes_lw=0, 
                       nodes_cmap_label='Function', nodes_vmin=None, nodes_vmax=None,
                       edges_color='grey', edges_width=0.5, edges_cmap='binary',
                       edges_alpha=0.1, edges_max_width=1, edges_min_width=0.1, 
                       sort_nodes=True, ascending=False, sort_by=None,
                       fontsize=12, prev_nodes_df=None):
    
    if prev_nodes_df is not None:
        axis = [x, y] if z is None else [x, y, z]
        nodes_df = minimize_nodes_distance(nodes_df, prev_nodes_df, axis)
    
    plot_nodes(axes, nodes_df=nodes_df, x=x, y=y, z=z,
               color=nodes_color, size=nodes_size, cmap=nodes_cmap, 
               alpha=nodes_alpha, zorder=2, max_size=nodes_max_size,
               min_size=nodes_min_size,
               edgecolor=nodes_edgecolor, lw=nodes_lw,
               label=None, clabel=nodes_cmap_label,
               sort=sort_nodes, sort_by=sort_by, ascending=ascending, 
               vmax=nodes_vmax, vmin=nodes_vmin, fontsize=fontsize,
               subset=None)
    
    if edges_df is not None:
        plot_edges(axes, nodes_df, edges_df, x=x, y=y, z=z,
                   color=edges_color, width=edges_width, cmap=edges_cmap,
                   alpha=edges_alpha, zorder=1, avoid_dups=True,
                   max_width=edges_max_width, min_width=edges_min_width)


def get_plotly_edges(nodes_df, edges_df, x=1, y=2, z=None,
                     avoid_dups=True):
    edge_x = []
    edge_y = []
    edge_z = []
    edges = get_edges_coords(nodes_df, edges_df, x=x, y=y, z=z,
                             avoid_dups=avoid_dups)
    ndim = edges.shape[2]
    for node1, node2 in zip(edges[:, 0, :], edges[:, 1, :]):
        edge_x.extend([node1[0], node2[0], None])
        edge_y.extend([node1[1], node2[1], None])
        if ndim > 2:
            edge_z.extend([node1[2], node2[2], None])
    return(edge_x, edge_y, edge_z)


def plot_interactive(nodes_df, edges_df=None, fpath=None, x='1', y='2', z=None,
                     nodes_color='f', nodes_size=4,
                     cmap='viridis', nodes_cmap_label='Function',
                     edges_width=0.5, edges_color='#888', edges_alpha=0.2,
                     text=None):
    # Create figure
    fig = go.Figure()

    # Create nodes plot
    colorbar = dict(thickness=25, title=nodes_cmap_label, xanchor='left',
                    titleside='right', len=0.8)
    marker = dict(showscale=True, colorscale=cmap, reversescale=False,
                  color=nodes_df[nodes_color], size=nodes_size, colorbar=colorbar,
                  line_width=2)
    if text is None:
        text = nodes_df.index
    
    if z is None:
        node_trace = go.Scatter(x=nodes_df[x], y=nodes_df[y],
                                mode='markers', hoverinfo='text',
                                marker=marker, text=text, name='Genotypes')
    else:
        node_trace = go.Scatter3d(x=nodes_df[x], y=nodes_df[y], z=nodes_df[z],
                                  mode='markers', hoverinfo='text',
                                  marker=marker, text=text, name='Genotypes')
    fig.add_trace(node_trace)

    # Create edges        
    if edges_df is not None:
        edges = get_plotly_edges(nodes_df, edges_df, x=x, y=y, z=z)
        if z is None:
            edge_trace = go.Scatter(x=edges[0], y=edges[1],
                                    line=dict(width=edges_width, color=edges_color),
                                    hoverinfo='none', mode='lines',
                                    opacity=edges_alpha, name='Mutations')
        else:
            edge_trace = go.Scatter3d(x=edges[0], y=edges[1], z=edges[2],
                                      line=dict(width=edges_width, color=edges_color),
                                      hoverinfo='none', mode='lines',
                                      opacity=edges_alpha, name='Mutations')
        fig.add_trace(edge_trace)
    
    # Update layout        
    scene = dict(xaxis_title='Diffusion axis {}'.format(x),
                 yaxis_title='Diffusion axis {}'.format(y))
    if z is not None:
        scene['zaxis_title'] = 'Diffusion axis {}'.format(z)
        
    fig.update_layout(title="Landscape visualization", 
                      hovermode='closest', template='simple_white',
                      xaxis_title='Diffusion axis {}'.format(x),
                      yaxis_title='Diffusion axis {}'.format(y),
                      scene=scene)
    
    save_plotly(fig, fpath=fpath)


def figure_visualization(nodes_df, edges_df=None, fpath=None, x='1', y='2', z=None,
                         nodes_color='f', nodes_size=None, nodes_cmap='viridis',
                         nodes_alpha=1, nodes_min_size=1, nodes_max_size=40,
                         nodes_edgecolor='black', nodes_lw=0, 
                         nodes_cmap_label='Function',
                         nodes_vmin=None, nodes_vmax=None,
                         edges_color='grey', edges_width=0.5, edges_cmap='binary',
                         edges_alpha=0.1, edges_max_width=1, edges_min_width=0.1, 
                         sort_nodes=True, sort_by=None, ascending=True,
                         fontsize=12, prev_nodes_df=None,
                         highlight_genotypes=None, is_prot=False,
                         highlight_size=200, palette='colorblind',
                         figsize=(10, 7.6), interactive=False, 
                         alphabet_type=None):
    
    if nodes_size is None:
        nodes_size = 15 if z is None else 4 if interactive else 50
    
    if interactive:
        text = nodes_df['protein'] if is_prot else nodes_df.index
        plot_interactive(nodes_df, edges_df=edges_df, fpath=fpath,
                         x=x, y=y, z=z,
                         nodes_color=nodes_color, nodes_size=nodes_size,
                         cmap=nodes_cmap, nodes_cmap_label=nodes_cmap_label,
                         edges_width=edges_width, edges_color=edges_color,
                         text=text)
    else:
        fig, axes = init_single_fig(figsize=figsize, is_3d=z is not None)
        plot_visualization(axes, nodes_df=nodes_df, edges_df=edges_df,
                           x=x, y=y, z=z,
                           nodes_color=nodes_color, nodes_size=nodes_size,
                           nodes_cmap=nodes_cmap, nodes_alpha=nodes_alpha,
                           nodes_min_size=nodes_min_size, nodes_max_size=nodes_max_size,
                           nodes_edgecolor=nodes_edgecolor, nodes_lw=nodes_lw, 
                           nodes_cmap_label=nodes_cmap_label, nodes_vmin=nodes_vmin,
                           nodes_vmax=nodes_vmax,
                           edges_color=edges_color, edges_width=edges_width,
                           edges_cmap=edges_cmap,
                           edges_alpha=edges_alpha, edges_max_width=edges_max_width,
                           edges_min_width=edges_min_width, 
                           sort_nodes=sort_nodes, sort_by=sort_by, ascending=ascending,
                           fontsize=fontsize,
                           prev_nodes_df=prev_nodes_df)
        
        if highlight_genotypes is not None:
            highlight_genotype_groups(axes, nodes_df, highlight_genotypes,
                                      is_prot=is_prot, x=x, y=y, z=z,
                                      alphabet_type=alphabet_type,
                                      size=highlight_size, palette=palette)
        savefig(fig, fpath)
