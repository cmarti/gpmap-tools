#!/usr/bin/env python
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import datashader as ds
import holoviews as hv

from holoviews.operation.datashader import datashade

from gpmap.src.seq import guess_space_configuration
from gpmap.src.plot.utils import get_lines_from_edges_df


def calc_ds_size(nodes_df, x, y, resolution, square=True):
    if square:
        xlim = nodes_df[x].min(), nodes_df[x].max()
        ylim = nodes_df[y].min(), nodes_df[y].max()
        dx, dy = xlim[1]- xlim[0], ylim[1] - ylim[0]
        w, h = resolution, resolution * dx / dy
    else:
        w, h = resolution, resolution
    return(int(w), int(h))

    
def plot_nodes(nodes_df, x='1', y='2', color='function', cmap='viridis',
               size=5, linewidth=0, edgecolor='black',
               vmin=None, vmax=None,
               sort_by=None, ascending=True,
               shade=True, resolution=800, square=True):
    if sort_by is not None:
        nodes_df = nodes_df.sort_values(sort_by, ascending=ascending)
        
    if vmin is None:
        vmin = nodes_df[color].min()
    if vmax is None:
        vmax = nodes_df[color].max()
    
    if shade:
        nodes = hv.Points(nodes_df, kdims=[x, y], label='Nodes')
        w, h = calc_ds_size(nodes_df, x, y, resolution, square=square)
        if sort_by is not None:
            dsg = datashade(nodes, cmap=cmap, width=w, height=h,
                            aggregator=ds.first(color))
        else:
            dsg = datashade(nodes, cmap=cmap, width=w, height=h,
                            aggregator=ds.max(color))
    else:
        hv.extension('matplotlib')
        colnames = [x, y]
        if color not in colnames: # avoid adding color in case is already a selected field
            colnames.append(color)
        scatter = hv.Scatter(nodes_df[colnames])
        dsg = scatter.opts(color=color, cmap=cmap, clim=(vmin, vmax),
                           s=size, linewidth=linewidth, 
                           edgecolor=edgecolor)
    aspect = 'square' if square else 'equal'
    dsg.opts(aspect=aspect)
    return(dsg)


def plot_edges(nodes_df, edges_df, x='1', y='2', cmap='grey',
               width=0.5, alpha=0.2, color='grey',
               shade=True, resolution=800, square=True):
    line_coords = get_lines_from_edges_df(nodes_df, edges_df, x=x, y=y, z=None)
    dsg = hv.Curve(line_coords)
    if shade:
        w, h = calc_ds_size(nodes_df, x, y, resolution, square=square)
        dsg = datashade(dsg, cmap=cmap, width=w, height=h)
    else:
        dsg = dsg.opts(color=color, linewidth=width, alpha=alpha)
    aspect = 'square' if square else 'equal'
    dsg.opts(aspect=aspect)
    return(dsg)


def plot_visualization(nodes_df, x='1', y='2', edges_df=None,
                       nodes_color='function', nodes_cmap='viridis',
                       nodes_size=5, nodes_vmin=None, nodes_vmax=None,
                       linewidth=0, edgecolor='black',
                       sort_by=None, ascending=False,
                       edges_width=0.5, edges_alpha=0.2, edges_color='grey',
                       edges_cmap='grey', background_color='white',
                       nodes_resolution=800, edges_resolution=1200,
                       shade_nodes=True, shade_edges=True, square=True):
    dsg = plot_nodes(nodes_df, x, y, nodes_color, nodes_cmap,
                     linewidth=linewidth, edgecolor=edgecolor,
                     size=nodes_size, vmin=nodes_vmin, vmax=nodes_vmax,
                     sort_by=sort_by, ascending=ascending,
                     resolution=nodes_resolution, shade=shade_nodes,
                     square=square)
    
    if edges_df is not None:
        edges_dsg = plot_edges(nodes_df, edges_df, x, y,
                               cmap=edges_cmap, width=edges_width, 
                               alpha=edges_alpha, color=edges_color,
                               resolution=edges_resolution,
                               shade=shade_edges, square=square)
        dsg = edges_dsg * dsg
    
    dsg.opts(xlabel='Diffusion axis {}'.format(x),
             ylabel='Diffusion axis {}'.format(y),
             bgcolor=background_color, padding=0.1)
    
    return(dsg)


def save_holoviews(dsg, fpath, fmt='png', figsize=None):
    fig = hv.render(dsg)
    if figsize is not None:
        fig.set_size_inches(figsize[0], figsize[1])
    savefig(fig, fpath, tight=False, fmt=fmt)


def figure_allele_grid_datashader(nodes_df, fpath, x='1', y='2', edges_df=None,
                                  positions=None, position_labels=None,
                                  edges_cmap='grey', background_color='white',
                                  nodes_resolution=800, edges_resolution=1200,
                                  fmt='png', figsize=None, square=True):
    if edges_df is not None:
        edges = plot_edges_datashader(nodes_df, edges_df, x, y,
                                      cmap=edges_cmap,
                                      resolution=edges_resolution,
                                      square=square)
    else:
        edges = None
        
    config = guess_space_configuration(nodes_df.index.values)
    length, n_alleles = config['length'], np.max(config['n_alleles'])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)

    plots = None
    
    nc = {i: np.array([seq[i] for seq in nodes_df.index])
          for i in range(length)}
    
    for i in range(n_alleles):
        for col, j in enumerate(positions):
            try:
                allele  = config['alphabet'][j][i]
                nodes_df['allele'] = (nc[col] == allele).astype(int)
            except IndexError:
                allele = ''
                nodes_df['allele'] = np.nan
                
            nodes = plot_nodes_datashader(nodes_df.copy(),
                                          x, y,
                                          color='allele', cmap='viridis',
                                          resolution=nodes_resolution,
                                          shade=True, square=square)
            nodes = nodes.relabel('{}{}'.format(j+1, allele))
            dsg = nodes if edges is None else edges * nodes
            
            dsg.opts(xlabel='Diffusion axis {}'.format(x),
                     ylabel='Diffusion axis {}'.format(y),
                     bgcolor=background_color,
                     title='{}{}'.format(position_labels[j], allele))
            
            if i < n_alleles - 1:
                dsg.opts(xlabel='')
            if col > 0:
                dsg.opts(ylabel='')
                
            if plots is None:
                plots = dsg
            else:
                plots += dsg
    dsg = plots.cols(length)
    fig = hv.render(dsg)
    if figsize is not None:
        fig.set_size_inches(*figsize)
    savefig(fig, fpath, tight=False, fmt=fmt)


def plot_hyperparam_cv(df, axes, x='log_a', y='logL', err_bars='stderr',
                       xlabel=r'$\log_{10}(a)$',
                       ylabel='log(L)',  show_folds=True, highlight='max',
                       legend_loc=1):
    
    sdf = df.groupby(x, as_index=False).agg({y: ('mean', 'std', 'count')})
    sdf.columns = ['x', 'mean', 'sd', 'count']
    sdf['stderr'] = sdf['sd'] / np.sqrt(sdf['count'])
    
    idx = sdf['mean'].argmax() if highlight == 'max' else sdf['mean'].argmin()
    x_star = sdf['x'][idx]
    y_star = sdf['mean'][idx]
    
    if show_folds:
        sns.lineplot(x=x, y=y, hue='fold', ax=axes, legend=False,
                     data=df.sort_values(x), alpha=0.4, linewidth=0.5,
                     zorder=1)
    axes.plot(sdf['x'], sdf['mean'], color='black', lw=1, zorder=1)
    axes.scatter(sdf['x'], sdf['mean'], color='black', s=15)
    axes.scatter(x_star, y_star, color='red', s=25, zorder=10, alpha=1)
    
    for a, m, s in zip(sdf['x'], sdf['mean'], sdf[err_bars]):
        color = 'red' if a == x_star else 'black'
        
        if a == x_star:
            label = '{}* = {:.1f}'.format(xlabel, x_star)
        else:
            label = None
        
        axes.plot((a, a), (m-s, m+s), lw=1, color=color, label=label)
    
    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    ylims = (ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.1)
    
    for r in sdf.loc[np.isinf(sdf['x']), :].to_dict(orient='index').values():
        x, y = r['x'], r['mean']
        label = r'{} = $\infty$'.format(xlabel)
        if y < 0:
            label = r'{} = -$\infty$'.format(xlabel)
        if not np.isnan(y):
            axes.plot(xlims, (y, y), lw=0.5, c='darkred', linestyle='--',
                      label=label)
    axes.legend(loc=legend_loc, fontsize=9)
    axes.set(xlabel=xlabel, ylabel='Out of sample {}'.format(ylabel),
             ylim=ylims, xlim=xlims)


def plot_density_vs_frequency(seq_density, axes):
    with warnings.catch_warnings():
        logf = np.log10(seq_density['frequency'])
        logq = np.log10(seq_density['Q_star'])
        warnings.simplefilter("ignore")
        data = pd.DataFrame({'logR': logf, 'logQ': logq}).dropna()
                             
    axes.scatter(data['logR'], data['logQ'],
                 color='black', s=5, alpha=0.4, zorder=2,
                 label='Observed sequences')
    
    mask = np.isinf(logf)
    zero_counts_logq = logq[mask]
    fake_logf = np.full(zero_counts_logq.shape, logf[mask == False].min() - 0.5) 
    axes.scatter(fake_logf, zero_counts_logq, marker='<',
                 color='red', s=5, alpha=0.2, zorder=2,
                 label='Unobserved sequences')
    
    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
    axes.plot(lims, lims, color='grey', linewidth=0.5, alpha=0.5, zorder=1)
    axes.set(xlabel=r'$log_{10}$(Frequency)', ylabel=r'$log_{10}$(Q*)', 
             xlim=lims, ylim=lims)
    axes.legend(loc=2, fontsize=9)


def plot_SeqDEFT_summary(log_Ls, seq_density=None, err_bars='stderr',
                         show_folds=True, legend_loc=1):
    '''
    Generates a 2 panel figure showing how the cross-validated likelihood
    changes with ``a`` hyperparameter and the best selected value for model
    fitting.  
    
    Parameters
    ----------
        log_Ls : pd.DataFrame of shape (num_a, 3)
            DataFrame containing the column names ``a``, ``logL`` and
            ``fold```
        
        seq_density : pd.DataFrame of shape (n_genotypes, >= 2)
            DataFrame with column names ``frequency``, ``Q_star`` with the 
            observed frequencies and estimated densities for each possible sequence
            respectively. If not provided only a 1 panel figure with the
            cross-validated likelihood curve will be provided
            
        err_bars : str
            What to show in the error bars: sd standard deviation across the 
            different folds or stderr for standard error of the mean
            
        show_folds: bool
            Whether to show the out of sample log likelihoods for the different
            folds in the cross-validation procedure separately
            
    Returns
    -------
        fig : matplotlib.figure object
            ``Figure`` object containing the resulting plots
    
    '''
    if seq_density is None:
        fig, axes = init_fig(1, 1, colsize=5, rowsize=4.3)
    else:
        fig, subplots = init_fig(1, 2, colsize=4, rowsize=3.5)
        plot_density_vs_frequency(seq_density, subplots[1])
        axes = subplots[0]
        
    plot_hyperparam_cv(log_Ls, axes, err_bars=err_bars,
                       show_folds=show_folds, legend_loc=legend_loc)
    
    if seq_density is not None:
        fig.tight_layout()
    return(fig)
