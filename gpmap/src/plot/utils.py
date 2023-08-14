#!/usr/bin/env python
import warnings
import numpy as np
import pandas as pd

from gpmap.src.settings import PLOTS_FORMAT
from gpmap.src.utils import check_error
from gpmap.src.seq import guess_space_configuration
from gpmap.src.genotypes import (get_edges_coords, get_nodes_df_highlight,
                                 minimize_nodes_distance)


def get_lines_from_edges_df(nodes_df, edges_df, x=1, y=2, z=None,
                            avoid_dups=True):
    edges = get_edges_coords(nodes_df, edges_df, x=x, y=y, z=z,
                             avoid_dups=avoid_dups)
    nans = np.full((edges.shape[0], 1), fill_value=np.nan)
    line_coords = np.vstack([np.hstack([edges[:, :, i], nans]).flatten()
                             for i in range(edges.shape[2])]).T
    return(line_coords)


def get_axis_lims(nodes_df, x, y, z=None):
    axis_max = max(nodes_df[x].max(), nodes_df[y].max())
    axis_min = min(nodes_df[x].min(), nodes_df[y].min())
    if z is not None:
        axis_max = max(axis_max, nodes_df[z].max())
        axis_min = min(axis_min, nodes_df[z].min())
    
    axis_range = axis_max - axis_min
    axis_lims = (axis_min - 0.05 * axis_range, axis_max + 0.05 * axis_range)
    return(axis_lims)


def get_allele_grid_config(sequences, positions=None, position_labels=None):
    config = guess_space_configuration(sequences)
    length, n_alleles = config['length'], np.max(config['n_alleles'])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)
    return({'seq_length': length, 'n_alleles': n_alleles,
            'alphabet': config['alphabet']})
