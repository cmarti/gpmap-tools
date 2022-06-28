#!/usr/bin/env python
import re
from itertools import product

import numpy as np
import pandas as pd

from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix

from gpmap.src.settings import PROT_AMBIGUOUS_VALUES, AMBIGUOUS_VALUES
from gpmap.src.utils import check_error, write_log
from gpmap.src.seq import extend_ambigous_seq, translate_seqs
from scipy.sparse._matrix_io import load_npz


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


def minimize_nodes_distance(nodes_df1, nodes_df2, axis):
    d = np.inf
    sel_coords = None
    
    coords1 = nodes_df1[axis]
    coords2 = nodes_df2[axis]
    
    for scalars in product([1, -1], repeat=len(axis)):
        c = np.vstack([v * s for v, s in zip(coords1.values.T, scalars)]).T
        distance = np.sqrt(np.sum((c - coords2) ** 2, 1)).mean(0)
        if distance < d:
            d = distance
            sel_coords = c
    
    nodes_df1[axis] = sel_coords
    return(nodes_df1)


def get_nodes_df_highlight(nodes_df, genotype_groups, is_prot=False,
                           alphabet_type='dna', codon_table='Standard'):
    # TODO: force protein to be in the table if we want to do highlight
    # protein subsequences to decouple the genetic code from plotting as 
    # it is key to visualization and should not be changed afterwards
    groups_dict = {}
    if is_prot:
        if 'protein' not in nodes_df.columns:
            nodes_df['protein'] = translate_seqs(nodes_df.index,
                                                 codon_table=codon_table)
        
        for group in genotype_groups:
            mapping = [PROT_AMBIGUOUS_VALUES] * len(group)
            for seq in extend_ambigous_seq(group, mapping):
                groups_dict[seq] = group
        nodes_df['group'] = [groups_dict.get(x, None) for x in nodes_df['protein']]
    else:
        nodes_df['group'] = np.nan
        for group in genotype_groups:
            mapping = [AMBIGUOUS_VALUES[alphabet_type]] * len(group)
            genotype_labels = extend_ambigous_seq(group, mapping)
            nodes_df.loc[genotype_labels, 'group'] = group
    nodes_df = nodes_df.dropna()
    return(nodes_df)


def filter_csr_matrix(matrix, idxs):
    return(matrix[idxs, :][:, idxs])


def dataframe_to_csr_matrix(edges_df):
    size = max(edges_df['i'].max(), edges_df['j'].max()) + 1
    idxs = np.arange(edges_df.shape[0])
    
    # idxs are store for filtering edges later on rather than just ones
    m = csr_matrix((idxs, (edges_df['i'], edges_df['j'])),
                   shape=(size, size))
    return(m)
    

def select_edges_from_genotypes(nodes_idxs, edges):
    if isinstance(edges, pd.DataFrame):
        m = filter_csr_matrix(dataframe_to_csr_matrix(edges),
                              nodes_idxs).tocoo()
        edges = edges.iloc[m.data, :].copy()
        edges['i'] = m.row
        edges['j'] = m.col
    else:
        if isinstance(edges, coo_matrix):
            edges = edges.tocsr()
        check_error(isinstance(edges, csr_matrix),
                    'edges must be a pd.DataFrame or sparse matrix')
        edges = filter_csr_matrix(edges, nodes_idxs).tocoo()
    return(edges)


def select_genotypes(nodes_df, genotypes, edges=None, is_idx=False):
    '''
    Selectes the selected genotypes from nodes_df with the corresponding 
    edges among the remaining genotypes if edges are provided
    
    Parameters
    ----------
        nodes_df: pd.DataFrame of shape (n_genotypes, n_features)
            DataFrame with the genotypes from a full sequence space as index
            Typically, it will contain, at least, the coordinates of the 
            visualization for each genotype, but it will keep any other column
            in the DataFrame for later use
        
        genotypes: array-like of shape (n_genotypes,)
            Array of ordered genotypes to select from the starting landscape
            It should contain the genotype labels by default, or indexes if
            option `is_idx` is provided 
            
        edges: pd.DataFrame of shape (n_edges, 2) or scipy.sparse.csr_matrix
               of shape (n_genotypes, n_genotypes)
            DataFrame or csr_matrix containing the adjacency relationships
            among genotypes provided in `nodes_df` in the discrete space
    
        is_idx: bool
            The genotypes argument is an array of indexes instead of an
            array of genotype labels to select genotypes
        
    Returns
    -------
        output: (nodes_df, edges)
            Filtered landscape containing the selected genotypes and the 
            adjacency relationships between them given as a tuple
    
    '''
    
    size = nodes_df.shape[0]
    nodes_df['index'] = np.arange(size)
    if is_idx:
        nodes_df = nodes_df.iloc[genotypes, :]
    else:
        nodes_df = nodes_df.loc[genotypes, :]
    
    if edges is not None:
        edges = select_edges_from_genotypes(nodes_df['index'], edges)
        return(nodes_df, edges)
    else:
        return(nodes_df)
    
    
def select_d_neighbors(nodes_df, genotype_labels, d=1, edges=None):
    # TODO: add option to select only edges starting from the selected genotypes
    seq_matrix = np.array([[s for s in seq] for seq in nodes_df.index])
    distances = np.array([np.vstack([allele != col for allele, col in zip(gt, seq_matrix.T)]).sum(0)
                          for gt in genotype_labels])
    genotypes = (distances <= d).any(0)
    return(select_genotypes(nodes_df, genotypes, edges=edges))


def select_genotypes_re(nodes_df, pattern, edges=None):
    exp = re.compile(pattern)
    genotypes = np.array([True if exp.search(seq) else False for seq in nodes_df.index])
    return(select_genotypes(nodes_df, genotypes, edges=edges))


def select_genotypes_ambiguous_seqs(nodes_df, seqs, alphabet_type, edges=None):
    genotypes = extend_ambigous_seq(seqs, AMBIGUOUS_VALUES[alphabet_type])
    return(select_genotypes(nodes_df, genotypes, edges=edges))


def guess_n_axis(nodes_df):
    i = 1
    while str(i) in nodes_df.columns:
        i += 1
    return(i-1)


def select_closest_genotypes(nodes_df, genotype, n_genotypes, edges=None, axis=None):
    if axis is None:
        axis = [str(x) for x in range(1, guess_n_axis(nodes_df) + 1)]
    reference_pos = nodes_df[axis].loc[genotype].values.flatten()
    sq_distance = 0
    for a, p in zip(axis, reference_pos):
        d = nodes_df[a] - p
        sq_distance += d * d

    sel_idxs = np.argsort(sq_distance)[:n_genotypes]
    return(select_genotypes(nodes_df, sel_idxs, edges=edges, is_idx=True))


def select_local_optima(nodes_df, edges_df, field_function='function'):
    f1 = nodes_df[field_function].values[edges_df['i'].values]
    f2 = nodes_df[field_function].values[edges_df['j'].values]
    edf = edges_df.loc[f1 > f2, :] 
    m = csr_matrix((np.ones(edf.shape[0]), (edf['i'].values, edf['j'].values)),
                   shape=(nodes_df.shape[0], nodes_df.shape[0])).sum(1)
    idx = np.where(m == m.max())[0]
    return(select_genotypes(nodes_df, idx, edges=edges_df, is_idx=True)[0])


def read_edges(fpath, log=None):
    if fpath is not None:
        write_log(log, 'Reading edges data from {}'.format(fpath))
        edges_format = fpath.split('.')[-1]
        if edges_format == 'npz':
            A = load_npz(fpath).tocoo()
            edges_df = pd.DataFrame({'i': A.row, 'j': A.col})
        elif edges_format == 'csv':
            edges_df = pd.read_csv(fpath)
        else:
            raise ValueError('edges format has to be ".npz" or ".csv"')
    else:
        write_log(log, 'No edges provided for plotting')
        edges_df = None
    
    return(edges_df)
    