#!/usr/bin/env python
import re
from itertools import product

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix

from gpmap.settings import PROT_AMBIGUOUS_VALUES, AMBIGUOUS_VALUES
from gpmap.utils import check_error, edges_df_to_csr_matrix
from gpmap.seq import extend_ambigous_seq, translate_seqs
from gpmap.space import SequenceSpace
from gpmap.matrix import filter_csr_matrix


def get_edges_coords(nodes_df, edges_df, x="1", y="2",
                     z=None, avoid_dups=True):
    edf = edges_df

    if avoid_dups:
        edf = edges_df.copy()
        idx = (edf["j"] > edf["i"]).values
        edf.loc[idx, "i"], edf.loc[idx, "j"] = (
            edf.loc[idx, "j"],
            edf.loc[idx, "i"],
        )
        edf.drop_duplicates(inplace=True)

    colnames = [x, y]
    if z is not None:
        colnames.append(z)

    nodes_coords = nodes_df[colnames].values
    edges_coords = np.stack(
        [nodes_coords[edf["i"]], nodes_coords[edf["j"]]], axis=2
    ).transpose((0, 2, 1))
    return edges_coords


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
    return nodes_df1


def get_nodes_df_highlight(
    nodes_df,
    genotype_groups,
    is_prot=False,
    alphabet_type="dna",
    codon_table="Standard",
):
    # TODO: force protein to be in the table if we want to do highlight
    # protein subsequences to decouple the genetic code from plotting as
    # it is key to visualization and should not be changed afterwards
    groups_dict = {}
    if is_prot:
        if "protein" not in nodes_df.columns:
            nodes_df["protein"] = translate_seqs(
                nodes_df.index, codon_table=codon_table
            )

        for group in genotype_groups:
            mapping = [PROT_AMBIGUOUS_VALUES] * len(group)
            for seq in extend_ambigous_seq(group, mapping):
                groups_dict[seq] = group
        nodes_df["group"] = [
            groups_dict.get(x, None) for x in nodes_df["protein"]
        ]
    else:
        nodes_df["group"] = np.nan
        for group in genotype_groups:
            mapping = [AMBIGUOUS_VALUES[alphabet_type]] * len(group)
            genotype_labels = extend_ambigous_seq(group, mapping)
            nodes_df.loc[genotype_labels, "group"] = group
    nodes_df = nodes_df.dropna()
    return nodes_df


def select_edges_from_genotypes(nodes_idxs, edges):
    if isinstance(edges, pd.DataFrame):
        m = filter_csr_matrix(edges_df_to_csr_matrix(edges),
                              nodes_idxs).tocoo()
        edges = edges.iloc[m.data, :].copy()
        edges["i"] = m.row
        edges["j"] = m.col
    else:
        if isinstance(edges, coo_matrix):
            edges = edges.tocsr()
        check_error(
            isinstance(edges, csr_matrix),
            "edges must be a pd.DataFrame or sparse matrix",
        )
        edges = filter_csr_matrix(edges, nodes_idxs).tocoo()
    return edges


def get_genotypes_from_region(nodes_df, max_values={}, min_values={}):
    """
    Filter and return the genotype labels that satisfy the specified 
    conditions based on maximum and minimum values for the columns in 
    the input DataFrame.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame with genotypes as the index and various features as columns.
        Typically, it contains at least the coordinates for visualization, 
        but it may also include other metadata.

    max_values : dict, optional
        Dictionary where keys are column names and values are the maximum 
        thresholds for filtering genotypes. Genotypes with values greater 
        than these thresholds in the specified columns will be excluded.

    min_values : dict, optional
        Dictionary where keys are column names and values are the minimum 
        thresholds for filtering genotypes. Genotypes with values less than 
        these thresholds in the specified columns will be excluded.

    Returns
    -------
    pd.Index
        Index containing the labels of genotypes that meet the specified 
        filtering criteria.
    """
    sel = np.full(nodes_df.shape[0], True)

    for col, max_value in max_values.items():
        sel = sel & (nodes_df[col] < max_value)

    for col, min_value in min_values.items():
        sel = sel & (nodes_df[col] > min_value)

    return nodes_df.index[sel]


def select_genotypes(nodes_df, genotypes, edges=None, is_idx=False):
    """
    Select the specified genotypes from ``nodes_df``, along with the corresponding
    edges among the remaining genotypes if ``edges`` are provided.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        A DataFrame containing genotypes as the index and various features as columns.
        Typically, it includes at least the coordinates for visualization, but it may
        also retain other metadata.

    genotypes : array-like
        An array of genotypes to select from the input landscape. By default, it should
        contain genotype labels, or indexes if the `is_idx` option is set to True.

    edges : pd.DataFrame or scipy.sparse.csr_matrix, optional
        A DataFrame or csr_matrix representing the adjacency relationships among genotypes
        provided in `nodes_df` within the discrete space. Defaults to None.

    is_idx : bool, optional
        Indicates whether the `genotypes` argument is an array of indexes instead of an
        array of genotype labels. Defaults to False.

    Returns
    -------
    pd.DataFrame or tuple
        If `edges` is None, returns a DataFrame containing the filtered genotypes.
        Otherwise, returns a tuple containing the filtered DataFrame and the adjacency
        relationships between the selected genotypes.
    """

    size = nodes_df.shape[0]

    if edges is not None:
        nodes_df["idx"] = np.arange(size)

    if is_idx:
        nodes_df = nodes_df.iloc[genotypes, :]
    else:
        nodes_df = nodes_df.loc[genotypes, :]

    if edges is not None:
        edges = select_edges_from_genotypes(nodes_df["idx"], edges)
        return (nodes_df.drop(["idx"], axis=1), edges)
    else:
        return nodes_df


def select_path(nodes_df, path):
    """
    Returns the nodes and edges DataFrames for plotting a path defined by a 
    sequence of genotypes.

    Parameters
    ----------
    nodes_df : pd.DataFrame of shape (n_genotypes, n_features)
        DataFrame containing the genotypes from a complete sequence space as the index.
        Typically, it includes at least the coordinates for visualizing each genotype,
        but it may also retain other columns for additional metadata.

    path : array-like
        Array-like object containing the sequence of states that form a path in the
        discrete space.

    Returns
    -------
    output : (nodes_df, edges)
        A tuple containing the filtered DataFrame with the selected genotypes in the path
        and the edges that define the connections between them.
    """
    ndf = nodes_df.loc[path, :]
    ncol = ndf.shape[0]
    edf = pd.DataFrame({"i": np.arange(0, ncol - 1), "j": np.arange(1, ncol)})
    return (ndf, edf)


def select_d_neighbors(nodes_df, genotype_labels, d=1, edges=None):
    """
    Select genotypes within a specified Hamming distance from given genotype labels.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing genotypes as the index.

    genotype_labels : list of str
        List of genotype sequences to compare against.

    d : int, optional
        Maximum Hamming distance to consider. Defaults to 1.

    edges : pd.DataFrame or scipy.sparse matrix, optional
        Additional parameter to pass to the `select_genotypes` function. Defaults to None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing selected genotypes within the specified Hamming distance.
    """

    seq_matrix = np.array([[s for s in seq] for seq in nodes_df.index])
    distances = np.array(
        [
            np.vstack(
                [allele != col for allele, col in zip(gt, seq_matrix.T)]
            ).sum(0)
            for gt in genotype_labels
        ]
    )
    genotypes = (distances <= d).any(0)
    return select_genotypes(nodes_df, genotypes, edges=edges)


def select_genotypes_re(nodes_df, pattern, edges=None):
    exp = re.compile(pattern)
    genotypes = np.array(
        [True if exp.search(seq) else False for seq in nodes_df.index]
    )
    return select_genotypes(nodes_df, genotypes, edges=edges)


def select_genotypes_ambiguous_seqs(nodes_df, seqs, alphabet_type, edges=None):
    genotypes = extend_ambigous_seq(seqs, AMBIGUOUS_VALUES[alphabet_type])
    return select_genotypes(nodes_df, genotypes, edges=edges)


def guess_n_axis(nodes_df):
    i = 1
    while str(i) in nodes_df.columns:
        i += 1
    return i - 1


def select_closest_genotypes(
    nodes_df, genotype, n_genotypes, edges=None, axis=None
):
    if axis is None:
        axis = [str(x) for x in range(1, guess_n_axis(nodes_df) + 1)]
    reference_pos = nodes_df[axis].loc[genotype].values.flatten()
    sq_distance = 0
    for a, p in zip(axis, reference_pos):
        d = nodes_df[a] - p
        sq_distance += d * d

    sel_idxs = np.argsort(sq_distance)[:n_genotypes]
    return select_genotypes(nodes_df, sel_idxs, edges=edges, is_idx=True)


def select_local_optima(nodes_df, edges_df, field_function="function"):
    f1 = nodes_df[field_function].values[edges_df["i"].values]
    f2 = nodes_df[field_function].values[edges_df["j"].values]
    edf = edges_df.loc[f1 > f2, :]
    m = csr_matrix(
        (np.ones(edf.shape[0]), (edf["i"].values, edf["j"].values)),
        shape=(nodes_df.shape[0], nodes_df.shape[0]),
    ).sum(1)
    idx = np.where(m == m.max())[0]
    return select_genotypes(nodes_df, idx, edges=edges_df, is_idx=True)[0]


def marginalize_landscape_positions(
    nodes_df, keep_pos=None, skip_pos=None, return_edges=False
):
    """
    Marginalize specific positions in the sequences and compute the average of 
    numeric values across the remaining genetic backgrounds.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        A DataFrame with sequence names as the index and at least one numeric
        column to compute the average across the selected genetic backgrounds.

    keep_pos : array-like, optional
        A list of 0-indexed positions to retain. The sequences will be averaged
        across all genetic backgrounds specified by the positions not included 
        in this list. If not provided, ``skip_pos`` must be specified.

    skip_pos : array-like, optional
        A list of 0-indexed positions to marginalize out. The sequences will be
        averaged across these positions. If not provided, `keep_pos` must be specified.

    return_edges : bool, optional, default=False
        If True, returns an additional DataFrame containing the edges of the
        reduced sequence space for visualization purposes.

    Returns
    -------
    nodes_df : pd.DataFrame
        A DataFrame containing the average value of every numeric column in the
        input DataFrame, with the subsequences at the desired positions as the index.

    edges_df : pd.DataFrame, optional
        A DataFrame containing the edges of the reduced sequence space. This is
        only returned if ``return_edges=True``.
    """

    # Select only numeric fields for averaging
    df = nodes_df.select_dtypes(include="number")

    # Check errors
    msg = "Positions to keep or marginalize out must be provided"
    check_error(keep_pos is not None or skip_pos is not None, msg=msg)

    msg = "Specify only positions to keep or avoid"
    check_error(keep_pos is None or skip_pos is None, msg=msg)

    # Select positions to keep: assumes constant sequence length
    if keep_pos is None:
        seq_length = len(df.index[0])
        keep_pos = [i for i in range(seq_length) if i not in skip_pos]

    df["keep_seq"] = ["".join([x[i] for i in keep_pos]) for x in df.index]
    out = df.groupby(["keep_seq"]).mean()

    if return_edges:
        space = SequenceSpace(X=out.index.values, y=np.ones(out.shape[0]))
        out = out, space.get_edges_df()
    return out
