#!/usr/bin/env python
from itertools import combinations

import pandas as pd
import numpy as np

from itertools import chain
from gpmap.linop import (
    ProjectionOperator,
    VjProjectionOperator,
    DeltaPOperator,
)
from gpmap.matrix import quad


class GPmapSummarizer(object):
    """
    Class for computing low-level descriptors of a complete genotype-phenotype
    map.

    Parameters
    ----------
    n_alleles : int
        Number of alleles per site.
    seq_length : int
        Number of sites in the sequence (sequence length).
    f : array-like, optional
        Phenotype values for every possible genotype, ordered lexicographically.
        If None, the phenotype vector can be provided later when calling
        instance methods.

    """

    def __init__(self, n_alleles: int, seq_length: int, f=None):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.positions = np.arange(seq_length)
        self.n = n_alleles**seq_length
        if f is not None:
            self.set_f(f)

    def validate_f(self, f):
        if f.shape[0] != self.n:
            raise ValueError(f"f must be of size {self.n} but is {f.shape[0]}")
        
    def set_f(self, f):
        self.validate_f(f)
        self.f = f
        self.total_variance = np.var(f)

    def get_f(self, f=None):
        if f is None and self.f is None:
            raise ValueError("f must be provided if not stored in the object")
        elif f is None:
            f = self.f
        else:
            self.validate_f(f)
        return f

    def calc_variance_perc(self, vcs):
        vcs["variance_perc"] = 100 * vcs["variance"] / vcs["variance"].sum()
        vcs["variance_perc_cum"] = np.cumsum(vcs["variance_perc"])
        return vcs
    
    def calc_root_mean_squared_epistatic_coeff(self, P=2, f=None):
        '''
        Compute root mean squared epistatic coefficient of order `P`
        across all possible combinations of P mutations in the complete
        genotype-phenotype map.

        Parameters
        ----------
        P : int
            The order of local epistatic coefficients to compute e.g. P=1
            reflects mutational effects, P=2 epistatic coefficients, etc.
            
        f : array-like, optional
            Phenotype values for every genotype in lexicographic order.
            If None, the instance attribute `self.f` is used. If both are None,
            a ValueError is raised.

        Returns
        -------
        rmsec : float
            Root mean squared epistatic coefficient of order `P`
        '''
        f = self.get_f(f)
        Delta = DeltaPOperator(self.n_alleles, self.seq_length, P)
        rmsec = np.sqrt(quad(Delta, f) / Delta.n_p_faces)
        return rmsec

    def calc_V_k_variance_components(self, f=None):
        """
        Compute variance components contributed by interactions of each order k.

        Calculates the total variance in the phenotype vector `f` explained by
        genetic interactions of order k for k = 1..seq_length. For each k this
        method projects `f` onto the corresponding subspace using
        ProjectionOperator and computes its norm.

        Parameters
        ----------
        f : array-like, optional
            Phenotype values for every genotype in lexicographic order.
            If None, the instance attribute `self.f` is used. If both are None,
            a ValueError is raised.

        Returns
        -------
        V_k_vcs : pd.DataFrame
            DataFrame with shape (seq_length, 4) and columns:

            - ``k``: interaction order (1..seq_length)
            - ``variance``: total variance explained by order k
            - ``variance_perc``: percentage of total variance explained by k
            - ``variance_perc_cum``: cumulative percentage up to and including k

        Notes
        -----
        Percentages are scaled so that the sum of ``variance_perc`` is 100.
        """
        f = self.get_f(f)

        vcs = []
        for k in np.arange(1, self.seq_length + 1):
            P_k = ProjectionOperator(self.n_alleles, self.seq_length, k=k)
            vcs.append({"k": k, "variance": quad(P_k, f)})
        vcs = pd.DataFrame(vcs)
        vcs = self.calc_variance_perc(vcs)
        return vcs

    def calc_V_U_variance_components(self, f=None):
        """
        Compute variance components contributed by interactions between every
        possible subset of sites U.

        Calculates the total variance in the phenotype vector `f` explained by
        genetic interactions involving all subsets of sites U. For each U this
        method projects `f` onto the corresponding subspace using
        VjProjectionOperator and computes its norm.

        Parameters
        ----------
        f : array-like, optional
            Phenotype values for every genotype in lexicographic order.
            If None, the instance attribute `self.f` is used. If both are None,
            a ValueError is raised.

        Returns
        -------
        V_U_vcs : pd.DataFrame
            DataFrame with shape (seq_length, 5) and columns:

            - ``U``: subset of sites
            - ``k``: interaction order (1..seq_length)
            - ``variance``: total variance explained by order k
            - ``variance_perc``: percentage of total variance explained by k
            - ``variance_perc_cum``: cumulative percentage up to and including k

        Notes
        -----
        Percentages are scaled so that the sum of ``variance_perc`` is 100.
        """
        f = self.get_f(f)

        V_U_vcs = []
        for k in range(1, 10):
            for U in combinations(self.positions, k):
                P_U = VjProjectionOperator(self.n_alleles, self.seq_length, U)
                V_U_vcs.append({"U": set(U), "k": k, "variance": quad(P_U, f)})
        V_U_vcs = pd.DataFrame(V_U_vcs)
        V_U_vcs = self.calc_variance_perc(V_U_vcs)
        return V_U_vcs

    def validate_U(self, V_U_vcs):
        sites_U = set(chain(*V_U_vcs["U"]))
        if sites_U > set(self.positions):
            raise ValueError(f"Unexpected sites in U {sites_U}")

    def calc_sites_variance_perc(self, V_U_vcs):
        """
        Compute the percentage variance explained by genetic interactions
        of every possible order involving every possible site from previously
        computed V_U variance components.

        Parameters
        ----------
        V_U_vcs : pd.DataFrame
            DataFrame with shape (seq_length, 5) and columns:

            - ``U``: subset of sites
            - ``k``: interaction order (1..seq_length)
            - ``variance``: total variance explained by order k
            - ``variance_perc``: percentage of total variance explained by k
            - ``variance_perc_cum``: cumulative percentage up to and including k

            This DataFrame is the output of ``calc_V_U_variance_components``.

        Returns
        -------
        vcs_perc : pd.DataFrame of shape (seq_length, seq_length)
            Table with the percentage variance explain by genetic interactions
            of order k (rows) involving sites p (columns).

        Notes
        -----
        Percentages are scaled so that the sum of ``variance_perc`` is 100.
        """
        self.validate_U(V_U_vcs)
        total_variance = V_U_vcs["variance"].sum()
        vcs = []
        ks = range(1, self.seq_length + 1)
        for k in ks:
            k_vcs = V_U_vcs.loc[V_U_vcs["k"] == k, :]
            row = []
            for p in self.positions:
                idx = [p in U for U in k_vcs["U"]]
                row.append(k_vcs.loc[idx, "variance"].sum())
            vcs.append(row)
        vcs = pd.DataFrame(vcs, index=ks, columns=self.positions)
        vcs_perc = 100 * vcs / total_variance
        return vcs_perc

    def calc_site_pairs_variance_perc(self, V_U_vcs, min_k=2):
        """
        Compute the percentage variance explained by genetic interactions
        of at least order `min_k` involving every possible pair of sites
        from previously computed V_U variance components.

        Parameters
        ----------
        V_U_vcs : pd.DataFrame
            DataFrame with shape (seq_length, 5) and columns:

            - ``U``: subset of sites
            - ``k``: interaction order (1..seq_length)
            - ``variance``: total variance explained by order k
            - ``variance_perc``: percentage of total variance explained by k
            - ``variance_perc_cum``: cumulative percentage up to and including k

            This DataFrame is the output of ``calc_V_U_variance_components``.

        Returns
        -------
        vcs_perc : pd.DataFrame
            Table with the percentage variance explain by genetic interactions
            of at least order `max_k` involving each possible pair of sites.

        Notes
        -----
        Percentages are scaled so that the sum of ``variance_perc`` is 100.
        """
        self.validate_U(V_U_vcs)
        if min_k < 1 or min_k > self.seq_length:
            msg = f"min_k={V_U_vcs} should be between 2 and {self.seq_length}"
            raise ValueError(msg)

        V_U_vcs_min_k = V_U_vcs.loc[V_U_vcs["k"] >= min_k, :]
        total_variance = V_U_vcs_min_k["variance"].sum()
        vcs = []
        for p, q in combinations(self.positions, 2):
            idx = [p in U and q in U for U in V_U_vcs_min_k["U"]]
            v = V_U_vcs_min_k.loc[idx, "variance"].sum()
            vcs.append({"site1": p, "site2": q, "variance": v})
        vcs = pd.DataFrame(vcs)
        vcs["variance_perc"] = 100 * vcs["variance"] / total_variance
        return vcs
