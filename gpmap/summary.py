from itertools import chain, combinations, product
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from gpmap.linop import (
    CovarianceDistanceOperator,
    CovarianceSitesOperator,
    DeltaPOperator,
    DeltaUDOperator,
    DeltaUOperator,
    ProjectionOperator,
    VUProjectionOperator,
)
from gpmap.matrix import kron, quad, reciprocal
from gpmap.seq import SequenceSpaceRelatedObject, get_product_states


class GPmapSummarizer:
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
        """
        Annotate a variance component table with percentage columns.

        Parameters
        ----------
        vcs : pd.DataFrame
            Table with a ``variance`` column that records the contribution of
            each component (otherwise ``KeyError`` is raised).

        Returns
        -------
        pd.DataFrame
            The same table with ``variance_perc`` and ``variance_perc_cum``
            added in place (summing to 100 by construction).
        """
        vcs["variance_perc"] = 100 * vcs["variance"] / vcs["variance"].sum()
        vcs["variance_perc_cum"] = np.cumsum(vcs["variance_perc"])
        return vcs

    def calc_root_mean_squared_epistatic_coeff(self, P=2):
        """
        Compute root mean squared epistatic coefficient of order `P`
        across all possible combinations of P mutations in the complete
        genotype-phenotype map.

        Parameters
        ----------
        P : int
            The order of local epistatic coefficients to compute e.g. P=1
            reflects mutational effects, P=2 epistatic coefficients, etc.

        Returns
        -------
        rmsec : float
            Root mean squared epistatic coefficient of order `P`
        """
        Delta = DeltaPOperator(self.n_alleles, self.seq_length, P)
        rmsec = np.sqrt(quad(Delta, self.f) / Delta.n_p_faces)
        return rmsec

    def calc_U_root_mean_squared_epistatic_coeffs(self, P=2):
        """
        Compute root mean squared P-way epistatic coefficient
        for each possible all possible combinations of P mutations
        in the complete genotype-phenotype map.

        Parameters
        ----------
        P : int
            The order of local epistatic coefficients to compute e.g. P=1
            reflects mutational effects, P=2 epistatic coefficients, etc.

        Returns
        -------
        rmsec : pd.DataFrame
            Root mean squared epistatic coefficient of order for each
            combination of sites U.
        """
        rmsecs = []
        for U in combinations(self.positions, P):
            asec = self.calc_U_avg_squared_epistatic_coeff(U)
            rmsec = np.sqrt(asec)
            rmsecs.append(list(U) + [rmsec])
        columns = [f"site{i + 1}" for i in range(P)] + ["rmsec"]
        rmsecs = pd.DataFrame(rmsecs, columns=columns)
        return rmsecs

    def calc_V_k_variance_components(self):
        """
        Compute variance components contributed by interactions of each order k.

        Calculates the total variance in the phenotype vector `f` explained by
        genetic interactions of order k for k = 1..seq_length. For each k this
        method projects `f` onto the corresponding subspace using
        ProjectionOperator and computes its norm.


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
        vcs = []
        for k in np.arange(1, self.seq_length + 1):
            P_k = ProjectionOperator(self.n_alleles, self.seq_length, k=k)
            vcs.append({"k": k, "variance": quad(P_k, self.f)})
        vcs = pd.DataFrame(vcs)
        vcs = self.calc_variance_perc(vcs)
        return vcs

    def calc_V_U_variance_components(self):
        """
        Compute variance components contributed by interactions between every
        possible subset of sites U.

        Calculates the total variance in the phenotype vector `f` explained by
        genetic interactions involving all subsets of sites U. For each U this
        method projects `f` onto the corresponding subspace using
        VUProjectionOperator and computes its norm.

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
        V_U_vcs = []
        for k in range(1, 10):
            for U in combinations(self.positions, k):
                P_U = VUProjectionOperator(self.n_alleles, self.seq_length, U)
                V_U_vcs.append({"U": set(U), "k": k, "variance": quad(P_U, self.f)})
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
            Table where the rows index interaction order (1..seq_length) and the
            columns index each site position. Each entry reports the percentage
            of the total variance explained by components of order ``k`` that
            involve site ``p``.

        Raises
        ------
        ValueError
            If ``V_U_vcs`` references sites outside ``self.positions``.

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
        of at least order ``min_k`` involving every possible pair of sites
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
        min_k : int, optional
            Minimum interaction order to include. Defaults to 2. Must satisfy
            1 <= ``min_k`` <= ``self.seq_length``.

        Returns
        -------
        vcs_perc : pd.DataFrame
            Table with columns ``site1``, ``site2``, ``variance``, and
            ``variance_perc`` that reports the percentage variance contributed
            by interactions of order >= ``min_k`` for each site pair.

        Raises
        ------
        ValueError
            If ``min_k`` is outside the range 1..``self.seq_length`` or if
            ``V_U_vcs`` references unexpected sites.

        Notes
        -----
        Percentages are scaled so that the sum of ``variance_perc`` is 100.
        """
        self.validate_U(V_U_vcs)
        if min_k < 1 or min_k > self.seq_length:
            msg = f"min_k={min_k} should be between 1 and {self.seq_length}"
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

    def calc_U_avg_squared_epistatic_coeff(self, U):
        """
        Compute the average of squared epistatic coefficients at sites U
        across all genetic backgrounds.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.

        Returns
        -------
        float
            The average of squared epistatic coefficients at sites U.
        """
        DeltaU = DeltaUOperator(self.n_alleles, self.seq_length, U)
        avg_squared_epistatic_coeff = quad(DeltaU, self.f) / DeltaU.n_U_faces
        return avg_squared_epistatic_coeff

    def calc_U_avg_squared_mean_epistatic_coeff(self, U):
        """
        Compute the average of squared mean epistatic coefficients at sites U
        across all genetic backgrounds.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.

        Returns
        -------
        float
            The average of squared mean epistatic coefficients at sites U.
        """
        k = len(U)
        a = self.n_alleles
        l = self.seq_length
        c = 2**k / (a ** (l - k) * (a - 1) ** k)
        W_U = VUProjectionOperator(self.n_alleles, self.seq_length, U)
        avg_sq_mean_epistatic_coeff = c * quad(W_U, self.f)
        return avg_sq_mean_epistatic_coeff

    def calc_Gamma_U_D(self, U, D):
        """
        Compute the Gamma statistic measuring the predictability of
        epistatic coefficients between sites U across genetic backgrounds
        differing at sites D.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.
        D : int
            Subset of sites at which genetic backgrounds differ.

        Returns
        -------
        float
            The Gamma statistic for the specified subset of sites and distance.
        """
        DeltaUD = DeltaUDOperator(self.n_alleles, self.seq_length, U, D)
        Gamma_U_D = DeltaUD.c * quad(DeltaUD, self.f)
        return Gamma_U_D

    def calc_gamma_U_D(self, U, D):
        """
        Compute the gamma statistic measuring the predictability of
        epistatic coefficients between sites U across genetic backgrounds
        differing at sites D normalized by the variance in epistatic
        coefficients at sites U.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.
        D : int
            Subset of sites at which genetic backgrounds differ.

        Returns
        -------
        float
            The gamma statistic for the specified subset of sites and distance.
        """
        Gamma_U_D = self.calc_Gamma_U_D(U, D)
        Gamma_U_D0 = self.calc_Gamma_U_D(U, [])
        gamma_U_D = Gamma_U_D / Gamma_U_D0
        return gamma_U_D

    def calc_covariance_U_D(self, U, D):
        """
        Compute the covariance in epistatic coefficients at sites U
        across genetic backgrounds differing at sites D.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.
        D : int
            Subset of sites at which genetic backgrounds differ.

        Returns
        -------
        float
            The covariance in epistatic coefficients for the specified
            subset of sites and distance.
        """
        Gamma_U_D = self.calc_Gamma_U_D(U, D)
        asmec = self.calc_U_avg_squared_mean_epistatic_coeff(U)
        covariance_U_D = Gamma_U_D - asmec
        return covariance_U_D

    def calc_correlation_U_D(self, U, D):
        """
        Compute the correlation in epistatic coefficients at sites U
        across genetic backgrounds differing at sites D.

        Parameters
        ----------
        U : tuple of int
            Subset of sites to consider epistatic coefficients.
        D : int
            Subset of sites at which genetic backgrounds differ.

        Returns
        -------
        float
            The correlation in epistatic coefficients for the specified
            subset of sites and distance.
        """
        covariance_U_D = self.calc_covariance_U_D(U, D)
        covariance_U_D0 = self.calc_covariance_U_D(U, [])
        correlation_U_D = covariance_U_D / covariance_U_D0
        return correlation_U_D

    def calc_gamma_i_to_j(self):
        """
        Compute the gamma statistic measuring the predictability of
        mutational effects at site i across genetic backgrounds
        differing at site j.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'site_i', 'site_j', and 'gamma'
            containing the gamma statistics for all pairs of sites.
        """
        gammas = []
        for i, j in combinations(self.positions, 2):
            gamma_ij = self.calc_gamma_U_D([i], [j])
            gammas.append({"site_i": i, "site_j": j, "gamma": gamma_ij})

            gamma_ji = self.calc_gamma_U_D([j], [i])
            gammas.append({"site_i": j, "site_j": i, "gamma": gamma_ji})

        return pd.DataFrame(gammas)


class GPDataSummarizer(SequenceSpaceRelatedObject):
    """
    Class for computing low-level descriptors of genotype-phenotype data
    (observed experimental data sampled from the full sequence space).

    This class extends SequenceSpaceRelatedObject and provides convenience
    routines to store observed data and compute covariance and local
    epistatic summaries using operators defined for the full sequence
    space. Unlike GPmapSummarizer which operates on a complete genotype-
    phenotype map, GPDataSummarizer works with a (possibly sparse)
    dataset of genotypes and corresponding phenotypes.

    Parameters
    ----------
    seq_length : int, optional
        Number of sites in the sequence (sequence length). Required unless
        provided via genotypes.
    alphabet : list of str, optional
        Alphabet for each site (list of characters). Required unless provided
        via genotypes or inferred.
    alphabet_type : str, default "custom"
        Type of alphabet (keeps compatibility with SequenceSpaceRelatedObject).
    genotypes : np.ndarray, optional
        Array of genotype strings (one per observation). If provided, seq_length
        and alphabet may be inferred from these.
    X : np.ndarray, optional
        Design / indicator matrix mapping observed genotypes to the full
        sequence-space basis. Shape (n_obs, n_full_genotypes).
    y : np.ndarray, optional
        Observed phenotype values corresponding to rows of X. Shape (n_obs,).
    y_var : np.ndarray, optional
        Measurement variances for each observation. If None, zeros are used.

    """

    def __init__(
        self,
        seq_length: Optional[int] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_type: str = "custom",
        genotypes: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        y_var: Optional[np.ndarray] = None,
    ):
        super().__init__(
            seq_length=seq_length,
            alphabet=alphabet,
            alphabet_type=alphabet_type,
            genotypes=genotypes,
        )
        if X is not None and y is not None:
            self.set_data(X, y, y_var=y_var)

    def set_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_var: Optional[np.ndarray] = None,
    ):
        """
        Store the observed genotypes and phenotypes on the summarizer.

        Parameters
        ----------
        X : np.ndarray
            Design matrix mapping each observation to the full sequence-space
            basis. Shape ``(n_obs, n_full_genotypes)``.
        y : np.ndarray
            Phenotype values for each observation. Shape ``(n_obs,)``.
        y_var : np.ndarray, optional
            Measurement variances for each observation. If ``None``, zeros are
            used internally; otherwise must have shape ``(n_obs,)``.

        Raises
        ------
        ValueError
            If ``X``/``y`` or ``y``/``y_var`` shapes mismatch.
        """
        if X.shape != y.shape:
            msg = f"X and y should have the same shape. Got {X.shape} and {y.shape}"
            raise ValueError(msg)
        if y_var is not None and X.shape != y_var.shape:
            msg = f"y and y_var should have the same shape. Got {y.shape} and {y_var.shape}"
            raise ValueError(msg)
        if y_var is None:
            y_var = np.zeros_like(y)

        self.X = X
        self.X_op = self.get_X_operator(X)
        self.y = y
        self.n_obs = y.shape[0]
        self.y_centered = y - y.mean()
        self.y_var = y_var
        self.y_var_mean = y_var.mean()
        self.y_mean_sq = y.mean() ** 2

    def calc_covariance_distance(self, centered: bool = False):
        """
        Compute empirical auto-covariance function depending on the Hamming
        distance between pairs of genotypes.

        Parameters
        ----------
        centered : bool, optional
            If True, return covariances computed on centered phenotypes (y - mean)
            and do not add back the phenotype mean square. If False (default),
            the phenotype mean square is added to produce raw (uncentered)
            second-moment estimates.

        Returns
        -------
        cov : np.ndarray, shape (seq_length + 1,)
            Covariance (or mean product) estimates for each Hamming distance
            d = 0..seq_length. Note: cov[0] is adjusted to remove the mean
            measurement variance (self.y_var_mean).
        ns : np.ndarray, shape (seq_length + 1,)
            Number of pairs of sequences at each distance class.
        """
        cov, ns = np.zeros(self.seq_length + 1), np.zeros(self.seq_length + 1)
        ones = np.ones(self.n_obs)
        for d in range(self.seq_length + 1):
            P = CovarianceDistanceOperator(
                self.n_alleles, self.seq_length, distance=d
            )
            A = self.X_op @ P @ self.X_op.transpose()
            sum_cov = quad(A, self.y_centered)
            ns[d] = quad(A, ones)
            cov_d = reciprocal(sum_cov, ns[d])
            if d == 0:
                cov_d -= self.y_var_mean

            cov[d] = cov_d

        if not centered:
            cov += self.y_mean_sq

        return (cov, ns)

    def calc_covariance_U_sites(self, centered: bool = False):
        """
        Compute empirical auto-covariance function depending on the combination
        of subsets U at which pairs of genotypes differ.

        Parameters
        ----------
        centered : bool, optional
            If True, compute covariances using centered phenotypes (y - mean)
            and do not add back the phenotype mean square. If False (default),
            add the phenotype mean square to produce raw (uncentered)
            second-moment estimates.

        Returns
        -------
        cov : np.ndarray, shape (2 ** self.seq_length,)
            Covariance (or mean product) estimates for each subset ``U`` in
            the same order returned by ``self.get_Us()``.
        ns : np.ndarray, shape (2 ** self.seq_length,)
            Number of observed genotype pairs that differ exactly on the sites
            specified by each subset ``U``.

        """
        cov, ns = [], []
        ones = np.ones(self.n_obs)

        for U in self.get_Us():
            U_sites = tuple(p for p, s in zip(self.positions, U) if s)
            P = CovarianceSitesOperator(
                self.n_alleles, self.seq_length, sites=U_sites
            )
            A = self.X_op @ P @ self.X_op.transpose()
            sum_cov = quad(A, self.y_centered)
            n = quad(A, ones)
            cov_U = reciprocal(sum_cov, n)
            if not U_sites:
                cov_U -= self.y_var_mean

            cov.append(cov_U)
            ns.append(n)

        cov, ns = np.array(cov), np.array(ns)
        if not centered:
            cov += self.y_mean_sq

        return (cov, ns)

    def calc_avg_squared_local_epistatic_coeff(self, P: int):
        """
        Compute the empirical average of squared local ``P``-way epistatic
        contrasts across the observed sequences.

        Parameters
        ----------
        P : int
            Order of the local epistatic coefficient to compute (number of sites
            in each local cubic hypercube / interaction order). Must be between
            1 and ``self.seq_length``.

        Returns
        -------
        float
            Average squared contrast aggregated over every fully observed local
            ``2**P`` genotype hypercube (faces) encountered in the data. Faces
            with any missing genotype are skipped.

        Raises
        ------
        ValueError
            If ``P`` is outside the interval ``[1, self.seq_length]``.
        ZeroDivisionError
            When no fully observed faces exist, causing ``n`` to stay zero.
        Notes
        -----
        The average is computed by enumerating all background assignments for
        the remaining ``self.seq_length - P`` sites, iterating over ordered
        allele pairs per target site, and applying the contrast operator built
        from ``kron([[-1, 1]] * P)``.
        """
        if P < 1 or P > self.seq_length:
            msg = (
                f"P must be between 1 and sequence length of {self.seq_length}"
            )
            raise ValueError(msg)

        v = dict(zip(self.X, self.y))

        background_seqs = list(
            product(self.alphabet, repeat=self.seq_length - P)
        )
        allele_pairs = list(combinations(self.alphabet, 2))
        allele_pairs_combs = list(product(allele_pairs, repeat=P))
        z = kron([[-1, 1]] * P)

        s, n = 0, 0
        sites_sets = combinations(self.positions, P)
        n_sets = comb(self.seq_length, P)
        for target_sites in tqdm(sites_sets, total=n_sets):
            background_sites = [
                s for s in self.positions if s not in target_sites
            ]
            for background_seq in background_seqs:
                bc = dict(zip(background_sites, background_seq))
                for pairs in allele_pairs_combs:
                    seqs = []
                    allele_combs = list(get_product_states(pairs))

                    for allele_comb in allele_combs:
                        seq = bc.copy()
                        seq.update(dict(zip(target_sites, allele_comb)))
                        seqs.append("".join([seq[i] for i in self.positions]))
                    try:
                        u = np.array([v[s] for s in seqs])
                    except KeyError:
                        continue
                    s += np.dot(u, z) ** 2
                    n += 1
        return s / n
