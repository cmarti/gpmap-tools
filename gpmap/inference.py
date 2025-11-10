#!/usr/bin/env python
from time import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, pearsonr
from itertools import product, combinations

from gpmap.aligner import VCKernelAligner
from gpmap.seq import get_product_states
from gpmap.matrix import quad, kron, reciprocal
from gpmap.gp import (
    GaussianProcessRegressor,
    GeneralizedGaussianProcessRegressor,
    MinimizerRegressor,
)
from gpmap.likelihood import SeqDEFTLikelihood
from gpmap.linop import (
    DeltaKernelBasisOperator,
    DeltaKernelRegularizerOperator,
    DeltaPOperator,
    DiagonalOperator,
    ProjectionOperator,
    VarianceComponentKernel,
    CovarianceDistanceOperator,
    CovarianceVjOperator,
)
from gpmap.utils import (
    check_error,
    get_cv_iter,
    get_CV_splits,
)


def calc_avg_local_epistatic_coeff(X, y, alphabet, seq_length, P):
    sites = np.arange(seq_length)
    v = dict(zip(X, y))

    background_seqs = list(product(alphabet, repeat=seq_length - P))
    allele_pairs = list(combinations(alphabet, 2))
    allele_pairs_combs = list(product(allele_pairs, repeat=P))
    z = kron([[-1, 1]] * P)

    s, n = 0, 0
    sites_sets = list(combinations(sites, P))
    for target_sites in tqdm(sites_sets):
        background_sites = [s for s in sites if s not in target_sites]
        for background_seq in background_seqs:
            bc = dict(zip(background_sites, background_seq))
            for pairs in allele_pairs_combs:
                seqs = []
                allele_combs = list(get_product_states(pairs))

                for allele_comb in allele_combs:
                    seq = bc.copy()
                    seq.update(dict(zip(target_sites, allele_comb)))
                    seqs.append("".join([seq[i] for i in sites]))
                try:
                    u = np.array([v[s] for s in seqs])
                except KeyError:
                    continue
                s += np.dot(u, z) ** 2
                n += 1
    return (s, n)


def _get_seq_values_and_obs_seqs(y, n_alleles, seq_length, idx=None):
    n = n_alleles**seq_length
    if idx is not None:
        seq_values, observed_seqs = np.zeros(n), np.zeros(n)
        seq_values[idx], observed_seqs[idx] = y, 1.0
    else:
        seq_values, observed_seqs = y, np.ones(n, dtype=float)
    return (seq_values, observed_seqs)


def calc_covariance_distance(y, n_alleles, seq_length, idx=None):
    seq_values, obs_seqs = _get_seq_values_and_obs_seqs(
        y, n_alleles, seq_length, idx=idx
    )

    cov, ns = np.zeros(seq_length + 1), np.zeros(seq_length + 1)
    for d in range(seq_length + 1):
        P = CovarianceDistanceOperator(n_alleles, seq_length, distance=d)
        Pquad = quad(P, seq_values)
        ns[d] = quad(P, obs_seqs)
        cov[d] = reciprocal(Pquad, ns[d])
    return (cov, ns)


def calc_covariance_vjs(y, n_alleles, seq_length, idx=None):
    lp1 = seq_length + 1
    seq_values, obs_seqs = _get_seq_values_and_obs_seqs(
        y, n_alleles, seq_length, idx=idx
    )

    cov, ns = [], []
    sites = np.arange(seq_length)
    sites_matrix = []
    for k in range(lp1):
        for j in combinations(sites, k):
            P = CovarianceVjOperator(n_alleles, seq_length, j=j)
            Pquad = quad(P, seq_values)
            nj = quad(P, obs_seqs)
            z = np.array([i not in j for i in range(seq_length)], dtype=float)

            cov.append(reciprocal(Pquad, nj))
            ns.append(nj)
            sites_matrix.append(z)

    sites_matrix = np.array(sites_matrix)
    cov, ns = np.array(cov), np.array(ns)
    return (cov, ns, sites_matrix)


class _DeltaPpriorGP(object):
    def get_C_lambdas(self):
        a = self.a
        msg = "a needs to be defined"
        check_error(a is not None, msg=msg)
        self.DP.calc_lambdas()
        lambdas = np.zeros(self.DP.lambdas.shape)
        lambdas[self.DP.P :] = (
            self.DP.lambdas[self.DP.P :] * a / self.DP.n_p_faces
        )

        if hasattr(self, "lambdas_P_inv") and self.lambdas_P_inv is not None:
            lambdas[: self.DP.P] = 1 / self.lambdas_P_inv

        return lambdas

    def get_K_sqrt(self):
        lambdas_inv = self.get_C_lambdas()
        lambdas = np.zeros_like(lambdas_inv)
        idx = lambdas_inv != 0
        lambdas[idx] = 1 / lambdas_inv[idx]
        W_sqrt = ProjectionOperator(
            self.n_alleles, self.seq_length, lambdas=lambdas
        ).matrix_sqrt()
        return W_sqrt


class MinimumEpistasisInterpolator(MinimizerRegressor, _DeltaPpriorGP):
    """
    Mininum epistasis interpolation model for sequence-function relationships.

    A class for performing Minimum Epistasis Interpolation (MEI) to infer
    complete genotype-phenotype maps from incomplete and noisy data. This
    model applies a prior that penalizes local epistatic coefficients of
    order P and infers the posterior distribution based on experimental
    data for a subset of sequences.

    Parameters
    ----------
    n_alleles : int, optional
        The number of alleles per site. If not provided, it will be inferred
        from the provided data.

    seq_length : int, optional
        The length of the genotype sequences. If not provided, it will be
        inferred from the provided data.

    genotypes : array-like, optional
        A list or array of genotypes to be used in the interpolation. If not
        provided, the model will infer the genotype space.

    alphabet_type : str, optional
        The type of alphabet used for genotypes. Default is "custom".

    P : int, optional
        The order of epistasis to consider. Default is 2. This determines the
        level of interaction between genetic sites that is penalized.

    a : float, optional
        The regularization parameter. If not provided, it will be inferred
        during the fitting process to best match the observed data.

    cg_rtol : float, optional
        The relative tolerance for the conjugate gradient solver. Default is
        1e-16. This controls the precision of the solver used in computations.
    """

    def __init__(
        self,
        n_alleles=None,
        seq_length=None,
        genotypes=None,
        alphabet_type="custom",
        P=2,
        a=None,
        cg_rtol=1e-16,
    ):
        super().__init__(
            seq_length=seq_length,
            n_alleles=n_alleles,
            genotypes=genotypes,
            alphabet_type=alphabet_type,
            cg_rtol=cg_rtol,
        )
        self.DP = DeltaPOperator(self.n_alleles, self.seq_length, P)
        self.s = self.DP.n_p_faces
        self.p = self.DP.n_p_faces_genotype
        self.initialized = True
        self.set_a(a)

    def set_a(self, a):
        self.a = a
        if a is not None:
            self.C = self.a / self.s * self.DP
        else:
            self.C = 1 / self.s * self.DP

    def smooth(self, y_pred):
        y_pred -= 1 / self.p * self.DP @ y_pred
        return y_pred

    def calc_posterior_covariance(self):
        if self.a is None:
            msg = "a must be defined to compute posterior covariance"
            raise ValueError(msg)
        else:
            return super().calc_posterior_covariance()

    def check_unique_solution(self):
        basis = self.DP.calc_kernel_basis()
        r1 = basis.rank
        r2 = np.linalg.matrix_rank(
            self.likelihood.Xop @ basis @ np.eye(basis.shape[1])
        )
        if r2 < r1:
            msg = "Minimum epistasis interpolation does not have a unique solution"
            raise ValueError(msg)

    def calc_posterior(self, X_pred=None, B=None):
        self.check_unique_solution()
        mean_post = self.calc_posterior_mean()
        if self.a is None:
            Sigma_post = None
        else:
            Sigma_post = self.calc_posterior_covariance()
        return self.transform_posterior(
            mean_post, Sigma_post, X_pred=X_pred, B=B
        )

    def fit(self, X, y, y_var=None):
        """
        Fits the Minimum Epistasis Interpolation (MEI) model hyperparameter
        to the provided data.

        This method infers the optimal regularization parameter `a` by computing
        the Minimum Epistasis Interpolation solution. It determines the value of `a`
        such that the expected average squared Pth epistatic coefficients match
        those of the MEI solution.

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Array containing the genotypes for which observations are provided
            in `y`.

        y : array-like of shape (n_obs,)
            Array containing the observed phenotypes corresponding to the
            genotypes in `X`.

        y_var : array-like of shape (n_obs,), optional
            Array containing the empirical or experimental variance for the
            measurements in `y`. If not provided, it is assumed to be uniform
            or unknown.
        """
        self.set_data(X, y)
        mean = self.calc_posterior_mean()
        a_star = self.DP.rank * self.s / quad(self.DP, mean)
        self.set_data(X, y, y_var=y_var)
        self.set_a(a_star)


class VCregression(GaussianProcessRegressor):
    """
    Variance Component regression model for sequence-function relationships.

    This model enables the inference and prediction of a scalar function in
    sequence spaces under a Gaussian Process prior. The prior is parameterized
    by the contribution of different orders of interaction to the observed
    genetic variability of a continuous phenotype.

    Parameters
    ----------
    n_alleles : int, optional
        The number of alleles per site. If not provided, it will be inferred
        from the data.

    seq_length : int, optional
        The length of the genotype sequences. If not provided, it will be
        inferred from the data.

    genotypes : array-like, optional
        A list or array of genotypes to be used in the interpolation.

    alphabet_type : str, optional
        The type of alphabet used for genotypes. Default is "custom".

    lambdas : array-like, optional
        Variance components for each order of interaction. If not provided,
        they will be inferred during fitting.

    beta : float, optional
        The regularization parameter for the kernel alignment. Default is 0.

    cross_validation : bool, optional
        Whether to perform cross-validation to select the best penalization
        constant for regularized variance component inference. Default is False.

    nfolds : int, optional
        The number of folds for cross-validation. Default is 5.

    cv_loss_function : str, optional
        The loss function to use during cross-validation. Options are
        "frobenius_norm", "logL", or "r2". Default is "frobenius_norm".

    num_beta : int, optional
        The number of beta values to evaluate during cross-validation. Default is 20.

    min_log_beta : float, optional
        The minimum log10(beta) value for cross-validation. Default is -2.

    max_log_beta : float, optional
        The maximum log10(beta) value for cross-validation. Default is 7.

    cg_rtol : float, optional
        The relative tolerance for the conjugate gradient solver. Default is 1e-16.

    progress : bool, optional
        Whether to display progress bars during fitting. Default is True.
    """

    def __init__(
        self,
        n_alleles=None,
        seq_length=None,
        genotypes=None,
        alphabet_type="custom",
        lambdas=None,
        beta=0,
        cross_validation=False,
        nfolds=5,
        cv_loss_function="frobenius_norm",
        num_beta=20,
        min_log_beta=-2,
        max_log_beta=7,
        cg_rtol=1e-16,
        progress=True,
    ):
        self.progress = progress
        self.beta = beta
        self.nfolds = nfolds
        self.num_reg = num_beta
        self.total_folds = self.nfolds * self.num_reg

        self.min_log_reg = min_log_beta
        self.max_log_reg = max_log_beta
        self.run_cv = cross_validation
        self.set_cv_loss_function(cv_loss_function)

        self.define_space(
            n_alleles=n_alleles,
            seq_length=seq_length,
            genotypes=genotypes,
            alphabet_type=alphabet_type,
        )

        if lambdas is not None:
            self.set_lambdas(lambdas)

        self.cg_rtol = cg_rtol

    def set_lambdas(self, lambdas=None, k=None):
        K = VarianceComponentKernel(
            self.n_alleles, self.seq_length, lambdas=lambdas, k=k
        )
        self.lambdas = K.lambdas
        super().__init__(base_kernel=K, progress=self.progress)

    def set_data(self, X, y, y_var=None, cov=None, ns=None):
        """
        Set the data for the Variance Component regression model.

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Array containing the genotypes for which observations are provided.

        y : array-like of shape (n_obs,)
            Array containing the observed phenotypes corresponding to the genotypes in `X`.

        y_var : array-like of shape (n_obs,), optional
            Array containing the empirical or experimental variance for the measurements in `y`.
            If not provided, it is assumed to be uniform or unknown.

        cov : array-like, optional
            Precomputed covariance matrix or related data. If not provided, it will be calculated
            from the input data.

        ns : array-like, optional
            Additional data or parameters related to the model. If not provided, it will be
            calculated from the input data.

        Notes
        -----
        - Providing `cov` and `ns` can save computational resources, as they will not be recalculated
          from the input data.
        """
        super().set_data(X, y, y_var=y_var)
        self.cov = cov
        self.ns = ns
        self.sigma2 = 0.0 if y_var is None else np.nanmin(y_var)

    def calc_covariance_distance(self, X, y):
        return calc_covariance_distance(
            y, self.n_alleles, self.seq_length, self.get_obs_idx(X)
        )

    def lambdas_to_variance(self, lambdas):
        variance_components = (lambdas * self.K.m_k)[1:]
        variance_components = variance_components / variance_components.sum()
        return variance_components

    def get_variance_components(self, lambdas=None):
        """
        Return the variance components as a DataFrame from :math:`\lambda`s.

        Parameters
        ----------
        lambdas : array-like, optional
            An array of eigenvalues representing the variance components. If not provided,
            the model's current `lambdas` attribute will be used.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the following columns:

            - ``k``: Index of the variance component (ranging from 0 to seq_length).
            - ``lambdas``: The input eigenvalues.
            - ``var_perc``: The percentage of variance explained by each component.
            - ``var_perc_cum``: The cumulative percentage of variance explained.
        """
        if lambdas is None:
            lambdas = self.lambdas
        s = self.seq_length + 1
        k = np.arange(s)
        vc_perc = np.zeros(s)
        vc_perc[1:] = self.lambdas_to_variance(lambdas) * 100
        df = pd.DataFrame(
            {
                "k": k,
                "lambdas": lambdas,
                "var_perc": vc_perc,
                "var_perc_cum": np.cumsum(vc_perc),
            }
        )
        return df

    def process_data(self, data):
        X, y, y_var = data
        cov, ns = self.calc_covariance_distance(X, y)
        return (X, y, y_var, cov, ns)

    def set_cv_loss_function(self, cv_loss_function):
        allowed_functions = ["frobenius_norm", "logL", "r2"]
        if cv_loss_function not in allowed_functions:
            msg = "Loss function {} not allowed. Choose from {}"
            raise ValueError(msg.format(cv_loss_function, allowed_functions))
        self.cv_loss_function = cv_loss_function

    def cv_fit(self, data, beta):
        X, y, y_var, cov, ns = data
        self.set_data(X=X, y=y, y_var=y_var, cov=cov, ns=ns)
        lambdas = self._fit(beta)
        return lambdas

    def cv_evaluate(self, data, lambdas):
        X, y, y_var, cov, ns = data

        if self.cv_loss_function == "frobenius_norm":
            # TODO: unclear how to properly deal with the variance here
            self.kernel_aligner.set_data(cov, ns, sigma2=y_var.min())
            loss = self.kernel_aligner.calc_loss(
                lambdas, beta=0, return_grad=False
            )

        else:
            self.set_lambdas(lambdas)
            ypred = self.predict(X)["f"].values

            if self.cv_loss_function == "logL":
                loss = -norm.logpdf(y, loc=ypred, scale=np.sqrt(y_var)).sum()
            elif self.cv_loss_function == "r2":
                loss = -(pearsonr(ypred, y)[0] ** 2)
            else:
                msg = "Allowed loss functions are [frobenius_norm, r2, logL]"
                raise ValueError(msg)

        return loss

    def _fit(self, beta=None):
        if beta is None:
            beta = self.beta

        cov, ns, sigma2 = self.cov, self.ns, self.sigma2
        if cov is None or ns is None:
            cov, ns = self.calc_covariance_distance(
                self.likelihood.X, self.likelihood.y
            )
            sigma2 = np.nanmin(self.likelihood.y_var)

        self.kernel_aligner.set_beta(beta)
        lambdas = self.kernel_aligner.fit(cov, ns, sigma2=sigma2)
        return lambdas

    def fit(self, X, y, y_var=None):
        """
        Infers the Variance Components from the provided data.

        This method infers the variance components, which represent the relative
        contribution of different orders of interaction to the variability in the
        sequence-function relationships. Variance components are determined through
        kernel alignment with the empirical distance-covariance function.

        After fitting, the optimal variance components (`lambdas`) are stored in
        the `VCregression.lambdas` attribute for use in predictions.

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Array containing the genotypes for which observations are provided
            in `y`.

        y : array-like of shape (n_obs,)
            Array containing the observed phenotypes corresponding to the
            genotypes in `X`.

        y_var : array-like of shape (n_obs,), optional
            Array containing the empirical or experimental variance for the
            measurements in `y`. If not provided, it is assumed to be uniform
            or unknown.
        """

        t0 = time()
        self.define_space(genotypes=X)
        self.kernel_aligner = VCKernelAligner(
            n_alleles=self.n_alleles, seq_length=self.seq_length
        )
        self.set_data(X, y, y_var=y_var)

        if self.run_cv:
            self.fit_beta_cv()
            self.set_data(X, y, y_var=y_var)

        lambdas = self._fit()

        self.fit_time = time() - t0
        self.set_lambdas(lambdas)
        self.vc_df = self.get_variance_components(lambdas)


class SeqDEFT(GeneralizedGaussianProcessRegressor, _DeltaPpriorGP):
    """
    Model for inference of a genotype-phenotype map from observations of sequences.

    Sequence Density Estimation using Field Theory (SeqDEFT) model for inferring
    a complete sequence probability distribution under a Gaussian Process prior.
    The prior is parameterized by the variance of local epistatic coefficients
    of order P.

    Parameters
    ----------
    P : int
        The order of local interaction coefficients penalized under the prior.
        For example, `P=2` penalizes local pairwise interactions across all
        possible faces of the Hamming graph, while `P=3` penalizes local
        3-way interactions across all possible cubes.

    a : float, optional, default=None
        A parameter related to the inverse variance of the P-order epistatic
        coefficients being penalized. Larger values induce stronger penalization,
        approximating the Maximum-Entropy model of order P-1. If `a=None`, the
        optimal value of `a` is determined through cross-validation.

    num_reg : int, optional, default=20
        The number of `a` values to evaluate during the cross-validation procedure.

    nfolds : int, optional, default=5
        The number of folds to use in the cross-validation procedure.

    lambdas_P_inv : array-like, optional, default=None
        The inverse of the variance components for the first P orders of interaction.
        If provided, these values are used to regularize the kernel basis.

    a_resolution : float, optional, default=0.1
        The resolution for determining the range of `a` values during cross-validation.

    max_a_max : float, optional, default=1e12
        The maximum value of `a` to consider during cross-validation.

    fac_max : float, optional, default=0.1
        A factor to determine the maximum value of `a` relative to the number of
        P-order faces in the Hamming graph.

    fac_min : float, optional, default=1e-6
        A factor to determine the minimum value of `a` relative to the number of
        P-order faces in the Hamming graph.

    optimization_opts : dict, optional, default={}
        A dictionary of options for the optimization procedure used to calculate
        the maximum entropy model.

    maxiter : int, optional, default=10000
        The maximum number of iterations for the optimization procedure.

    gtol : float, optional, default=1e-6
        The gradient tolerance for the optimization procedure.

    ftol : float, optional, default=1e-8
        The function tolerance for the optimization procedure.
    """

    def __init__(
        self,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        genotypes=None,
        P=2,
        a=None,
        num_reg=20,
        nfolds=5,
        lambdas_P_inv=None,
        a_resolution=0.1,
        max_a_max=1e12,
        fac_max=0.1,
        fac_min=1e-6,
        optimization_opts={},
        maxiter=10000,
        gtol=1e-6,
        ftol=1e-8,
    ):
        super().__init__(
            seq_length=seq_length,
            n_alleles=n_alleles,
            genotypes=genotypes,
            alphabet_type=alphabet_type,
        )
        self.P = P
        self.nfolds = nfolds

        msg = '"a" can only be None or >= 0'
        check_error(a is None or a >= 0, msg=msg)

        # Attributes to generate a values
        self.num_reg = num_reg
        self.total_folds = self.nfolds * self.num_reg

        # Parameters to generate a grid in SeqDEFT, but should be generalizable
        # by just defining a distance metric for phi
        self.a_resolution = a_resolution
        self.max_a_max = max_a_max
        self.fac_max = fac_max
        self.fac_min = fac_min
        self.baseline_phi = None

        # Optimization attributes
        opts = {"ftol": ftol, "gtol": gtol, "maxiter": maxiter}
        optimization_opts.update(opts)
        self.optimization_opts = optimization_opts
        self.likelihood = SeqDEFTLikelihood(self.genotypes)
        self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
        self.s = self.DP.n_p_faces
        self.kernel_basis = DeltaKernelBasisOperator(
            self.n_alleles, self.seq_length, self.P
        )
        self.set_lambdas_P_inv(lambdas_P_inv)
        self.set_a(a)

    def set_lambdas_P_inv(self, lambdas_P_inv):
        if lambdas_P_inv is None:
            self.lambdas_P_inv = None
            self.kernel_regularizer = None
        else:
            msg = "lambdas_P_inv={} size is different from P={}"
            msg = msg.format(lambdas_P_inv.shape[0], self.P)
            check_error(lambdas_P_inv.shape[0] == self.P, msg)

            self.lambdas_P_inv = lambdas_P_inv
            self.kernel_regularizer = DeltaKernelRegularizerOperator(
                self.kernel_basis, self.lambdas_P_inv
            )

    def set_a(self, a):
        self.a = a

        if a is not None and np.isfinite(a):
            check_error(
                a >= 0, msg='"a" must be larger or equal than 0 and finite'
            )
            if self.lambdas_P_inv is None:
                self.C = a / self.s * self.DP
            else:
                lambdas = a / self.s * self.DP.lambdas
                lambdas[: self.P] = self.lambdas_P_inv
                self.C = ProjectionOperator(
                    self.n_alleles, self.seq_length, lambdas=lambdas
                )

    def cv_set_data(self, X, y):
        self.likelihood.set_data(
            X,
            y=y,
            offset=self.baseline,
            positions=self.positions,
            phylo_correction=self.phylo_correction,
            adjust_freqs=self.adjust_freqs,
            allele_freqs=self.allele_freqs,
        )

    def cv_fit(self, data, a, phi0=None):
        X, y, _ = data
        self.cv_set_data(X=X, y=y)
        self.set_a(a)
        phi = self.calc_posterior_max(phi0=phi0)
        return phi

    def cv_evaluate(self, data, phi):
        X, y, _ = data
        self.cv_set_data(X=X, y=y)
        logL = self.likelihood.calc_logL(phi)
        return logL

    def get_cv_logL_df(self, cv_logL):
        with np.errstate(divide="ignore"):
            cv_log_L = pd.DataFrame(cv_logL)
            cv_log_L["log_a"] = np.log10(cv_log_L["a"])
            cv_log_L["sd"] = self.a_to_sd(cv_log_L["a"])
            cv_log_L["log_sd"] = np.log10(cv_log_L["sd"])
        return cv_log_L

    def get_ml_a(self, cv_logL_df):
        df = cv_logL_df.groupby("a")["logL"].mean()
        return df.index[np.argmax(df)]

    def calc_cv_loss(self, cv_iter, total_folds):
        phi0_cache = {}
        for a, fold, train, test in tqdm(cv_iter, total=total_folds):
            phi = self.cv_fit(train, a, phi0=phi0_cache.get(fold, None))
            if np.isinf(a):
                phi0_cache[fold] = phi
            loss = self.cv_evaluate(test, phi)
            yield ({"a": a, "fold": fold, "logL": loss})

    def fit_a_cv(self, phi_inf=None):
        a_values = np.append(np.inf, self.get_a_values(phi_inf=phi_inf))
        total_folds = a_values.shape[0] * self.nfolds

        cv_splits = get_CV_splits(
            X=self.likelihood.X,
            y=self.likelihood.y,
            y_var=None,
            nfolds=self.nfolds,
        )
        cv_iter = get_cv_iter(cv_splits, a_values)
        cv_logL = self.calc_cv_loss(cv_iter, total_folds)
        self.logL_df = self.get_cv_logL_df(cv_logL)
        a = self.get_ml_a(self.logL_df)
        return a

    def phi_to_b(self, phi):
        return self.kernel_basis.transpose() @ phi

    def b_to_phi(self, b):
        return self.kernel_basis @ b

    def a_to_sd(self, a):
        return np.sqrt(self.DP.n_p_faces / a)

    def sd_to_a(self, sd):
        return self.DP.n_p_faces / sd**2

    def calc_maximum_entropy_model(self, b0=None):
        res = minimize(
            fun=self.calc_loss,
            jac=True,
            method="L-BFGS-B",
            x0=b0,
            options=self.optimization_opts,
        )

        if not res.success:
            raise ValueError(res.message)

        self.opt_res = res
        return res.x

    def calc_posterior_max(self, phi0=None):
        phi0 = self.get_phi0(phi0=phi0)

        if self.a == 0:
            with np.errstate(divide="ignore"):
                phi = -np.log(self.likelihood.R)
            self.opt_res = None

        elif np.isfinite(self.a):
            phi = self.calc_posterior_mean(phi0=phi0)

        else:
            b0 = self.phi_to_b(phi0)
            b = self.calc_maximum_entropy_model(b0=b0)
            phi = self.b_to_phi(b)

        return phi

    def mcmc(
        self,
        n_samples=1000,
        n_chains=4,
        progress=True,
        target_accept=0.9,
        **kwargs,
    ):
        def logp(x):
            loss, grad = self.calc_loss(x, return_grad=True)
            return (-loss, -grad)

        def logp_grad(x):
            return -self.calc_grad(x)

        samples = []
        sampler = HMC(logp, logp_grad, step_size=0.1, path_length=10)
        for _ in range(n_chains):
            x0 = self.sample_prior()
            for s in sampler.sample(x0=x0, n_samples=n_samples):
                samples.append(s)

            # start = {"x": self.sample_prior(self.a)}
            # sampler = NUTS(logp, start=start, grad_logp=True,
            #                 target_accept=0.9, **kwargs)
            # for s in sampler.sample(n_chains=1, num=2 * n_samples,
            #                         progress_bar=progress, burn=n_samples):
            #     samples.append(s[0])
        samples = np.array(samples)
        return samples

    def set_baseline(self, X=None, baseline_phi=None):
        if baseline_phi is None:
            self.baseline = None
        else:
            msg = "Sequences `X` associated to the baseline must be provided"
            check_error(X is not None, msg=msg)
            self.baseline = (
                pd.Series(baseline_phi, index=X).loc[self.genotypes].values
            )

    def set_data(
        self,
        X,
        y=None,
        positions=None,
        baseline_X=None,
        baseline_phi=None,
        phylo_correction=False,
        adjust_freqs=False,
        allele_freqs=None,
    ):
        """
        Set the data for the SeqDEFT model, including observed sequences,
        weights, and optional baseline information.

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Array containing the observed sequences.

        y : array-like of shape (n_obs,), optional
            Array containing the weights for each observed sequence. By default,
            each sequence is assigned a weight of 1. These weights can be computed
            using phylogenetic correction.

        positions : array-like of shape (n_pos,), optional
            If provided, subsequences at these positions in the input sequences
            will be used as input. Default is None.

        baseline_X : array-like of shape (n_genotypes,), optional
            Array containing the sequences associated with `baseline_phi`. Default
            is None.

        baseline_phi : array-like of shape (n_genotypes,), optional
            Array containing the baseline values (`baseline_phi`) to include in
            the model. Default is None.

        phylo_correction : bool, optional, default=False
            Whether to apply phylogenetic correction using the full-length sequences.

        adjust_freqs : bool, optional, default=False
            Whether to adjust densities by the expected allele frequencies in the
            full-length sequences.

        allele_freqs : dict or codon_table, optional
            Dictionary containing the expected allele frequencies for each allele
            in the set of possible sequences, or a codon table to generate expected
            amino acid frequencies. If `None`, these frequencies will be calculated
            from the full-length observed sequences. Default is None.
        """
        self.positions = positions
        self.adjust_freqs = adjust_freqs
        self.phylo_correction = phylo_correction
        self.allele_freqs = allele_freqs
        self.set_baseline(baseline_X, baseline_phi)
        self.likelihood.set_data(
            X,
            y=y,
            offset=self.baseline,
            positions=self.positions,
            phylo_correction=self.phylo_correction,
            adjust_freqs=self.adjust_freqs,
            allele_freqs=self.allele_freqs,
        )

    def fit(
        self,
        X,
        y=None,
        baseline_phi=None,
        baseline_X=None,
        positions=None,
        phylo_correction=False,
        adjust_freqs=False,
        allele_freqs=None,
    ):
        """
        Infers the SeqDEFT model hyperparameter `a` from the provided data.

        This method determines the optimal regularization parameter `a` by evaluating
        the log-likelihood of held-out sequences under a grid search for `a` in
        cross-validation settings.

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Array containing the observed sequences.

        y : array-like of shape (n_obs,)
            Array containing the weights for each observed sequence.
            By default, each sequence is assigned a weight of 1. These weights
            can be computed using phylogenetic correction.

        baseline_X : array-like of shape (n_genotypes,), optional
            Array containing the sequences associated with `baseline_phi`.

        baseline_phi : array-like of shape (n_genotypes,), optional
            Array containing the baseline values (`baseline_phi`) to include
            in the model.

        positions : array-like of shape (n_pos,), optional
            If provided, subsequences at these positions in the input sequences
            will be used as input.

        phylo_correction : bool, optional, default=False
            Whether to apply phylogenetic correction using the full-length sequences.

        adjust_freqs : bool, optional, default=False
            Whether to adjust densities by the expected allele frequencies in the
            full-length sequences.

        allele_freqs : dict or codon_table, optional
            Dictionary containing the expected allele frequencies for each allele
            in the set of possible sequences, or a codon table to generate expected
            amino acid frequencies. If `None`, these frequencies will be calculated
            from the full-length observed sequences.
        """
        self.set_data(
            X,
            y=y,
            positions=positions,
            baseline_X=baseline_X,
            baseline_phi=baseline_phi,
            phylo_correction=phylo_correction,
            adjust_freqs=adjust_freqs,
            allele_freqs=allele_freqs,
        )

        if self.a is None:
            a = self.fit_a_cv()
            self.set_a(a)
            self.likelihood.set_data(
                X,
                y=y,
                offset=self.baseline,
                positions=self.positions,
                phylo_correction=self.phylo_correction,
                adjust_freqs=self.adjust_freqs,
                allele_freqs=allele_freqs,
            )

    def predict(self, X_pred=None, calc_variance=False):
        """
        Compute the Maximum a Posteriori (MAP) estimate of the phenotype for
        the specified genotypes or the entire genotype space.

        Parameters
        ----------
        X_pred : array-like of shape (n_genotypes,), optional
            Array containing the genotypes for which the phenotype predictions
            are desired. If `X_pred` is None, predictions are computed for the
            entire sequence space.

        calc_variance : bool, optional, default=False
            If True, the posterior variances and standard deviations for each
            genotype are also computed and included in the output.

        Returns
        -------
        pred : pd.DataFrame of shape (n_genotypes, n_columns)
            A DataFrame containing the predicted phenotypes for each input
            genotype in the column ``f``. If ``calc_variance=True``, additional
            columns are included:

            - ``f_var``: Posterior variance for each genotype.
            - ``f_std``: Posterior standard deviation for each genotype.
            - ``ci_95_lower``: Lower bound of the 95% credible interval.
            - ``ci_95_upper``: Upper bound of the 95% credible interval.
            The genotype labels are used as the row index.

            If neither ``X_pred`` nor ``calc_variance`` are provided, the output
            DataFrame includes additional columns:

            - ``freq``: Empirical frequencies of the genotypes.
            - ``Q_star``: Estimated genotype probabilities.

        Examples
        --------
        Predict phenotypes for the entire genotype space:

        >>> pred = model.predict()

        Predict phenotypes for specific genotypes with variance:

        >>> pred = model.predict(X_pred=["AAA", "AAC"], calc_variance=True)
        """

        if X_pred is not None or calc_variance:
            return super().predict(X_pred, calc_variance)
        else:
            phi = self.calc_posterior_max()
            output = self.likelihood.phi_to_output(phi)
            return output

    def calc_loss_inf_a(self, b, return_grad=True, store_hess=True):
        basis = self.kernel_basis
        phi = basis @ b
        res = self.likelihood.calc_loss_grad_hess(phi)
        loss, grad, hess = res

        if self.kernel_regularizer is not None:
            res = self.kernel_regularizer.calc_loss_grad_hess_b(b)
            loss_reg, grad_reg, hess_reg = res
            loss += loss_reg

        if return_grad:
            grad = basis.transpose() @ grad
            if self.kernel_regularizer is not None:
                grad += grad_reg

            if store_hess:
                self.hess = basis.transpose() @ DiagonalOperator(hess) @ basis
                if self.kernel_regularizer is not None:
                    self.hess += hess_reg
            return (loss, grad)
        else:
            return loss

    def calc_loss(self, x, return_grad=True):
        if np.isinf(self.a):
            return self.calc_loss_inf_a(x, return_grad=return_grad)
        else:
            return super().calc_loss(x, return_grad=return_grad)

    # Optional methods
    def calc_a_max(self, phi_inf):
        a_tmp = self.a
        a_max = self.DP.n_p_faces * self.fac_max

        self.set_a(a_max)
        phi_max = self.calc_posterior_max(phi0=phi_inf)
        distance = D_geo(phi_max, phi_inf)

        while distance > self.a_resolution and a_max < self.max_a_max:
            a_max *= 10
            self.set_a(a_max)
            phi_max = self.calc_posterior_max(phi0=phi_inf)
            distance = D_geo(phi_max, phi_inf)

        self.set_a(a_tmp)
        return a_max

    def calc_a_min(self, phi_inf=None):
        a_tmp = self.a
        a_min = self.DP.n_p_faces * self.fac_min

        self.set_a(0)
        phi_0 = self.calc_posterior_max(0)

        self.set_a(a_min)
        phi_min = self.calc_posterior_max(phi0=phi_inf)

        distance = D_geo(phi_min, phi_0)

        while distance > self.a_resolution:
            a_min /= 10
            self.set_a(a_min)
            phi_min = self.calc_posterior_max(phi0=phi_inf)
            distance = D_geo(phi_min, phi_0)

        self.set_a(a_tmp)
        return a_min

    def get_a_values(self, phi_inf=None):
        if phi_inf is None:
            self.set_a(np.inf)
            phi_inf = self.calc_posterior_max()

        a_min = self.calc_a_min(phi_inf)
        a_max = self.calc_a_max(phi_inf)
        a_values = np.geomspace(a_min, a_max, self.num_reg)
        a_values = np.hstack([0, a_values, np.inf])
        self.total_folds = self.nfolds * (self.num_reg + 2)
        return a_values

    def simulate(self, N, seed=None):
        """
        Simulates data under the specified `a` penalization for
        local P-epistatic coefficients.

        Parameters
        ----------
        N : int
            Number of total sequences to sample.

        seed : int, optional (default=None)
            Random seed to use for simulation.

        Returns
        -------
        phi : array-like of shape (N,)
            Vector containing the true phi values from which samples were generated.

        X : array-like of shape (N,)
            Vector containing the sampled sequences from the probability
            distribution.

        Examples
        --------

        >>> model = SeqDEFT(n_alleles=4, seq_length=5, P=2, a=1.0)
        >>> phi, X = model.simulate(N=100, seed=42)
        """
        phi, X = self._simulate(seed=seed, N=N)
        return phi, X


def D_geo(phi1, phi2):
    logQ1 = -phi1 - logsumexp(-phi1)
    logQ2 = -phi2 - logsumexp(-phi2)
    s = np.exp(logsumexp(0.5 * (logQ1 + logQ2)))
    x = min(s, 1)
    return 2 * np.arccos(x)


class HMC(object):
    def __init__(self, logp, logp_grad, step_size, path_length):
        self.logp = logp
        self.logp_grad = logp_grad

        self.step_size = step_size
        self.path_length = path_length
        self.max_steps = 100
        self.n_steps = min(
            int(self.path_length / self.step_size) - 1, self.max_steps
        )

        # self.m = m
        # self.f = f
        # self.sqrt_1mf2 = np.sqrt(1 - self.f ** 2)
        # self.scales = 1 / np.sqrt(hess_diag)

        # step-size tunning parameters
        self.window = 10
        self.gamma_old = 1
        self.gamma_new = 1

    def sample(self, x0, n_samples=1000):
        # Initiate iteration
        position = x0
        logp, logp_grad = self.logp(position)
        momentum = self.sample_momentum(position)
        energy = self.calc_energy(position, momentum, logp)

        # HMC iterations
        self.acceptance_rates = []
        self.num_acceptance = 0

        # Warmup
        # tunning = DualAveragingStepSize(self.step_size)
        for i in tqdm(range(2 * n_samples)):
            momentum = self.sample_momentum(position)
            new_position, new_logp_grad, new_energy = self.leapfrog(
                position, momentum, logp_grad
            )

            p_accept = min(1, np.exp(new_energy - energy))
            if np.random.uniform() < p_accept:
                position, logp_grad, energy = (
                    new_position,
                    new_logp_grad,
                    new_energy,
                )
                self.num_acceptance += 1

            if i < n_samples:
                if i % self.window == 0:
                    self.tune_step_size()
                # self.step_size = tunning.update(p_accept)
            # elif i == n_samples:
            #     self.num_acceptance = 0
            #     self.step_size = tunning.update(p_accept, smoothed=True)
            else:
                yield (position)

        # # Sampling
        # for _ in tqdm(range(n_samples)):
        #     phi, psi, grad, log_P = self.step(phi, psi, grad, log_P)
        #     psi *= -1
        #     yield(phi)

    def leapfrog(self, position, momentum, logp_grad):
        step_sizes = self.step_size  # * self.scales
        position, momentum = np.copy(position), np.copy(momentum)

        momentum -= step_sizes * self.logp_grad(position) / 2
        for _ in range(self.n_steps):
            position += step_sizes * momentum
            momentum -= step_sizes * self.logp_grad(position)

        position += step_sizes * momentum
        logp, logp_grad = self.logp(position)
        momentum -= step_sizes * logp_grad / 2
        momentum *= -1

        energy = self.calc_energy(position, momentum, logp)
        return (position, logp_grad, energy)

    def sample_momentum(self, position):
        return np.random.normal(size=position.shape)

    def calc_energy(self, position, momentum, logp=None):
        if logp is None:
            logp = self.logp(position)[0]
        return -logp - np.sum(momentum**2) / 2

    def tune_step_size(self):
        acceptance_rate = self.num_acceptance / self.window
        new_step_size = self.update_step_size(self.step_size, acceptance_rate)

        exponent = 1 / (self.gamma_old + self.gamma_new)
        step_size = (
            self.step_size**self.gamma_old * new_step_size**self.gamma_new
        ) ** exponent
        self.n_steps = min(int(self.path_length / step_size) - 1, self.n_steps)
        self.acceptance_rates.append(acceptance_rate)
        self.num_acceptance = 0
        self.step_size = step_size

    def update_step_size(self, step_size, acceptance_rate):
        if acceptance_rate < 0.001:
            step_size *= 0.1
        elif 0.001 <= acceptance_rate < 0.05:
            step_size *= 0.5
        elif 0.05 <= acceptance_rate < 0.2:
            step_size *= 0.7
        elif 0.2 <= acceptance_rate < 0.5:
            step_size *= 0.8
        elif 0.5 <= acceptance_rate < 0.6:
            step_size *= 0.9
        elif 0.6 <= acceptance_rate <= 0.7:
            step_size *= 1
        elif 0.7 < acceptance_rate <= 0.8:
            step_size *= 1.1
        elif 0.8 < acceptance_rate <= 0.9:
            step_size *= 1.5
        elif 0.9 < acceptance_rate <= 0.95:
            step_size *= 2
        elif 0.95 < acceptance_rate:
            step_size *= 3
        return step_size

    def compute_R_hat(self, samples):
        # Copy the multi_phi_samples
        num_chains, G, num_samples_per_chain = samples.shape

        num_subchains, len_subchain = (
            2 * num_chains,
            int(num_samples_per_chain / 2),
        )

        # Re-shape multi_phi_samples into a
        # shape of (num_subchains, G, len_subchain)
        a = []
        for k in range(num_chains):
            a.append(samples[k, :, :len_subchain])
            a.append(samples[k, :, len_subchain:])
        multi_phi_samples_reshaped = np.array(a)

        # Compute R_hat for each component of phi
        R_hats = []
        for i in range(G):
            # Collect the (sub)chains of samples of phi_i
            i_collector = np.zeros([len_subchain, num_subchains])
            for j in range(num_subchains):
                i_collector[:, j] = multi_phi_samples_reshaped[j, i, :]

            # Compute the between-(sub)chain variance
            mean_0 = i_collector.mean(axis=0)
            mean_01 = mean_0.mean()
            B = (
                len_subchain
                / (num_subchains - 1)
                * np.sum((mean_0 - mean_01) ** 2)
            )

            # Compute the within-(sub)chain variance
            s2 = np.zeros(num_subchains)
            for j in range(num_subchains):
                s2[j] = (
                    1
                    / (len_subchain - 1)
                    * np.sum((i_collector[:, j] - mean_0[j]) ** 2)
                )
            W = s2.mean()

            # Estimate the marginal posterior variance
            var = (len_subchain - 1) / len_subchain * W + 1 / len_subchain * B

            # Compute R_hat
            R_hat = np.sqrt(var / W)

            # Save
            R_hats.append(R_hat)

        # Return
        return np.array(R_hats)


class DualAveragingStepSize:
    """update stepsize for the leapfrog function during tuning steps"""

    def __init__(
        self,
        initial_step_size,
        target_accept=0.7,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
    ):
        # proposals are biased upwards to stay away from 0.
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept, smoothed=False):
        # Running tally of absolute error. Can be positive or negative. Want to
        # be 0.
        self.error_sum += self.target_accept - p_accept

        # This is the next proposed (log) step size. Note it is biased towards
        # mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)

        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t**-self.kappa

        # Smoothed average step size
        self.log_averaged_step = (
            eta * log_step + (1 - eta) * self.log_averaged_step
        )

        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        if smoothed:
            return np.exp(self.log_averaged_step)
        else:
            return np.exp(log_step)
