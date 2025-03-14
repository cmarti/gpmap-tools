#!/usr/bin/env python
from time import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, pearsonr

from gpmap.aligner import VCKernelAligner
from gpmap.gp import (
    GaussianProcessRegressor,
    GeneralizedGaussianProcessRegressor,
    MinimizerRegressor,
    SequenceInterpolator,
)
from gpmap.likelihood import SeqDEFTLikelihood
from gpmap.linop import (
    DeltaKernelBasisOperator,
    DeltaKernelRegularizerOperator,
    DeltaPOperator,
    DiagonalOperator,
    ProjectionOperator,
    VarianceComponentKernel,
    calc_avg_local_epistatic_coeff,
    calc_covariance_distance,
)
from gpmap.seq import (
    get_subsequences,
)
from gpmap.utils import (
    calc_cv_loss,
    check_error,
    get_cv_iter,
    get_CV_splits,
)


class MinimumEpistasisInterpolator(SequenceInterpolator):
    def __init__(
        self,
        P=2,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        cg_rtol=1e-16,
    ):
        self.P = P
        super().__init__(
            n_alleles=n_alleles,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            cg_rtol=cg_rtol,
        )

    def define_precision_matrix(self):
        self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
        self.C = 1 / self.DP.n_p_faces * self.DP
        self.p = self.DP.n_p_faces_genotype

    def smooth(self, y_pred):
        y_pred -= 1 / self.p * self.DP @ y_pred
        return y_pred


class MinimumEpistasisRegression(MinimizerRegressor):
    def __init__(
        self,
        P,
        a=None,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        nfolds=5,
        num_reg=20,
        min_log_reg=-2,
        max_log_reg=6,
        progress=True,
        cg_rtol=1e-4,
    ):
        self.a = a
        self.P = P

        self.nfolds = nfolds
        self.num_reg = num_reg
        self.total_folds = self.nfolds * self.num_reg
        self.min_log_reg = min_log_reg
        self.max_log_reg = max_log_reg
        super().__init__(
            n_alleles=n_alleles,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            progress=progress,
            cg_rtol=cg_rtol,
        )

    def init(
        self,
        seq_length=None,
        n_alleles=None,
        genotypes=None,
        alphabet_type="custom",
    ):
        if not self.initialized:
            self.define_space(
                seq_length=seq_length,
                n_alleles=n_alleles,
                genotypes=genotypes,
                alphabet_type=alphabet_type,
            )
            self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
            self.s = self.DP.n_p_faces
            self.initialized = True
            self.set_a(self.a)

    def set_a(self, a):
        self.a = a
        if a is not None:
            self.C = self.a / self.s * self.DP

    def cv_fit(self, data, a):
        X, y, y_var = data
        self.set_a(a)
        self.set_data(X, y, y_var)
        y_pred = self.calc_posterior()[0]
        return y_pred

    def cv_evaluate(self, data, y_pred):
        X, y, y_var = data
        pred_idx = self.get_obs_idx(X)
        logL = norm.logpdf(y, loc=y_pred[pred_idx], scale=np.sqrt(y_var)).mean()
        return logL

    def a_to_sd(self, a):
        return np.sqrt(self.DP.n_p_faces / a)

    def sd_to_a(self, sd):
        return self.DP.n_p_faces / sd**2

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

    def fit_a_cv(self):
        a_values = self.get_regularization_constants()
        cv_splits = get_CV_splits(
            X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds
        )
        cv_iter = get_cv_iter(cv_splits, a_values)
        cv_logL = calc_cv_loss(
            cv_iter,
            self.cv_fit,
            self.cv_evaluate,
            total_folds=a_values.shape[0] * self.nfolds,
            param_label="a",
            loss_label="logL",
        )
        self.logL_df = self.get_cv_logL_df(cv_logL)
        a = self.get_ml_a(self.logL_df)
        return a

    def fit(self, X, y, y_var=None, cross_validation=False):
        """
        Infers the optimal `a` from the provided data, this is,
        the magnitude of Pth order local epistatic coefficients
        that maximize predictive performance in held out data

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `y`

        y : array-like of shape (n_obs,)
            Vector containing the observed phenotypes corresponding to `X`
            sequences

        y_var : array-like of shape (n_obs,)
            Vector containing the empirical or experimental known variance for
            the measurements in `y`

        Returns
        -------
        a : float
            Optimal `a` value maximing the cross-validated log-likelihood

        """
        self.set_data(X, y, y_var=y_var)

        if self.a is None:
            if cross_validation:
                a = self.fit_a_cv()
                self.set_data(X, y, y_var=y_var)
            else:
                # TODO: find a better solution for this
                alphabet = np.unique(self.alphabet)
                s, n = calc_avg_local_epistatic_coeff(
                    X,
                    y,
                    alphabet=alphabet,
                    seq_length=self.seq_length,
                    P=self.P,
                )
                a = self.DP.rank * n / s
            self.set_a(a)


class VCregression(GaussianProcessRegressor):
    """
    Variance Component regression model that allows inference and prediction
    of a scalar function in sequence spaces under a Gaussian Process prior
    parametrized by the contribution of the different orders of interaction
    to the observed genetic variability of a continuous phenotype

    It requires the use of the same number of alleles per sites

    """

    def __init__(
        self,
        lambdas=None,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
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

        if seq_length is not None and (
            n_alleles is not None or alphabet_type != "custom"
        ):
            self.define_space(
                n_alleles=n_alleles,
                seq_length=seq_length,
                alphabet_type=alphabet_type,
            )

        if lambdas is not None:
            self.set_lambdas(lambdas)

        self.cg_rtol = cg_rtol

    def set_lambdas(self, lambdas=None, k=None):
        K = VarianceComponentKernel(
            self.n_alleles, self.seq_length, lambdas=lambdas, k=k
        )
        self._K_BB = None
        self.lambdas = K.lambdas
        super().__init__(base_kernel=K, progress=self.progress)

    def set_data(self, X, y, y_var=None, cov=None, ns=None):
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

    def get_variance_component_df(self, lambdas):
        s = self.seq_length + 1
        k = np.arange(s)
        vc_perc = np.zeros(s)
        vc_perc[1:] = self.lambdas_to_variance(lambdas)
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
            ypred = self.predict(X)["y"].values

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
            cov, ns = self.calc_covariance_distance(self.X, self.y)
            sigma2 = np.nanmin(self.y_var)

        self.kernel_aligner.set_beta(beta)
        lambdas = self.kernel_aligner.fit(cov, ns, sigma2=sigma2)
        return lambdas

    def fit(self, X, y, y_var=None):
        """
        Infers the variance components from the provided data, this is,
        the relative contribution of the different orders of interaction
        to the variability in the sequence-function relationships

        Stores learned `lambdas` in the attribute VCregression.lambdas
        to use internally for predictions and returns them as output

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `y`

        y : array-like of shape (n_obs,)
            Vector containing the observed phenotypes corresponding to `X`
            sequences

        y_var : array-like of shape (n_obs,)
            Vector containing the empirical or experimental known variance for
            the measurements in `y`

        Returns
        -------
        lambdas: array-like of shape (seq_length + 1,)
            Variances for each order of interaction k inferred from the data

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
        self.vc_df = self.get_variance_component_df(lambdas)


class SeqDEFT(GeneralizedGaussianProcessRegressor):
    """
    Sequence Density Estimation using Field Theory model that allows inference
    of a complete sequence probability distribution under a Gaussian Process prior
    parameterized by variance of local epistatic coefficients of order P

    It requires the use of the same number of alleles per sites

    Parameters
    ----------
    P : int
        Order of the local interaction coefficients that we are penalized
        under the prior i.e. `P=2` penalizes local pairwise interaction
        across all posible faces of the Hamming graph while `P=3` penalizes
        local 3-way interactions across all possible cubes.

    a : float (None)
        Parameter related to the inverse of the variance of the P-order
        epistatic coefficients that are being penalized. Larger values
        induce stronger penalization and approximation to the
        Maximum-Entropy model of order P-1. If `a=None` the best a is found
        through cross-validation

    num_reg : int (20)
        Number of a values to evaluate through cross-validation

    nfolds: int (5)
        Number of folds to use in the cross-validation procedure

    """

    def __init__(
        self,
        P,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        genotypes=None,
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
        super().__init__()
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

        self.initialized = False
        self._init(
            n_alleles=n_alleles,
            seq_length=seq_length,
            alphabet_type=alphabet_type,
            genotypes=genotypes,
        )
        self.set_lambdas_P_inv(lambdas_P_inv)
        self.set_a(a)

    def _init(
        self,
        seq_length=None,
        n_alleles=None,
        genotypes=None,
        alphabet_type="custom",
    ):
        if not self.initialized:
            self.define_space(
                seq_length=seq_length,
                n_alleles=n_alleles,
                genotypes=genotypes,
                alphabet_type=alphabet_type,
            )
            self.likelihood = SeqDEFTLikelihood(self.genotypes)
            self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
            self.s = self.DP.n_p_faces
            self.kernel_basis = DeltaKernelBasisOperator(
                self.n_alleles, self.seq_length, self.P
            )
            self.initialized = True

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
            X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds
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
            x0 = self.simulate_phi()
            for s in sampler.sample(x0=x0, n_samples=n_samples):
                samples.append(s)

            # start = {"x": self.simulate_phi(self.a)}
            # sampler = NUTS(logp, start=start, grad_logp=True,
            #                 target_accept=0.9, **kwargs)
            # for s in sampler.sample(n_chains=1, num=2 * n_samples,
            #                         progress_bar=progress, burn=n_samples):
            #     samples.append(s[0])
        samples = np.array(samples)
        return samples

    def get_C_lambdas(self):
        a = self.a
        self.DP.calc_lambdas()
        lambdas = np.zeros(self.DP.lambdas.shape)
        lambdas[self.P :] = a * self.DP.lambdas[self.P :] / self.DP.n_p_faces

        if self.lambdas_P_inv is not None:
            lambdas[: self.P] = 1 / self.lambdas_P_inv

        return lambdas

    def simulate_phi(self):
        """
        Simulates data under the specified `a` penalization for
        local P-epistatic coefficients

        Returns
        -------
        phi : array-like of shape (n_genotypes,)
            Vector containing values for the latent phenotype or field
            sampled from the prior characterized by `a`
        """

        x = np.random.normal(size=self.n_genotypes)
        lambdas_inv = self.get_C_lambdas()
        lambdas = np.zeros_like(lambdas_inv)
        idx = lambdas_inv != 0
        lambdas[idx] = 1 / lambdas_inv[idx]
        W_sqrt = ProjectionOperator(
            self.n_alleles, self.seq_length, lambdas=lambdas
        ).matrix_sqrt()
        phi = W_sqrt @ x
        return phi

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
        self.X = X
        self.y = y
        self.y_var = None

        self.init(genotypes=get_subsequences(X, positions=positions))
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
        Infers the sequence-function relationship under the specified
        \Delta^{(P)} prior

        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the observed sequences

        y : array-like of shape (n_obs,)
            Vector containing the weights for each observed sequence.
            By default, each sequence takes a weight of 1. These weights
            can be calculated using phylogenetic correction

        baseline_X: array-like of shape (n_genotypes,)
            Vector containing the sequences associated with baseline_phi

        baseline_phi: array-like of shape (n_genotypes,)
            Vector containing the baseline_phi to include in the model

        positions: array-like of shape (n_pos,)
            If provided, subsequences at these positions in the provided
            input sequences will be used as input

        phylo_correction: bool (False)
            Apply phylogenetic correction using the full length sequences

        adjust_freqs: bool (False)
            Whether to correct densities by the expected allele frequencies
            in the full length sequences

        allele_freqs: dict or codon_table
            Dictionary containing the allele expected frequencies frequencies
            for every allele in the set of possible sequences or the codon
            table to use to genereate expected aminoacid frequencies
            If `None`, they will be calculated from the full length observed
            sequences.

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

    def simulate(self, N, phi=None, seed=None):
        """
        Simulates data under the specified `a` penalization for
        local P-epistatic coefficients

        Parameters
        ----------
        N : int
            Number of total sequences to sample

        phi : array-like of shape (n_genotypes,)
            Vector containing values for the field underlying the probability
            distribution from which to sample sequences. If provided, they
            will be used instead of sampling them from the prior characterized
            by the given `a`.

        seed: int (None)
            Random seed to use for simulation

        Returns
        -------
        X : array-like of shape (N,)
            Vector containing the sampled sequences from the probability distribution
        """

        if seed is not None:
            np.random.seed(seed)

        if phi is not None:
            check_error(
                phi.shape == (self.n_genotypes,),
                msg='Ensure "phi" has the shape (n_genotypes,)',
            )
        else:
            check_error(
                self.a is not None, '"a" must be provided if "phi=None"'
            )
            phi = self.simulate_phi()

        X = self.likelihood.sample(phi, N)
        return X


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
        tunning = DualAveragingStepSize(self.step_size)
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

        # Re-shape multi_phi_samples into a shape of (num_subchains, G, len_subchain)
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
