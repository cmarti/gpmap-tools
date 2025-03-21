#!/usr/bin/env python
from time import time

import numpy as np
import pandas as pd
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import norm
from scipy.optimize import minimize

from gpmap.linop import (
    DiagonalOperator,
    IdentityOperator,
    InverseOperator,
    SubMatrixOperator,
    get_diag,
)
from gpmap.matrix import quad
from gpmap.seq import (
    get_alphabet,
    get_seqs_from_alleles,
    guess_space_configuration,
)
from gpmap.utils import (
    calc_cv_loss,
    check_error,
    get_cv_iter,
    get_CV_splits,
)


class SeqGaussianProcessRegressor(object):
    def __init__(self, expand_alphabet=True):
        self.expand_alphabet = expand_alphabet

    def get_regularization_constants(self):
        return 10 ** (
            np.linspace(self.min_log_reg, self.max_log_reg, self.num_reg)
        )

    def fit_beta_cv(self):
        beta_values = self.get_regularization_constants()
        cv_splits = get_CV_splits(
            X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds
        )
        cv_iter = get_cv_iter(
            cv_splits, beta_values, process_data=self.process_data
        )
        cv_loss = calc_cv_loss(
            cv_iter,
            self.cv_fit,
            self.cv_evaluate,
            total_folds=beta_values.shape[0] * self.nfolds,
        )
        self.cv_loss_df = pd.DataFrame(cv_loss)

        with np.errstate(divide="ignore"):
            self.cv_loss_df["log_beta"] = np.log10(self.cv_loss_df["beta"])

        loss = self.cv_loss_df.groupby("beta")["loss"].mean()
        self.beta = loss.index[np.argmin(loss)]

    def define_space(
        self,
        seq_length=None,
        n_alleles=None,
        genotypes=None,
        alphabet_type="custom",
    ):
        if genotypes is not None:
            configuration = guess_space_configuration(
                genotypes,
                ensure_full_space=False,
                force_regular=True,
                force_regular_alleles=False,
            )
            seq_length = configuration["length"]
            alphabet = [sorted(a) for a in configuration["alphabet"]]
            n_alleles = configuration["n_alleles"][0]
            n_alleles = len(alphabet[0])
        else:
            msg = "Either seq_length or genotypes must be provided"
            check_error(seq_length is not None, msg=msg)
            alphabet = get_alphabet(
                n_alleles=n_alleles, alphabet_type=alphabet_type
            )
            n_alleles = len(alphabet)
            alphabet = [alphabet] * seq_length

        self.set_config(n_alleles, seq_length, alphabet)

    def get_obs_idx(self, seqs):
        obs_idx = self.genotype_idxs[seqs]
        return obs_idx

    def set_config(self, n_alleles, seq_length, alphabet):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.alphabet = alphabet
        self.n_genotypes = n_alleles**seq_length
        self.genotypes = np.array(list(get_seqs_from_alleles(alphabet)))
        self.genotype_idxs = pd.Series(
            np.arange(self.n_genotypes), index=self.genotypes
        )

    def make_contrasts(self, contrast_matrix):
        """
        Computes the posterior distribution of linear combinations of genotypes
        under the specific Gaussian Process prior.

        Parameters
        ----------
        contrast_matrix: pd.DataFrame of shape (n_genotypes, n_contrasts)
            DataFrame containing the linear combinations of genotypes for
            which to compute the summary of the posterior distribution

        Returns
        -------
        contrasts: pd.DataFrame of shape (n_contrasts, 5)
            DataFrame containing the summary of the posterior for each of
            the posterior standard deviation, lower and upper bound
            for the 95 % credible interval and the posterior
            probability for each quantity to be larger or smaller than 0.

        """
        X_pred = contrast_matrix.index.values
        contrast_names = contrast_matrix.columns.values
        B = contrast_matrix.values.T

        if B.shape[0] == 1:
            B = np.vstack([B, B])
            contrast_names = np.append(contrast_names, [None])

        m, Sigma = self.calc_posterior(X_pred=X_pred, B=B)
        variances = get_diag(Sigma, progress=True)
        stderr = np.sqrt(variances)
        posterior = norm(m, stderr)
        p = posterior.cdf(0.0)
        p = np.max(np.vstack([p, 1 - p]), axis=0)
        dm = 2 * stderr
        result = pd.DataFrame(
            {
                "estimate": m,
                "std": stderr,
                "ci_95_lower": m - dm,
                "ci_95_upper": m + dm,
                "p(|x|>0)": p,
            },
            index=contrast_names,
        )
        result = result.loc[contrast_matrix.columns.values, :]
        return result

    def predict(self, X_pred=None, calc_variance=False):
        """
        Compute the Maximum a Posteriori (MAP) estimate of the phenotype at
        the provided or all genotypes

        Parameters
        ----------
        X_pred : array-like of shape (n_genotypes,)
            Vector containing the genotypes for which we want to predict the
            phenotype. If `n_genotypes == None` then predictions are provided
            for the whole sequence space

        calc_variance : bool (False)
            Option to also return the posterior variances for each individual
            genotype

        Returns
        -------
        function : pd.DataFrame of shape (n_genotypes, 1)
                   Returns the phenotypic predictions for each input genotype
                   in the column ``ypred`` and genotype labels as row names.
                   If ``calc_variance=True``, then it has an additional
                   column with the posterior variances for each genotype
        """

        t0 = time()
        post_mean, Sigma = self.calc_posterior(X_pred=X_pred)

        seqs = self.genotypes if X_pred is None else X_pred
        pred = pd.DataFrame({"y": post_mean}, index=seqs)

        if calc_variance:
            pred["y_var"] = get_diag(Sigma, progress=True)
            pred["std"] = np.sqrt(pred["y_var"])
            pred["ci_95_lower"] = pred["y"] - 2 * pred["std"]
            pred["ci_95_upper"] = pred["y"] + 2 * pred["std"]

        self.pred_time = time() - t0
        return pred


class SequenceInterpolator(SeqGaussianProcessRegressor):
    def __init__(
        self,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        cg_rtol=1e-16,
        **kwargs,
    ):
        self.cg_rtol = cg_rtol
        self.kwargs = kwargs

        self.initialized = False
        if seq_length is not None and (
            n_alleles is not None or alphabet_type != "custom"
        ):
            self.init(
                n_alleles=n_alleles,
                seq_length=seq_length,
                alphabet_type=alphabet_type,
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
                n_alleles=n_alleles,
                seq_length=seq_length,
                alphabet_type=alphabet_type,
                genotypes=genotypes,
            )
            self.define_precision_matrix()
            self.initialized = True

    def set_data(self, X, y):
        if np.any(np.isnan(y)):
            msg = "y vector contains nans"
            raise ValueError(msg)

        self.init(genotypes=X)
        self.X = X
        self.y = y

        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)

        z = np.full(self.n_genotypes, True)
        z[self.obs_idx] = False
        self.pred_idx = np.where(z)[0]

    def calc_cost(self, y):
        return quad(self.C, y)

    def calc_posterior_mean(self):
        C_II = SubMatrixOperator(
            self.C, row_idx=self.pred_idx, col_idx=self.pred_idx
        )
        C_II_inv = InverseOperator(C_II, method="cg")
        C_IB = SubMatrixOperator(
            self.C, row_idx=self.pred_idx, col_idx=self.obs_idx
        )
        b = C_IB @ self.y

        y_pred = np.zeros(self.n_genotypes)
        y_pred[self.obs_idx] = self.y
        y_pred[self.pred_idx] = -C_II_inv @ b
        return y_pred

    def predict(self):
        """
        Compute the Maximum a Posteriori (MAP) estimate of the phenotype at
        the provided or all genotypes

        Returns
        -------
        function : pd.DataFrame of shape (n_genotypes, 1)
                   Returns the phenotypic predictions for each input genotype
                   in the column ``ypred`` and genotype labels as row names.
                   If ``calc_variance=True``, then it has an additional
                   column with the posterior variances for each genotype
        """

        t0 = time()
        post_mean = self.calc_posterior_mean()
        pred = pd.DataFrame({"y": post_mean}, index=self.genotypes)
        self.pred_time = time() - t0
        return pred


class GaussianProcessRegressor(SeqGaussianProcessRegressor):
    def __init__(self, base_kernel, progress=True):
        self.K = base_kernel
        self.progress = progress

    def set_data(self, X, y, y_var=None):
        self.define_space(genotypes=X)
        self.X = X
        self.y = y

        if np.any(np.isnan(y)):
            msg = "y vector contains nans"
            raise ValueError(msg)

        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)

        if y_var is None:
            y_var = np.zeros(y.shape[0])

        if np.any(np.isnan(y_var)):
            msg = "y_var vector contains nans"
            raise ValueError(msg)

        self.y_var = y_var
        self.D_var = DiagonalOperator(y_var)
        self._K_BB = None

    def calc_posterior(self, X_pred=None, B=None):
        pred_idx = (
            np.arange(self.n_genotypes)
            if X_pred is None
            else self.get_obs_idx(X_pred)
        )

        K_aB = self.K.compute(x1=pred_idx, x2=self.obs_idx)
        K_aa = self.K.compute(x1=pred_idx, x2=pred_idx)
        K_Ba = self.K.compute(x1=self.obs_idx, x2=pred_idx)
        K_BB = self.K.compute(self.obs_idx, self.obs_idx, self.D_var)

        K_BB_inv = InverseOperator(K_BB, method="cg")
        mean_post = K_aB @ K_BB_inv @ self.y
        Sigma_post = K_aa - K_aB @ K_BB_inv @ K_Ba

        if B is not None:
            B = aslinearoperator(B)
            mean_post = B @ mean_post
            Sigma_post = B @ Sigma_post @ B.T

        return (mean_post, Sigma_post)

    def sample(self):
        a = np.random.normal(size=self.K.shape[1])
        y = self.K.matrix_sqrt() @ a
        return y

    def simulate(self, sigma=0, p_missing=0):
        """
        Simulates data under the specified Variance component priors

        Parameters
        ----------
        sigma : real
            Standard deviation of the experimental noise additional to the
            variance components

        p_missing : float between 0 and 1
            Probability of randomly missing genotypes in the simulated output
            data

        Returns
        -------
        data : pd.DataFrame of shape (n_genotypes, 3)
            DataFrame with the columns ``y_true``, ``y``and ``var``
            corresponding to the true function at each genotype, the
            observed values and the variance of the measurement respectively
            for each sequence or genotype indicated in the ``DataFrame.index``

        """
        y_true = self.sample()
        y = np.random.normal(y_true, sigma) if sigma > 0 else y_true
        y_var = np.full(self.n_genotypes, sigma**2, dtype=float)

        data = pd.DataFrame(
            {"y_true": y_true, "y": y, "y_var": y_var}, index=self.genotypes
        )

        if p_missing > 0:
            idxs = np.random.uniform(size=y.shape[0]) < p_missing
            data.loc[idxs, ["y", "y_var"]] = np.nan
        return data


class MinimizerRegressor(SeqGaussianProcessRegressor):
    def __init__(
        self,
        n_alleles=None,
        seq_length=None,
        alphabet_type="custom",
        progress=True,
        cg_rtol=1e-4,
    ):
        self.progress = progress
        self.cg_rtol = cg_rtol
        self.initialized = False

        if seq_length is not None:
            self.init(
                n_alleles=n_alleles,
                seq_length=seq_length,
                alphabet_type=alphabet_type,
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

    def set_data(self, X, y, y_var=None):
        if np.any(np.isnan(y)):
            msg = "y vector contains nans"
            raise ValueError(msg)

        if y_var is None:
            y_var = np.zeros(y.shape[0])

        if np.any(np.isnan(y_var)):
            msg = "y_var vector contains nans"
            raise ValueError(msg)

        self.init(genotypes=X)
        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)

        self.X = X
        self.y = y
        self.y_var = y_var

        y_var_inv_star = np.zeros(self.n_genotypes)
        y_var_inv_star[self.obs_idx] = 1 / y_var
        self.D_var_inv_star = DiagonalOperator(y_var_inv_star)
        self.b = np.zeros(self.n_genotypes)
        self.b[self.obs_idx] = y / y_var
        self._A_inv = None

    def calc_loss_prior(self, v):
        return quad(self.C, v)

    @property
    def A_inv(self):
        if not hasattr(self, "_A_inv") or self._A_inv is None:
            A = self.C + self.D_var_inv_star
            self._A_inv = InverseOperator(A, method="cg")
        return self._A_inv

    def calc_posterior_mean(self):
        mean_post = self.A_inv @ self.b
        return mean_post

    def calc_posterior_covariance(self, **kwargs):
        return self.A_inv

    def calc_posterior(self, X_pred=None, B=None):
        mean_post = self.calc_posterior_mean()
        Sigma_post = self.calc_posterior_covariance(mean_post=mean_post)

        if X_pred is not None:
            pred_idx = self.get_obs_idx(X_pred)
            mean_post = mean_post[pred_idx]
            Sigma_post = SubMatrixOperator(
                Sigma_post, row_idx=pred_idx, col_idx=pred_idx
            )

        if B is not None:
            B = aslinearoperator(B)
            mean_post = B @ mean_post
            Sigma_post = B @ Sigma_post @ B.T

        return (mean_post, Sigma_post)


class GeneralizedGaussianProcessRegressor(MinimizerRegressor):
    def get_phi0(self, phi0=None):
        phi = np.zeros(self.n_genotypes) if phi0 is None else phi0
        return phi

    def calc_grad(self, phi, return_grad=True, store_hess=True):
        Cphi = self.C @ phi
        grad = Cphi + self.likelihood.calc_grad(phi)
        return grad

    def calc_loss(self, phi, return_grad=True, store_hess=True):
        # Compute loss from the likelihood
        res = self.likelihood.calc_loss_grad_hess(phi)
        data_loss, data_grad, data_hess = res

        # Compute loss
        Cphi = self.C @ phi
        loss = 0.5 * np.dot(phi, Cphi) + data_loss

        if not return_grad:
            return loss

        # Compute gradient
        grad = Cphi + data_grad

        # Store hessian
        if store_hess:
            hess = self.C + DiagonalOperator(data_hess)
            self.hess = hess
        return (loss, grad)

    def calc_posterior_mean(self, phi0=None):
        phi0 = self.get_phi0(phi0)
        res = minimize(
            fun=self.calc_loss,
            jac=True,
            x0=phi0,
            method="L-BFGS-B",
            options=self.optimization_opts,
        )

        if not res.success:
            raise ValueError(res.message)

        phi = res.x
        self.opt_res = res
        return phi

    def calc_posterior_covariance(self, mean_post):
        w = self.likelihood.calc_loss_grad_hess(mean_post)[2]
        D = DiagonalOperator(1 / np.sqrt(w))
        A = D @ self.C @ D + IdentityOperator(self.n_genotypes)
        Sigma = D @ InverseOperator(A, method="cg") @ D
        return Sigma
