#!/usr/bin/env python
from time import time

import numpy as np
import pandas as pd
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import norm
from scipy.optimize import minimize

from gpmap.likelihood import GaussianLikelihood
from gpmap.linop import (
    DiagonalOperator,
    IdentityOperator,
    InverseOperator,
    SelIdxOperator,
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
            X=self.likelihood.X,
            y=self.likelihood.y,
            y_var=self.likelihood.y_var,
            nfolds=self.nfolds,
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
        """
        Define the genotype space configuration for the object.

        This method sets up the genotype space by either inferring it from
        provided genotypes or constructing it based on sequence length and
        the number of alleles.

        seq_length : int, optional
            The length of the sequences in the genotype space. Required if
            `genotypes` is not provided. Defaults to None.
        n_alleles : int, optional
            The number of alleles per position in the sequence. Required if
            `genotypes` is not provided. Defaults to None.
        genotypes : list of str, optional
            A list of genotypes to infer the space configuration from. If
            provided, `seq_length` and `n_alleles` will be inferred from
            this list. Defaults to None.
        alphabet_type: str
            The type of alphabet to use when constructing the genotype space.
            Options include "custom", "dna", "rna", or "protein". Defaults to "custom".

        Raises
        ------
        ValueError
            If neither `seq_length` nor `genotypes` is provided.

        Notes
        -----
        - If `genotypes` is provided, the method will attempt to infer the
          sequence length, alphabet, and number of alleles from the given
          genotypes.
        - If `genotypes` is not provided, the method will construct the
          genotype space using the specified `seq_length`, `n_alleles`, and
          `alphabet_type`.

        """
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

    def sample_prior(self):
        """
        Generate a sample from the prior distribution.

        This method samples from the prior distribution by drawing random values
        from a standard normal distribution and transforming them using the square
        root of the covariance matrix. The resulting sample represents a realization
        of the prior distribution over genotypes.

        Returns
        -------
            f: numpy.ndarray
                A 1D array of shape (n_genotypes,) representing a sample from the prior
                distribution. Each element corresponds to a genotype's value drawn from
                the prior.

        """
        a = np.random.normal(size=self.n_genotypes)
        f = self.get_K_sqrt() @ a
        return f

    def _simulate(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        phi = self.sample_prior()
        y = self.likelihood.sample(phi, **kwargs)
        return phi, y

    def simulate(self, X=None, y_var=0.0, p_missing=0, seed=None):
        """
        Simulates data under the specified prior allowing for the addition of
        experimental Gaussian noise and the random omission of genotypes in
        the output data.

        X : array-like, optional
            Input sequences for which to generate the measurements `y`.
            If `None`, genotypes are randomly selected based on the missing
            probability `p_missing`. Default is `None`.
        y_var : float or array-like, optional
            Standard deviation of the experimental noise to be added to the
            variance components. If a float is provided, it is broadcast to
            match the shape of `X`. If an array is provided, its shape must
            match either the number of genotypes or the shape of `X`.
            Default is `0.`.
        p_missing : float, optional
            Probability (between 0 and 1) of randomly omitting genotypes in the
            simulated output data. Default is `0`.
        seed : float, optional
            Random seed for reproducibility. Default is `None`.

        Returns
        -------
        f : array-like
            The true simulated measurements without experimental noise.
        X : array-like
            The input sequences used for the simulation.
        y : array-like
            The simulated measurements with experimental noise added.
        y_var : array-like
            The standard deviation of the experimental noise for each input
            sequence.

        Raises
        ------
        ValueError
            If the shape of `y_var` does not match the expected dimensions.

        Examples
        --------
        Simulate data with default parameters:
         
        >>> f, X, y, y_var = gp.simulate()

        Simulate data with custom noise and missing probability:

        >>> f, X, y, y_var = gp.simulate(y_var=0.1, p_missing=0.2, seed=42)

        """

        if X is None:
            u = np.random.uniform(size=self.n_genotypes)
            X = self.genotypes[u > p_missing]

        if not hasattr(self, "likelihood"):
            self.likelihood = GaussianLikelihood(self.genotypes)
            self.likelihood.set_data(X, None, None)

        if isinstance(y_var, float):
            y_var = np.full(X.shape, y_var)
        elif y_var.shape[0] == self.n_genotypes:
            y_var = self.likelihood.Xop @ y_var
        elif y_var.shape[0] != X.shape[0]:
            msg = "y_var shape should match X shape"
            raise ValueError(msg)
        self.likelihood.set_data(X, None, y_var)
        f, y = self._simulate(seed=seed)
        return (f, X, y, y_var)

    def _transform_posterior(self, mean, cov, B):
        mean = B @ mean
        if cov is not None:
            cov = B @ cov @ B.transpose()
        return (mean, cov)

    def transform_posterior(self, mean_post, Sigma_post, X_pred=None, B=None):
        if X_pred is not None:
            pred_idx = self.get_obs_idx(X_pred)
            Z = SelIdxOperator(self.n_genotypes, pred_idx)
            mean_post, Sigma_post = self._transform_posterior(
                mean_post, Sigma_post, Z
            )

        if B is not None:
            B = aslinearoperator(B)
            mean_post, Sigma_post = self._transform_posterior(
                mean_post, Sigma_post, B
            )

        return (mean_post, Sigma_post)

    def calc_posterior(self, X_pred=None, B=None):
        """
        Calculate the posterior distribution for the given inputs.

        This method computes the posterior mean and covariance for the 
        specified prediction points or the entire sequence space. Optionally, 
        a linear transformation can be applied to the posterior distribution.

        Parameters
        ----------
        X_pred : array-like, optional
            Prediction points (genotypes) where the posterior distribution 
            is evaluated. If `None`, the posterior is computed for the entire 
            sequence space. Default is `None`.
        B : array-like or scipy.sparse.linalg.LinearOperator, optional
            Linear transformation to apply to the posterior distribution over 
            the complete space of possible sequences. If `None`, no 
            transformation is applied. Default is `None`.

        Returns
        -------
        mu : numpy.ndarray
            The transformed posterior mean, either for the specified prediction 
            points or the entire sequence space.
        Sigma : numpy.ndarray or scipy.sparse.linalg.LinearOperator
            The transformed posterior covariance matrix, either for the 
            specified prediction points or the entire sequence space.

        Notes
        -----
        - The posterior mean and covariance are computed using the Gaussian 
          Process prior and the observed data.
        - If `X_pred` is provided, the posterior is evaluated only for the 
          specified prediction points.
        - If `B` is provided, the posterior distribution is transformed using 
          the specified linear operator or matrix.

        Examples
        --------
        Compute the posterior distribution for the entire sequence space:

        >>> mu, Sigma = gp.calc_posterior()

        Compute the posterior distribution for specific prediction points:

        >>> mu, Sigma = gp.calc_posterior(X_pred=["AAA", "AAC"])

        Apply a linear transformation to the posterior distribution:

        >>> B = np.array([[1, 0, 0], [0, 1, 0]])
        >>> mu, Sigma = gp.calc_posterior(B=B)
        """

        mean_post = self.calc_posterior_mean()
        Sigma_post = self.calc_posterior_covariance()

        return self.transform_posterior(
            mean_post, Sigma_post, X_pred=X_pred, B=B
        )

    def make_contrasts(self, contrast_matrix):
        """
        Computes the posterior distribution of linear combinations of genotypes
        under the specified Gaussian Process prior.

        This method calculates the posterior mean, standard deviation, 
        95% credible intervals, and the posterior probability for each 
        linear combination of genotypes defined in the contrast matrix.

        Parameters
        ----------
        contrast_matrix : pd.DataFrame of shape (n_genotypes, n_contrasts)
            A DataFrame where each column represents a linear combination 
            of genotypes (contrast) for which the posterior distribution 
            is to be computed. The index should correspond to the genotypes.

        Returns
        -------
        contrasts : pd.DataFrame of shape (n_contrasts, 5)
            A DataFrame summarizing the posterior distribution for each 
            contrast. The columns include:

            - ``estimate``: Posterior mean for each contrast.

            - ``std``: Posterior standard deviation for each contrast.

            - ``ci_95_lower``: Lower bound of the 95% credible interval.

            - ``ci_95_upper``: Upper bound of the 95% credible interval.

            - ``p(|x|>0)``: Posterior probability that the absolute value 
              of the contrast is greater than 0.

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

        Notes
        -----
        - The MAP estimate is computed using the posterior mean.
        - If `calc_variance` is enabled, the credible intervals are calculated 
          as mean ± 2 * standard deviation.

        Examples
        --------
        Predict phenotypes for the entire genotype space:

        >>> pred = model.predict()

        Predict phenotypes for specific genotypes with variance:
        
        >>> pred = model.predict(X_pred=["AAA", "AAC"], calc_variance=True)
        """

        t0 = time()
        post_mean, Sigma = self.calc_posterior(X_pred=X_pred)

        seqs = self.genotypes if X_pred is None else X_pred
        pred = pd.DataFrame({"f": post_mean}, index=seqs)

        if calc_variance:
            pred["f_var"] = get_diag(Sigma, progress=True)
            pred["f_std"] = np.sqrt(pred["f_var"])
            pred["ci_95_lower"] = pred["f"] - 2 * pred["f_std"]
            pred["ci_95_upper"] = pred["f"] + 2 * pred["f_std"]

        self.pred_time = time() - t0
        return pred


class GaussianProcessRegressor(SeqGaussianProcessRegressor):
    def __init__(self, base_kernel, progress=True):
        self.K = base_kernel
        self.n_genotypes = self.K.shape[0]
        self.progress = progress

    def set_data(self, X, y, y_var=None):
        self.define_space(genotypes=X)
        self.likelihood = GaussianLikelihood(self.genotypes)
        self.likelihood.set_data(X, y, y_var)
        self.X = self.likelihood.Xop
        self.X_t = self.X.transpose()

    @property
    def K_xx_inv(self):
        if np.any(self.likelihood.y_var == 0.):
            K_xx = self.X @ self.K @ self.X_t + self.likelihood.D_var
            K_xx_inv = InverseOperator(K_xx, method="cg")
        else:
            # This formulation is better conditioned with heterogenous variances
            D_sqrt = self.likelihood.D_var_inv_sqrt
            Identity = IdentityOperator(self.likelihood.n_obs)
            A = D_sqrt @ self.X @ self.K @ self.X_t @ D_sqrt + Identity
            K_xx_inv = D_sqrt @ InverseOperator(A, method="cg") @ D_sqrt
        return(K_xx_inv)

    def calc_posterior_mean(self):
        mean_post = self.K @ self.X_t @ self.K_xx_inv @ self.likelihood.y
        return mean_post

    def calc_posterior_covariance(self):
        Sigma_post = (
            self.K - self.K @ self.X_t @ self.K_xx_inv @ self.X @ self.K
        )
        return Sigma_post

    def get_K_sqrt(self):
        return self.K.matrix_sqrt()


class MinimizerRegressor(SeqGaussianProcessRegressor):
    def __init__(
        self,
        seq_length=None,
        n_alleles=None,
        genotypes=None,
        alphabet_type="custom",
        progress=True,
        cg_rtol=1e-16,
    ):
        self.progress = progress
        self.cg_rtol = cg_rtol
        self.define_space(
            seq_length=seq_length,
            n_alleles=n_alleles,
            genotypes=genotypes,
            alphabet_type=alphabet_type,
        )

    def set_data(self, X, y, y_var=None):
        self.define_space(genotypes=X)
        self.likelihood = GaussianLikelihood(self.genotypes)
        self.likelihood.set_data(X, y, y_var)

    def calc_loss_prior(self, v):
        return quad(self.C, v)

    def calc_posterior_mean(self):
        X_t = self.likelihood.Xop.transpose()
        y = self.likelihood.y
        if self.likelihood.zero_var:
            Z = self.likelihood.Zop
            Z_t = self.likelihood.Zop.transpose()
            C_zz_inv = InverseOperator(Z @ self.C @ Z_t, method="cg")
            b = Z @ self.C @ X_t @ y
            mean_post = X_t @ y - Z_t @ C_zz_inv @ b
        else:
            A = InverseOperator(self.C + self.likelihood.D, method="cg")
            mean_post = A @ X_t @ self.likelihood.D_var_inv @ y
        return mean_post

    def calc_posterior_covariance(self):
        if self.likelihood.zero_var:
            Z = self.likelihood.Zop
            Z_t = self.likelihood.Zop.transpose()
            C_zz_inv = InverseOperator(Z @ self.C @ Z_t, method="cg")
            Sigma_post = Z_t @ C_zz_inv @ Z
        else:
            D = self.likelihood.D
            Sigma_post = InverseOperator(self.C + D, method="cg")
        return Sigma_post


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
        self.opt_res = res
        mean_post = res.x
        return mean_post

    def calc_posterior_covariance(self, mean_post):
        w = self.likelihood.calc_loss_grad_hess(mean_post)[2]
        D = DiagonalOperator(1 / np.sqrt(w))
        A = D @ self.C @ D + IdentityOperator(self.n_genotypes)
        Sigma_post = D @ InverseOperator(A, method="cg") @ D
        return Sigma_post

    def calc_posterior(self, X_pred=None, B=None):
        mean_post = self.calc_posterior_mean()
        Sigma_post = self.calc_posterior_covariance(mean_post)

        return self.transform_posterior(
            mean_post, Sigma_post, X_pred=X_pred, B=B
        )
