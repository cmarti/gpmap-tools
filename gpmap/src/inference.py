#!/usr/bin/env python
import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm
from functools import partial
from scipy.optimize import minimize
from scipy.special import logsumexp, loggamma
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from scipy.sparse.linalg import aslinearoperator

from gpmap.src.settings import PHI_LB, PHI_UB
from gpmap.src.utils import (check_error, get_CV_splits, safe_exp, get_cv_iter,
                             calc_cv_loss)
from gpmap.src.seq import (guess_space_configuration, get_alphabet,
                           get_seqs_from_alleles,  calc_msa_weights,
                           get_subsequences, calc_allele_frequencies,
                           calc_expected_logp, calc_genetic_code_aa_freqs)
from gpmap.src.linop import (DeltaPOperator, calc_covariance_distance,
                             DeltaKernelBasisOperator, ProjectionOperator,
                             VarianceComponentKernel, SubMatrixOperator,
                             DiagonalOperator, IdentityOperator, InverseOperator)
from gpmap.src.matrix import inv_dot, inv_quad, quad
from gpmap.src.aligner import VCKernelAligner


class SeqGaussianProcessRegressor(object):
    def __init__(self, expand_alphabet=True):
        self.expand_alphabet = expand_alphabet
    
    def get_regularization_constants(self):
        return(10**(np.linspace(self.min_log_reg, self.max_log_reg, self.num_reg)))
    
    def fit_beta_cv(self):
        beta_values = self.get_regularization_constants()
        cv_splits = get_CV_splits(X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds)
        cv_iter = get_cv_iter(cv_splits, beta_values, process_data=self.process_data)
        cv_loss = calc_cv_loss(cv_iter, self.cv_fit, self.cv_evaluate,
                               total_folds=beta_values.shape[0] * self.nfolds)
        self.cv_loss_df = pd.DataFrame(cv_loss)
        
        with np.errstate(divide='ignore'):
            self.cv_loss_df['log_beta'] = np.log10(self.cv_loss_df['beta'])
            
        loss = self.cv_loss_df.groupby('beta')['loss'].mean()
        self.beta = loss.index[np.argmin(loss)]
    
    def define_space(self, seq_length=None, n_alleles=None, genotypes=None,
                     alphabet_type='custom'):
        if genotypes is not None:
            configuration = guess_space_configuration(genotypes,
                                                      ensure_full_space=False,
                                                      force_regular=True,
                                                      force_regular_alleles=False)
            seq_length = configuration['length']
            alphabet = [sorted(a) for a in configuration['alphabet']]
            n_alleles = configuration['n_alleles'][0]
            n_alleles = len(alphabet[0])
        else:
            msg = 'Either seq_length or genotypes must be provided'
            check_error(seq_length is not None, msg=msg)
            alphabet = get_alphabet(n_alleles=n_alleles,
                                    alphabet_type=alphabet_type)
            n_alleles = len(alphabet)
            alphabet = [alphabet] * seq_length
        
        self.set_config(n_alleles, seq_length, alphabet)
    
    def get_obs_idx(self, seqs):
        obs_idx = self.genotype_idxs[seqs]
        return(obs_idx)
    
    def set_config(self, n_alleles, seq_length, alphabet):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.alphabet = alphabet
        self.n_genotypes = n_alleles ** seq_length
        self.genotypes = np.array(list(get_seqs_from_alleles(alphabet)))
        self.genotype_idxs = pd.Series(np.arange(self.n_genotypes),
                                       index=self.genotypes)
    
    def calc_posterior_variance(self, X_pred=None, cg_rtol=1e-4):
        pred_idx = np.arange(self.n_genotypes) if X_pred is None else self.get_obs_idx(X_pred)
        if self.progress:
            pred_idx = tqdm(pred_idx)

        post_vars = np.array([self.calc_posterior_variance_i(i, cg_rtol=cg_rtol)
                              for i in pred_idx])        
        return(post_vars)

    def make_contrasts(self, contrast_matrix, cg_rtol=1e-4):
        """
        Computes the posterior distribution of linear combinations of genotypes
        under the specific Gaussian Process prior.

        Parameters
        ----------
        contrast_matrix: pd.DataFrame of shape (n_genotypes, n_contrasts)
            DataFrame containing the linear combinations of genotypes for
            which to compute the summary of the posterior distribution

        cg_rtol: float
            Relative tolerance of Conjugate Gradient algorithm used
            to compute the posterior

        Returns
        -------
        contrasts: pd.DataFrame of shape (n_contrasts, 5)
            DataFrame containing the summary of the posterior for each of
            the provided contrasts. This includes the estimate, 
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

        m, S = self.calc_posterior(X_pred=X_pred, B=B, cg_rtol=cg_rtol)
        S = S @ np.eye(S.shape[1])
        stderr = np.sqrt(np.diag(S))
        posterior = norm(m, stderr)
        p = posterior.cdf(0.)
        p = np.max(np.vstack([p, 1-p]), axis=0)
        dm = 2 * stderr
        result = pd.DataFrame({'estimate' : m, 'std': stderr,
                               'ci_95_lower': m - dm, 'ci_95_upper': m + dm,
                               'p(|x|>0)' : p}, index=contrast_names)
        result = result.loc[contrast_matrix.columns.values, :]
        return(result)

    def predict(self, X_pred=None, calc_variance=False, cg_rtol=1e-4):
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
        ypred = self.calc_posterior_mean(cg_rtol=cg_rtol)
        pred = pd.DataFrame({'y': ypred}, index=self.genotypes)
        if X_pred is not None:
            pred = pred.loc[X_pred, :]
        if calc_variance:
            pred['y_var'] = self.calc_posterior_variance(X_pred=X_pred, cg_rtol=cg_rtol)
            pred['std'] = np.sqrt(pred['y_var'])
            pred['ci_95_lower'] = pred['y'] - 2 * pred['std']
            pred['ci_95_upper'] = pred['y'] + 2 * pred['std']

        self.pred_time = time() - t0
        return(pred)
        
        
class SequenceInterpolator(SeqGaussianProcessRegressor):
    def __init__(self, n_alleles=None, seq_length=None, alphabet_type='custom',
                 cg_rtol=1e-16, **kwargs):
        self.cg_rtol = cg_rtol
        self.kwargs = kwargs
        
        self.initialized = False
        if seq_length is not None and (n_alleles is not None or alphabet_type != 'custom'):
            self.init(n_alleles=n_alleles, seq_length=seq_length, alphabet_type=alphabet_type)
            
    def init(self, seq_length=None, n_alleles=None, genotypes=None, alphabet_type='custom'):
        if not self.initialized:
            self.define_space(n_alleles=n_alleles, seq_length=seq_length,
                              alphabet_type=alphabet_type, genotypes=genotypes)
            self.define_precision_matrix()
            self.initialized = True
    
    def set_data(self, X, y):
        if np.any(np.isnan(y)):
            msg = 'y vector contains nans'
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
        return(quad(self.C, y))
        
    def predict(self):
        C_II = SubMatrixOperator(self.C, row_idx=self.pred_idx, col_idx=self.pred_idx)
        C_II_inv = InverseOperator(C_II, method='cg', rtol=self.cg_rtol)
        C_IB = SubMatrixOperator(self.C, row_idx=self.pred_idx, col_idx=self.obs_idx)
        b = C_IB @ self.y
        
        y_pred = np.zeros(self.n_genotypes)
        y_pred[self.obs_idx] = self.y
        y_pred[self.pred_idx] = -C_II_inv @ b
        return(y_pred)


class MinimumEpistasisInterpolator(SequenceInterpolator):
    def __init__(self, P=2, n_alleles=None, seq_length=None, alphabet_type='custom',
                 cg_rtol=1e-16):
        self.P = P
        super().__init__(n_alleles=n_alleles, seq_length=seq_length,
                         alphabet_type=alphabet_type, cg_rtol=cg_rtol)
    
    def define_precision_matrix(self):
        self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
        self.C = 1 / self.DP.n_p_faces * self.DP
        self.p = self.DP.n_p_faces_genotype
        
    def smooth(self, y_pred):
        y_pred -= 1 / self.p * self.DP @ y_pred
        return(y_pred)


class GaussianProcessRegressor(SeqGaussianProcessRegressor):
    def __init__(self, base_kernel, progress=True):
        self.K = base_kernel
        self.progress = progress
    
    def set_data(self, X, y, y_var=None):
        self.define_space(genotypes=X)
        self.X = X
        self.y = y
        
        if np.any(np.isnan(y)):
            msg = 'y vector contains nans'
            raise ValueError(msg)
        
        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)

        if y_var is None:
            y_var = np.zeros(y.shape[0])    
            
        if np.any(np.isnan(y_var)):
            msg = 'y_var vector contains nans'
            raise ValueError(msg)
                
        self.y_var = y_var
        self.D_var = DiagonalOperator(y_var)
        self._K_BB = None
    
    @property
    def K_BB(self):
        if not hasattr(self, '_K_BB') or self._K_BB is None:
            self._K_BB = self.K.compute(self.obs_idx, self.obs_idx, self.D_var)
        return(self._K_BB)
    
    def calc_posterior_mean(self, B=None, cg_rtol=1e-8):
        K_aB = self.K.compute(x2=self.obs_idx)
        y_pred = K_aB @ inv_dot(self.K_BB, self.y, method='cg', rtol=cg_rtol)
        return(y_pred)
    
    def calc_posterior_variance_i(self, i, cg_rtol=1e-4):
        K_i = self.K.get_column(i)
        K_ii = K_i[i]
        K_Bi = K_i[self.obs_idx]
        post_var = K_ii - inv_quad(self.K_BB, K_Bi, method='cg', rtol=cg_rtol)
        return(post_var)
    
    def calc_posterior(self, X_pred=None, B=None, cg_rtol=1e-4):
        pred_idx = np.arange(self.n_genotypes) if X_pred is None else self.get_obs_idx(X_pred)
        if B is None:
            B = IdentityOperator(pred_idx.shape[0])
        else:
            B = aslinearoperator(B)

        K_aB = self.K.compute(x1=pred_idx, x2=self.obs_idx)
        W_aB = B @ K_aB

        # Compute mean
        K_BB_inv = InverseOperator(self.K_BB, method='cg', rtol=cg_rtol)
        mean_post = W_aB @ K_BB_inv @ self.y

        # Compute covariance
        K_aa = self.K.compute(x1=pred_idx, x2=pred_idx)
        Sigma_post = B @ K_aa @ B.transpose() - W_aB @ K_BB_inv @ W_aB.transpose()
        return(mean_post, Sigma_post)
    
    def sample(self):
        a = np.random.normal(size=self.K.shape[1])
        y = self.K.matrix_sqrt() @ a
        return(y)
    
    def simulate(self, sigma=0, p_missing=0):
        '''
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
            DataFrame with the columns ``y_true``, ``y``and ``var`` corresponding
            to the true function at each genotype, the observed values and the
            variance of the measurement respectively for each sequence or 
            genotype indicated in the ``DataFrame.index`` 
        
        '''
        y_true = self.sample()
        y = np.random.normal(y_true, sigma) if sigma > 0 else y_true
        y_var = np.full(self.n_genotypes, sigma ** 2, dtype=float)
        
        data = pd.DataFrame({'y_true': y_true,
                             'y': y, 
                             'y_var': y_var},
                            index=self.genotypes)

        if p_missing > 0:
            idxs = np.random.uniform(size=y.shape[0]) < p_missing
            data.loc[idxs, ['y', 'y_var']]  = np.nan
        return(data)
    
#     ### Ongoing attempt to do evidence maximization ###
#     def neg_marginal_log_likelihood(self, params):
#         self.K.set_params(params)
#         self.K_inv_y = self.K.inv_dot(self.y)
#         mll1 = np.dot(self.y, self.K_inv_y) / 2
#         mll2 = self.K.calc_log_det(method='SLQ', n_vectors=5, degree=10) / 2
# #         mll3 = self.n / 2 * np.log(2 * np.pi)
#         mll = mll1 + mll2 
#         print(params, mll1, mll2)
#         return(mll)
    
#     def neg_marginal_log_likelihood_grad(self, params):
#         if np.any(params != self.K.get_params()):
#             self.K.set_params(params)
#             self.K_inv_y = self.K.inv_dot(self.y)
            
#         grad = []
#         for K_grad in self.K.grad():
            
#             def matvec(x):
#                 v = K_grad.dot(x)
#                 return(np.dot(self.K_inv_y, v) * self.K_inv_y - self.K.inv_dot(v))
            
#             op = ExtendedLinearOperator(matvec=matvec,
#                                         shape=self.K.shape, dtype=self.K.dtype)
#             grad.append(op.calc_trace(exact=False, n_vectors=10))
#         return(-np.array(grad))
    
#     def fit(self, X, y, y_var=None):
#         self.set_data(X, y, y_var=y_var)
        
#         x0 = self.K.get_params0()
#         f = self.neg_marginal_log_likelihood
#         f_grad = self.neg_marginal_log_likelihood_grad
#         res = minimize(fun=f, x0=x0,
#                        jac=f_grad, method='L-BFGS-B')
#         params = res.x
#         self.K.set_params(params)
#         return(params)


class MinimizerRegressor(SeqGaussianProcessRegressor):
    def __init__(self, n_alleles=None, seq_length=None, alphabet_type='custom',
                 progress=True, cg_rtol=1e-4):
        self.progress = progress
        self.cg_rtol = cg_rtol
        self.initialized = False
        
        if seq_length is not None:
            self.init(n_alleles=n_alleles, seq_length=seq_length,
                      alphabet_type=alphabet_type)
    
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        if not self.initialized:
            self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                              genotypes=genotypes, alphabet_type=alphabet_type)
            
    def set_data(self, X, y, y_var=None):
        if np.any(np.isnan(y)):
            msg = 'y vector contains nans'
            raise ValueError(msg)

        if y_var is None:
            y_var = np.zeros(y.shape[0])    
            
        if np.any(np.isnan(y_var)):
            msg = 'y_var vector contains nans'
            raise ValueError(msg)
        
        self.init(genotypes=X)
        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)
        
        self.X = X
        self.y = y
        self.y_var = y_var
        
        self.y_star = np.zeros(self.n_genotypes)
        self.y_star[self.obs_idx] = y
        self.y_var_inv_star = np.zeros(self.n_genotypes)
        self.y_var_inv_star[self.obs_idx] = 1 / y_var
        self.D_var_inv_star = DiagonalOperator(self.y_var_inv_star)
        self._A_inv = None
    
    def calc_loss_prior(self, v):
        return(quad(self.C, v))
    
    @property
    def A_inv(self):
        if not hasattr(self, '_A_inv') or self._A_inv is None:
            A = self.C + self.D_var_inv_star
            self._A_inv = InverseOperator(A, method='cg', rtol=self.cg_rtol)
        return(self._A_inv)

    def calc_posterior_mean(self, B=None, cg_rtol=1e-8):
        mean_post = self.A_inv @ self.D_var_inv_star @ self.y_star
        if B is not None:
            mean_post = B @ mean_post
        return(mean_post)
    
    def calc_posterior_variance_i(self, i, cg_rtol=1e-4):
        i = np.array([i])
        post_var = SubMatrixOperator(self.A_inv, row_idx=i, col_idx=i) @ np.eye(1)
        return(post_var[0, 0])
        
    def calc_posterior(self, X_pred=None, B=None):
        mean_post = self.A_inv @ self.D_var_inv_star @ self.y_star
        Sigma_post = self.A_inv
        
        if X_pred is not None:
            pred_idx = self.get_obs_idx(X_pred)
            mean_post = mean_post[pred_idx]
            Sigma_post = SubMatrixOperator(Sigma_post, row_idx=pred_idx, col_idx=pred_idx)
        
        if B is not None:
            B = aslinearoperator(B)
            mean_post = B @ mean_post
            Sigma_post = B @ Sigma_post @ B.T
            
        return(mean_post, Sigma_post)


class MinimumEpistasisRegression(MinimizerRegressor):
    def __init__(self, P, a=None, n_alleles=None, seq_length=None, alphabet_type='custom',
                 nfolds=5, num_reg=20, min_log_reg=-2, max_log_reg=6, 
                 progress=True, cg_rtol=1e-4):
        self.a = a
        self.P = P

        self.nfolds = nfolds
        self.num_reg = num_reg
        self.total_folds = self.nfolds * self.num_reg
        self.min_log_reg = min_log_reg
        self.max_log_reg = max_log_reg
        super().__init__(n_alleles=n_alleles, seq_length=seq_length,
                         alphabet_type=alphabet_type, progress=progress,
                         cg_rtol=cg_rtol)
    
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        if not self.initialized:
            self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                              genotypes=genotypes, alphabet_type=alphabet_type)
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
        return(y_pred)

    def cv_evaluate(self, data, y_pred):
        X, y, y_var = data
        pred_idx = self.get_obs_idx(X)
        logL = norm.logpdf(y, loc=y_pred[pred_idx], scale=np.sqrt(y_var)).mean()
        return(logL)

    def _a_to_sd(self, a):
        return(np.sqrt(self.DP.n_p_faces / a))
    
    def _sd_to_a(self, sd):
        return(self.DP.n_p_faces / sd ** 2)
    
    def get_cv_logL_df(self, cv_logL):
        with np.errstate(divide = 'ignore'):
            cv_log_L = pd.DataFrame(cv_logL)
            cv_log_L['log_a'] = np.log10(cv_log_L['a'])
            cv_log_L['sd'] = self._a_to_sd(cv_log_L['a'])
            cv_log_L['log_sd'] = np.log10(cv_log_L['sd'])
        return(cv_log_L)
    
    def get_ml_a(self, cv_logL_df):
        df = cv_logL_df.groupby('a')['logL'].mean()
        return(df.index[np.argmax(df)])
    
    def fit_a_cv(self):
        a_values = self.get_regularization_constants()
        cv_splits = get_CV_splits(X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds)
        cv_iter = get_cv_iter(cv_splits, a_values)
        cv_logL = calc_cv_loss(cv_iter, self.cv_fit, self.cv_evaluate,
                               total_folds=a_values.shape[0] * self.nfolds,
                               param_label='a', loss_label='logL')
        self.logL_df = self.get_cv_logL_df(cv_logL)    
        self.set_a(self.get_ml_a(self.logL_df))
    
    def fit(self, X, y, y_var=None):
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
            self.fit_a_cv()
            self.set_data(X, y, y_var=y_var)


class VCregression(GaussianProcessRegressor):
    '''
        Variance Component regression model that allows inference and prediction
        of a scalar function in sequence spaces under a Gaussian Process prior
        parametrized by the contribution of the different orders of interaction
        to the observed genetic variability of a continuous phenotype
        
        It requires the use of the same number of alleles per sites
            
    '''
    def __init__(self, lambdas=None, n_alleles=None, seq_length=None, alphabet_type='custom',
                 beta=0, cross_validation=False, nfolds=5, cv_loss_function='frobenius_norm',
                 num_beta=20, min_log_beta=-2, max_log_beta=7,  cg_rtol=1e-16, progress=True):
        self.progress = progress
        self.beta = beta
        self.nfolds = nfolds
        self.num_reg = num_beta
        self.total_folds = self.nfolds * self.num_reg
        
        self.min_log_reg = min_log_beta
        self.max_log_reg = max_log_beta
        self.run_cv = cross_validation
        self.set_cv_loss_function(cv_loss_function)

        if seq_length is not None and (n_alleles is not None or alphabet_type != 'custom'):
            self.define_space(n_alleles=n_alleles, seq_length=seq_length,
                              alphabet_type=alphabet_type)

        if lambdas is not None:
            self.set_lambdas(lambdas)
        
        self.cg_rtol = cg_rtol
        
    def set_lambdas(self, lambdas=None, k=None):
        K = VarianceComponentKernel(self.n_alleles, self.seq_length,
                                    lambdas=lambdas, k=k)
        self._K_BB = None
        self.lambdas = K.lambdas
        super().__init__(base_kernel=K, progress=self.progress)
    
    def set_data(self, X, y, y_var=None, cov=None, ns=None):
        super().set_data(X, y, y_var=y_var)
        self.cov = cov
        self.ns = ns
        self.sigma2 = 0. if y_var is None else np.nanmin(y_var)

    def calc_covariance_distance(self, X, y):
        return(calc_covariance_distance(y, self.n_alleles, self.seq_length, 
                                        self.get_obs_idx(X)))
    
    def lambdas_to_variance(self, lambdas):
        variance_components = (lambdas * self.K.m_k)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)
    
    def get_variance_component_df(self, lambdas):
        s = self.seq_length + 1
        k = np.arange(s)
        vc_perc = np.zeros(s)
        vc_perc[1:] = self.lambdas_to_variance(lambdas)
        df = pd.DataFrame({'k': k, 'lambdas': lambdas,
                           'var_perc': vc_perc,
                           'var_perc_cum': np.cumsum(vc_perc)})
        return(df)

    def process_data(self, data):
        X, y, y_var = data
        cov, ns = self.calc_covariance_distance(X, y)
        return(X, y, y_var, cov, ns)
    
    def set_cv_loss_function(self,cv_loss_function):
        allowed_functions = ['frobenius_norm', 'logL', 'r2']
        if cv_loss_function not in allowed_functions:
            msg = 'Loss function {} not allowed. Choose from {}'
            raise ValueError(msg.format(cv_loss_function, allowed_functions))
        self.cv_loss_function = cv_loss_function
            
    def cv_fit(self, data, beta):
        X, y, y_var, cov, ns = data
        self.set_data(X=X, y=y, y_var=y_var, cov=cov, ns=ns)
        lambdas = self._fit(beta)
        return(lambdas)
    
    def cv_evaluate(self, data, lambdas):
        X, y, y_var, cov, ns = data
        
        if self.cv_loss_function == 'frobenius_norm':
            # TODO: unclear how to properly deal with the variance here
            self.kernel_aligner.set_data(cov, ns, sigma2=y_var.min())
            loss = self.kernel_aligner.calc_loss(lambdas, beta=0, return_grad=False)

        else:
            self.set_lambdas(lambdas)
            ypred = self.predict(X)['y'].values

            if self.cv_loss_function == 'logL':
                loss = -norm.logpdf(y, loc=ypred, scale=np.sqrt(y_var)).sum()
            elif self.cv_loss_function == 'r2':
                loss = -pearsonr(ypred, y)[0] ** 2
            else:
                msg = 'Allowed loss functions are [frobenius_norm, r2, logL]'
                raise ValueError(msg)
            
        return(loss)

    def _fit(self, beta=None):
        if beta is None:
            beta = self.beta
            
        cov, ns, sigma2 = self.cov, self.ns, self.sigma2
        if cov is None or ns is None:
            cov, ns = self.calc_covariance_distance(self.X, self.y)
            sigma2 = np.nanmin(self.y_var)

        self.kernel_aligner.set_beta(beta)
        lambdas = self.kernel_aligner.fit(cov, ns, sigma2=sigma2)
        return(lambdas)
    
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
        self.kernel_aligner = VCKernelAligner(n_alleles=self.n_alleles,
                                              seq_length=self.seq_length)
        self.set_data(X, y, y_var=y_var)
        
        if self.run_cv:
            self.fit_beta_cv()
            self.set_data(X, y, y_var=y_var)
        
        lambdas = self._fit()
        
        self.fit_time = time() - t0
        self.set_lambdas(lambdas)
        self.vc_df = self.get_variance_component_df(lambdas)


class SeqDEFT(SeqGaussianProcessRegressor):
    '''
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
    
    '''
    def __init__(self, P, n_alleles=None, seq_length=None, alphabet_type='custom',
                 a=None, num_reg=20, nfolds=5,
                 a_resolution=0.1, max_a_max=1e12, fac_max=0.1, fac_min=1e-6,
                 optimization_opts={}, maxiter=1000, gtol=1e-3, ftol=1e-8):
        super().__init__()
        self.P = P
        self.a = a
        self._a = a
        self.a_is_fixed = a is not None
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
        
        # Optimization attributes
        opts = {'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter}
        self.optimization_opts = optimization_opts.update(opts)
        self.initialized = False

        if seq_length is not None:
            self.init(n_alleles=n_alleles, seq_length=seq_length,
                      alphabet_type=alphabet_type)
        elif alphabet_type != 'custom' or n_alleles != None:
            msg = '`seq_length` must be specified together with'
            msg += '`n_alleles` or `alphabet_type`'
            raise ValueError(msg)
        
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        if not self.initialized:
            self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                              genotypes=genotypes, alphabet_type=alphabet_type)
            self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
            self.kernel_basis = DeltaKernelBasisOperator(self.n_alleles, self.seq_length, self.P)
            self.baseline_phi = np.zeros(self.n_genotypes)
            self.initialized = True

    def get_a_values(self):
        return(self.get_regularization_constants())
    
    def cv_fit(self, data, a, phi0=None):
        X, y, _ = data
        self._set_data(X=X, y=y)
        phi = self._fit(a, phi0=phi0)
        return(phi)
    
    def cv_evaluate(self, data, phi):
        X, y, _ = data
        self._set_data(X=X, y=y)
        logL = self.calc_logL(phi)
        return(logL)
    
    def get_cv_logL_df(self, cv_logL):
        with np.errstate(divide = 'ignore'):
            cv_log_L = pd.DataFrame(cv_logL)
            cv_log_L['log_a'] = np.log10(cv_log_L['a'])
            cv_log_L['sd'] = self._a_to_sd(cv_log_L['a'])
            cv_log_L['log_sd'] = np.log10(cv_log_L['sd'])
        return(cv_log_L)
    
    def get_ml_a(self, cv_logL_df):
        df = cv_logL_df.groupby('a')['logL'].mean()
        return(df.index[np.argmax(df)])
    
    def fit_a_cv(self, phi_inf=None):
        if phi_inf is None:
            phi_inf = self._fit(np.inf)
        a_values = self.get_a_values(phi_inf=phi_inf)

        cv_splits = get_CV_splits(X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds)
        cv_iter = get_cv_iter(cv_splits, a_values)
        cv_logL = calc_cv_loss(cv_iter, partial(self.cv_fit, phi0=phi_inf), self.cv_evaluate,
                               total_folds=a_values.shape[0] * self.nfolds, param_label='a',
                               loss_label='logL')
        self.logL_df = self.get_cv_logL_df(cv_logL)    
        self.a = self.get_ml_a(self.logL_df)
    
    def _phi_to_b(self, phi):
        return(self.kernel_basis.transpose_dot(phi))
    
    def _b_to_phi(self, b):
        return(self.kernel_basis.dot(b))

    def _a_to_sd(self, a):
        return(np.sqrt(self.DP.n_p_faces / a))
    
    def _sd_to_a(self, sd):
        return(self.DP.n_p_faces / sd ** 2)
    
    def get_x0(self, x0=None):
        if x0 is None:
            x0 = np.log(self.n_genotypes) * np.ones(self.n_genotypes)
        
        if np.isinf(self._a):
            x0 = self._phi_to_b(x0)
        return(x0)
        
    def get_res_phi(self, res):
        if not res.success:
            print(res.message)
        out = res.x
        
        if np.isinf(self._a):
            out = self._b_to_phi(out)
        return(out)
    
    def _fit(self, a, phi0=None):
        check_error(a >= 0, msg='"a" must be larger or equal than 0')
        self._a = a
        
        if a == 0 and hasattr(self, '_fit_a_0'):
            phi = self._fit_a_0()
            
        elif np.isfinite(a) and self.n_genotypes > 1e4:
            x0 = self.get_x0(phi0)
            opts = {k: v for k, v in self.optimization_opts.items()
                    if k != 'ftol'}
            res = minimize(fun=self.calc_loss, jac=True, 
                           hessp=self.calc_loss_finite_hessp,
                           x0=x0, method='trust-krylov', options=opts)
        else:
            x0 = self.get_x0(phi0)
            res = minimize(fun=self.calc_loss, jac=True, 
                           x0=x0, method='L-BFGS-B',
                           options=self.optimization_opts)
        phi = self.get_res_phi(res)
        self.opt_res = res
        return(phi)
    
    def set_baseline_phi(self, X, baseline_phi):
        msg = 'baseline_phi needs to be the size of n_genotypes'
        check_error(baseline_phi.shape[0] == self.n_genotypes, msg=msg)

        msg = 'Sequences `X` associated to the baseline_phi must be provided'
        check_error(X is not None, msg=msg)

        self.baseline_phi = pd.Series(baseline_phi, index=X).loc[self.genotypes].values
    
    def _get_lambdas(self, a):
        self.DP.calc_lambdas()
        lambdas = np.zeros(self.DP.lambdas.shape)
        lambdas[self.P:] = self.DP.n_p_faces / (a * self.DP.lambdas[self.P:])
        return(lambdas)

    def simulate_phi(self, a):
        '''
        Simulates data under the specified `a` penalization for
        local P-epistatic coefficients
        
        Parameters
        ----------
        a : float
            Parameter related to the inverse of the variance of the P-order
            epistatic coefficients that are being penalized. Larger values
            induce stronger penalization and approximation to the 
            Maximum-Entropy model of order P-1.
        
        Returns
        -------
        phi : array-like of shape (n_genotypes,)
            Vector containing values for the latent phenotype or field
            sampled from the prior characterized by `a`
        '''
        
        lambdas = self._get_lambdas(a)
        x = np.random.normal(size=self.n_genotypes)
        W_sqrt = ProjectionOperator(self.n_alleles, self.seq_length,
                                    lambdas=lambdas).matrix_sqrt()
        phi = W_sqrt @ x
        return(phi)
    
    def fill_zeros_counts(self, X, y):
        obs = pd.DataFrame({'x': X, 'y': y}).groupby(['x'])['y'].sum().reset_index()
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        try:
            data.loc[obs['x'].values] = obs['y'].values
        except KeyError:
            msg = 'Sequences outside of sequence space found'
            raise KeyError(msg)
        return(data)
    
    def _set_data(self, X, y=None, allele_freqs=None, **kwargs):
        if y is None:
            y = calc_msa_weights(X, phylo_correction=self.phylo_correction)
        
        if self.adjust_freqs:
            if allele_freqs is None:
                allele_freqs = calc_allele_frequencies(X, y=y)
            elif isinstance(allele_freqs, dict):
                self.allele_freqs = allele_freqs
            else:
                self.allele_freqs = calc_genetic_code_aa_freqs(allele_freqs)
            
        self.X = get_subsequences(X, positions=self.positions)
        self.y = y
        self.y_var = None
        
        data = self.fill_zeros_counts(self.X, y).values
        self.N = data.sum()
        self.R = (data / self.N)
        self.counts = data
        self.multinomial_constant = loggamma(self.counts.sum() + 1) - loggamma(self.counts + 1).sum()
        self.obs_idx = data > 0.
    
    def set_data(self, X, y=None, positions=None, phylo_correction=False,
                 adjust_freqs=False, allele_freqs=None):
        self.init(genotypes=get_subsequences(X, positions=positions))
        self.positions = positions
        self.adjust_freqs = adjust_freqs
        self.phylo_correction = phylo_correction
        self._set_data(X, y=y, allele_freqs=allele_freqs)
    
    def fit(self, X, y=None, baseline_phi=None, baseline_X=None,
            positions=None, phylo_correction=False,
            force_fit_a=True, adjust_freqs=False, allele_freqs=None):
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
            
        force_fit_a : bool 
            Whether to re-fit ``a`` using cross-validation even if it is already
            defined a priori
        
        adjust_freqs: bool (False)
            Whether to correct densities by the expected allele frequencies
            in the full length sequences
        
        allele_freqs: dict or codon_table
            Dictionary containing the allele expected frequencies frequencies
            for every allele in the set of possible sequences or the codon
            table to use to genereate expected aminoacid frequencies
            If `None`, they will be calculated from the full length observed
            sequences.
            
        Returns
        -------
        
        landscape : pd.DataFrame (n_genotypes, 2)
            DataFrame containing the estimated function for each possible
            sequence in the space
        
        """
        self.set_data(X, y=y, positions=positions,
                      phylo_correction=phylo_correction,
                      adjust_freqs=adjust_freqs, allele_freqs=allele_freqs)
        self.set_baseline_phi(baseline_X, baseline_phi)
        phi_inf = self._fit(np.inf)
        
        if not self.a_is_fixed and force_fit_a:
            self.fit_a_cv(phi_inf=phi_inf)
            self._set_data(X, y)
        
        # Fit model with a_star or provided a
        if np.isfinite(self.a):
            phi = self._fit(self.a, phi0=phi_inf)
        else:
            phi = phi_inf

        output = self.phi_to_output(phi)
        return(output)
    
    def calc_regularization(self, phi):
        regularizer = 0
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.any() > 0:
                regularizer += np.sum((phi[flags] - PHI_UB)**2)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.any() > 0:
                regularizer += np.sum((phi[flags] - PHI_LB)**2)
        return(regularizer)
    
    def calc_regularization_grad(self, phi):
        regularizer = np.zeros(self.n_genotypes)
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi[flags] - PHI_UB)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi[flags] - PHI_LB)
        return(regularizer)
    
    def calc_regularization_hess(self, phi):
        regularizer = np.zeros(self.n_genotypes)
        flags = (phi > PHI_UB) | (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer[flags] = 2
        return(DiagonalOperator(regularizer))
    
    def calc_logL(self, phi):
        c = self.multinomial_constant
        obs_phi = self.phi_to_obs_phi(phi)
        logq = self.phi_to_logQ(obs_phi)
        return(c + np.dot(self.counts[self.obs_idx], logq[self.obs_idx]))
    
    def phi_to_obs_phi(self, phi):
        return(phi + self.baseline_phi)
    
    def obs_phi_to_exp_phi(self, obs_phi):
        return(self.N * safe_exp(-obs_phi))
    
    def calc_data_loss(self, obs_phi, exp_phi):
        return(self.N * np.dot(self.R, obs_phi) + exp_phi.sum())
    
    def calc_obs_phi_exp_phi_data_loss(self, phi):
        obs_phi = self.phi_to_obs_phi(phi)
        exp_phi = self.obs_phi_to_exp_phi(obs_phi)
        loss = self.calc_data_loss(obs_phi, exp_phi)
        if hasattr(self, 'calc_regularization'):
            loss += self.calc_regularization(obs_phi)
        return(obs_phi, exp_phi, loss)
    
    def calc_data_loss_grad(self, exp_phi):
        return(self.counts - exp_phi)
    
    def calc_loss_finite_a(self, phi, return_grad=True, store_hess=True):
        # Compute loss from the prior
        a_over_s = self._a / self.DP.n_p_faces
        C = a_over_s * self.DP
        Cphi = C @ phi
        loss = 0.5 * np.dot(phi, Cphi)

        # Compute loss from the likelihood
        res = self.calc_obs_phi_exp_phi_data_loss(phi)
        obs_phi, exp_phi, data_loss = res
        loss += data_loss
        print('Loss evaluation', loss)
        
        # Store hessian
        if store_hess:
            hess = C + DiagonalOperator(exp_phi)
            if hasattr(self, 'calc_regularization'):
                hess += self.calc_regularization_hess(obs_phi)
            self.hess = hess
        
        # Compute gradient
        if return_grad:
            grad = Cphi + self.calc_data_loss_grad(exp_phi)
            if hasattr(self, 'calc_regularization'):
                grad += self.calc_regularization_grad(obs_phi)
            return(loss, grad)
    
        return(loss)

    def calc_loss_finite_hessp(self, phi, p):
        return(self.hess @ p)
    
    def calc_loss_inf_a(self, b, return_grad=True):
        phi = self._b_to_phi(b)

        # Calculate loss
        res = self.calc_obs_phi_exp_phi_data_loss(phi)
        obs_phi, exp_phi, loss = res
        
        # Calculate gradient
        print('Loss evaluation', loss)
        if return_grad:
            # TODO: review for non-zero baselines
            grad = self.calc_data_loss_grad(exp_phi)
            if hasattr(self, 'calc_regularization'):
                grad += self.calc_regularization_grad(obs_phi)
            grad = self._phi_to_b(grad)
            return(loss, grad)
        
        return(loss)
    
    def calc_loss(self, x, return_grad=True):
        if np.isinf(self._a):
            return(self.calc_loss_inf_a(x, return_grad=return_grad))
        else:
            return(self.calc_loss_finite_a(x, return_grad=return_grad))
    
    def phi_to_output(self, phi):
        Q_star = self.phi_to_Q(phi)
        seq_densities = pd.DataFrame({'frequency': self.R, 'phi': phi,
                                      'Q_star': Q_star},
                                     index=self.genotypes)
        if self.adjust_freqs:
            exp_logp = calc_expected_logp(self.genotypes, self.allele_freqs)
            logp_adj = np.log(Q_star) - exp_logp
            seq_densities['adjusted_Q_star'] = self.phi_to_Q(-logp_adj)
        
        return(seq_densities)
    
    # Optional methods
    def calc_a_max(self, phi_inf):
        a_max = self.DP.n_p_faces * self.fac_max
        
        phi_max = self._fit(a_max, phi0=phi_inf)
        distance = D_geo(phi_max, phi_inf)
        
        while distance > self.a_resolution and a_max < self.max_a_max:
            a_max *= 10
            phi_max = self._fit(a_max, phi0=phi_inf)
            distance = D_geo(phi_max, phi_inf)
            
        return(a_max)
    
    def calc_a_min(self, phi_inf):
        a_min = self.DP.n_p_faces * self.fac_min
        
        phi_0 = self._fit(0)
        phi_min = self._fit(a_min, phi0=phi_inf)
        
        distance = D_geo(phi_min, phi_0)
        
        while distance > self.a_resolution:
            a_min /= 10
            phi_min = self._fit(a_min, phi0=phi_inf)
            distance = D_geo(phi_min, phi_0)
        return(a_min)

    def get_a_values(self, phi_inf=None):
        if phi_inf is None:
            phi_inf = self._fit(np.inf)
        a_min = self.calc_a_min(phi_inf) 
        a_max = self.calc_a_max(phi_inf)
        a_values = np.geomspace(a_min, a_max, self.num_reg)
        a_values = np.hstack([0, a_values, np.inf])
        self.total_folds = self.nfolds * (self.num_reg + 2)
        return(a_values)

    def phi_to_logQ(self, phi):
        return(-phi - logsumexp(-phi))
    
    def phi_to_Q(self, phi):
        return(np.exp(self.phi_to_logQ(phi + self.baseline_phi)))
    
    def _fit_a_0(self):
        with np.errstate(divide='ignore'):
            phi = -np.log(self.R)
        return(phi)
    
    def simulate(self, N, a=None, phi=None, seed=None):
        '''
        Simulates data under the specified `a` penalization for
        local P-epistatic coefficients
        
        Parameters
        ----------
        N : int
            Number of total sequences to sample
        
        a : float
            Parameter related to the inverse of the variance of the P-order
            epistatic coefficients that are being penalized. Larger values
            induce stronger penalization and approximation to the 
            Maximum-Entropy model of order P-1.
        
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
        '''
        
        if seed is not None:
            np.random.seed(seed)
        
        if phi is not None:
            check_error(phi.shape == (self.n_genotypes,),
                        msg='Ensure "phi" has the shape (n_genotypes,)')
        else:
            check_error(a is not None, '"a" must be provided if "phi=None"')
            phi = self.simulate_phi(a)
            
        Q = self.phi_to_Q(phi)
        X = np.random.choice(self.genotypes, size=N, replace=True, p=Q)
        return(X)
    

def D_geo(phi1, phi2):
    logQ1 = -phi1 - logsumexp(-phi1)
    logQ2 = -phi2 - logsumexp(-phi2)
    s = np.exp(logsumexp(0.5 * (logQ1 + logQ2)))
    x = min(s, 1)
    return 2 * np.arccos(x)
