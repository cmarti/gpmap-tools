#!/usr/bin/env python
import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats.stats import pearsonr
from scipy.stats import norm

from gpmap.src.settings import PHI_LB, PHI_UB
from gpmap.src.utils import check_error, get_CV_splits, safe_exp
from gpmap.src.seq import (guess_space_configuration, get_alphabet,
                           get_seqs_from_alleles,  calc_msa_weights,
                           get_subsequences, calc_allele_frequencies,
                           calc_expected_logp, calc_genetic_code_aa_freqs)
from gpmap.src.linop import (DeltaPOperator, calc_covariance_distance,
                             ExtendedLinearOperator, DeltaKernelBasisOperator,
                             KernelOperator, ProjectionOperator,
                             VarianceComponentKernel,
                             DiagonalOperator)
from gpmap.src.matrix import inv_dot, inv_quad
from gpmap.src.aligner import VCKernelAligner


class SeqGaussianProcessRegressor(object):
    def __init__(self, expand_alphabet=True):
        self.expand_alphabet = expand_alphabet
    
    def get_regularization_constants(self):
        return(10**(np.linspace(self.min_log_reg, self.max_log_reg, self.num_reg)))
    
    def get_cv_iter(self, hyperparam_values):
        for fold, train, validation in get_CV_splits(X=self.X, y=self.y,
                                                     y_var=self.y_var,
                                                     nfolds=self.nfolds):
            for param in hyperparam_values:
                yield(param, fold, train, validation)
    
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
        

class GaussianProcessRegressor(SeqGaussianProcessRegressor):
    def __init__(self, base_kernel, progress=True):
        self.K = base_kernel
        self.progress = progress
    
    def set_data(self, X, y, y_var=None):
        self.define_space(genotypes=X)
        self.X = X
        self.y = y
        self.n_obs = y.shape[0]
        self.obs_idx = self.get_obs_idx(X)

        if y_var is None:
            y_var = np.zeros(y.shape[0])    
                
        self.y_var = y_var
        self.D_var = DiagonalOperator(y_var)
    
    @property
    def K_BB(self):
        if not hasattr(self, '_K_BB') or self._K_BB is None:
            self._K_BB = self.K.compute(self.obs_idx, self.obs_idx, self.D_var)
        return(self._K_BB)
    
    def calc_posterior_mean(self, cg_rtol=1e-8):
        K_aB = self.K.compute(x2=self.obs_idx)
        y_pred = K_aB @ inv_dot(self.K_BB, self.y, method='cg', rtol=cg_rtol)
        return(y_pred)
    
    def calc_posterior_variance_i(self, i, cg_rtol=1e-4):
        K_i = self.K.get_column(i)
        K_ii = K_i[i]
        K_Bi = K_i[self.obs_idx]
        post_var = K_ii - inv_quad(self.K_BB, K_Bi, method='cg', rtol=cg_rtol)
        return(post_var)
    
    def calc_posterior_variance(self, X_pred=None, cg_rtol=1e-4):
        pred_idx = np.arange(self.n_genotypes) if X_pred is None else self.get_obs_idx(X_pred)
        if self.progress:
            pred_idx = tqdm(pred_idx)

        post_vars = np.array([self.calc_posterior_variance_i(i, cg_rtol=cg_rtol)
                              for i in pred_idx])        
        return(post_vars)
    
    def predict(self, X_pred=None, calc_variance=False, cg_rtol=1e-16):
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
        msg = 'Make sure `lambdas` are specified for making predictions'
        check_error(hasattr(self, 'lambdas'), msg=msg)
        
        t0 = time()
        ypred = self.calc_posterior_mean(cg_rtol=cg_rtol)
        pred = pd.DataFrame({'ypred': ypred}, index=self.genotypes)
        if X_pred is not None:
            pred = pred.loc[X_pred, :]
        if calc_variance:
            pred['var'] = self.calc_posterior_variance(X_pred=X_pred, cg_rtol=cg_rtol)
        self.pred_time = time() - t0
        return(pred)
    
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
                 num_beta=20, min_log_beta=-2, max_log_beta=7,  cg_rtol=1e-16):
        self.beta = beta
        self.nfolds = nfolds
        self.num_reg = num_beta
        self.total_folds = self.nfolds * self.num_reg
        
        self.min_log_reg = min_log_beta
        self.max_log_reg = max_log_beta
        self.run_cv = cross_validation
        self.cv_loss_function = cv_loss_function

        if lambdas is not None:
            self.define_space(n_alleles=n_alleles, seq_length=seq_length,
                              alphabet_type=alphabet_type)
            self.set_lambdas(lambdas)
        
        self.cg_rtol = cg_rtol
        
    def set_lambdas(self, lambdas=None, k=None):
        K = VarianceComponentKernel(self.n_alleles, self.seq_length,
                                    lambdas=lambdas, k=k)
        self._K_BB = None
        self.lambdas = K.lambdas
        super().__init__(base_kernel=K)
    
    def set_data(self, X, y, y_var=None, cov=None, ns=None):
        super().set_data(X, y, y_var=y_var)
        self.cov = cov
        self.ns = ns

    def calc_covariance_distance(self, X, y):
        return(calc_covariance_distance(y, self.n_alleles,
                                        self.seq_length, 
                                        self.get_obs_idx(X)))
    
    def lambdas_to_variance(self, lambdas):
        variance_components = (lambdas * self.K.m_k)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)

    def get_cv_iter(self, hyperparam_values):
        for fold, train, validation in get_CV_splits(X=self.X, y=self.y,
                                                     y_var=self.y_var,
                                                     nfolds=self.nfolds):
            X_train, y_train, y_var_train = train 
            train_cov, train_ns = self.calc_covariance_distance(X_train, y_train)
            train = X_train, y_train, y_var_train, train_cov, train_ns

            X_test, y_test, y_var_test = validation
            test_cov, test_ns = self.calc_covariance_distance(X_train, y_train)
            validation = X_test, y_test, y_var_test, test_cov, test_ns

            for param in hyperparam_values:
                yield(param, fold, train, validation)

    def calc_cv_loss(self, cv_data):
        for beta, fold, train, test in tqdm(cv_data, total=self.total_folds):
            X_train, y_train, y_var_train, train_cov, train_ns = train 
            X_test, y_test, y_var_test, test_cov, test_ns = test

            # Find best lambdas
            self.set_data(X=X_train, y=y_train, y_var=y_var_train,
                          cov=train_cov, ns=train_ns)
            lambdas = self._fit(beta)

            # Calculate loss in test data
            if self.cv_loss_function == 'frobenius_norm':
                self.kernel_aligner.set_data(test_cov, test_ns)
                loss = self.kernel_aligner.calc_mse(lambdas)

            else:
                self.set_lambdas(lambdas)
                ypred = self.predict(X_test)['ypred'].values

                if self.cv_loss_function == 'logL':
                    loss = -norm.logpdf(y_test, loc=ypred, scale=np.sqrt(y_var_test)).sum()
                elif self.cv_loss_function == 'r2':
                    loss = -pearsonr(ypred, y_test)[0] ** 2
                else:
                    msg = 'Allowed loss functions are [frobenius_norm, r2, logL]'
                    raise ValueError(msg)
            
            yield({'beta': beta, 'fold': fold, 'loss': loss})

    def fit_beta_cv(self):
        beta_values = self.get_regularization_constants()
        cv_data = self.get_cv_iter(beta_values)
        cv_loss = self.calc_cv_loss(cv_data)
         
        self.cv_loss_df = pd.DataFrame(cv_loss)
        with np.errstate(divide='ignore'):
            self.cv_loss_df['log_beta'] = np.log10(self.cv_loss_df['beta'])
        loss = self.cv_loss_df.groupby('beta')['loss'].mean()
        self.beta = loss.index[np.argmin(loss)]
    
    def _fit(self, beta=None):
        if beta is None:
            beta = self.beta
            
        cov, ns = self.cov, self.ns
        if cov is None or ns is None:
            cov, ns = self.calc_covariance_distance(self.X, self.y)
        self.kernel_aligner.set_beta(beta)
        lambdas = self.kernel_aligner.fit(cov, ns)
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


class DeltaPEstimator(SeqGaussianProcessRegressor):
    def __init__(self, P, a=None, num_reg=20, nfolds=5,
                 a_resolution=0.1, max_a_max=1e12, fac_max=0.1, fac_min=1e-6,
                 opt_method='L-BFGS-B', optimization_opts={}, scale_by=1,
                 gtol=1e-3, ftol=1e-8):
        super().__init__()
        self.P = P
        self.a = a
        self.a_is_fixed = a is not None
        self.nfolds = nfolds
        
        msg = '"a" can only be None or >= 0'
        check_error(a is None or a >= 0, msg=msg)
        
        # Attributes to generate a values
        self.num_reg = num_reg
        self.total_folds = self.nfolds * self.num_reg
        
        # Default bounds for a in absence of a more clever method
        self.min_log_reg = -4
        self.max_log_reg = 10
        
        # Parameters to generate a grid in SeqDEFT, but should be generalizable
        # by just defining a distance metric for phi
        self.a_resolution = a_resolution
        self.max_a_max = max_a_max
        self.scale_by = scale_by
        self.fac_max = fac_max
        self.fac_min = fac_min
        
        # Optimization attributes
        self.opt_method = opt_method
        optimization_opts['gtol'] = gtol
        optimization_opts['ftol'] = ftol
        self.optimization_opts = optimization_opts
        
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.set_baseline_phi()
        self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
        self.kernel_basis = DeltaKernelBasisOperator(self.n_alleles, self.seq_length, self.P)

    def get_a_values(self):
        return(self.get_regularization_constants())
    
    def calc_loss_and_grad_finite(self, phi):
        DP_dot_phi = self.DP.dot(phi)
        loss = self._a / (2 * self.DP.n_p_faces) * np.sum(phi * DP_dot_phi)
        loss += self.calc_neg_log_likelihood(phi)
        if hasattr(self, 'calc_regularization'):
            loss += self.calc_regularization(phi)
        
        grad = self._a / self.DP.n_p_faces * DP_dot_phi
        grad += self.calc_neg_log_likelihood_grad(phi)
        if hasattr(self, 'calc_regularization'):
            grad += self.calc_regularization_grad(phi)
            
        return(loss, grad)
    
    def calc_loss_and_grad_inf(self, b):
        phi = self._b_to_phi(b)
        
        # Calculate loss
        loss = self.calc_neg_log_likelihood(phi)
        if hasattr(self, 'calc_regularization'):
            loss += self.calc_regularization(phi)
        
        # Calculate gradient
        grad = self.calc_neg_log_likelihood_grad(phi)
        if hasattr(self, 'calc_regularization'):
            grad += self.calc_regularization_grad(phi)
        grad = self._phi_to_b(grad)
        return(loss, grad)
    
    def calc_loss_and_grad(self, x):
        if np.isinf(self._a):
            return(self.calc_loss_and_grad_inf(x))
        else:
            return(self.calc_loss_and_grad_finite(x))
        
    def calc_cv_logL(self, cv_data, phi_initial=None):
        for a, fold, train, test in tqdm(cv_data, total=self.total_folds):
            (X_train, y_train, y_var_train), (X_test, y_test, y_var_test) = train, test

            self._set_data(X=X_train, y=y_train, y_var=y_var_train)
            phi = self._fit(a, phi_initial=phi_initial)
            
            self._set_data(X=X_test, y=y_test, y_var=y_var_test)
            test_logL = -self.calc_neg_log_likelihood(phi)
            yield({'a': a, 'fold': fold, 'logL': test_logL})

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
        cv_data = self.get_cv_iter(a_values)
        cv_logL = self.calc_cv_logL(cv_data, phi_initial=phi_inf) 
        
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
    
    def _fit(self, a, phi_initial=None):
        check_error(a >= 0, msg='"a" must be larger or equal than 0')
        self._a = a
        
        if a == 0 and hasattr(self, '_fit_a_0'):
            phi = self._fit_a_0()
            
        else:
            x0 = self.get_x0(phi_initial)
            res = minimize(fun=self.calc_loss_and_grad, jac=True,
                           x0=x0, method=self.opt_method,
                           options=self.optimization_opts)
            phi = self.get_res_phi(res)
        
        # a, N = a * scale_by, N *scale_by    
        return(phi)
    
    def set_baseline_phi(self, baseline_phi=None, X=None):
        if baseline_phi is None:
            self.baseline_phi = np.zeros(self.n_genotypes)
        else:
            msg = 'baseline_phi needs to be the size of n_genotypes'
            check_error(baseline_phi.shape[0] == self.n_genotypes, msg=msg)

            if X is None:
                self.baseline_phi = baseline_phi
            else:
                baseline_phi = pd.Series(baseline_phi, index=X)
                self.baseline_phi = baseline_phi.loc[self.genotypes].values
    
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


class SeqDEFT(DeltaPEstimator):
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
    def fill_zeros_counts(self, X, y):
        obs = pd.DataFrame({'x': X, 'y': y}).groupby(['x'])['y'].sum().reset_index()
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        data.loc[obs['x'].values] = obs['y'].values
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

        baseline_phi: array-like of shape (n_genotypes,)
            Vector containing the baseline_phi to include in the model

        baseline_X: array-like of shape (n_genotypes,)
            Vector containing the sequences associated with baseline_phi
        
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
        self.set_baseline_phi(baseline_phi=baseline_phi, X=baseline_X)
        phi_inf = self._fit(np.inf)
        
        if not self.a_is_fixed and force_fit_a:
            self.fit_a_cv(phi_inf=phi_inf)
            self._set_data(X, y)
        
        # Fit model with a_star or provided a
        phi = self._fit(self.a, phi_initial=phi_inf)
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
    
    def calc_neg_log_likelihood(self, phi):
        phi_aux = phi.copy() + self.baseline_phi
        S3 = self.N * np.sum(safe_exp(-phi_aux))
        
        phi_aux[np.logical_and(np.isinf(phi_aux), self.R == 0)] = 0
        S2 = self.N * np.sum(self.R * phi_aux)
        return(S2 + S3)
    
    def calc_neg_log_likelihood_grad(self, phi):
        # TODO: review gradient calculation in case of baseline_phi
        phi_aux = phi.copy() + self.baseline_phi
        grad_S2 =  self.N * self.R
        grad_S3 = -self.N * safe_exp(-phi_aux)
        return(grad_S2 + grad_S3)
    
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
        
        phi_max = self._fit(a_max, phi_initial=phi_inf)
        distance = D_geo(phi_max, phi_inf)
        
        while distance > self.a_resolution and a_max < self.max_a_max:
            a_max *= 10
            phi_max = self._fit(a_max, phi_initial=phi_inf)
            distance = D_geo(phi_max, phi_inf)
            
        return(a_max)
    
    def calc_a_min(self, phi_inf):
        a_min = self.DP.n_p_faces * self.fac_min
        
        phi_0 = self._fit(0)
        phi_min = self._fit(a_min, phi_initial=phi_inf)
        
        distance = D_geo(phi_min, phi_0)
        
        while distance > self.a_resolution:
            a_min /= 10
            phi_min = self._fit(a_min, phi_initial=phi_inf)
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
