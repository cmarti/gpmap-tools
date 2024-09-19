#!/usr/bin/env python
from functools import partial
from time import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.stats.stats import pearsonr

from gpmap.src.aligner import VCKernelAligner
from gpmap.src.gp import (
    GaussianProcessRegressor,
    GeneralizedGaussianProcessRegressor,
    MinimizerRegressor,
    SequenceInterpolator,
)
from gpmap.src.likelihood import SeqDEFTLikelihood
from gpmap.src.linop import (
    DeltaKernelBasisOperator,
    DeltaKernelRegularizerOperator,
    DeltaPOperator,
    ProjectionOperator,
    VarianceComponentKernel,
    calc_covariance_distance,
)
from gpmap.src.seq import (
    get_subsequences,
)
from gpmap.src.utils import (
    calc_cv_loss,
    check_error,
    get_cv_iter,
    get_CV_splits,
)


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


class SeqDEFT(GeneralizedGaussianProcessRegressor):
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
                 a=None, num_reg=20, nfolds=5, lambdas_P_inv=None,
                 a_resolution=0.1, max_a_max=1e12, fac_max=0.1, fac_min=1e-6,
                 optimization_opts={}, maxiter=1000, gtol=1e-3, ftol=1e-8):
        super().__init__()
        self.P = P
        self.a = a
        self._a = a
        self.set_lambdas_P_inv(lambdas_P_inv)
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
        self.baseline_phi = None
        
        # Optimization attributes
        opts = {'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter}
        optimization_opts.update(opts)
        self.optimization_opts = optimization_opts
        self.initialized = False

        if seq_length is not None:
            self.init(n_alleles=n_alleles, seq_length=seq_length,
                      alphabet_type=alphabet_type)
        elif alphabet_type != 'custom' or n_alleles is not None:
            msg = '`seq_length` must be specified together with'
            msg += '`n_alleles` or `alphabet_type`'
            raise ValueError(msg)
    
    def set_lambdas_P_inv(self, lambdas_P_inv):
        if lambdas_P_inv is None:
            self.lambdas_P_inv = None
        else:
            msg = 'lambdas_P_inv={} size is different from P={}'.format(lambdas_P_inv.shape[0], self.P)
            check_error(lambdas_P_inv.shape[0] == self.P, msg)
            self.lambdas_P_inv = lambdas_P_inv
    
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        if not self.initialized:
            self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                              genotypes=genotypes, alphabet_type=alphabet_type)
            self.likelihood = SeqDEFTLikelihood(self.genotypes)
            self.DP = DeltaPOperator(self.n_alleles, self.seq_length, self.P)
            self.kernel_basis = DeltaKernelBasisOperator(self.n_alleles, self.seq_length, self.P)
            self.initialized = True
            
            if self.lambdas_P_inv is None:
                self.kernel_regularizer = None
            else:    
                self.kernel_regularizer = DeltaKernelRegularizerOperator(self.kernel_basis,
                                                                         self.lambdas_P_inv)
            
    def cv_fit(self, data, a, phi0=None):
        X, y, _ = data
        self._set_data(X=X, y=y)
        phi = self._fit(a, phi0=phi0)
        return(phi)
    
    def cv_evaluate(self, data, phi):
        X, y, _ = data
        self._set_data(X=X, y=y)
        logL = self.likelihood.calc_logL(phi)
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
    
    def calc_cv_loss(self, cv_iter, total_folds):
        phi0_cache = {}
        for a, fold, train, test in tqdm(cv_iter, total=total_folds):
            phi = self.cv_fit(train, a, phi0=phi0_cache.get(fold, None))
            if np.isinf(a):
                phi0_cache[fold] = phi
            loss = self.cv_evaluate(test, phi)
            yield({'a': a, 'fold': fold, 'logL': loss})
    
    def fit_a_cv(self, phi_inf):
        a_values = np.append(np.inf, self.get_a_values(phi_inf=phi_inf))
        total_folds = a_values.shape[0] * self.nfolds

        cv_splits = get_CV_splits(X=self.X, y=self.y, y_var=self.y_var, nfolds=self.nfolds)
        cv_iter = get_cv_iter(cv_splits, a_values)
        cv_logL = self.calc_cv_loss(cv_iter, total_folds)
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
            raise ValueError(res.message)
        out = res.x
        
        if np.isinf(self._a):
            out = self._b_to_phi(out)
        return(out)
    
    def _fit(self, a, phi0=None):
        check_error(a >= 0, msg='"a" must be larger or equal than 0')
        self._a = a
        
        if a == 0 and hasattr(self, '_fit_a_0'):
            with np.errstate(divide='ignore'):
                phi = -np.log(self.R)
            self.opt_res = None
            
        elif np.isfinite(a):
            phi0 = self.get_x0(phi0)
            phi = self.calc_posterior_mean(phi0=phi0)
            
        else:
            x0 = self.get_x0(phi0)
            res = minimize(fun=self.calc_loss, jac=True, 
                           x0=x0, method='L-BFGS-B',
                           options=self.optimization_opts)
            phi = self.get_res_phi(res)
            self.opt_res = res
        return(phi)
    
    def _get_lambdas(self, a):
        self.DP.calc_lambdas()
        lambdas = np.zeros(self.DP.lambdas.shape)
        lambdas[self.P:] = self.DP.n_p_faces / (a * self.DP.lambdas[self.P:])
        if self.lambdas_P_inv is not None:
            lambdas[:self.P] = 1 / self.lambdas_P_inv
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
    
    def _set_data(self, X, y=None, allele_freqs=None):
        '''
        This is the part of data definition that may change during cross-validation
        '''
        self.likelihood.set_data(X, y=y, offset=self.baseline, 
                                 positions=self.positions,
                                 phylo_correction=self.phylo_correction,
                                 adjust_freqs=self.adjust_freqs,
                                 allele_freqs=allele_freqs)

    def set_baseline(self, X=None, baseline_phi=None):
        if baseline_phi is None:
            self.baseline = None
        else:
            msg = 'Sequences `X` associated to the baseline must be provided'
            check_error(X is not None, msg=msg)
            self.baseline = pd.Series(baseline_phi, index=X).loc[self.genotypes].values
    
    def set_data(self, X, y=None, positions=None,
                 baseline_X=None, baseline_phi=None,
                 phylo_correction=False,
                 adjust_freqs=False, allele_freqs=None):
        self.X = X
        self.y = y
        self.y_var = None
        
        self.init(genotypes=get_subsequences(X, positions=positions))
        self.positions = positions
        self.adjust_freqs = adjust_freqs
        self.phylo_correction = phylo_correction
        self.set_baseline(baseline_X, baseline_phi)
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
                      baseline_X=baseline_X, baseline_phi=baseline_phi,
                      phylo_correction=phylo_correction,
                      adjust_freqs=adjust_freqs, allele_freqs=allele_freqs)
        phi_inf = self._fit(np.inf)
        
        if not self.a_is_fixed and force_fit_a:
            self.fit_a_cv(phi_inf)
            self._set_data(X, y)
        
        # Fit model with a_star or provided a
        if np.isfinite(self.a):
            phi = self._fit(self.a, phi0=phi_inf)
        else:
            phi = phi_inf

        output = self.likelihood.phi_to_output(phi)
        return(output)
    
    def calc_loss_finite_a(self, phi, return_grad=True, store_hess=True):
        a_over_s = self._a / self.DP.n_p_faces
        
        if self.kernel_regularizer is None:
            self.C = a_over_s * self.DP
        else:
            self.C = a_over_s * self.DP + self.kernel_regularizer
            
        return(super().calc_loss(phi, return_grad=return_grad, store_hess=store_hess))

    def calc_loss_hessp(self, phi, p):
        if np.isfinite(self._a):
            return(self.hess @ p)
        else:
            msg = 'Hessian product not implemented for infinite a'
            raise ValueError(msg)
    
    def calc_loss_inf_a(self, b, return_grad=True):
        phi = self._b_to_phi(b)
        res = self.likelihood.calc_loss_grad_hess(phi)
        loss, grad, _ = res

        if self.kernel_regularizer is not None:
            Wb = self.kernel_regularizer.beta_dot(b)
            loss += np.dot(b, Wb)
        
        if return_grad:
            grad = self._phi_to_b(grad)
            if self.kernel_regularizer is not None:
                grad += 2 * Wb
            return(loss, grad)
        else:
            return(loss)
    
    def calc_loss(self, x, return_grad=True):
        if np.isinf(self._a):
            return(self.calc_loss_inf_a(x, return_grad=return_grad))
        else:
            return(self.calc_loss_finite_a(x, return_grad=return_grad))
    
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
            
        X = self.likelihood.sample(phi, N)
        return(X)
    

def D_geo(phi1, phi2):
    logQ1 = -phi1 - logsumexp(-phi1)
    logQ2 = -phi2 - logsumexp(-phi2)
    s = np.exp(logsumexp(0.5 * (logQ1 + logQ2)))
    x = min(s, 1)
    return 2 * np.arccos(x)
