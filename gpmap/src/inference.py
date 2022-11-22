#!/usr/bin/env python
import random
from itertools import product, combinations
from tqdm import tqdm
from _functools import partial

import numpy as np
import pandas as pd

from numpy.linalg.linalg import matrix_power
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve import minres
from scipy.optimize._minimize import minimize
from scipy.special._basic import comb, factorial
from scipy.linalg.decomp_svd import orth
from scipy.linalg.basic import solve
from scipy.sparse.linalg import cg
from scipy.special._logsumexp import logsumexp

from gpmap.src.settings import U_MAX, PHI_LB, PHI_UB
from gpmap.src.utils import (get_sparse_diag_matrix, check_error,
                             calc_matrix_polynomial_dot, Frob, grad_Frob,
                             reciprocal, calc_Kn_matrix, calc_cartesian_product,
                             calc_cartesian_prod_freqs)
from gpmap.src.seq import (guess_space_configuration, get_alphabet,
                           get_seqs_from_alleles)


class LandscapeEstimator(object):
    def __init__(self, expand_alphabet=True):
        self.expand_alphabet = expand_alphabet
    
    def define_space(self, seq_length=None, n_alleles=None, genotypes=None,
                     alphabet_type='custom'):
        alphabet = None
        
        if genotypes is not None:
            configuration = guess_space_configuration(genotypes,
                                                      ensure_full_space=False)
            seq_length = configuration['length']
            alphabet_type = configuration['alphabet_type']
            n_alleles = configuration['n_alleles']
            
            if np.unique(n_alleles).shape[0] > 1:
                if self.expand_alphabet:
                    alphabet = set()
                    for alleles in configuration['alphabet']:
                        alphabet = alphabet.union(alleles)
                    alphabet = [sorted(alphabet)] * seq_length
                else:
                    msg = 'All sites must have the same number of alleles'
                    msg += 'per site or set "expand_alphabet=True" for filling'
                    msg += 'in the missing alleles observed at other sites'
                    raise ValueError(msg)
            else:
                alphabet = configuration['alphabet']
                
            self.space = SequenceSpace(seq_length=seq_length, alphabet=alphabet,
                                       alphabet_type='custom')
            n_alleles = len(alphabet[0])
        
        elif n_alleles is None or seq_length is None:
            msg = 'Either genotypes or both seq_length and n_alleles must '
            msg += 'be provided'
            raise ValueError(msg)
        
        if alphabet is None:
            alphabet = get_alphabet(n_alleles=n_alleles,
                                    alphabet_type=alphabet_type)
        
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.n_genotypes = n_alleles ** seq_length
        self.genotypes = np.array(list(get_seqs_from_alleles([alphabet] * seq_length)))
        self.genotype_idxs = pd.Series(np.arange(self.n_genotypes),
                                       index=self.genotypes)
    
    def calc_eigenvalues(self):
        lambdas = np.arange(self.seq_length + 1) * self.n_alleles
        return(lambdas)
    
    def calc_eigenvalue_multiplicity(self):
        self.lambdas_multiplicity = np.array([comb(self.seq_length, k) * (self.n_alleles - 1) ** k
                                              for k in range(self.seq_length + 1)], dtype=int)
    
    def calc_eig_vandermonde_matrix(self, lambdas=None):
        if lambdas is None:
            lambdas = np.arange(self.seq_length + 1) * self.n_alleles
        V = np.vstack([lambdas ** i for i in range(self.seq_length + 1)]).T
        return(V)
    
    def calc_polynomial_coeffs(self, numeric=False):
        '''
        Calculates the coefficients of the polynomial in L that represent 
        projection matrices into each of the kth eigenspaces.
        
        Returns
        -------
        B : array-like of shape (seq_length + 1, seq_length + 1)
            Matrix containing the b_i,k coefficients for power i on rows
            and order k on columns. One can obtain the coefficients for any
            combination of $\lambda_k$ values by scaling the coefficients 
            for each eigenspace by its eigenvalue and adding them up across
            different powers
        '''
        if numeric:
            self.B = np.linalg.inv(self.calc_eig_vandermonde_matrix())

        else:        
            l = self.seq_length
            lambdas = self.calc_eigenvalues()
            s = l + 1
            B = np.zeros((s, s))
            
            idx = np.arange(s)
            
            for k in idx:
                k_idx = idx != k
                k_lambdas = lambdas[k_idx]
                norm_factor = 1 / np.prod(k_lambdas - lambdas[k])
    
                for power in idx:
                    p = np.sum([np.product(v) for v in combinations(k_lambdas, l - power)])
                    B[power, k] = norm_factor * (-1) ** (power) * p
            
            self.B = B
            
        return(self.B)
    
    def get_polynomial_coeffs(self, k=None, lambdas=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)
        
        if lambdas is not None:
            if not hasattr(self, 'B') or self.B.shape[0] != self.seq_length + 1:
                self.calc_polynomial_coeffs()
            coeffs = self.B.dot(lambdas)
        else:
            coeffs = self.B[:, k].ravel() 
        
        return(coeffs)

    def project(self, y, k=None, lambdas=None):
        '''
        Projects the function ``y`` into the ``k``th eigenspace of the 
        graph Laplacian
        
        Parameters
        ----------
        y : array-like of shape (n_genotypes,)
            Vector with the function values associated to each of the genotypes
            ordered as in ``genotypes`` attribute
        
        k : int
            Order of the eigenspace in which to project the function ``y``
        
        lambdas : array-like of shape (seq_length + 1,)
            Vector containing the variance components by which to scale 
            the projection of ``y`` into each of the ``k``th eigenspaces
            
        Returns
        -------
        yk : array-like of shape (n_genotypes,)
            Vector containing the pure ``k``th order component of the function
            ``y`` or the sum of the projections into each component scaled by
            ``lambda[k]`` if ``lambdas`` is provided 
        '''
        msg = 'k and lambdas cannot be provided simultaneously'
        check_error(lambdas is None or k is None, msg=msg)
        
        coeffs = self.get_polynomial_coeffs(k=k, lambdas=lambdas)
        projection = calc_matrix_polynomial_dot(coeffs, self.M, y)
        return(projection)
    
    def calc_L_powers_unique_entries_matrix(self):
        """Construct entries of powers of L. 
        Column: powers of L. 
        Row: Hamming distance"""
        
        # TODO: replace this with the analytical solution from the Lagrange
        # polynomials

        l, a, s = self.seq_length, self.n_alleles, self.seq_length + 1
    
        # Construct C
        C = np.zeros([s, s])
        for i in range(s):
            for j in range(s):
                if i == j:
                    C[i, j] = i * (a - 2)
                if i == j + 1:
                    C[i, j] = i
                if i == j - 1:
                    C[i, j] = (l - j + 1) * (a - 1)
    
        # Construct D
        D = np.array(np.diag(l * (a - 1) * np.ones(s), 0))
    
        # Construct B
        B = D - C
    
        # Construct u
        u = np.zeros(s)
        u[0], u[1] = l * (a - 1), -1
    
        # Construct MAT column by column
        MAT = np.zeros([s, s])
        MAT[0, 0] = 1
        for j in range(1, s):
            MAT[:, j] = matrix_power(B, j-1).dot(u)
    
        # Invert MAT
        self.L_powers_unique_entries_inv = np.linalg.inv(MAT)
        self.L_powers_unique_entries = MAT
        

class VCregression(LandscapeEstimator):
    '''
        Variance Component regression model that allows inference and prediction
        of a scalar function in sequence spaces under a Gaussian Process prior
        parametrized by the contribution of the different orders of interaction
        to the observed genetic variability of a continuous phenotype
        
        It requires the use of the same number of alleles per sites
        
        Parameters
        ----------
        p : np.array of shape (seq_length, n_alleles)
            Stationary frequencies of alleles under a simple random walk
            to use for the generalized or skewed variance component regression.
            If not provided, regular VC regression will be applied
            
    '''
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom', ps=None):
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.calc_sparse_matrix_for_polynomial(ps=ps)
        self.calc_polynomial_coeffs()
        
        # For estimating lambdas
        self.calc_L_powers_unique_entries_matrix()
        self.calc_W_kd_matrix()
    
    def calc_eigenvalues(self):
        lambdas = np.arange(self.seq_length + 1)
        if not self.skewed:
            lambdas *= self.n_alleles
        return(lambdas)
    
    def calc_sparse_matrix_for_polynomial(self, ps=None):
        '''calc L or Q'''
        
        self.skewed = ps is not None
        self.ps = ps

        if self.skewed:
            sites_matrices = [calc_Kn_matrix(p=p) for p in ps]
            self.D_pi = get_sparse_diag_matrix(calc_cartesian_prod_freqs(ps))
        else:
            sites_matrices = [calc_Kn_matrix(self.n_alleles)] * self.seq_length
        
        M = calc_cartesian_product(sites_matrices)
        M = get_sparse_diag_matrix(M.sum(1).A1.flatten()) - M
        self.M = M
        
    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(1 / a**l * s)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
    
    def get_obs_idx(self, seqs):
        obs_idx = self.genotype_idxs[seqs]
        return(obs_idx)

    def set_data(self, X, y, variance=None):
        self.obs_idx = self.get_obs_idx(X)
        self.y = y
        self.n_obs = y.shape[0]

        if variance is None:
            variance = np.zeros(y.shape[0])        
        self.variance = variance
    
    def fit(self, X, y, variance=None,
            cross_validation=False, nfolds=10):
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
        
        variance : array-like of shape (n_obs,)
            Vector containing the empirical or experimental known variance for
            the measurements in `y`
            
        cross_validation : bool (False)
            Whether to use cross-validation and regularize the variance
            components so that their contribution tends to decay exponentially
            
        nfolds : int (10)
            Number of folds to use for cross-validation
        
        Returns
        -------
        lambdas: array-like of shape (seq_length + 1,)
            Variances for each order of interaction k inferred from the data
        
        """
        if not hasattr(self, 'space'):
            self.init(genotypes=X)
        
        self.set_data(X, y, variance=variance)
        
        if cross_validation:
            self.second_order_diff_matrix = self.calc_second_order_diff_matrix()
            self.lambdas = self.estimate_lambdas_cv(self.obs_idx, y, variance,
                                                    nfolds=nfolds)
        else:
            self.lambdas = self.estimate_lambdas(self.obs_idx, y)
        return(self.lambdas)
    
    def estimate_lambdas(self, obs_idx, y, betas=None,
                         second_order_diff_matrix=None):
        """solve for lambdas using least squares with the given rho_d 
        and N_d"""
        rho_d, N_d = self.compute_empirical_rho(obs_idx, y)
        
        # Calculate matrices for calculating Frobenius norm given lambdas
        M = self.construct_M(N_d)
        a = self.construct_a(rho_d, N_d) 

        if betas is None:        
            # Minimize the objective function Frob with lambda > 0
            M_inv = np.linalg.inv(M)
            lambda0 = np.array(np.dot(M_inv, a)).ravel()
            res = minimize(fun=Frob, jac=grad_Frob, args=(M, a),
                           x0=lambda0, method='L-BFGS-B',
                           bounds=[(0, None)] * (self.seq_length + 1))
            lambdas = res.x
            
        else:
            lambdas = []
            for beta in betas:
                res = minimize(fun=self.Frob_reg, args=(M, a, beta, second_order_diff_matrix),
                               x0=np.zeros(self.seq_length + 1), method='Powell',
                               options={'xtol': 1e-8, 'ftol': 1e-8})
                lambdas.append(np.exp(res.x))
                
        return(lambdas)
    
    def get_cv_indexes(self, n_obs, nfolds=10):
        order = np.arange(n_obs)
        random.shuffle(order)
        folds = np.array_split(order, nfolds)
        
        for i in range(nfolds):
            training_idx = np.hstack(folds[:i] + folds[i+1:])
            validation_idx = folds[i]
            yield(training_idx, validation_idx)
    
    def estimate_lambdas_cv(self, obs_idx, y, variance, nfolds=10, betas=None):
        """
        Estimate lambdas using regularized least squares with 
        regularization parameter chosen using cross-validation
    
        """
        if betas is None:
            betas = 10 ** np.arange(-2, 6, .5)
            
        mses = []
        for training_idx, validation_idx in tqdm(self.get_cv_indexes(y.shape[0], nfolds),
                                                 total=nfolds):
            y_train, y_validation = y[training_idx], y[validation_idx]

            # Estimate lambdas from training data
            lambdas = self.estimate_lambdas(obs_idx=training_idx,
                                            y=y_train, betas=betas)
            
            # Compute covariance on validation and substract the empirical variance
            rho, n = self.compute_empirical_rho(validation_idx, y_validation)
            rho[0] -= np.mean(variance[validation_idx])
            
            # Compute MSE between the expected and observed covariance matrix
            mses.append([np.sum((self.W_kd.T.dot(lambdas[i]) - rho) ** 2 * (1 / np.sum(n)) * n)
                         for i in range(betas.shape[0])])
        
        # Select the beta with best MSE across folds    
        mses = np.array(mses)
        mses = np.mean(mses, axis=0)
        betaopt = betas[np.argmin(mses)]
        
        # Re-estimate data
        lambdas = self.estimate_lambdas(obs_idx, y, betas=[betaopt])[0]
        self.beta_mse = dict(zip(betas, mses))
        return(lambdas)

    def get_noise_diag_matrix(self):
        self.E_sparse = get_sparse_diag_matrix(self.variance)
    
    def get_gt_to_data_matrix(self, idx=None, subset_idx=None):
        if idx is None:
            idx = self.genotype_idxs.values
        
        if subset_idx is None:
            obs_idx = idx.copy()
        else:
            obs_idx = idx[subset_idx]
            
        n_obs = obs_idx.shape[0]
        gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
                             shape=(self.n_genotypes, n_obs))
        return(gt2data)
    
    def compute_empirical_rho(self, obs_idx, y):
        n_obs = y.shape[0]
        gt2data = self.get_gt_to_data_matrix(obs_idx)
        
        seq_values = gt2data.dot(y)
        observed_seqs = gt2data.dot(np.ones(n_obs))

        # Compute rho_d and N_d
        size = self.seq_length + 1
        correlation, distance_class_ns = np.zeros(size), np.zeros(size)
        for d in range(size):
            c_k = self.L_powers_unique_entries_inv[:, d]
            
            polynomial = calc_matrix_polynomial_dot(c_k, self.M, observed_seqs)
            distance_class_n = np.sum(observed_seqs * polynomial)
            
            polynomial = calc_matrix_polynomial_dot(c_k, self.M, seq_values)
            correlation[d] = reciprocal(np.sum(seq_values * polynomial), distance_class_n)
            
            distance_class_ns[d] = distance_class_n
            
        return(correlation, distance_class_ns)

    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i:i + 3] = [-1, 2, -1]
        return(Diff2.T.dot(Diff2))
    
    def Frob_reg(self, theta, M, a, beta, second_order_diff_matrix):
        """cost function for regularized least square method for inferring 
        lambdas"""
        Frob1 = np.exp(theta).dot(M).dot(np.exp(theta))
        Frob2 = 2 * np.exp(theta).dot(a)
        return(Frob1 - Frob2 + beta * theta[1:].dot(self.second_order_diff_matrix).dot(theta[1:]))
    
    def construct_M(self, N_d):
        size = self.seq_length + 1
        M = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                for d in range(size):
                    M[i, j] += N_d[d] * self.W_kd[i, d] * self.W_kd[j, d]
        return(M)
    
    def construct_a(self, rho_d, N_d):
        size = self.seq_length + 1
        a = np.zeros(size)
        for i in range(size):
            for d in range(size):
                a[i] += N_d[d] * self.W_kd[i, d] * rho_d[d]
        return(a)
    
    def lambdas_to_variance(self, lambdas):
        self.calc_eigenvalue_multiplicity()
        variance_components = (lambdas * self.lambdas_multiplicity)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)
    
    def predict(self, Xpred=None, X=None, y=None, variance=None, lambdas=None,
                ps=None, estimate_variance=False):
        """
        Compute the Maximum a Posteriori (MAP) estimate of the phenotype at 
        the provided or all genotypes
        
        Parameters
        ----------
        Xpred : array-like of shape (n_genotypes,)
            Vector containing the genotypes for which we want to predict the
            phenotype. If `n_genotypes == None` then predictions are provided
            for the whole sequence space
        
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `y`
            
        y : array-like of shape (n_obs,)
            Vector containing the observed phenotypes corresponding to `X`
            sequences
        
        variance : array-like of shape (n_obs,)
            Vector containing the empirical or experimental known variance for
            the measurements in `y`
            
        lambdas : array-like of shape (seq_length + 1,)
            Vector containing variance components `lambdas` to use to make 
            predictions given the observed `X`, `y` sequence function
            relationship 
            
        estimate_variance : bool (False)
            Option to also return the posterior variances for each individual
            genotype
        
        Returns
        -------
        function : pd.DataFrame of shape (n_genotypes, 1)
                   Returns the phenotypic predictions for each input genotype
                   in the column ``ypred`` and genotype labels as row names.
                   If ``estimate_variance=True``, then it has an additional
                   column with the posterior variances for each genotype
        """
        
        if np.all([X is not None, y is not None,
                   variance is not None, lambdas is not None]):
            self.init(genotypes=X, ps=ps)
            self.set_data(X=X, y=y, variance=variance)
            self.lambdas = lambdas
            
        elif np.any([X is not None, y is not None, lambdas is not None]):
            msg = 'X, y and lambdas must all be provided for prediction or'
            msg = ' the model should have been previously fitted'
            raise ValueError(msg)
        
        self.get_noise_diag_matrix()
        self.gt2data = self.get_gt_to_data_matrix(self.obs_idx)
    
        # Minimize error given lambdas
        coeffs = self.get_polynomial_coeffs(lambdas=self.lambdas)
        Kop = LinearOperator((self.n_obs, self.n_obs),
                             matvec=partial(self.K_BB_E_dot, coeffs=coeffs))
        a_star = minres(Kop, self.y, tol=1e-9)[0]
        
        ypred = calc_matrix_polynomial_dot(coeffs, self.M, self.gt2data.dot(a_star))
        pred = pd.DataFrame({'ypred': ypred}, index=self.genotypes)

        if Xpred is not None:
            pred = pred.loc[Xpred, :]

        if estimate_variance:
            pred['var'] = self.estimate_posterior_variance(Xpred=pred.index)
        
        return(pred)
    
    def K_BB_E_dot(self, v, coeffs):
        """ multiply by the m by m matrix K_BB + E"""
        yhat = calc_matrix_polynomial_dot(coeffs, self.M, self.gt2data.dot(v))
        ynoise = yhat + self.gt2data.dot(self.E_sparse.dot(v))
        return(self.gt2data.T.dot(ynoise))
    
    def estimate_posterior_variance(self, Xpred=None):
        """compute posterior variances for a list of sequences"""
        # TODO: fix response: cannot give negative variance estimates in test
        # TODO: replace rho by covariance matrix
        if Xpred is None:
            Xpred = self.space.state_labels
        n_pred, pred_idx = Xpred.shape[0], self.get_obs_idx(Xpred)
    
        # Build linear operator
        self.gt2data = self.get_gt_to_data_matrix(self.obs_idx)
        coeffs = self.get_polynomial_coeffs(lambdas=self.lambdas)
        Kop = LinearOperator((self.n_obs, self.n_obs),
                             matvec=partial(self.K_BB_E_dot, coeffs=coeffs))
        
        # Compute posterior variance for each target sequence
        covariance_distance = self.W_kd.T.dot(self.lambdas)
        K_ii = covariance_distance[0]
        post_vars = []
        for i in tqdm(pred_idx, total=n_pred):
            vec = np.zeros(self.n_genotypes)
            vec[i] = 1
            K_Bi = calc_matrix_polynomial_dot(coeffs, self.M, vec)[self.obs_idx]
            # Posibility to optimize with a good preconditioner
            alph = cg(Kop, K_Bi)[0]
            post_vars.append(K_ii - np.sum(K_Bi * alph))
    
        return(np.array(post_vars))
    
    def simulate(self, lambdas, sigma=0, p_missing=0):
        '''
        Simulates data under the specified Variance component priors
        
        Parameters
        ----------
        lambdas : array-like of shape (seq_length + 1,)
            Vector containing the variance of each of the ``seq_length``
            components characterizing the prior
        
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
        
        a = np.random.normal(size=self.n_genotypes)
        if hasattr(self, 'D_pi'):
            a = self.D_pi.dot(a)
        
        # TODO: find out why this needs to be sqrt
        yhat = self.project(a, lambdas=np.sqrt(lambdas))
        y = np.random.normal(yhat, sigma) if sigma > 0 else yhat
        variance = np.full(self.n_genotypes, sigma**2)
        
        if p_missing > 0:
            n = int((1 - p_missing) * self.n_genotypes)
            sel_idxs = np.random.choice(np.arange(self.n_genotypes), n)
            y[sel_idxs] = np.nan
            variance[sel_idxs] = np.nan
        
        data = pd.DataFrame({'y_true': yhat, 'y': y, 'var': variance},
                            index=self.genotypes)
        return(data)


class DeltaPEstimator(LandscapeEstimator):
    def __init__(self, P, a=None, num_a=20, nfolds=5,
                 a_resolution=0.1, max_a_max=1e12, fac_max=0.1, fac_min=1e-6,
                 opt_method='L-BFGS-B', optimization_opts={}, scale_by=1,
                 gtol=1e-3):
        self.P = P
        self.a = a
        self.a_is_fixed = a is not None
        self.nfolds = nfolds
        
        msg = '"a" can only be None or >= 0'
        check_error(a is None or a >= 0, msg=msg)
        
        # Attributes to generate a values
        self.num_a = num_a
        
        # Default bounds for a in absence of a more clever method
        self.min_log_a = -4
        self.max_log_a = 10
        
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
        optimization_opts['ftol'] = 0
        self.optimization_opts = optimization_opts
        
    def calc_n_p_faces(self, length, P, n_alleles):
        return(comb(length, P) * comb(n_alleles, 2) ** P * n_alleles ** (length - P))
        
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.check_P()
        
        self.space.calc_laplacian()
        self.n_p_faces = self.calc_n_p_faces(self.seq_length, self.P, self.n_alleles) 
        self.n_genotypes = self.space.n_states
        self.L = self.space.laplacian
    
    @property
    def D_kernel_basis_orth_sparse(self):
        if not hasattr(self, '_D_kernel_basis_orth_sparse'):
            self._D_kernel_basis_orth_sparse = self.construct_D_kernel_basis()
        return(self._D_kernel_basis_orth_sparse)
    
    def get_a_values(self):
        return(np.exp(np.linspace(self.min_log_a, self.max_log_a, self.num_a)))
    
    def check_P(self):
        if self.P == (self.seq_length + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= self.P <= self.seq_length:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
    
    def L_opt(self, phi, p=0):
        return(self.L.dot(phi) - p * self.n_alleles * phi)
    
    def D_opt(self, phi):
        Dphi = phi.copy()
        for p in range(self.P):
            Dphi = self.L_opt(Dphi, p)
        return Dphi / factorial(self.P)
    
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
    
    def calc_neg_log_prior_prob(self, phi, a):
        S1 = a / (2*self.n_p_faces) * np.sum(phi * self.D_opt(phi))
        regularizer = self.calc_regularization(phi)
        return(S1 + regularizer)
    
    def calc_neg_log_prior_prob_grad(self, phi, a):
        grad_S1 = a / self.n_p_faces * self.D_opt(phi)
        regularizer = self.calc_regularization_grad(phi)
        return(grad_S1 + regularizer)
    
    def S(self, phi, a):
        S1 = self.calc_neg_log_prior_prob(phi, a)
        S2 = self.calc_neg_log_likelihood(phi) 
        return(S1 + S2)
    
    def S_grad(self, phi, a):
        S1_grad = self.calc_neg_log_prior_prob_grad(phi, a)
        S2_grad = self.calc_neg_log_likelihood_grad(phi) 
        return(S1_grad + S2_grad)
    
    def S_inf(self, b):
        phi = self.D_kernel_basis_orth_sparse.dot(b)
        S2 = self.calc_neg_log_likelihood(phi)
        regularizer = self.calc_regularization(phi)
        return(S2 + regularizer)
    
    def S_grad_inf(self, b):
        phi = self.D_kernel_basis_orth_sparse.dot(b)
        S2_grad = self.calc_neg_log_likelihood_grad(phi)
        regularizer = self.calc_regularization_grad(phi)
        return(self.D_kernel_basis_orth_sparse.T.dot(S2_grad + regularizer))
    
    def construct_D_kernel_basis2(self, max_value_to_zero=1e-12):
        # TODO: think on a way to optimize this for large landscapes
        
        # Generate bases and sequences
        bases = np.array(list(range(self.n_alleles)))
        seqs = np.array(list(product(bases, repeat=self.seq_length)))
    
        # Construct D kernel basis
        
        # Basis of kernel W(0)
        D_kernel_basis = np.ones([self.n_genotypes, 1])
        
        for p in range(1, self.P):
            # Basis of kernel W(1)
            if p == 1:
                W1_dim = self.seq_length*(self.n_alleles-1)
                W1_basis = np.zeros([self.n_genotypes,W1_dim])
                for site in range(self.seq_length):
                    W1_basis[:,site*(self.n_alleles-1):(site+1)*(self.n_alleles-1)] = pd.get_dummies(seqs[:,site], drop_first=True).values
                D_kernel_basis = np.hstack((D_kernel_basis, W1_basis))
    
            # Basis of kernel W(>=2)
            if p >= 2:
                W2_dim = int(comb(self.seq_length,p) * (self.n_alleles-1)**p)
                W2_basis = np.ones([self.n_genotypes,W2_dim])
                site_groups = list(combinations(range(self.seq_length), p))
                base_groups = list(product(range(1,self.n_alleles), repeat=p))  # because we have dropped first base
                col = 0
                for site_group in site_groups:
                    for base_group in base_groups:
                        for i in range(p):
                            site, base_idx = site_group[i], base_group[i]-1  # change 'base' to its 'idx'
                            W2_basis[:,col] *= W1_basis[:,site*(self.n_alleles-1)+base_idx]
                        col += 1
                D_kernel_basis = np.hstack((D_kernel_basis, W2_basis))
    
        D_kernel_basis = orth(D_kernel_basis)    
        D_kernel_basis[np.abs(D_kernel_basis) < max_value_to_zero] = 0
        D_kernel_basis_orth_sparse = csr_matrix(D_kernel_basis)
        return(D_kernel_basis_orth_sparse)

    def get_Vj(self, j):
        if not hasattr(self, 'vj'):
            site_L = self.n_alleles * np.eye(self.n_alleles) - np.ones((self.n_alleles, self.n_alleles))
            v0 = np.full((self.n_alleles, 1), 1 / np.sqrt(self.n_alleles)) 
            v1 = orth(site_L)
            
            msg = 'Basis for subspaces V0 and V1 are not orthonormal'
            check_error(np.allclose(v1.T.dot(v0), 0), msg)
            
            self.vj = {0: v0, 1: v1,
                       (0,): v0, (1,): v1}

        if j not in self.vj:
            # self.vj[j] = np.tensordot(self.vj[j[0]], self.get_Vj(j[1:]), axes=0)
            self.vj[j] = np.vstack([np.hstack([x * self.get_Vj(j[1:]) for x in row])
                                    for row in self.vj[j[0]]])
        return(self.vj[j])

    def construct_D_kernel_basis(self, max_value_to_zero=1e-12):
        basis = [np.full((self.n_genotypes, 1), 1 / np.sqrt(self.n_genotypes))]
        for p in range(1, self.P):
            for idxs in combinations(range(self.seq_length), p):
                idxs = np.array(idxs)
                j = np.zeros(self.seq_length, dtype=int)
                j[idxs] = 1
                basis.append(self.get_Vj(tuple(j)))
        basis = np.hstack(basis)
        basis[np.abs(basis) < max_value_to_zero] = 0
        basis = csr_matrix(basis)
        return(basis)
    
    def construct_D_spectrum(self):
        D_eig_vals, D_multis = np.zeros(self.seq_length+1), np.zeros(self.seq_length+1)
        for k in range(self.seq_length+1):
            lambda_k = k * self.n_alleles
            Lambda_k = 1
            for p in range(self.P):
                Lambda_k *= lambda_k - p * self.n_alleles
            m_k = comb(self.seq_length,k) * (self.n_alleles-1)**k
            D_eig_vals[k], D_multis[k] = Lambda_k/factorial(self.P), m_k
    
        self.D_eig_vals, self.D_multis = D_eig_vals, D_multis
        
    def get_cv_iter(self, a_values):
        for train, validation in self.split_cv():        
            for a in a_values:
                yield(a, train, validation)   
    
    def fit_a_cv(self):
        cv_log_L = []
        a_values = self.get_a_values()
        cv_data = self.get_cv_iter(a_values)
        
        for a, train, test in tqdm(cv_data, total=self.nfolds * self.num_a):
            X_train, y_train = train
            X_test, y_test = test

            self.set_data(X=X_train, y=y_train)
            phi = self._fit(a)
            
            self.set_data(X=X_test, y=y_test)
            test_logL = -self.calc_neg_log_likelihood(phi) 
            
            cv_log_L.append({'a': a, 'logL': test_logL})
        
        cv_log_L = pd.DataFrame(cv_log_L)
        cv_log_L = cv_log_L.groupby('a', as_index=False).agg({'logL': ('mean', 'std')})
        cv_log_L.columns = ['a', 'log_likelihood_mean', 'log_likelihood_sd']
        self.cv_log_L = cv_log_L
        self.a = cv_log_L['a'][np.argmax(cv_log_L['log_likelihood_mean'])]
    
    def _fit(self, a, phi_initial=None):
        check_error(a >= 0, msg='"a" must be larger or equal than 0')
        
        if phi_initial is None and a > 0:
            phi_initial = np.zeros(self.n_genotypes)
        
        if a == 0 and hasattr(self, '_fit_a_0'):
            phi = self._fit_a_0()
            
        elif a == np.inf:
            b_initial = self.D_kernel_basis_orth_sparse.T.dot(phi_initial)
            res = minimize(fun=self.S_inf, jac=self.S_grad_inf,
                           x0=b_initial, method=self.opt_method,
                           options=self.optimization_opts)
            if not res.success:
                self.report(res.message)
            b_a = res.x
            phi = self.D_kernel_basis_orth_sparse.dot(b_a)
            
        else:
            res = minimize(fun=self.S, jac=self.S_grad, args=(a,),
                           x0=phi_initial, method=self.opt_method,
                           options=self.optimization_opts)
            if not res.success:
                print(res.message)
            phi = res.x

        # a, N = a * scale_by, N *scale_by    
        return(phi)
    
    def fit(self, X, y, force_fit_a=True):
        """
        Infers the sequence-function relationship under the specified
        \Delta^{(P)} prior 
        
        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `counts`. Missing sequences are assumed to have observed 0 times
            
        y : array-like of shape (n_obs,) or (n_obs, n_cols)
            Vector or matrix containing the expected type of data for each model
            
        force_fit_a : bool 
            Whether to re-fit ``a`` using cross-validation even if it is already
            defined a priori
            
        Returns
        -------
        
        landscape : pd.DataFrame (n_genotypes, 2)
            DataFrame containing the estimated function for each possible
            sequence in the space
        
        """
        self.init(genotypes=X)
        self.set_data(X, y)
        
        if not self.a_is_fixed and force_fit_a:
            self.fit_a_cv()
            self.set_data(X, y)
        
        # Fit model with a_star
        phi = self._fit(self.a)
        output = self.phi_to_output(phi)
        return(output)
    

class SeqDEFT(DeltaPEstimator):
    # Required methods
    def set_data(self, X, y):
        self.X = X
        self.y = y
        
        data = self.fill_zeros_counts(X, y)
        self.N = data.sum()
        self.R = (data / self.N)
    
    def calc_neg_log_likelihood(self, phi):
        S2 = self.N * np.sum(self.R * phi)
        S3 = self.N * np.sum(safe_exp(-phi))
        return(S2 + S3)
    
    def calc_neg_log_likelihood_grad(self, phi):
        grad_S2 = self.N * self.R
        grad_S3 = self.N * safe_exp(-phi)
        return(grad_S2 - grad_S3)
    
    def split_cv(self):
        seqs = self.counts_to_seqs()
        np.random.shuffle(seqs)
        n_test = np.round(self.N / self.nfolds).astype(int)
        
        for i in np.arange(0, seqs.shape[0], n_test):
            test = np.unique(seqs[i:i+n_test], return_counts=True)
            train = np.unique(np.append(seqs[:i], seqs[i+n_test:]),
                              return_counts=True)
            yield(train, test)
            
    def phi_to_output(self, phi):
        Q_star = self.phi_to_Q(phi)
        seq_densities = pd.DataFrame({'frequency': self.R, 'Q_star': Q_star},
                                     index=self.genotypes)
        return(seq_densities)
    
    # Optional methods
    def calc_a_max(self, phi_inf):
        a_max = self.n_p_faces * self.fac_max
        
        phi_max = self._fit(a_max, phi_initial=phi_inf)
        distance = D_geo(phi_max, phi_inf)
        
        while distance > self.a_resolution and a_max < self.max_a_max:
            a_max *= 10
            phi_max = self._fit(a_max, phi_initial=phi_inf)
            distance = D_geo(phi_max, phi_inf)
            
        return(a_max)
    
    def calc_a_min(self, phi_inf):
        a_min = self.n_p_faces * self.fac_min
        
        phi_0 = self._fit(0)
        phi_min = self._fit(a_min, phi_initial=phi_inf)
        
        distance = D_geo(phi_min, phi_0)
        
        while distance > self.a_resolution:
            a_min /= 10
            phi_min = self._fit(a_min, phi_initial=phi_inf)
            distance = D_geo(phi_min, phi_0)
        return(a_min)

    def get_a_values(self):
        phi_inf = self._fit(np.inf)
        a_min = self.calc_a_min(phi_inf) 
        a_max = self.calc_a_max(phi_inf)
        a_values = np.geomspace(a_min, a_max, self.num_a)
        return(a_values)

    def fill_zeros_counts(self, X, y):
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        data.loc[X] = y
        data = data.astype(int)
        return(data)

    def phi_to_logQ(self, phi):
        return(-phi - logsumexp(-phi))
    
    def phi_to_Q(self, phi):
        return(np.exp(self.phi_to_logQ(phi)))
    
    def _fit_a_0(self):
        with np.errstate(divide='ignore'):
            phi = -np.log(self.R)
        return(phi)
    
    def counts_to_seqs(self):
        seqs = []
        for seq, counts in zip(self.X, self.y):
            seqs.extend([seq] * counts)
        seqs = np.array(seqs)
        return(seqs)
    
    # TODO: fix and refactor simulation code
    def simulate(self, N, a_true, random_seed=None):
        # Set random seed
        np.random.seed(random_seed)
    
        # Simulate phi from prior distribution
        v = np.random.normal(size=self.n_genotypes)
        phi_true = np.zeros(self.n_genotypes)
        
        for k in range(self.P, self.seq_length+1):
            eta_k = np.sqrt(self.n_p_faces) / np.sqrt(a_true * self.D_eig_vals[k])
            self.solve_b_k(k)
            phi_true += eta_k * self.W_k_opt(v)
    
        # Construct Q_true from the simulated phi
        Q_true = np.exp(-phi_true) / np.sum(np.exp(-phi_true))
    
        # Simulate N data points from Q_true
        data = np.random.choice(self.n_genotypes, size=N, replace=True, p=Q_true)
    
        # Obtain count data
        values, counts = np.unique(data, return_counts=True)
        Ns = np.zeros(self.n_genotypes)
        Ns[values] = counts
    
        # Normalize count data
        R = Ns / N
    
        # Save N and R
        data_dict = {'N': int(N), 'R': R, 'Q_true': Q_true}
    
        # Return
        return data_dict
    
    def solve_b_k(self, k):
        # Tabulate w_k(d)
        w_k = np.zeros(self.seq_length+1)
        for d in range(self.seq_length+1):
            w_k[d] = self.w(k, d)
    
        # Solve for b_k
        b_k = solve(self.MAT, w_k)
        
        self.b_k = b_k
    
    def w(self, k, d):
        ss = 0
        for q in range(self.seq_length+1):
            ss += (-1)**q * (self.n_alleles-1)**(k-q) * comb(d,q) * comb(self.seq_length-d,k-q)
        return 1/self.n_alleles**self.seq_length * ss
    
    def W_k_opt(self, v):
        max_power = len(self.b_k) - 1
        Lsv = np.zeros([self.n_genotypes,len(self.b_k)])
        Lsv[:,0] = self.b_k[0] * v
        power = 1
        while power <= max_power:
            v = self.L_opt(v)
            Lsv[:,power] = self.b_k[power] * v
            power += 1
        Wkv = Lsv.sum(axis=1)
        return Wkv


def D_geo(phi1, phi2):
    logQ1 = -phi1 - logsumexp(-phi1)
    logQ2 = -phi2 - logsumexp(-phi2)
    s = np.exp(logsumexp(0.5 * (logQ1 + logQ2)))
    x = min(s, 1)
    return 2 * np.arccos(x)


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)
