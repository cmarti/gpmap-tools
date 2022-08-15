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
from gpmap.src.space import SequenceSpace
from gpmap.src.utils import get_sparse_diag_matrix, check_error
from gpmap.src.seq import guess_space_configuration


def reciprocal(x, y):
    """calculate reciprocal of variable, if variable=0, return 0"""
    if y == 0:
        return 0
    else:
        return x / y


def Frob(lambdas, M, a):
    """calculate the cost function given lambdas and a"""
    Frob1 = np.dot(lambdas, M).dot(lambdas)
    Frob2 = 2 * np.sum(lambdas * a)
    return Frob1 - Frob2


def grad_Frob(lambdas, M, a):
    """gradient of the function Frob(lambdas, M, a)"""
    grad_Frob1 = 2 * M.dot(lambdas)
    grad_Frob2 = 2 * a
    return grad_Frob1 - grad_Frob2


class LandscapeEstimator(object):
    
    def define_space(self, seq_length=None, n_alleles=None, genotypes=None,
                     alphabet_type='custom'):
        alphabet = None
        
        if genotypes is not None:
            configuration = guess_space_configuration(genotypes,
                                                      ensure_full_space=False)
            seq_length = configuration['length']
            alphabet_type = configuration['alphabet_type']
            alphabet = set()
            for alleles in configuration['alphabet']:
                alphabet = alphabet.union(alleles)
            alphabet = [sorted(alphabet)] * seq_length
            self.space = SequenceSpace(seq_length=seq_length, alphabet=alphabet,
                                       alphabet_type='custom')
            n_alleles = len(alphabet[0])
        
        elif n_alleles is None or seq_length is None:
            msg = 'Either genotypes or both seq_length and n_alleles must '
            msg += 'be provided'
            raise ValueError(msg)
        
        else:
            self.space = SequenceSpace(seq_length=seq_length, n_alleles=n_alleles,
                                       alphabet_type=alphabet_type)
        
        self.seq_length = seq_length
        self.n_alleles = n_alleles
    
    @property
    def genotypes(self):
        return(self.space.genotypes)
    
    
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
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.space.calc_laplacian()
        self.n_genotypes = self.space.n_states
        self.L = self.space.laplacian
        self.calc_L_powers_unique_entries_matrix()
        self.calc_W_kd_matrix()
    
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
    
    def calc_collapsed_covariance_matrix(self, lambdas):
        return(np.dot(self.W_kd.T, lambdas))
    
    def calc_L_powers_coeffs(self, lambdas):
        covariance = self.calc_collapsed_covariance_matrix(lambdas)
        return(np.dot(self.L_powers_unique_entries_inv, covariance))
    
    def calc_L_powers(self, v):
        powers = [v]
        for _ in range(self.seq_length):
            powers.append(self.L.dot(powers[-1]))
        return(np.vstack(powers).T)
    
    def get_obs_idx(self, seqs):
        obs_idx = self.space.state_idxs[seqs]
        return(obs_idx)


    def set_data(self, X, y, variance):
        obs_idx = self.get_obs_idx(X)
        if variance is None:
            variance = np.zeros(obs_idx.shape[0])
        
        self.obs_idx = obs_idx
        self.y = y
        self.variance = variance
        self.n_obs = y.shape[0]
    
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
            idx = self.space.state_idxs.values
        
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
            distance_class_n = np.sum(observed_seqs * self.calc_L_polynomial_product(c_k, observed_seqs))
            correlation[d] = reciprocal(np.sum(seq_values * self.calc_L_polynomial_product(c_k, seq_values)), 
                                        distance_class_n)
            distance_class_ns[d] = distance_class_n
            
        return(correlation, distance_class_ns)

    def calc_L_polynomial_product(self, coefficients, v):
        powers = self.calc_L_powers(v)
        return(powers.dot(coefficients))
    
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
    
    def W_dot(self, v, L_powers_coeffs):
        """multiply by the whole covariance matrix. b_k is set by lambdas"""
        return(self.calc_L_powers(v).dot(L_powers_coeffs))
    
    def K_BB_E_dot(self, v, L_powers_coeffs):
        """ multiply by the m by m matrix K_BB + E"""
        return(self.gt2data.T.dot(self.W_dot(self.gt2data.dot(v), L_powers_coeffs)) + self.E_sparse.dot(v))

    def predict(self, Xpred=None, X=None, y=None, variance=None, lambdas=None,
                estimate_variance=False):
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
                   in the column `function` and genotype labels as row names.
                   If `estimate_variance` is True, then it has an additional
                   column with the posterior variances for each genotype
        """
        
        if np.all([X is not None, y is not None,
                   variance is not None, lambdas is not None]):
            self.init(genotypes=X)
            self.set_data(X=X, y=y, variance=variance)
            self.lambdas = lambdas
        elif np.any([X is not None, y is not None, lambdas is not None]):
            msg = 'X, y and lambdas must all be provided for prediction or'
            msg = ' the model should have been previously fitted'
            raise ValueError(msg)
        
        self.get_noise_diag_matrix()
        L_powers_coeffs = self.calc_L_powers_coeffs(self.lambdas)
        self.gt2data = self.get_gt_to_data_matrix(self.obs_idx)
    
        # Solve 'a' with ys
        matvec = partial(self.K_BB_E_dot, L_powers_coeffs=L_powers_coeffs)
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=matvec)
        a_star = minres(Kop, self.y, tol=1e-9)[0]
    
        # Compute posterior mean
        f_star = self.W_dot(self.gt2data.dot(a_star), L_powers_coeffs)
        self.residuals = self.y - self.gt2data.T.dot(f_star)
        self.res_var = self.residuals.var()
        
        pred = pd.DataFrame({'function': f_star}, index=self.space.genotypes)
        if estimate_variance:
            pred['variance'] = self.estimate_posterior_variance()
        
        if Xpred is not None:
            pred = pred.loc[Xpred, :]
        
        return(pred)
    
    def estimate_posterior_variance(self, Xpred=None):
        """compute posterior variances for a list of sequences"""
        # TODO: fix response: cannot give negative variance estimates in test
        # TODO: replace rho by covariance matrix
        if Xpred is None:
            Xpred = self.space.state_labels
        n_pred = Xpred.shape[0]
        pred_idx = self.get_obs_idx(Xpred)
    
        # Construct rho & b_k
        rho = self.W_kd.T.dot(self.lambdas)
        L_powers_coeffs = np.dot(self.L_powers_unique_entries_inv, rho)
        self.gt2data = self.get_gt_to_data_matrix(self.obs_idx)
        matvec = partial(self.K_BB_E_dot, L_powers_coeffs=L_powers_coeffs)
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=matvec)
        
        # Compute posterior variance for each target sequence
        K_ii = rho[0]
        post_vars = []
        for i in tqdm(pred_idx, total=n_pred):
            vec = np.zeros(self.n_genotypes)
            vec[i] = 1
            K_Bi = self.W_dot(vec, L_powers_coeffs)[self.obs_idx]
            alph = cg(Kop, K_Bi)[0]
            post_vars.append(K_ii - np.sum(K_Bi * alph))
    
        return np.array(post_vars)
    
    def calc_eigenvalue_multiplicity(self):
        self.lambdas_multiplicity = np.array([comb(self.seq_length, k) * (self.n_alleles - 1) ** k
                                              for k in range(self.seq_length + 1)], dtype=int)
    
    def lambdas_to_variance(self, lambdas):
        self.calc_eigenvalue_multiplicity()
        variance_components = (lambdas * self.lambdas_multiplicity)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)
    
    def simulate(self, lambdas, sigma=0, p_missing=0):
        f_random = np.random.normal(size=self.n_genotypes)
        L_powers_coeffs = self.calc_L_powers_coeffs(np.sqrt(lambdas))
        f = self.W_dot(f_random, L_powers_coeffs)
        
        if sigma > 0:
            f_obs = np.random.normal(f, sigma)
        else:
            f_obs = f
            
        variance = np.full(self.n_genotypes, sigma**2)
        seqs = self.space.genotypes
        
        if p_missing > 0:
            n = int((1 - p_missing) * self.n_genotypes)
            sel_idxs = np.random.choice(np.arange(self.n_genotypes), n)
            
            f_obs = f_obs[sel_idxs]
            variance = variance[sel_idxs]
            seqs = seqs[sel_idxs]
        
        self.real_function = pd.DataFrame({'function': f}, index=self.space.state_labels)     
        data = pd.DataFrame({'function': f_obs, 'variance': variance}, index=seqs)
        return(data)


class SeqDEFT(LandscapeEstimator):
    def __init__(self, P):
        self.P = P
    
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
        
        self.calc_L_powers_unique_entries_matrix()

    @property
    def D_kernel_basis_orth_sparse(self):
        if not hasattr(self, '_D_kernel_basis_orth_sparse'):
            self._D_kernel_basis_orth_sparse = self.construct_D_kernel_basis()
        return(self._D_kernel_basis_orth_sparse)
    
    def check_P(self):
        if self.P == (self.seq_length + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= self.P <= self.seq_length:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
    
    def get_phi_0(self, options, scale_by):
        if not hasattr(self, 'phi_0'):
            self.phi_0 = self._fit(0, options=options, scale_by=scale_by)
        return(self.phi_0)
    
    def get_phi_inf(self, options, scale_by, gtol=1e-3):
        if not hasattr(self, 'phi_inf'):
            self.phi_inf = self._fit(np.inf, options=options,
                                     scale_by=scale_by, gtol=gtol)
        return(self.phi_inf)
    
    def calc_a_max(self, max_a_max=1e12, resolution=0.1, fac_max=1,
                   options=None, scale_by=1, gtol=1e-3):
        a_max = self.n_p_faces * fac_max
        phi_inf = self.get_phi_inf(options, scale_by)
        phi_max = self._fit(a_max, phi_initial=phi_inf, options=options,
                            scale_by=scale_by, gtol=gtol)
        distance = D_geo(phi_max, phi_inf)
        
        while distance > resolution and a_max < max_a_max:
            a_max *= 10
            phi_max = self._fit(a_max, phi_initial=phi_inf, options=options,
                                scale_by=scale_by, gtol=gtol)
            distance = D_geo(phi_max, phi_inf)
            
        return(a_max)
    
    def calc_a_min(self, resolution=0.1, fac_min=1e-6, options=None, scale_by=1,
                   gtol=1e-3):
        a_min = self.n_p_faces * fac_min
        phi_0 = self.get_phi_0(options, scale_by)
        phi_inf = self.get_phi_inf(options, scale_by, gtol=gtol)
        phi_min = self._fit(a_min, phi_initial=self.phi_inf, options=options,
                            scale_by=scale_by, gtol=gtol)
        distance = D_geo(phi_min, phi_0)
        
        while distance > resolution:
            a_min /= 10
            phi_min = self._fit(a_min, phi_initial=phi_inf, options=options,
                                scale_by=scale_by, gtol=gtol)
            distance = D_geo(phi_min, phi_0)
        return(a_min)

    def find_a_bounds(self, max_a_max=1e12, resolution=0.1, fac_max=0.1,
                      fac_min=1e-6, options=None, scale_by=1, gtol=1e-3):
        a_max = self.calc_a_max(max_a_max, resolution, fac_max, options, scale_by, gtol=gtol)
        a_min = self.calc_a_min(resolution, fac_min, options, scale_by, gtol=gtol)
        return(a_min, a_max)
    
    def get_a_values(self, resolution=0.1, max_a_max=1e12, num_a=20,
                     options=None, scale_by=1,
                     fac_max=0.1, fac_min=1e-6, gtol=1e-3):
        a_min, a_max = self.find_a_bounds(max_a_max, resolution, fac_max, fac_min,
                                          options, scale_by, gtol=gtol)
        a_values = np.geomspace(a_min, a_max, num_a)
        return(a_values)

    def set_data(self, X, counts):
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        data.loc[X] = counts
        data = data.astype(int)
        
        self.counts = data
        self.N = data.sum()
        self.R = (data / self.N)
        self.data_dict = {'N': self.N, 'R': self.R}

    def counts_to_data_dict(self, counts):
        n = counts.sum()
        return({'N': n, 'R': counts / n})
    
    def fit(self, X, counts, a_value=None, nfolds=5,
            resolution=0.1, max_a_max=1e12, 
            num_a=20, options=None, scale_by=1,
            fac_max=0.1, fac_min=1e-6, gtol=1e-3):
        
        """
        Infers Sequence density from the provided count data
        
        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `counts`. Missing sequences are assumed to have observed 0 times
            
        counts : array-like of shape (n_obs,)
            Vector containing the number of times each sequence in `X` was
            observed
        
        a_value: float (None)
            Hyperparameter value to use to interpolate between empirical
            frequencies and maximum entropy model. It is estimated
            using Cross-validation unless provided in this argument
                        
        nfolds : int (5)
            Number of folds to use for cross-validation for hyperparameter `a`
            optimization
        
        num_a: int(20)
            Number of geometrically spaced hyperparameter `a` values to 
            trace the MAP curve for optimization
        
        TODO: fill other parameters 
            
            
        Returns
        -------
        
        seq_densities: pd.DataFrame (n_genotypes, 2)
            DataFrame containing the observed frequency and the estimated
            density for each possible sequence
        
        """
        if not hasattr(self, 'space'):
            self.init(genotypes=X)
        
        self.set_data(X, counts)
        phi_inf = self.get_phi_inf(options, scale_by, gtol=gtol)

        if a_value is None:
            a_values = self.get_a_values(resolution=resolution, max_a_max=max_a_max,
                                         num_a=num_a, options=options, scale_by=scale_by,
                                         fac_max=fac_max, fac_min=fac_min, gtol=gtol)
            ll = self.compute_log_Ls(a_values, nfolds, options, scale_by, gtol=gtol)
            a_value = ll.iloc[np.argmax(ll['log_likelihood']), :]['a']
        
        # Fit model with a_star
        phi = self._fit(a_value, phi_initial=phi_inf, options=options,
                        scale_by=scale_by, gtol=gtol)
        log_Q_star = -phi - logsumexp(-phi)
        Q_star = np.exp(log_Q_star)
        seq_densities = pd.DataFrame({'frequency': self.R, 'Q_star': Q_star},
                                     index=self.genotypes)
        assert(np.allclose(Q_star.sum(), 1))
        return(seq_densities)
    
    def L_opt(self, phi, p=0):
        return self.L.dot(phi) - p * self.n_alleles * phi
    
    def get_phi_initial(self, phi_initial=None):
        if phi_initial is None:
            Q_initial = np.ones(self.n_genotypes) / self.n_genotypes
            phi_initial = -np.log(Q_initial)
        return(phi_initial)

    def get_data(self, data_dict=None):
        # Get N and R
        if data_dict is None:
            N, R = self.N, self.R
        else: 
            N, R = data_dict['N'], data_dict['R']
        return(N, R)
    
    def _fit_a_inf(self, N, R, phi_initial, method, options):
        b_initial = self.D_kernel_basis_orth_sparse.T.dot(phi_initial)
        res = minimize(fun=self.S_inf, jac=self.grad_S_inf, args=(N, R),
                       x0=b_initial, method=method, options=options)
        if not res.success:
            self.report(res.message)
        b_a = res.x
        phi = self.D_kernel_basis_orth_sparse.dot(b_a)
        return(phi)
    
    def _fit_a(self, N, R, a, phi_initial, method, options):
        res = minimize(fun=self.S, jac=self.grad_S, args=(a,N,R),
                       x0=phi_initial, method=method, options=options)
        if not res.success:
            self.report(res.message)
        return(res.x)
    
    def _fit(self, a, phi_initial=None, data_dict=None,
             method='L-BFGS-B', options=None, scale_by=1, gtol=1e-3):
        N, R = self.get_data(data_dict=data_dict)
        
        # Solution is the same if we scale these values: may be useful if 
        # running into numerical problems
        a, N = a / scale_by, N / scale_by
        phi_initial = self.get_phi_initial(phi_initial=phi_initial)
        
        if options is None:
            options = {}
        options['gtol'] = gtol
        options['ftol'] = 0
        
        if a == 0:
            with np.errstate(divide='ignore'):
                phi = -np.log(R)
        elif 0 < a < np.inf:
            phi = self._fit_a(N, R, a, phi_initial, method, options)
        elif a == np.inf:
            phi = self._fit_a_inf(N, R, phi_initial, method, options)
        else:
            raise(ValueError('"a" not in the right range.'))
    
        # a, N = a * scale_by, N *scale_by
        return(phi)
    
    def calc_Q(self, phi):
        return(np.exp(-phi) / np.sum(np.exp(-phi)))
    
    def D_opt(self, phi):
        Dphi = phi.copy()
        for p in range(self.P):
            Dphi = self.L_opt(Dphi, p)
        return Dphi / factorial(self.P)
    
    def S(self, phi, a, N, R):
        S1 = a/(2*self.n_p_faces) * np.sum(phi * self.D_opt(phi))
        S2 = N * np.sum(R * phi)
        S3 = N * np.sum(safe_exp(-phi))
        regularizer = 0
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer += np.sum((phi - PHI_UB)[flags]**2)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer += np.sum((phi - PHI_LB)[flags]**2)
        result = S1 + S2 + S3 + regularizer
        return(result)
    
    def grad_S(self, phi, a, N, R):
        grad_S1 = a/self.n_p_faces * self.D_opt(phi)
        grad_S2 = N * R
        grad_S3 = N * safe_exp(-phi)
        regularizer = np.zeros(self.n_genotypes)
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi - PHI_UB)[flags]
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi - PHI_LB)[flags]
        result = grad_S1 + grad_S2 - grad_S3 + regularizer
        return(result)
    
    def S_inf(self, b, N, R):
        phi = self.D_kernel_basis_orth_sparse.dot(b)
        S_inf1 = N * np.sum(R * phi)
        S_inf2 = N * np.sum(safe_exp(-phi))
        regularizer = 0
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer += np.sum((phi - PHI_UB)[flags]**2)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer += np.sum((phi - PHI_LB)[flags]**2)
        return S_inf1 + S_inf2 + regularizer
    
    def grad_S_inf(self, b, N, R):
        phi = self.D_kernel_basis_orth_sparse.dot(b)
        grad_S_inf1 = N * R
        grad_S_inf2 = N * safe_exp(-phi)
        regularizer = np.zeros(self.n_genotypes)
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi - PHI_UB)[flags]
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi - PHI_LB)[flags]
        return self.D_kernel_basis_orth_sparse.T.dot(grad_S_inf1 - grad_S_inf2 + regularizer)
    
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
    
    def calc_log_likelihood(self, a, counts=None, Q=None):
        if counts is None:
            counts = self.counts
        if Q is None:
            Q = self.Q_star
        
        if a == 0:
            N_logQ = np.zeros(self.n_genotypes)
            N_flags, Q_flags = (counts == 0), (Q == 0)
            flags = ~N_flags * Q_flags
            N_logQ[flags] = -np.inf
            flags = ~N_flags * ~Q_flags
            N_logQ[flags] = counts[flags] * np.log(Q[flags])
            if any(N_logQ == -np.inf):
                log_L = -np.inf
            else:
                log_L = np.sum(N_logQ)
        else:
            log_L = np.sum(counts * np.log(Q))
        return(log_L)
    
    def phi_to_Q(self, phi):
        return(np.exp(-phi) / np.sum(np.exp(-phi)))
    
    def calculate_cv_fold_logL(self, a, train, validation, phi,
                               options=None, scale_by=1, gtol=1e-3):
        data_dict = self.counts_to_data_dict(train)
        phi = self._fit(a, phi_initial=phi, data_dict=data_dict,
                                         options=options, scale_by=scale_by,
                                         gtol=gtol)
        Q = self.phi_to_Q(phi)
        logL = self.calc_log_likelihood(a, validation, Q)
        return(logL)
    
    def get_cv_iter(self, cv_fold, a_values):
        for k, (train, validation) in enumerate(self.split_cv(cv_fold)):        
            for i, a in enumerate(a_values):
                yield(k, i, a, train, validation)   
    
    def compute_log_Ls(self, a_values, nfolds=5,
                       options=None, scale_by=1, gtol=1e-3):
        phi_inf = self.get_phi_inf(options, scale_by, gtol=gtol)
    
        log_Lss = np.zeros([nfolds, a_values.shape[0]])
        cv_data = self.get_cv_iter(nfolds, a_values)
        n_iters = nfolds * a_values.shape[0]
        
        for k, i, a, train, validation in tqdm(cv_data, total=n_iters):    
            log_Lss[k,i] = self.calculate_cv_fold_logL(a, train, validation, phi_inf,
                                                       options=options, scale_by=scale_by,
                                                       gtol=gtol)
        
        log_Ls = pd.DataFrame({'a': a_values, 'log_likelihood': log_Lss.mean(axis=0)})
        self.log_Ls = log_Ls
        return(log_Ls)
    
    def expand_counts(self):
        obs = []
        for i, c in enumerate(self.counts):
            obs.extend([i] * c)
        obs = np.array(obs)
        np.random.shuffle(obs)
        return(obs)
    
    def count_obs(self, obs):
        v, c = np.unique(obs, return_counts=True)
        counts = pd.DataFrame(c, index=v).reindex(np.arange(self.n_genotypes)).fillna(0).astype(int).values[:, 0]
        return(counts)
    
    def split_cv(self, cv_fold):
        obs = self.expand_counts()
        n_valid = np.round(self.N / cv_fold).astype(int)
        for _ in range(cv_fold):
            np.random.shuffle(obs)
            train = self.count_obs(obs[n_valid:])
            validation = self.count_obs(obs[:n_valid])
            yield(train, validation) 


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
