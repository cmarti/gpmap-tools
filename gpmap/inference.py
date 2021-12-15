#!/usr/bin/env python
import random
import warnings
from itertools import product, combinations
from os.path import exists, join
from tqdm import tqdm
from _functools import partial

import pystan
import numpy as np
import pandas as pd

from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve import minres
from scipy.sparse._matrix_io import save_npz, load_npz
from scipy.optimize._minimize import minimize
from scipy.special._basic import comb, factorial
from scipy.linalg.decomp_svd import orth
from scipy.linalg.basic import solve
from scipy.sparse.linalg import cg
from numpy.linalg.linalg import cholesky

from gpmap.plot_utils import init_fig, savefig, arrange_plot
from gpmap.settings import CACHE_DIR, U_MAX, PHI_LB, PHI_UB
from gpmap.base import SequenceSpace, get_sparse_diag_matrix
from scipy.special._logsumexp import logsumexp



VC_STAN = '''
data {
  int<lower=0> G;                 // number of total genotypes
  int<lower=0> O;                 // number of observed genotypes
  
  int<lower=0> N;                 // number of non-zero entries in L
  int<lower=0> M;                 // number of rows with non-zero entries in L
  
  int<lower=0> D;                 // number of possible hamming distances
  
  int<lower=0> L_v[N];            // indexes of L non-zero values
  int<lower=0> L_u[M];            // row indexes of L non-zero values
  vector[N] L_w;                  // L non-zero values
  
  matrix[D, D] W_kd;              // Krawchauk matrix   
  matrix[D, D] M_inv;             // L polynomials entries matrix inverse
  vector[D] m_k;                  // eigenvalue multiplicities
    
  int<lower=0, upper=G> idx[O];   // indexes of observed phenotypes
  real y[O];                      // observed phenotypic values
  real<lower=0> sigma;
  
}

parameters {
    real<lower=0> log_lambda_beta;
    real log_lambda0;
    
    real ymean;
    
    vector[G] y_raw;
}

transformed parameters {
    vector<lower=0>[D] lambdas;
    matrix[G, D] L_powers;
    vector[D] beta;
    vector[G] yhat;
    
    L_powers[,1] = y_raw;
    lambdas[1] = 0;
    for(i in 2:D){
        L_powers[,i] = csr_matrix_times_vector(G, G, L_w, L_v, L_u, L_powers[,i-1]);
        lambdas[i] = exp(log_lambda0 - i * log_lambda_beta);
    }
    
    beta = M_inv * (W_kd' * lambdas);
    yhat = ymean + L_powers * beta;
}


model {
    ymean ~ normal(0, 1);
    
    log_lambda0 ~ normal(0, 5);
    log_lambda_beta ~ normal(0, 5);
    
    y_raw ~ std_normal();
    y ~ normal(yhat[idx], sigma);
}

'''



def gd(x, y):
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


class VCregression(SequenceSpace):
    def __init__(self, length, n_alleles, log=None, alphabet_type='dna'):
        self.init(length, n_alleles, log=log, alphabet_type=alphabet_type)
        
        self.calc_MAT()
        self.calc_laplacian()
        self.calc_W_kd_matrix()
    
    def get_sequence_index(self, seqs):
        return(self.genotype_idxs[seqs].values)
    
    def load_data(self, f_obs, variance=None, seqs=None):
        self.obs = f_obs
        self.n_obs = f_obs.shape[0]

        if variance is None:
            variance = np.zeros(self.n_obs)
        self.variance = variance
        
        if seqs is None:
            if self.n_obs != self.n_genotypes:
                msg = 'Number of observations must match number of genotypes if'
                msg += '"seqs" argument is not provided'
                raise(ValueError(msg))
            self.obs_idx = np.arange(self.n_genotypes)
        else:
            self.obs_idx = self.get_sequence_index(seqs)
            
        self.report('Data loaded with {} sequences'.format(self.n_obs))
        
        self.get_noise_diag_matrix()
        self.get_gt_to_data_matrix()
        
    def load_dataframe(self, df):
        f_obs = df['mean'].values
        variance = df['var'].values if 'var' in df.columns else None
        seqs = df.index.values 
        self.load_data(f_obs=f_obs, variance=variance, seqs=seqs)
    
    def get_obs(self, idx=None):
        if idx is None:
            return(self.obs)
        else:
            return(self.obs[idx])
    
    def get_noise_diag_matrix(self):
        self.E_sparse = get_sparse_diag_matrix(self.variance)
    
    def get_gt_to_data_matrix(self, obs_idx=None):
        if obs_idx is None:
            obs_idx = self.obs_idx
        else:
            obs_idx = self.obs_idx[obs_idx]
            
        n_obs = obs_idx.shape[0]
            
        gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
                             shape=(self.n_genotypes, n_obs))
        return(gt2data)
    
    def compute_empirical_rho(self, obs_idx=None):
        gt2data = self.get_gt_to_data_matrix(obs_idx)
        obs = self.get_obs(obs_idx)
        n_obs = obs.shape[0]
        
        seq_values = gt2data.dot(obs)
        observed_seqs = gt2data.dot(np.ones(n_obs))

        # Compute rho_d and N_d
        size = self.length + 1
        rho_d, N_d = np.zeros(size), np.zeros(size)
        for d in range(size):
            c_k = self.MAT_inv[:, d]
            rho_d[d] = np.sum(seq_values * self.R_d_opt(c_k, seq_values))
            N_d[d] = np.sum(observed_seqs * self.R_d_opt(c_k, observed_seqs))
    
        # Divide by the number of observations in each distance class
        rho_d = [gd(rho_d[i], N_d[i]) for i in range(size)]
        return(rho_d, N_d)
    
    def calc_R_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.length - 2, self.length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i:i + 3] = [-1, 2, -1]
        self.Rmat = Diff2.T.dot(Diff2)
    
    def R_d_opt(self, c_k, v):
        """multiply by d-adjacenty matrix"""
        v = v.copy()
        max_power = len(c_k) - 1
        Lsv = np.zeros([self.n_genotypes, len(c_k)])
        
        for power in range(max_power + 1):
            if power > 0:
                v = self.L.dot(v)
            Lsv[:, power] = c_k[power] * v
            
        Rdv = Lsv.sum(axis=1)
        return(Rdv)
    
    def Frob_reg(self, theta, M, a, beta):
        """cost function for regularized least square method for inferring 
        lambdas"""
        Frob1 = np.exp(theta).dot(M).dot(np.exp(theta))
        Frob2 = 2 * np.exp(theta).dot(a)
        return Frob1 - Frob2 + beta * theta[1:].dot(self.Rmat).dot(theta[1:])
    
    def construct_M_and_a(self, rho_d, N_d):
        # Construct M
        size = self.length + 1
        M = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                for d in range(size):
                    M[i, j] += N_d[d] * self.W_kd[i, d] * self.W_kd[j, d]
    
        # Construct a
        a = np.zeros(size)
        for i in range(size):
            for d in range(size):
                a[i] += N_d[d] * self.W_kd[i, d] * rho_d[d]
        return(M, a)
    
    def solve_for_lambda(self, obs_idx=None, betas=None):
        """solve for lambdas using least squares with the given rho_d 
        and N_d"""
        rho_d, N_d = self.compute_empirical_rho(obs_idx=obs_idx)
        M, a = self.construct_M_and_a(rho_d, N_d)

        if betas is None:        
            # Minimize the objective function Frob with lambda > 0
            M_inv = np.linalg.inv(M)
            lambda0 = np.array(np.dot(M_inv, a)).ravel()
            res = minimize(fun=Frob, jac=grad_Frob, args=(M, a), x0=lambda0,
                           method='L-BFGS-B',
                           bounds=[(0, None)] * (self.length + 1))
            if not res.success:
                self.report(res.message)
            lambdas = res.x
        else:
            lambdas = []
            for beta in betas:
                res = minimize(fun=self.Frob_reg, args=(M, a, beta),
                               x0=np.zeros(self.length + 1), method='Powell',
                               options={'xtol': 1e-8, 'ftol': 1e-8})
                lambdas.append(np.exp(res.x))
        return(lambdas)
    
    def get_cv_indexes(self, nfolds=10):
        order = np.arange(self.n_obs)
        random.shuffle(order)
        folds = np.array_split(order, nfolds)
        
        for i in range(nfolds):
            training_idx = np.hstack(folds[:i] + folds[i+1:])
            validation_idx = folds[i]
            yield(training_idx, validation_idx)
    
    def solve_for_lambda_cv(self, nfolds=10, betas=None):
        """
        Solve for optimal lambdas using regularized least squares with 
        regularization parameter chosen using crossvalidation
    
        """
        if betas is None:
            betas = 10 ** np.arange(-2, 6, .5)
            
        mses = []
        for training_idx, validation_idx in tqdm(self.get_cv_indexes(nfolds),
                                                 total=nfolds):
            lambdas = self.solve_for_lambda(obs_idx=training_idx, betas=betas)
            rho, n = self.compute_empirical_rho(obs_idx=validation_idx)
            rho[0] -= np.mean(self.variance[validation_idx])
            mses.append([np.sum((self.W_kd.T.dot(lambdas[i]) - rho) ** 2 * (1 / np.sum(n)) * n)
                         for i in range(betas.shape[0])])
        mses = np.array(mses)
        mses = np.mean(mses, axis=0)
        betaopt = betas[np.argmin(mses)]
        
        lambdas = self.solve_for_lambda(betas=[betaopt])[0]
        return(lambdas)

    def estimate_lambdas(self, regularize=False):
        if regularize:
            self.calc_R_matrix()
            lambdas = self.solve_for_lambda_cv()
        else:
            lambdas = self.solve_for_lambda()
        return(lambdas)
    
    def lambdas_to_variance(self, lambdas):
        self.calc_eigenvalue_multiplicity()
        variance_components = (lambdas * self.lambdas_multiplicity)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)
    
    def estimate_variance_components(self, regularize=False):
        self.report('Estimating variance components')
        lambdas = self.estimate_lambdas(regularize=regularize)
        return(lambdas)
    
    def W_dot(self, v, L_powers_coeffs):
        """multiply by the whole covariance matrix. b_k is set by lambdas"""
        return(self.calc_laplacian_powers(v).dot(L_powers_coeffs))
    
    def K_BB_E_dot(self, v, L_powers_coeffs):
        """ multiply by the m by m matrix K_BB + E"""
        return(self.gt2data.T.dot(self.W_dot(self.gt2data.dot(v), L_powers_coeffs)) + self.E_sparse.dot(v))

    def estimate_f(self, lambdas):
        """compute the MAP"""
        self.report('Estimating mean phenotypic values')
        L_powers_coeffs = self.calc_L_powers_coeffs(lambdas)
        self.gt2data = self.get_gt_to_data_matrix()
    
        # Solve 'a' with ys
        matvec = partial(self.K_BB_E_dot, L_powers_coeffs=L_powers_coeffs)
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=matvec)
        a_star = minres(Kop, self.obs, tol=1e-9)[0]
    
        # Compute posterior mean
        f_star = self.W_dot(self.gt2data.dot(a_star), L_powers_coeffs)
        self.residuals = self.obs - self.gt2data.T.dot(f_star)
        self.res_var = self.residuals.var()
        return(f_star)
    
    def estimate_f_variance(self, lambdas):
        """compute posterior variances for a list of sequences"""
        # TODO: fix response: cannot give negative variance estimates in test
    
        # Construct rho & b_k
        rho = self.W_kd.T.dot(lambdas)
        L_powers_coeffs = np.dot(self.MAT_inv, rho)
        self.gt2data = self.get_gt_to_data_matrix()
        
        K_ii = rho[0]
        K_Bi = np.zeros([self.n_obs, self.n_genotypes])
        for i in range(self.n_genotypes):
            vec = np.zeros(self.n_genotypes)
            vec[self.obs_idx] = 1
            K_Bi[:, i] = self.W_dot(vec, L_powers_coeffs)[self.obs_idx]
    
        # Compute posterior variance for each target sequence
        matvec = partial(self.K_BB_E_dot, L_powers_coeffs=L_powers_coeffs)
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=matvec)
        
        post_vars = []
        self.report("computing posterior variance")
        for i in tqdm(range(self.n_genotypes), total=self.n_genotypes):
            alph = cg(Kop, K_Bi[:, i])[0]
            post_vars.append(K_ii - np.sum(K_Bi[:, i] * alph))
    
        # Return
        return np.array(post_vars)
    
    def simulate(self, lambdas, sigma=0, p_missing=0):
        f_random = np.random.normal(size=self.n_genotypes)
        L_powers_coeffs = self.calc_L_powers_coeffs(np.sqrt(lambdas))
        f = self.W_dot(f_random, L_powers_coeffs)
        
        f_obs = np.random.normal(f, sigma)
        variance = np.full(self.n_genotypes, sigma**2)
        seqs = None
        
        if p_missing > 0:
            n = int((1 - p_missing) * self.n_genotypes)
            sel_idxs = np.random.choice(np.arange(self.n_genotypes), n)
            f_obs = f_obs[sel_idxs]
            variance = variance[sel_idxs]
            seqs = self.genotype_labels[sel_idxs]
            
        self.load_data(f_obs, variance, seqs)
    
    def calc_cov_dense(self, lambdas):
        covariance = np.dot(self.W_kd.T, lambdas)
        D = self.calc_distance_matrix()
        return(covariance[D])
    
    def simulate_naive(self, lambdas, sigma=0, chol=False): 
        covariance = self.calc_cov_dense(lambdas)
        if chol:
            Lcov = cholesky(covariance + sigma * np.identity(self.n_genotypes))
            f_obs = np.dot(Lcov, np.random.normal(0, 1, size=self.n_genotypes))
        else:
            m = np.zeros(self.n_genotypes)
            f_obs = np.random.multivariate_normal(m, covariance)
            f_obs += np.random.normal(0, sigma, size=self.n_genotypes)
        variance = np.full(self.n_genotypes, sigma**2)
        self.load_data(f_obs, variance)
    
    def simulate_eig(self, lambdas, sigma=0):
        l, q = np.round(self.L_lambdas).astype(int), self.L_q
        l_u = np.unique(l)
        
        f = 0
        for k, l_k in zip(range(1, self.length + 1), l_u):
            idx = l == l_k
            q_k = q[:, idx]
            a_k = np.random.normal(size=idx.sum())
            f_k = np.sqrt(lambdas[k]) * np.dot(q_k, a_k)
            f += f_k
            
        f_obs = np.random.normal(f, sigma)
        variance = np.full(self.n_genotypes, sigma**2)
        self.load_data(f_obs, variance)
        
    def get_stan_data(self, sigma):
        stan_data = {'G': self.n_genotypes,
                     'O': self.n_obs,
                     'N': self.L.data.shape[0],
                     'M': self.L.indptr.shape[0],
                     'D': self.length + 1,
                     
                     'L_v': self.L.indices + 1,
                     'L_u': self.L.indptr + 1,
                     'L_w': self.L.data,
                     
                     'W_kd': self.W_kd,
                     'M_inv': self.MAT_inv,
                     'm_k': self.lambdas_multiplicity,
                     
                     'idx': self.obs_idx + 1,
                     'y': self.obs,
                     'sigma': sigma}
        return(stan_data)
    
    def stan_fit(self, sigma):
        data = self.get_stan_data(sigma)

        model = pystan.StanModel(model_code=VC_STAN)
        fit = model.sampling(data=data, chains=4, iter=2000)
        return(fit.extract())


class SeqDEFT(SequenceSpace):
    def __init__(self, length, P, n_alleles=None,
                 alphabet_type='rna', parameters_only=False,
                 with_kernel_basis=True, log=None):
        self.init(length, n_alleles=n_alleles, alphabet_type=alphabet_type, log=log)
        self.set_P(P)
        
        # Number of faces
        self.s = comb(self.length, P) * comb(self.n_alleles, 2) ** P * self.n_alleles ** (self.length - P)
        
        # Prepare D kernel basis
        self.calc_laplacian()
        self.calc_MAT()
        
        if not parameters_only:
            if with_kernel_basis:
                self.prepare_D_kernel_basis()
            self.construct_D_spectrum()
    
    def set_P(self, P):
        if P == (self.length + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= P <= self.length:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
        self.P = P
    
    def load_data(self, counts):
        self.counts = counts
        self.N = counts.sum()
        self.R = (counts / self.N).values
        self.data_dict = {'N': self.N,
                          'R': self.R}
    
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
        a_max = self.s * fac_max
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
        a_min = self.s * fac_min
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

    def counts_to_data_dict(self, counts):
        n = counts.sum()
        return({'N': n, 'R': counts / n})
    
    def fit(self, counts=None, cv_fold=5, seed=None, resolution=0.1, max_a_max=1e12, 
            num_a=20, options=None, scale_by=1, fac_max=0.1, fac_min=1e-6, gtol=1e-3):
        if seed is not None:
            np.random.seed(seed)
            
        if counts is not None:
            self.load_data(counts)
            
        a_min, a_max = self.find_a_bounds(max_a_max, resolution, fac_max, fac_min,
                                          options, scale_by, gtol=gtol)
        phi_inf = self.get_phi_inf(options, scale_by, gtol=gtol)
        
        ll = self.compute_log_Ls(a_max, a_min, num_a, cv_fold, options,
                                 scale_by, gtol=gtol)
        self.a_star = ll.iloc[np.argmax(ll['log_likelihood']), :]['a']
        
        phi = self._fit(self.a_star, phi_initial=phi_inf, options=options,
                        scale_by=scale_by, gtol=gtol)
        
        self.log_Q_star = -phi - logsumexp(-phi)
        self.Q_star = np.exp(self.log_Q_star)
        assert(np.allclose(self.Q_star.sum(), 1))
    
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
        return phi
    
    def calc_Q(self, phi):
        return(np.exp(-phi) / np.sum(np.exp(-phi)))
    
    def D_opt(self, phi):
        Dphi = phi.copy()
        for p in range(self.P):
            Dphi = self.L_opt(Dphi, p)
        return Dphi / factorial(self.P)
    
    def S(self, phi, a, N, R):
        S1 = a/(2*self.s) * np.sum(phi * self.D_opt(phi))
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
        grad_S1 = a/self.s * self.D_opt(phi)
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
    
    def prepare_D_kernel_basis(self):
        # If the matrix desired has been made already, load it. Otherwise, construct and save it
        file_name1 = 'D_kernel_basis_alpha'+str(self.n_alleles)+'_l'+str(self.length)+'_P'+str(self.P)+'.npz'
        file_name2 = 'D_kernel_basis_orth_alpha'+str(self.n_alleles)+'_l'+str(self.length)+'_P'+str(self.P)+'.npz'
        
        fpath1 = join(CACHE_DIR, file_name1)
        fpath2 = join(CACHE_DIR, file_name2)
        
        if exists(fpath1) and exists(fpath2):
    
            self.report('Loading D kernel basis ...')
            D_kernel_basis_sparse = load_npz(fpath1)
            D_kernel_basis_orth_sparse = load_npz(fpath2)
    
            D_kernel_dim = 0
            for p in range(self.P):
                D_kernel_dim += int(comb(self.length, p) * (self.n_alleles-1)**p)
    
        else:
    
            self.report('Constructing D kernel basis ...')
            D_kernel_dim, D_kernel_basis_sparse, D_kernel_basis_orth_sparse = self.construct_D_kernel_basis()
    
            save_npz(fpath1, D_kernel_basis_sparse)
            save_npz(fpath2, D_kernel_basis_orth_sparse)
        
        self.D_kernel_dim = D_kernel_dim
        self.D_kernel_basis_sparse = D_kernel_basis_sparse
        self.D_kernel_basis_orth_sparse = D_kernel_basis_orth_sparse
    
    def construct_D_kernel_basis(self):
    
        # Generate bases and sequences
        bases = np.array(list(range(self.n_alleles)))
        seqs = np.array(list(product(bases, repeat=self.length)))
    
        # Construct D kernel basis
        for p in range(self.P):
    
            # Basis of kernel W(0)
            if p == 0:
                W0_dim = 1
                W0_basis = np.ones([self.n_genotypes,W0_dim])
                D_kernel_basis = W0_basis
    
            # Basis of kernel W(1)
            if p == 1:
                W1_dim = self.length*(self.n_alleles-1)
                W1_basis = np.zeros([self.n_genotypes,W1_dim])
                for site in range(self.length):
                    W1_basis[:,site*(self.n_alleles-1):(site+1)*(self.n_alleles-1)] = pd.get_dummies(seqs[:,site], drop_first=True).values
                D_kernel_basis = np.hstack((D_kernel_basis, W1_basis))
    
            # Basis of kernel W(>=2)
            if p >= 2:
                W2_dim = int(comb(self.length,p) * (self.n_alleles-1)**p)
                W2_basis = np.ones([self.n_genotypes,W2_dim])
                site_groups = list(combinations(range(self.length), p))
                base_groups = list(product(range(1,self.n_alleles), repeat=p))  # because we have dropped first base
                col = 0
                for site_group in site_groups:
                    for base_group in base_groups:
                        for i in range(p):
                            site, base_idx = site_group[i], base_group[i]-1  # change 'base' to its 'idx'
                            W2_basis[:,col] *= W1_basis[:,site*(self.n_alleles-1)+base_idx]
                        col += 1
                D_kernel_basis = np.hstack((D_kernel_basis, W2_basis))
    
        # Get kernel dimension
        D_kernel_dim = D_kernel_basis.shape[1]
    
        # Make D kernel basis orthonormal
        D_kernel_basis_orth = orth(D_kernel_basis)
    
        # Save D_kernel_basis and D_kernel_basis_orth as a sparse matrix
        D_kernel_basis_sparse = csr_matrix(D_kernel_basis)
        D_kernel_basis_orth_sparse = csr_matrix(D_kernel_basis_orth)
    
        # Return
        return D_kernel_dim, D_kernel_basis_sparse, D_kernel_basis_orth_sparse
    
    def construct_D_spectrum(self):
        self.report('Constructing D spectrum ...')

        D_eig_vals, D_multis = np.zeros(self.length+1), np.zeros(self.length+1)
        for k in range(self.length+1):
            lambda_k = k * self.n_alleles
            Lambda_k = 1
            for p in range(self.P):
                Lambda_k *= lambda_k - p * self.n_alleles
            m_k = comb(self.length,k) * (self.n_alleles-1)**k
            D_eig_vals[k], D_multis[k] = Lambda_k/factorial(self.P), m_k
    
        self.D_eig_vals, self.D_multis = D_eig_vals, D_multis

    def simulate(self, N, a_true, random_seed=None):
        # Set random seed
        np.random.seed(random_seed)
    
        # Simulate phi from prior distribution
        v = np.random.normal(size=self.n_genotypes)
        phi_true = np.zeros(self.n_genotypes)
        
        for k in range(self.P, self.length+1):
            eta_k = np.sqrt(self.s) / np.sqrt(a_true * self.D_eig_vals[k])
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
        w_k = np.zeros(self.length+1)
        for d in range(self.length+1):
            w_k[d] = self.w(k, d)
    
        # Solve for b_k
        b_k = solve(self.MAT, w_k)
        
        self.b_k = b_k
    
    def w(self, k, d):
        ss = 0
        for q in range(self.length+1):
            ss += (-1)**q * (self.n_alleles-1)**(k-q) * comb(d,q) * comb(self.length-d,k-q)
        return 1/self.n_alleles**self.length * ss
    
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
    
    def compute_log_Ls(self, a_max, a_min, num_a, cv_fold=5,
                       options=None, scale_by=1, gtol=1e-3):
        self.report('Running {} fold cross validation:'.format(cv_fold))
        a_values = np.geomspace(a_min, a_max, num_a)
        phi_inf = self.get_phi_inf(options, scale_by, gtol=gtol)
    
        log_Lss = np.zeros([cv_fold, num_a])
        cv_data = self.get_cv_iter(cv_fold, a_values)
        for k, i, a, train, validation in tqdm(cv_data, total=cv_fold * num_a):    
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
    
    def plot_a_optimization(self, axes):
        aa = self.log_Ls['a'].values
        log_Ls = self.log_Ls['log_likelihood'].values
        
        a_star = aa[log_Ls.argmax()]
        max_log_L = log_Ls.max()
        
        axes.scatter(np.log10(aa), log_Ls, color='blue', s=15, zorder=1)
        axes.scatter(np.log10(a_star), max_log_L, color='red', s=15, zorder=2)
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        x = xlims[0] + 0.05 * (xlims[1]- xlims[0])
        y = ylims[0] + 0.9 * (ylims[1]- ylims[0])
        axes.annotate('a* = {:.1f}'.format(a_star), xy=(x, y))
        axes.set_xlabel(r'$log_{10}$ (a)')
        axes.set_ylabel('Out of sample log(L)')
    
    def plot_density_vs_frequency(self, axes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.DataFrame({'logR': np.log10(self.R),
                                 'logQ': np.log10(self.Q_star)}).dropna()
                                 
        axes.scatter(data['logR'], data['logQ'],
                     color='black', s=5, alpha=0.4, zorder=2)
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
        axes.plot(lims, lims, color='grey', linewidth=0.5, alpha=0.5, zorder=1)
        arrange_plot(axes, xlabel=r'$log_{10}$(Frequency)',
                     ylabel=r'$log_{10}$(Q*)',
                     xlims=lims, ylims=lims)
    
    def plot_summary(self, fname):
        fig, subplots = init_fig(1, 2)
        
        self.plot_a_optimization(subplots[0])
        self.plot_density_vs_frequency(subplots[1])
        
        savefig(fig, fname)
        
    def write_output(self, fpath):
        output = pd.DataFrame({'phi_star': self.phi_star,
                               'Q_star': self.Q_star},
                              index=self.genotype_labels)
        output.to_csv(fpath)


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
