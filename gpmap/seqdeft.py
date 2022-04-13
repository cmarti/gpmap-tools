import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import time
import warnings
from os.path import join, exists
from itertools import combinations, product

from numpy.linalg import eigh, solve
from numpy.random import choice, normal, random, seed
from scipy.linalg import orth
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, dia_matrix, load_npz, save_npz
from scipy.sparse.linalg import eigsh
from scipy.special import comb, factorial

from gpmap.settings import CACHE_DIR
from gpmap.base import SequenceSpace
from gpmap.utils import write_log
from gpmap.plot import init_fig, arrange_plot, savefig

U_MAX = 500
PHI_UB, PHI_LB = 100, 0


class SeqDEFT(object):
    def __init__(self, n_alleles, length, P, parameters_only=False,
                 with_kernel_basis=True, log=None):
        if P == (length + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= P <= length:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
    
        self.alpha = n_alleles
        self.l = length
        self.P = P
        self.G = n_alleles ** self.l
        self.s = comb(self.l, P) * comb(n_alleles, 2) ** P * n_alleles ** (self.l - P)
        self.log = log
        
        # Prepare D kernel basis
        self.calc_laplacian()
        if not parameters_only:
            if with_kernel_basis:
                self.prepare_D_kernel_basis()
            self.construct_D_spectrum()
    
    def report(self, msg):
        write_log(self.log, msg)

    def init_df_map(self):
        self.df_map = pd.DataFrame(columns=['a', 'phi'])
    
    def append_df_map(self, a, phi):
        self.df_map = self.df_map.append({'a': a, 'phi': phi}, ignore_index=True)
        
    def calc_a_max(self, max_a_max=1e12, resolution=0.1, fac_max=0.1, options=None, scale_by=1):
        a_max = self.s * fac_max
        
        self.report('Computing a_max = %f ...' % a_max)
        phi_max = self.estimate_MAP_solution(a_max, phi_initial=self.phi_inf, options=options, scale_by=scale_by)
        
        self.report('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, self.phi_inf))
        while D_geo(phi_max, self.phi_inf) > resolution and a_max < max_a_max:
            a_max *= 10
        
            self.report('Computing a_max = %f ...' % a_max)
            phi_max = self.estimate_MAP_solution(a_max, phi_initial=self.phi_inf, options=options, scale_by=scale_by)
            
            self.report('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, self.phi_inf))
        return(a_max, phi_max)
    
    def calc_a_min(self, resolution=0.1, fac_min=1e-6, options=None, scale_by=1):
        a_min = self.s * fac_min
        self.report('Computing a_min = %f ...' % a_min)
        phi_min = self.estimate_MAP_solution(a_min, phi_initial=self.phi_inf, options=options, scale_by=scale_by)
        self.report('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, self.phi_0))
        while D_geo(phi_min, self.phi_0) > resolution:
            a_min /= 10
            self.report('Computing a_min = %f ...' % a_min)
            phi_min = self.estimate_MAP_solution(a_min, phi_initial=self.phi_inf, options=options, scale_by=scale_by)
            self.report('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, self.phi_0))
            
        return(a_min, phi_min)

    def find_a_bounds(self, max_a_max=1e12, resolution=0.1, fac_max=0.1, fac_min=1e-6, options=None, scale_by=1):
        # Compute a = inf end
        self.report('Computing a = inf ...')
        self.phi_inf = self.estimate_MAP_solution(np.inf, options=options,
                                                  scale_by=scale_by)
        self.append_df_map(np.inf, self.phi_inf)
        
        self.report('Computing a = 0 ...')
        self.phi_0 = self.estimate_MAP_solution(0, options=options,
                                                scale_by=scale_by)
        self.append_df_map(a=0, phi=self.phi_0)
    
        self.a_max, phi_max = self.calc_a_max(max_a_max, resolution, fac_max,
                                              options, scale_by)
        self.append_df_map(self.a_max, phi_max)
        
        self.a_min, phi_min = self.calc_a_min(resolution, fac_min, options,
                                              scale_by)
        self.append_df_map(self.a_min, phi_min)

    def automatic_trace_MAP_curve(self, resolution=0.1, options=None, scale_by=1):
        # Gross-partition the MAP curve
        self.report('Gross-partitioning the MAP curve ...')
        aa = np.geomspace(self.a_min, self.a_max, 10)
        for i in range(len(aa)-2, 0, -1):
            a = aa[i]
            self.report('Computing a = %f ...' % a)
            phi_a = self.estimate_MAP_solution(a, phi_initial=self.phi_inf, options=options, scale_by=scale_by)
            self.append_df_map(a, phi_a)

        # Fine-partition the MAP curve to achieve desired resolution
        self.report('Fine-partitioning the MAP curve ...')
        flag = True
        while flag:
            self.df_map = self.df_map.sort_values(by='a')
            aa, phis = self.df_map['a'].values, self.df_map['phi'].values
            flag = False
            
            for i in range(len(self.df_map)-1):
                a_i, a_j = aa[i], aa[i+1]
                phi_i, phi_j = phis[i], phis[i+1]
                
                if D_geo(phi_i, phi_j) > resolution:
                    a = np.geomspace(a_i, a_j, 3)[1]
                    self.report('Computing a = %f ...' % a)
                    phi_a = self.estimate_MAP_solution(a, phi_initial=self.phi_inf,
                                                       options=options, scale_by=scale_by)
                    self.append_df_map(a, phi_a)
                    flag = True
                    
    def geom_trace_MAP_curve(self, num_a=20, options=None, scale_by=1,
                             a_values=None):
        self.report('Partitioning the MAP curve into %d points ...' % num_a)
        if a_values is None:
            if not hasattr(self, 'a_min') or not hasattr(self, 'a_max'):
                msg = 'Ensure either a values are provided '
                msg += 'or boundaries are calculated'
                raise ValueError(msg)
            a_values = np.geomspace(self.a_min, self.a_max, num_a)
            
        for i, a in enumerate(a_values):
            self.report('Computing a_%d = %f ...' % (i, a))
            phi_a = self.estimate_MAP_solution(a, phi_initial=self.phi_inf,
                                               options=options, scale_by=scale_by)
            self.append_df_map(a, phi_a)
    
    def trace_MAP_curve(self, resolution=0.1, num_a=20, fac_max=0.1,
                        fac_min=1e-6, options=None, scale_by=1, max_a_max=1e8,
                        a_values=None):
        self.init_df_map()
        if a_values is None:
            self.find_a_bounds(max_a_max=max_a_max, resolution=resolution,
                               fac_max=fac_max, fac_min=fac_min, options=options,
                               scale_by=scale_by)
        
        if num_a is None:
            self.automatic_trace_MAP_curve(resolution=resolution, options=options,
                                           scale_by=scale_by)
        else:
            self.geom_trace_MAP_curve(num_a=num_a, options=options,
                                      scale_by=scale_by, a_values=a_values)
    
        self.df_map.sort_values(by='a', inplace=True)
        self.df_map.reset_index(drop=True, inplace=True)
    
    def counts_to_data_dict(self, counts):
        n = counts.sum()
        return({'N': n, 'R': counts / n})
    
    def load_data(self, counts):
        self.counts = counts
        self.N = counts.sum()
        self.R = (counts / self.N).values
        self.data_dict = {'N': self.N,
                          'R': self.R}
        
    def fit(self, counts=None, cv_fold=5, seed=None, resolution=0.1, max_a_max=1e12, 
            num_a=20, options=None, scale_by=1, fac_max=0.1, fac_min=1e-6, 
            a_values=None):
        if seed is not None:
            np.random.seed(seed)
            
        if counts is not None:
            self.load_data(counts)
        self.trace_MAP_curve(resolution=resolution, max_a_max=max_a_max,
                             num_a=num_a, options=options, scale_by=scale_by,
                             fac_min=fac_min, fac_max=fac_max, a_values=a_values)
        self.compute_log_Ls(cv_fold)
        
        aa = self.df_map['a'].values
        phis = self.df_map['phi'].values
        log_Ls = self.df_map['log_L'].values
        
        self.a_star = aa[log_Ls.argmax()]
        self.phi_star = phis[log_Ls.argmax()]
        self.Q_star = np.exp(-self.phi_star) / np.sum(np.exp(-self.phi_star))
        assert(np.allclose(self.Q_star.sum(), 1))
    
    def calc_laplacian(self):
        gpmap = SequenceSpace()
        gpmap.init(self.l, self.alpha, alphabet_type='custom')
        gpmap.calc_laplacian()
        self.L = gpmap.L
    
    def L_opt(self, phi, p=0):
        return self.L.dot(phi) - p * self.alpha * phi
    
    def estimate_MAP_solution(self, a, phi_initial=None, data_dict=None,
                              method='L-BFGS-B', options=None, scale_by=1):
        # Get N and R
        if data_dict is None:
            N, R = self.N, self.R
        else: 
            N, R = data_dict['N'], data_dict['R']
    
        # Do scaling
        a /= scale_by
        N /= scale_by
    
        # Set initial guess of phi if it is not provided
        if phi_initial is None:
            Q_initial = np.ones(self.G) / self.G
            phi_initial = -np.log(Q_initial)
    
        # Find the MAP estimate of phi
        if a == 0:
    
            with np.errstate(divide='ignore'):
                phi_a = -np.log(R)
    
        elif 0 < a < np.inf:
            res = minimize(fun=self.S, jac=self.grad_S, args=(a,N,R),
                           x0=phi_initial, method=method, options=options)
            if not res.success:
                self.report(res.message)
            phi_a = res.x
    
        elif a == np.inf:
            b_initial = self.D_kernel_basis_orth_sparse.T.dot(phi_initial)
            res = minimize(fun=self.S_inf, jac=self.grad_S_inf, args=(N,R),
                           x0=b_initial, method=method, options=options)
            if not res.success:
                self.report(res.message)
            b_a = res.x
            phi_a = self.D_kernel_basis_orth_sparse.dot(b_a)
    
        else:
    
            self.report('"a" not in the right range.')
            sys.exit()
    
        # Undo scaling
        a *= scale_by
        N *= scale_by
    
        # Return
        return phi_a
    
    def calc_Q(self, phi):
        return(np.exp(-phi) / np.sum(np.exp(-phi)))
    
    def D_opt(self, phi):
        Dphi = phi.copy()
        for p in range(self.P):
            Dphi = self.L_opt(Dphi, p)
        return Dphi/factorial(self.P)
    
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
        regularizer = np.zeros(self.G)
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
        regularizer = np.zeros(self.G)
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
        file_name1 = 'D_kernel_basis_alpha'+str(self.alpha)+'_l'+str(self.l)+'_P'+str(self.P)+'.npz'
        file_name2 = 'D_kernel_basis_orth_alpha'+str(self.alpha)+'_l'+str(self.l)+'_P'+str(self.P)+'.npz'
        
        fpath1 = join(CACHE_DIR, file_name1)
        fpath2 = join(CACHE_DIR, file_name2)
        
        if exists(fpath1) and exists(fpath2):
    
            self.report('Loading D kernel basis ...')
            D_kernel_basis_sparse = load_npz(fpath1)
            D_kernel_basis_orth_sparse = load_npz(fpath2)
    
            D_kernel_dim = 0
            for p in range(self.P):
                D_kernel_dim += int(comb(self.l, p) * (self.alpha-1)**p)
    
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
        bases = np.array(list(range(self.alpha)))
        seqs = np.array(list(product(bases, repeat=self.l)))
    
        # Construct D kernel basis
        for p in range(self.P):
    
            # Basis of kernel W(0)
            if p == 0:
                W0_dim = 1
                W0_basis = np.ones([self.G,W0_dim])
                D_kernel_basis = W0_basis
    
            # Basis of kernel W(1)
            if p == 1:
                W1_dim = self.l*(self.alpha-1)
                W1_basis = np.zeros([self.G,W1_dim])
                for site in range(self.l):
                    W1_basis[:,site*(self.alpha-1):(site+1)*(self.alpha-1)] = pd.get_dummies(seqs[:,site], drop_first=True).values
                D_kernel_basis = np.hstack((D_kernel_basis, W1_basis))
    
            # Basis of kernel W(>=2)
            if p >= 2:
                W2_dim = int(comb(self.l,p) * (self.alpha-1)**p)
                W2_basis = np.ones([self.G,W2_dim])
                site_groups = list(combinations(range(self.l), p))
                base_groups = list(product(range(1,self.alpha), repeat=p))  # because we have dropped first base
                col = 0
                for site_group in site_groups:
                    for base_group in base_groups:
                        for i in range(p):
                            site, base_idx = site_group[i], base_group[i]-1  # change 'base' to its 'idx'
                            W2_basis[:,col] *= W1_basis[:,site*(self.alpha-1)+base_idx]
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

        D_eig_vals, D_multis = np.zeros(self.l+1), np.zeros(self.l+1)
        for k in range(self.l+1):
            lambda_k = k * self.alpha
            Lambda_k = 1
            for p in range(self.P):
                Lambda_k *= lambda_k - p * self.alpha
            m_k = comb(self.l,k) * (self.alpha-1)**k
            D_eig_vals[k], D_multis[k] = Lambda_k/factorial(self.P), m_k
    
        self.D_eig_vals, self.D_multis = D_eig_vals, D_multis

    def simulate(self, N, a_true, random_seed=None):
    
        # Set random seed
        seed(random_seed)
    
        # Simulate phi from prior distribution
        v = normal(loc=0, scale=1, size=self.G)
        self.construct_MAT()
        phi_true = np.zeros(self.G)
        for k in range(self.P, self.l+1):
            # eta_k = ? for k < self.P
            eta_k = np.sqrt(self.s) / np.sqrt(a_true * self.D_eig_vals[k])
            self.solve_b_k(k)
            phi_true += eta_k * self.W_k_opt(v)
    
        # Construct Q_true from the simulated phi
        Q_true = np.exp(-phi_true) / np.sum(np.exp(-phi_true))
    
        # Simulate N data points from Q_true
        data = choice(self.G, size=N, replace=True, p=Q_true)
    
        # Obtain count data
        values, counts = np.unique(data, return_counts=True)
        Ns = np.zeros(self.G)
        Ns[values] = counts
    
        # Normalize count data
        R = Ns / N
    
        # Save N and R
        data_dict = {'N': int(N), 'R': R, 'Q_true': Q_true}
    
        # Return
        return data_dict
    
    def construct_MAT(self):
        # Construct C
        C = np.zeros([self.l+1,self.l+1])
        for i in range(self.l+1):
            for j in range(self.l+1):
                if i == j:
                    C[i,j] = i * (self.alpha-2)
                if i == j+1:
                    C[i,j] = i
                if i == j-1:
                    C[i,j] = (self.l-j+1) * (self.alpha-1)
    
        # Construct D
        D = np.array(np.diag(self.l*(self.alpha-1)*np.ones(self.l+1), 0))
    
        # Construct B
        B = D - C
    
        # Construct u
        u = np.zeros(self.l+1)
        u[0], u[1] = self.l*(self.alpha-1), -1
    
        # Construct MAT column by column
        MAT = np.zeros([self.l+1,self.l+1])
        MAT[0,0] = 1
        for j in range(1, self.l+1):
            MAT[:,j] = np.array(np.mat(B)**(j-1) * np.mat(u).T).ravel()
    
        self.MAT = MAT
    
    def solve_b_k(self, k):
        # Tabulate w_k(d)
        w_k = np.zeros(self.l+1)
        for d in range(self.l+1):
            w_k[d] = self.w(k, d)
    
        # Solve for b_k
        b_k = solve(self.MAT, w_k)
        
        self.b_k = b_k
    
    def w(self, k, d):
        ss = 0
        for q in range(self.l+1):
            ss += (-1)**q * (self.alpha-1)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
        return 1/self.alpha**self.l * ss
    
    def W_k_opt(self, v):
        max_power = len(self.b_k) - 1
        Lsv = np.zeros([self.G,len(self.b_k)])
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
            N_logQ = np.zeros(self.G)
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
                               options=None, scale_by=1):
        data_dict = self.counts_to_data_dict(train)
        phi = self.estimate_MAP_solution(a, phi_initial=phi, data_dict=data_dict,
                                         options=options, scale_by=scale_by)
        Q = self.phi_to_Q(phi)
        logL = self.calc_log_likelihood(a, validation, Q)
        return(logL)
    
    def compute_log_Ls(self, cv_fold=5, options=None, scale_by=1):
        self.report('Running {} fold cross validation:'.format(cv_fold))
        log_Lss = np.zeros([cv_fold,len(self.df_map)])
        
        for k, (train, validation) in enumerate(self.split_cv(cv_fold)):
            self.report('\t# {}'.format(k))
    
            for i in range(len(self.df_map)):
                a, phi_a = self.df_map['a'].values[i], self.df_map['phi'].values[i]
                log_Lss[k,i] = self.calculate_cv_fold_logL(a, train, validation, phi_a,
                                                           options=options, scale_by=scale_by)
        log_Ls = log_Lss.mean(axis=0)
        self.df_map['log_L'] = log_Ls
    
    def expand_counts(self):
        obs = []
        for i, c in enumerate(self.counts):
            obs.extend([i] * c)
        obs = np.array(obs)
        np.random.shuffle(obs)
        return(obs)
    
    def count_obs(self, obs):
        v, c = np.unique(obs, return_counts=True)
        counts = pd.DataFrame(c, index=v).reindex(np.arange(self.G)).fillna(0).astype(int).values[:, 0]
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
        aa = self.df_map['a'].values[1:-1]
        log_Ls = self.df_map['log_L'].values[1:-1]
        
        a_star = aa[log_Ls.argmax()]
        max_log_L = log_Ls.max()
        
        axes.scatter(np.log10(aa), log_Ls, color='blue', s=15, zorder=1)
        axes.scatter(np.log10(a_star), max_log_L, color='red', s=15, zorder=2)
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        x = xlims[0] + 0.05 * (xlims[1]- xlims[0])
        y = ylims[0] + 0.9 * (ylims[1]- ylims[0])
        axes.annotate(s='a* = {:.1f}'.format(a_star), xy=(x, y))
        axes.set_xlabel(r'$log_{10}$ (a)')
        axes.set_ylabel('Out of sample log(L)')
    
    def plot_density_vs_frequency(self, axes):
        axes.scatter(np.log10(self.R), np.log10(self.Q_star),
                     color='black', s=5, alpha=0.4, zorder=2)
        # sns.histplot(x=np.log10(self.R+1e-6), y=np.log10(self.Q_star+1e-6),
        #              ax=axes, bins=100)
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
        plt.plot(lims, lims, color='grey', linewidth=0.5, alpha=0.5, zorder=1)
        arrange_plot(axes, xlabel=r'$log_{10}$(Frequency)',
                     ylabel=r'$log_{10}$(Q*)',
                     xlims=lims, ylims=lims)
    
    def plot_summary(self, fname):
        fig, subplots = init_fig(1, 2)
        
        self.plot_a_optimization(subplots[0])
        self.plot_density_vs_frequency(subplots[1])
        
        savefig(fig, fname)
        

#
# Data importation
#

def import_data(path, coding_dict, ignore_sites=None):

    # Read in processed data
    df = pd.read_csv(path, sep='\t', names=['sequence', 'count'], dtype=str)

    # Get flags for the sites of interest
    if ignore_sites is not None:
        flags = np.full(l+len(ignore_sites), True)
        flags[ignore_sites] = False

    # Obtain count data
    Ns = np.zeros(self.G)
    for i in range(len(df)):
        sequence, count = df.loc[i,'sequence'], int(df.loc[i,'count'])
        try:  # sequences with letters not included in coding_dict will be ignored
            seq = [coding_dict[letter] for letter in sequence]
            if ignore_sites is not None:
                seq = np.array(seq)[flags]
            pos = sequence_to_position(seq)
            Ns[pos] = count
        except:
            pass

    # Normalize count data
    N = np.sum(Ns)
    R = Ns / N

    # Save N and R
    data_dict = {'N': int(N), 'R': R}

    # Return
    return data_dict


#
# Data simulation
#




#
# MAP estimation
#



def compute_log_Es(data_dict, df_map):

    # Set global parameters for later use
    global Delta

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Construct D matrix
    Delta = D_mat()

    # Compute terms (~ "log Z_ratio")
    terms = np.zeros(len(df_map))

    for i in range(len(df_map)):

        a, phi_a = df_map['a'].values[i], df_map['phi'].values[i]

        if a == 0:

            terms[i] = -np.inf

        elif 0 < a < np.inf:

            S_a = S(phi_a, a, N, R)
            H_a = hess_S(phi_a, a, N, R)
            H_a_eig_vals = eigh(H_a)[0]
            terms[i] = - S_a + (self.G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * np.sum(np.log(H_a_eig_vals))

        elif a == np.inf:

            b_a = D_kernel_basis_orth_sparse.T.dot(phi_a)
            S_a = S_inf(b_a, N, R)
            Ne_sparse = csr_matrix(N*np.exp(-phi_a))
            Ne_ker = ((D_kernel_basis_orth_sparse.T.multiply(Ne_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()
            Ne_ker_eig_vals = eigh(Ne_ker)[0]
            D_row_eig_vals, D_row_multis = D_eig_vals[self.P:], D_multis[self.P:]
            terms[i] = - S_a - 1/2 * (np.sum(np.log(Ne_ker_eig_vals)) + np.sum(D_row_multis * np.log(D_row_eig_vals)))

        else:

            print('"a" not in the right range.')
            sys.exit()

    # Compute log_Es
    term_inf = terms[(df_map['a'] == np.inf)]
    log_Es = terms - term_inf

    # Save log_Es
    df_map['log_E'] = log_Es

    # Report execution time
    if self.time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def compute_log_Es_bounds(data_dict, df_map):

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Compute the diagonal element of D
    u_0 = np.zeros(self.G)
    u_0[0] = 1
    D_ii = np.sum(u_0 * D_opt(u_0))

    # Compute terms (~ "log Z_ratio")
    terms_lb, terms_ub = np.zeros(len(df_map)), np.zeros(len(df_map))

    for i in range(len(df_map)):

        a, phi_a = df_map['a'].values[i], df_map['phi'].values[i]

        if a == 0:

            terms_lb[i] = -np.inf
            terms_ub[i] = terms_lb[i]

        elif 0 < a < np.inf:

            S_a = S(phi_a, a, N, R)
            log_det_lb = np.sum(np.log(N * np.exp(-phi_a)))
            log_det_ub = np.sum(np.log(a/s * D_ii + N * np.exp(-phi_a)))
            terms_lb[i] = - S_a + (self.G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * log_det_ub
            terms_ub[i] = - S_a + (self.G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * log_det_lb

        elif a == np.inf:

            b_a = D_kernel_basis_orth_sparse.T.dot(phi_a)
            S_a = S_inf(b_a, N, R)
            Ne_sparse = csr_matrix(N*np.exp(-phi_a))
            Ne_ker = ((D_kernel_basis_orth_sparse.T.multiply(Ne_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()
            Ne_ker_eig_vals = eigh(Ne_ker)[0]
            D_row_eig_vals, D_row_multis = D_eig_vals[self.P:], D_multis[self.P:]
            terms_lb[i] = - S_a - 1/2 * (np.sum(np.log(Ne_ker_eig_vals)) + np.sum(D_row_multis * np.log(D_row_eig_vals)))
            terms_ub[i] = terms_lb[i]

        else:

            print('"a" not in the right range.')
            sys.exit()

    # Compute log_Es bounds
    term_inf = terms_lb[(df_map['a'] == np.inf)]
    log_Es_lb, log_Es_ub = terms_lb - term_inf, terms_ub - term_inf

    # Save log_Es bounds
    df_map['log_E_lb'], df_map['log_E_ub'] = log_Es_lb, log_Es_ub

    # Report execution time
    if self.time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map




def compute_rms_log_p_association(phi, p):
    if any(phi == np.inf):
        rms_log_p_association = np.inf
    else:
        Dphi = phi.copy()
        for i in range(p):
            Dphi = L_opt(Dphi, i)
        Dphi /= factorial(p)
        s_p = comb(l,p) * comb(self.alpha,2)**p * self.alpha**(l-p)
        rms_log_p_association = np.sqrt(abs(1/s_p * np.sum(phi * Dphi)))
    return rms_log_p_association


def compute_marginal_probability(phi):
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    Q_ker = D_kernel_basis_sparse.T.dot(Q)
    df_marginal_probs = pd.DataFrame(columns=['sites', 'bases', 'probability'])
    c = 0
    for p in range(self.P):
        site_groups = list(combinations(range(l), p))
        base_groups = list(product(range(1,self.alpha), repeat=p))  # because we have dropped first base
        for site_group in site_groups:
            for base_group in base_groups:
                df_marginal_probs = df_marginal_probs.append({'sites': site_group, 'bases': base_group,
                                                              'probability': Q_ker[c]}, ignore_index=True)
                c += 1
    return df_marginal_probs


#
# Posterior sampling
#


def posterior_sampling(phi, a, data_dict, num_samples, method, args, random_seed=None):

    # Set start time
    start_time = time.perf_counter()

    # Set random seed
    seed(random_seed)

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Perform posterior sampling
    if method == 'hmc':
        phi_initial, phi_samples, acceptance_rates = hamiltonian_monte_carlo(phi, a, N, R, num_samples, args)

    else:
        print('"method" not recognized.')
        sys.exit()

    # Report execution time
    if self.time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return phi_initial, phi_samples, acceptance_rates


def hamiltonian_monte_carlo(phi_star, a_star, N, R, num_samples, args):

    # Get HMC parameters
    e = args['e']
    L = args['L']
    Le = args['Le']
    L_max = args['L_max']
    m = args['m']
    f = args['f']
    window = args['window']
    gamma_old = args['gamma_old']
    gamma_new = args['gamma_new']
    perturbation = args['perturbation']
    num_warmup = args['num_warmup']
    num_thinning = args['num_thinning']

    num_draws = num_warmup + num_samples * num_thinning

    # Compute scales
    u_0 = np.zeros(self.G)
    u_0[0] = 1
    D_ii = np.sum(u_0 * D_opt(u_0))
    H_ii = a_star/s * D_ii + N * np.exp(-phi_star)
    scales = 1 / np.sqrt(H_ii)

    # Other settings
    phi_initial = phi_star + 2*(random(self.G)-0.5) * perturbation * scales
    a = a_star

    warnings.filterwarnings('error')

    if a == 0:

        phi_initial, phi_samples, acceptance_rates = None, None, None

    elif 0 < a < np.inf:

        # Initiate iteration
        phi_old = phi_initial.copy()
        S_phi_old = S(phi_old, a, N, R)
        grad_S_phi_old = grad_S(phi_old, a, N, R)
        psi_old = normal(loc=0, scale=np.sqrt(m), size=self.G)

        # HMC iterations
        phi_samples, acceptance_rates = np.zeros([self.G,num_samples]), []
        num_acceptance = 0
        k, c = 1, 0
        while k <= num_draws:

            try:

                # Update psi
                psi = normal(loc=0, scale=np.sqrt(m), size=self.G)
                psi_old = f * psi_old + np.sqrt(1-f**2) * psi

                # Set multiple stepsizes
                es = e * scales

                # Leapfrog steps
                phi, psi = phi_old.copy(), psi_old.copy()
                psi -= 1/2 * es * grad_S_phi_old
                for leapfrog_step in range(L-1):
                    phi += es / m * psi
                    grad_S_phi = grad_S(phi, a, N, R)
                    psi -= es * grad_S_phi
                phi += es / m * psi
                grad_S_phi = grad_S(phi, a, N, R)
                psi -= 1/2 * es * grad_S_phi
                psi *= -1

                # Compute probability ratio
                S_phi = S(phi, a, N, R)
                log_P = - S_phi - 1/2 * np.sum(psi**2) / m
                log_P_old = - S_phi_old - 1/2 * np.sum(psi_old**2) / m
                log_r = log_P - log_P_old

                # Accept/Reject proposed phi
                if log_r > np.log(random()):
                    phi_old = phi.copy()
                    S_phi_old = S_phi.copy()
                    grad_S_phi_old = grad_S_phi.copy()
                    psi_old = psi.copy()
                    num_acceptance += 1
                else:
                    phi_old = phi_old.copy()
                    S_phi_old = S_phi_old.copy()
                    grad_S_phi_old = grad_S_phi_old.copy()
                    psi_old = psi_old.copy()

                # Save phi and negate psi
                if (k > num_warmup) and (k % num_thinning == 0):
                    phi_samples[:,c] = phi_old
                    c += 1
                psi_old *= -1

                # Adapt e and L
                if k % window == 0:
                    acceptance_rate = num_acceptance / window
                    e_new = tune_hmc_stepsize(e, acceptance_rate)
                    e = (e**gamma_old * e_new*gamma_new)**(1/(gamma_old+gamma_new))
                    L = min(int(Le/e), L_max)
                    acceptance_rates.append(acceptance_rate)
                    num_acceptance = 0

                k += 1

            except Warning:

                phi_old = phi_old.copy()
                S_phi_old = S_phi_old.copy()
                grad_S_phi_old = grad_S_phi_old.copy()
                psi_old = psi_old.copy()
                e *= 0.95
                L = min(int(Le/e), L_max)

    elif a == np.inf:

        phi_initial, phi_samples, acceptance_rates = None, None, None

    else:

        print('"a" not in the right range.')
        sys.exit()

    # Return
    return phi_initial, phi_samples, acceptance_rates


def tune_hmc_stepsize(e, acceptance_rate):
    if acceptance_rate < 0.001:
        e *= 0.1
    elif 0.001 <= acceptance_rate < 0.05:
        e *= 0.5
    elif 0.05 <= acceptance_rate < 0.2:
        e *= 0.7
    elif 0.2 <= acceptance_rate < 0.5:
        e *= 0.8
    elif 0.5 <= acceptance_rate < 0.6:
        e *= 0.9
    elif 0.6 <= acceptance_rate <= 0.7:
        e *= 1
    elif 0.7 < acceptance_rate <= 0.8:
        e *= 1.1
    elif 0.8 < acceptance_rate <= 0.9:
        e *= 1.5
    elif 0.9 < acceptance_rate <= 0.95:
        e *= 2
    elif 0.95 < acceptance_rate:
        e *= 3
    return e


def compute_R_hat(multi_phi_samples0):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, self.G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    num_subchains, len_subchain = 2*num_chains, int(num_samples_per_chain/2)

    # Re-shape multi_phi_samples into a shape of (num_subchains, self.G, len_subchain)
    a = []
    for k in range(num_chains):
        a.append(multi_phi_samples[k,:,:len_subchain])
        a.append(multi_phi_samples[k,:,len_subchain:])
    multi_phi_samples_reshaped = np.array(a)

    # Compute R_hat for each component of phi
    R_hats = []
    for i in range(self.G):

        # Collect the (sub)chains of samples of phi_i
        i_collector = np.zeros([len_subchain,num_subchains])
        for j in range(num_subchains):
            i_collector[:,j] = multi_phi_samples_reshaped[j,i,:]

        # Compute the between-(sub)chain variance
        mean_0 = i_collector.mean(axis=0)
        mean_01 = mean_0.mean()
        B = len_subchain/(num_subchains-1) * np.sum((mean_0 - mean_01)**2)

        # Compute the within-(sub)chain variance
        s2 = np.zeros(num_subchains)
        for j in range(num_subchains):
            s2[j] = 1/(len_subchain-1) * np.sum((i_collector[:,j] - mean_0[j])**2)
        W = s2.mean()

        # Estimate the marginal posterior variance
        var = (len_subchain-1)/len_subchain * W + 1/len_subchain * B

        # Compute R_hat
        R_hat = np.sqrt(var/W)

        # Save
        R_hats.append(R_hat)

    # Return
    return np.array(R_hats)


def plot_trajectory(i, multi_phi_samples0, phi_map, colors, save_fig=False):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, self.G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    # Plot trajectory of the i-th component of phi
    plt.figure(figsize=(6,5))
    for k in range(num_chains):
        plt.plot(range(num_samples_per_chain), multi_phi_samples[k,i,:], color=colors[k], alpha=0.4, zorder=1)

    if phi_map is not None:
        plt.hlines(y=phi_map[i], xmin=0, xmax=num_samples_per_chain, color='black', zorder=2)

    plt.xlabel('Sample #', fontsize=14)
    plt.ylabel(r'$\phi_{%d}$'%i, fontsize=16)
    plt.xlim(0, num_samples_per_chain)
    if save_fig:
        plt.savefig('trajectory_%d'%i, dpi=200)
    plt.show()


def combine_samples(multi_phi_samples0):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, self.G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    # Combine phi samples
    phi_samples = multi_phi_samples[0,:,:]
    for k in range(1, num_chains):
        phi_samples = np.hstack((phi_samples, multi_phi_samples[k,:,:]))

    # Return
    return phi_samples


def plot_distribution(i, phi_samples_list, phi_map, num_bins, colors, save_fig=False):

    # Plot distribution of the i-th component of phi
    plt.figure(figsize=(6,5))
    hist_max = 0
    for k in range(len(phi_samples_list)):
        hist, bin_edges = np.histogram(phi_samples_list[k][i,:], bins=num_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = np.linspace(bin_edges[0]+bin_width/2, bin_edges[-1]-bin_width/2, len(bin_edges)-1)
        plt.bar(bin_centers, hist, width=bin_width, color=colors[k], alpha=0.5, edgecolor=colors[k], zorder=1)
        hist_max = max(hist_max, max(hist))

    if phi_map is not None:
        plt.vlines(x=phi_map[i], ymin=0, ymax=1.2*hist_max, color='black', zorder=2)

    plt.xlabel(r'$\phi_{%d}$'%i, fontsize=16)
    plt.ylim(0, 1.2*hist_max)
    if save_fig:
        plt.savefig('distribution_%d'%i, dpi=200)
    plt.show()


#
# Analysis tools: computing pairwise association
#


def compute_log_ORs(phi, site_i, site_j, site_i_mut=None, site_j_mut=None, condition={}, coding_dict=None):

    # If coding dictionary is provided, convert letters to codes
    if coding_dict is not None:
        if (site_i_mut is not None) and (site_j_mut is not None):
            site_i_mut = [coding_dict[letter] for letter in site_i_mut]
            site_j_mut = [coding_dict[letter] for letter in site_j_mut]
        for key in condition.keys():
            value = [coding_dict[letter] for letter in condition[key]]
            condition[key] = value

    # Generate bases
    bases = list(range(self.alpha))

    # Get background sites
    bg_sites = list(set(range(l)) - {site_i,site_j})

    # Get allowable bases for each background site
    bg_sites_bases = []
    for bg_site in bg_sites:
        if bg_site in condition.keys():
            bg_sites_bases.append(condition[bg_site])
        else:
            bg_sites_bases.append(bases)

    # Generate background sequences
    bg_seqs = product(*bg_sites_bases)

    # Generate all possible 2x2 faces that can be formed by site i (mut) and site j (mut)
    if (site_i_mut is not None) and (site_j_mut is not None):
        faces = [list(product(site_i_mut, site_j_mut))]
    else:
        base_pairs = list(combinations(bases, 2))
        base_pair_products = list(product(base_pairs, base_pairs))
        faces = []
        for base_pair_product in base_pair_products:
            faces.append(list(product(*base_pair_product)))

    # For each background sequence, compute log_OR on all faces formed by site i (mut) and site j (mut)
    log_ORs, associated_seqs = [], []
    for bg_seq in bg_seqs:
        for face in faces:
            face_phis, face_seqs = [], []
            for k in range(4):
                face_vertex_k_seq = np.full(l, -1, dtype=int)
                face_vertex_k_seq[bg_sites] = bg_seq
                face_vertex_k_seq[[site_i,site_j]] = face[k]
                face_vertex_k_pos = sequence_to_position(face_vertex_k_seq)
                face_phis.append(phi[face_vertex_k_pos])
                face_seqs.append(face_vertex_k_seq)
            log_ORs.append(-((face_phis[3]-face_phis[1])-(face_phis[2]-face_phis[0])))
            associated_seqs.append(face_seqs)

    # If coding dictionary is provided, convert codes to letters
    if coding_dict is not None:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        TMP = []
        for seqs in associated_seqs:
            tmp = []
            for seq in seqs:
                tmp.append(''.join([rev_coding_dict[code] for code in seq]))
            TMP.append(tmp)
        associated_seqs = TMP

    # Save log_ORs and associated sequences in a dataframe
    df_log_ORs = pd.DataFrame()
    df_log_ORs['log_OR'], df_log_ORs['associated_seqs'] = log_ORs, associated_seqs
    df_log_ORs = df_log_ORs.sort_values(by='log_OR', ascending=False).reset_index(drop=True)

    # Return
    return df_log_ORs


#
# Analysis tools: making visualization
#


def make_visualization(Q, markov_chain, K=20, tol=1e-9, reuse_Ac=False, path='sparse_matrix/Ac/'):

    # Set start time
    Start_time = time.perf_counter()

    # If reuse existing A and c, load them. Otherwise, construct A and c from scratch and save them
    if reuse_Ac:

        print('Loading A and c ...')
        start_time = time.perf_counter()
        A_sparse = load_npz(path+'A.npz')
        c = joblib.load(path+'c.pkl')
        if self.time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

    else:

        print('Constructing A and c ...')
        start_time = time.perf_counter()
        A_sparse, c = construct_Ac(Q, markov_chain)
        if self.time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        save_npz(path+'A.npz', A_sparse)
        joblib.dump(c, path+'c.pkl')

    # Compute the dominant eigenvalues and eigenvectors of A
    print('Computing dominant eigenvalues and eigenvectors of A ...')
    start_time = time.perf_counter()
    eig_vals_tilt, eig_vecs_tilt = eigsh(A_sparse, K, which='LM', tol=tol)
    if self.time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))

    # Check accuracy of the eigenvalues and eigenvectors of A
    df_check = pd.DataFrame(columns=['eigenvalue', 'colinearity', 'max_difference'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        Au = A_sparse.dot(u)
        max_diff = abs(Au-lda*u).max()
        Au /= np.sqrt(np.sum(Au**2))
        colin = np.sum(Au*u)
        df_check = df_check.append({'eigenvalue': lda, 'colinearity': colin, 'max_difference': max_diff}, ignore_index=True)
    df_check = df_check.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Obtain the eigenvalues and eigenvectors of T, and use them to construct visualization coordinates
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(self.G,self.G))
    df_visual = pd.DataFrame(columns=['eigenvalue', 'coordinate'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        if lda < 1:
            eig_val = c * (lda - 1)
            eig_vec = Diag_Q_inv_sparse.dot(u)
            coordinate = eig_vec / np.sqrt(-eig_val)
            df_visual = df_visual.append({'eigenvalue': eig_val, 'coordinate': coordinate}, ignore_index=True)
        else:
            df_visual = df_visual.append({'eigenvalue': 0, 'coordinate': np.full(self.G,np.nan)}, ignore_index=True)
    df_visual = df_visual.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Report execution time
    if self.time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - Start_time))

    # Return
    return df_visual, df_check


def construct_Ac(Q, markov_chain):

    # Choose a model for the reversible Markov chain
    if markov_chain == 'evolutionary':
        T_ij = T_evolutionary
    elif markov_chain == 'Metropolis':
        T_ij = T_Metropolis
    elif markov_chain == 'power_law':
        T_ij = T_power_law
    else:
        print('markov_chain "model" not recognized.')
        sys.exit()

    # Generate bases and sequences
    bases = list(range(self.alpha))
    seqs = list(product(bases, repeat=l))

    # Construct transition matrix (or rate matrix) T
    row_ids, col_ids, values = [], [], []
    for i in range(self.G):
        tmp = []
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    # Blocking transitions between +1 & -1 state for 'aneuploidy data' subsets
                    # k = np.where(np.array(seqs[i]) != np.array(seqs[j]))[0][0]
                    # if (seqs[i][k]==1 and seqs[j][k]==2) or (seqs[i][k]==2 and seqs[j][k]==1):
                    #     value = 0
                    # else:
                    value = T_ij(Q[i], Q[j])
                    row_ids.append(i)
                    col_ids.append(j)
                    values.append(value)
                    tmp.append(value)
        row_ids.append(i)
        col_ids.append(i)
        values.append(-np.sum(tmp))

    # Save T as a sparse matrix
    T_sparse = csr_matrix((values, (row_ids, col_ids)), shape=(self.G,self.G))

    # Construct a symmetric matrix T_tilt from T
    Diag_Q_sparse = dia_matrix((np.sqrt(Q), np.array([0])), shape=(self.G,self.G))
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(self.G,self.G))
    T_tilt_sparse = Diag_Q_sparse.dot(T_sparse * Diag_Q_inv_sparse)

    # Choose the value of c
    c = 0
    for i in range(self.G):
        sum_i = abs(T_tilt_sparse[i,i])
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    sum_i += abs(T_tilt_sparse[i,j])
        c = max(c, sum_i)

    # Construct A and save it as a sparse matrix
    I_sparse = dia_matrix((np.ones(self.G), np.array([0])), shape=(self.G,self.G))
    A_sparse = I_sparse + 1/c * T_tilt_sparse

    # Return
    return A_sparse, c


def T_evolutionary(Q_i, Q_j, par=1):
    if Q_i == Q_j:
        return 1
    else:
        return par * (np.log(Q_j)-np.log(Q_i)) / (1 - np.exp(-par * (np.log(Q_j)-np.log(Q_i))))


def T_Metropolis(Q_i, Q_j):
    if Q_j > Q_i:
        return 1
    else:
        return Q_j/Q_i


def T_power_law(Q_i, Q_j, par=1/2):
    return Q_j**par / Q_i**(1-par)


def get_nodes(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual['coordinate'].values[kx]*xflip, df_visual['coordinate'].values[ky]*yflip

    # Save the coordinates
    df_nodes = pd.DataFrame()
    df_nodes['node'], df_nodes['x'], df_nodes['y'] = range(self.G), x, y

    # Return
    return df_nodes


def get_edges(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual['coordinate'].values[kx]*xflip, df_visual['coordinate'].values[ky]*yflip

    # Generate bases and sequences
    bases = list(range(self.alpha))
    seqs = list(product(bases, repeat=l))

    # Get coordinates of all edges (i > j)
    nodes_i, nodes_j, edges = [], [], []
    for i in range(self.G):
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    if i > j:
                        nodes_i.append(i)
                        nodes_j.append(j)
                        edges.append([(x[i],y[i]), (x[j],y[j])])

    # Save the coordinates
    df_edges = pd.DataFrame()
    df_edges['node_i'], df_edges['node_j'], df_edges['edge'] = nodes_i, nodes_j, edges

    # Return
    return df_edges


#
# Analysis tools: others
#


def find_local_max(phi, data_dict=None, coding_dict=None, threshold=0):

    # Get counts if data dictionary is provided
    if data_dict is not None:
        N, R = data_dict['N'], data_dict['R']
        Ns = N * R

    # Generate bases and sequences
    bases = list(range(self.alpha))
    seqs = list(product(bases, repeat=l))

    # Find local maxima
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    local_max_seqs, local_max_probs, local_max_cnts = [], [], []
    for i in range(self.G):
        if Q[i] > threshold:
            js = []
            for site in range(l):
                for base in bases:
                    seq_i = np.array(seqs[i])
                    if base != seq_i[site]:
                        seq_i[site] = base
                        j = sequence_to_position(seq_i)
                        js.append(j)
            if all(np.greater(np.ones(l*(self.alpha-1))*Q[i], np.take(Q,js))):
                local_max_seqs.append(seqs[i])
                local_max_probs.append(Q[i])
                if data_dict is not None:
                    local_max_cnts.append(int(Ns[i]))

    # If coding dictionary is provided, convert codes to letters
    if coding_dict is not None:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        tmp = []
        for seq in local_max_seqs:
            tmp.append(''.join([rev_coding_dict[code] for code in seq]))
        local_max_seqs = tmp

    # Save local maxima in a dataframe
    df_local_max = pd.DataFrame()
    df_local_max['sequence'], df_local_max['probability'] = local_max_seqs, local_max_probs
    if data_dict is not None:
        df_local_max['count'] = local_max_cnts
    df_local_max = df_local_max.sort_values(by='probability', ascending=False).reset_index(drop=True)

    # Return
    return df_local_max


def compute_entropy(phi):
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    if any(Q == 0):
        flags = (Q != 0)
        entropy = -np.sum(Q[flags] * np.log2(Q[flags]))
    else:
        entropy = -np.sum(Q * np.log2(Q))
    return entropy


#
# Utility functions
#


def sequence_to_position(seq, coding_dict=None):
    if coding_dict is None:
        return int(np.sum(seq * seq_to_pos_converter))
    else:
        tmp = [coding_dict[letter] for letter in seq]
        return int(np.sum(tmp * seq_to_pos_converter))


def position_to_sequence(pos, coding_dict=None):
    if coding_dict is None:
        return sequences[pos]
    else:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        tmp = sequences[pos]
        return ''.join([rev_coding_dict[code] for code in tmp])


def D_geo(phi1, phi2):
    Q1 = np.exp(-phi1) / np.sum(np.exp(-phi1))
    Q2 = np.exp(-phi2) / np.sum(np.exp(-phi2))
    x = min(np.sum(np.sqrt(Q1 * Q2)), 1)
    return 2 * np.arccos(x)




def sample_from_data(N, data_dict, random_seed=None):

    # Set random seed
    seed(random_seed)

    # Generate raw data
    raw_data = generate_raw_data(data_dict, random_seed)

    # Sample N points from raw data
    sample = choice(raw_data, size=N, replace=False)

    # Turn sample into count data
    values, counts = np.unique(sample, return_counts=True)
    Ns = np.zeros(self.G)
    Ns[values] = counts

    # Make sure the amount of sample is correct
    if np.sum(Ns) != N:
        print('"sample" not correctly drawn from data.')

    # Save N and R
    R = Ns / N
    sample_dict = {'N': int(N), 'R': R}

    # Return
    return sample_dict


#
# Basic functions
#


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)

def hess_S(phi, a, N, R):
    hess_S1 = a/s * Delta
    hess_S2 = N * np.diag(safe_exp(-phi), 0)
    return np.array(hess_S1 + hess_S2)


def hess_S_inf(b, N, R):
    phi = D_kernel_basis_orth_sparse.dot(b)
    hess_S_inf_sparse = csr_matrix(N*np.exp(-phi))
    return ((D_kernel_basis_orth_sparse.T.multiply(hess_S_inf_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()


def L_mat():
    L = np.zeros([self.G,self.G])
    for i in range(self.G):
        for j in range(i+1):
            u_i, u_j = np.zeros(self.G), np.zeros(self.G)
            u_i[i], u_j[j] = 1, 1
            L[i,j] = np.sum(u_i * L_opt(u_j))
            L[j,i] = L[i,j]
    return L


def D_mat():
    D = np.zeros([self.G,self.G])
    for i in range(self.G):
        for j in range(i+1):
            u_i, u_j = np.zeros(self.G), np.zeros(self.G)
            u_i[i], u_j[j] = 1, 1
            D[i,j] = np.sum(u_i * D_opt(u_j))
            D[j,i] = D[i,j]
    return D
