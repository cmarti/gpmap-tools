#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp

from gpmap.utils import write_pickle, load_pickle
from gpmap.src.utils import check_symmetric, get_sparse_diag_matrix


class RandomWalk(object):
    def __init__(self):
        return()


class TimeReversibleRandomWalk(RandomWalk):
    def __init__(self, space, log=None):
        self.space = space
        self.log = log
    
    def _ensure_time_reversibility(self, rate_matrix, tol=1e-8):
        '''D_pi^{1/2} Q D_pi^{-1/2} has to be symmetric'''
        self.report('Checking numerical time reversibility')
        sandwich_rate_m = self.diag_freq.dot(rate_matrix).dot(self.diag_freq_inv)
        sandwich_rate_m = (sandwich_rate_m + sandwich_rate_m.T) / 2
        check_symmetric(sandwich_rate_m, tol=tol)
        rate_matrix = self.diag_freq_inv.dot(sandwich_rate_m).dot(self.diag_freq)
        return(rate_matrix, sandwich_rate_m)
    
    def _calc_sandwich_aux_matrix(self):
        '''
        This method calculates an auxiliary matrix to ensure that the calculated
        eigenvalues are close to 0 to avoid numerical problems
        '''
        self.report('Calculating A = I - 1/c * T for eigen-decomposition')
        self.upper_bound_eigenvalue = np.abs(self.sandwich_rate_matrix).sum(1).max()
        I = identity(self.n_genotypes) 
        sandwich_aux_matrix = I + 1 / self.upper_bound_eigenvalue * self.sandwich_rate_matrix
        return(sandwich_aux_matrix)
    
    def _check_eigendecomposition(self, matrix, eigenvalues, right_eigenvectors, tol=1e-3):
        self.report('Testing eigendecomposition of T')
        for i in range(self.n_components):
            u = right_eigenvectors[:, i]
            v1 = matrix.dot(u)
            v2 = eigenvalues[i] * u
            abs_err = np.mean(np.abs(v1 - v2)) 
            
            if abs_err > tol:
                msg = 'Numeric error in eigendecomposition: abs error = {:.5f} > {:.5f}'
                raise ValueError(msg.format(abs_err, tol))
            
        self.report('Eigendecomposition is correct')
    
    def calc_eigendecomposition(self, n_components=10, tol=1e-14, eig_tol=1e-2,
                                maxiter=30):
        n_components = min(n_components, self.n_genotypes-1)
        self.n_components = n_components
        sandwich_aux_matrix = self._calc_sandwich_aux_matrix()
        
        self.report('Calculating {} eigenvalue-eigenvector pairs'.format(n_components))
        v0 = self.diag_freq.dot(self.genotypes_stationary_frequencies)
        lambdas, q = eigsh(sandwich_aux_matrix, n_components,
                           v0=v0, which='LM', tol=tol)
        
        # Reverse order
        lambdas = lambdas[::-1]
        q = np.fliplr(q)
        
        # Store results
        self.eigenvalues = self.upper_bound_eigenvalue * (lambdas - 1)
        self.right_eigenvectors = self.diag_freq_inv.dot(q)
        self._check_eigendecomposition(self.rate_matrix,
                                       self.eigenvalues,
                                       self.right_eigenvectors,
                                       tol=eig_tol)

    def _calc_decay_rates(self):
        return(-1 / self.eigenvalues[1:])
    
    def _calc_projection(self):
        self.report('Scaling projection axis')
        projection = []
        for i in range(self.n_components):
            eigenvalue = self.eigenvalues[i]
            right_eigenvector = self.right_eigenvectors[:, i]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projection.append(right_eigenvector / np.sqrt(-eigenvalue))
                
        colnames = [str(x) for x in np.arange(1, len(projection))]
        self.nodes_df = pd.DataFrame(np.vstack(projection[1:]).T,
                                     index=self.genotypes,
                                     columns=colnames)
        self.nodes_df['f'] = self.f
        self.nodes_df['stationary_freq'] = self.genotypes_stationary_frequencies
        
        i, j = self._get_edges()
        self.edges_df = pd.DataFrame({'i': i, 'j': j})
        
        if hasattr(self, 'prot'):
            self.nodes_df['protein'] = self.prot
        decay_rates = self._calc_decay_rates()
        k = np.arange(1, decay_rates.shape[0] + 1)
        self.decay_df = pd.DataFrame({'k': k, 'decay_rates': decay_rates})
        
    def calc_visualization(self, Ns=None, meanf=None, 
                           perc_function=None, n_components=10,
                           tol=1e-12, eig_tol=1e-2):
        if Ns is None and meanf is None and perc_function is None:
            msg = 'Either Ns or the expected mean function or its percentile '
            msg += 'in equilibrium is required to calculate the rate matrix'
            raise ValueError(msg)
        
        if meanf is not None or perc_function is not None:
            Ns = self.calc_Ns(stationary_function=meanf, perc=perc_function)
            
        self.calc_stationary_frequencies(Ns)
        self.calc_rate_matrix(Ns, tol=tol)
        self.calc_eigendecomposition(n_components, tol=tol, eig_tol=eig_tol)
        self._calc_projection()
    
    def save(self, fpath):
        attrs = ['Ns', 'nodes_df', 'edges_df', 'log_decay_rates', 'f',
                 'genotypes_stationary_frequencies', 'n_alleles', 'length',
                 'alphabet_type']
        data = {attr: getattr(self, attr) for attr in attrs}
        write_pickle(data, fpath)
    
    def write_tables(self, prefix):
        self.nodes_df.to_csv('{}.nodes.csv'.format(prefix))
        self.edges_df.to_csv('{}.edges.csv'.format(prefix), index=False)
        self.decay_df.to_csv('{}.decay_rates.csv'.format(prefix),
                             index=False)
        
    def load(self, fpath, log=None):
        self.log = log
        self.report('Loading visualization data from {}'.format(fpath))
        data = load_pickle(fpath)
        self.init(data['length'], data['n_alleles'],
                  alphabet_type=data['alphabet_type'], log=log)
        for attr, value in data.items():
            setattr(self, attr, value)
    

def filter_genotypes(nodes_df, genotypes, edges_df=None):
    size = nodes_df.shape[0]
    nodes_df['index'] = np.arange(size)
    nodes_df = nodes_df.loc[genotypes, :]
    
    if edges_df is not None:
        m = csr_matrix((edges_df.index, (edges_df['i'], edges_df['j'])),
                       shape=(size, size))
        m = m[nodes_df['index'], :][:, nodes_df['index']].tocoo()
        edges_df = edges_df.iloc[m.data, :].copy()
        edges_df['i'] = m.row
        edges_df['j'] = m.col
        return(nodes_df, edges_df)
    else:
        return(nodes_df)


class WMWSWalk(TimeReversibleRandomWalk):
    def _calc_delta_function(self, rows, cols):
        return(self.space.function[cols] - self.space.function[rows])
    
    def calc_stationary_frequencies(self, Ns, store_sqrt_diag_matrices=True):
        log_freqs = Ns * self.f
        self.log_total = logsumexp(log_freqs)
        self.log_genotypes_stationary_frequencies = log_freqs - self.log_total
        freqs = np.exp(self.log_genotypes_stationary_frequencies)
        
        # Store equilibrium frequencies
        self.genotypes_stationary_frequencies = freqs
        
        if store_sqrt_diag_matrices:
            sqrt_freqs = np.sqrt(self.genotypes_stationary_frequencies)
            self.diag_freq = get_sparse_diag_matrix(sqrt_freqs) 
            self.diag_freq_inv = get_sparse_diag_matrix(1 / sqrt_freqs)
        
        return(freqs)
    
    def calc_stationary_function(self):
        self.fmean = np.sum(self.f * self.genotypes_stationary_frequencies)
        return(self.fmean)
    
    def calc_Ns(self, stationary_function=None, perc=None, tol=1e-8, maxiter=100, 
                max_attempts=10):
        if perc is not None:
            stationary_function = np.percentile(self.f, perc)
        elif stationary_function is None:
            msg = 'Either stationary_function or percentile must be provided'
            raise ValueError(msg)
            
        msg = 'Optimizing Ns to reach a stationary state with mean(f)={}'
        self.report(msg.format(stationary_function))
        
        def f(param):
            Ns = np.exp(param)
            self.calc_stationary_frequencies(Ns, store_sqrt_diag_matrices=False)
            x = self.calc_stationary_function()
            rel_error = (stationary_function - x) ** 2
            return(rel_error)
        
        for _ in range(max_attempts):
            result = minimize(f, x0=0, tol=tol,
                              options={'maxiter': maxiter})
            if f(result.x[0]) < tol:
                break
        else:
            msg = 'Could not find the Ns that yields the desired mean function'
            raise ValueError(msg)

        Ns = np.exp(result.x[0])
        return(Ns)
    
    def _calc_rate(self, delta_function, Ns):
        S = Ns * delta_function
        return(S / (1 - np.exp(-S)))

    def _calc_rate_vector(self, delta_function, Ns):
        rate = np.ones(delta_function.shape[0])
        idxs = np.isclose(delta_function, 0) == False
        rate[idxs] = self._calc_rate(delta_function[idxs], Ns)
        return(rate)
    
    def calc_rate_matrix(self, Ns, tol=1e-8):
        self.report('Calculating rate matrix with Ns={}'.format(Ns))
        i, j = self.space.get_neighbor_pairs()
        delta_function = self._calc_delta_function(i, j)
        size = (self.n_states, self.n_states)
        rate_ij = self._calc_rate_vector(delta_function, Ns)
        rate_matrix = csr_matrix((rate_ij, (i, j)), shape=size)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate_matrix.setdiag(-rate_matrix.sum(1).A1)
    
        rate_matrix, sandwich_rate_m = self._ensure_time_reversibility(rate_matrix, tol=tol)
        self.Ns = Ns
        self.rate_matrix = rate_matrix
        self.sandwich_rate_matrix = sandwich_rate_m
