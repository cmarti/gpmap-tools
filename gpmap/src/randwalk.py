#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp

from gpmap.utils import write_pickle, load_pickle, check_eigendecomposition
from gpmap.src.utils import (check_symmetric, get_sparse_diag_matrix, check_error,
                             write_log)


class RandomWalk(object):
    def __init__(self):
        return()
    
    def report(self, msg):
        write_log(self.log, msg)


class TimeReversibleRandomWalk(RandomWalk):
    def __init__(self, space, log=None):
        self.space = space
        self.log = log
    
    def set_stationary_freqs(self, stationary_freqs):
        self.stationary_freqs = stationary_freqs
        sqrt_freqs = np.sqrt(stationary_freqs)
        self.diag_freq = get_sparse_diag_matrix(sqrt_freqs) 
        self.diag_freq_inv = get_sparse_diag_matrix(1 / sqrt_freqs)
    
    def _ensure_time_reversibility(self, rate_matrix, tol=1e-8):
        '''D_pi^{1/2} Q D_pi^{-1/2} has to be symmetric'''
        self.report('Checking numerical time reversibility')
        sandwich_rate_m = self.diag_freq.dot(rate_matrix).dot(self.diag_freq_inv)
        sandwich_rate_m = (sandwich_rate_m + sandwich_rate_m.T) / 2
        check_symmetric(sandwich_rate_m, tol=tol)
        rate_matrix = self.diag_freq_inv.dot(sandwich_rate_m).dot(self.diag_freq)
        return(rate_matrix, sandwich_rate_m)
    
    def calc_eigendecomposition(self, n_components=10, tol=1e-14, eig_tol=1e-2):
        n_components = min(n_components, self.space.n_states - 1)
        self.n_components = n_components
        
        # Transform matrix shifting eigenvalues close to 0 to avoid numerical problems
        upper_bound = np.abs(self.sandwich_rate_matrix).sum(1).max()
        sandwich_aux_matrix = identity(self.space.n_states) + 1 / upper_bound * self.sandwich_rate_matrix
        
        self.report('Calculating {} eigenvalue-eigenvector pairs'.format(n_components))
        v0 = self.diag_freq.dot(self.stationary_freqs)
        lambdas, q = eigsh(sandwich_aux_matrix, n_components,
                           v0=v0, which='LM', tol=tol)
        
        # Reverse order
        lambdas = lambdas[::-1]
        q = np.fliplr(q)
        
        # Undo eigenvalue shifting
        self.eigenvalues = upper_bound * (lambdas - 1)
        
        # Store right eigenvectors of the rate matrix
        self.right_eigenvectors = self.diag_freq_inv.dot(q)
        check_eigendecomposition(self.rate_matrix, self.eigenvalues,
                                 self.right_eigenvectors, tol=eig_tol)

    def calc_diffusion_axis(self):
        self.report('Scaling projection axis')
        scaling_factors = get_sparse_diag_matrix(1 / np.sqrt(-self.eigenvalues[1:]))
        self.nodes_df = pd.DataFrame(scaling_factors.dot(self.right_eigenvectors[:, 1:].T).T,
                                     index=self.space.state_labels,
                                     columns=np.arange(1, self.n_components).astype(str))
        self.nodes_df['function'] = self.space.function
        self.nodes_df['stationary_freq'] = self.stationary_freqs
        
    def calc_decay_rates(self):
        decay_rates = -1 / self.eigenvalues[1:]
        k = np.arange(1, decay_rates.shape[0] + 1)
        self.decay_rates_df = pd.DataFrame({'k': k, 'decay_rates': decay_rates})
    
    def calc_visualization(self, Ns=None, mean_function=None, 
                           mean_function_perc=None, n_components=10,
                           tol=1e-12, eig_tol=1e-2):
        if Ns is None and mean_function is None and mean_function_perc is None:
            msg = 'One of [Ns,  mean_function, mean_function_perc]'
            msg += 'is required to calculate the rate matrix'
            raise ValueError(msg)
        
        self.set_Ns(Ns=Ns, mean_function=mean_function,
                    mean_function_perc=mean_function_perc)
        self.calc_stationary_frequencies()
        self.calc_rate_matrix(tol=tol)
        self.calc_eigendecomposition(n_components, tol=tol, eig_tol=eig_tol)
        self.calc_diffusion_axis()
        self.calc_decay_rates()
    
    def save(self, fpath):
        attrs = ['Ns', 'nodes_df', 'edges_df', 'log_decay_rates', 'f',
                 'stationary_freqs', 'n_alleles', 'length',
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
    

class WMWSWalk(TimeReversibleRandomWalk):
    '''Class for Weak Mutation Weak Selection Random Walk on a SequenceSpace
    
    ...

    Attributes
    ----------
    space : DiscreteSpace class
        Space on which the random walk takes place
    Ns : real 
        Scaled effective population size for the evolutionary model
    rate_matrix : csr_matrix
        Rate matrix defining the continuous time process

    Methods
    -------
    set_Ns():
        Method to specify the scaled effective population size Ns, either directly
        or by specifying the mean function at stationarity or the percentile
        it represents from the distribution of functions across sequence space
        
    calc_stationary_frequencies():
        Calculates the stationary frequencies of the genotypes under the random
        walk specified on the discrete space
    
    calc_rate_matrix():
        Calculates the rate matrix for the continuous time process given 
        the scaled effective population size (Ns) or average phenotype at
        stationarity.
    
    
    '''
    def _calc_delta_function(self, rows, cols):
        return(self.space.function[cols] - self.space.function[rows])
    
    def _calc_stationary_frequencies(self, Ns):
        log_phi = Ns * self.space.function
        log_total = logsumexp(log_phi)
        log_stationary_freqs = log_phi - log_total
        return(np.exp(log_stationary_freqs))
    
    def calc_stationary_frequencies(self):
        '''Calculates the genotype stationary frequencies using Ns stored in 
        the object and stores the corresponding diagonal matrices with the
        sqrt transformation and its inverse
        '''
        check_error(hasattr(self, 'Ns'),
                    'Ns must be set for calculating stationary frequencies')
        self.set_stationary_freqs(self._calc_stationary_frequencies(self.Ns))
        return(self.stationary_freqs)
    
    def calc_stationary_mean_function(self):
        check_error(hasattr(self, 'stationary_freqs'),
                    'Calculate the stationary frequencies first')
        self.fmean = np.sum(self.space.function * self.stationary_freqs)
        return(self.fmean)
    
    def set_Ns(self, Ns=None, mean_function=None, mean_function_perc=None,
               tol=1e-4, maxiter=100, max_attempts=10):
        if Ns is not None:
            self.Ns = Ns
            return(Ns)
        
        if mean_function_perc is not None:
            mean_function = np.percentile(self.space.function, mean_function_perc)
        elif mean_function is None:
            msg = 'Either stationary_function or percentile must be provided'
            raise ValueError(msg)
            
        msg = 'Optimizing Ns to reach a stationary state with mean(f)={}'
        self.report(msg.format(mean_function))
        
        function = self.space.function
        def calc_stationary_function_error(logNs):
            freqs = self._calc_stationary_frequencies(np.exp(logNs))
            x = np.sum(function * freqs)
            sq_error = (mean_function - x) ** 2
            return(sq_error)
        
        for _ in range(max_attempts):
            result = minimize(calc_stationary_function_error, x0=0, tol=tol,
                              options={'maxiter': maxiter})
            Ns = np.exp(result.x[0])
            freqs = self._calc_stationary_frequencies(Ns)
            inferred_mean_function = np.sum(function * freqs)
            if calc_stationary_function_error(result.x[0]) < tol:
                break
        else:
            msg = 'Could not find the Ns that yields the desired mean function '
            msg += '= {:.2f}. Best guess is {:.2f}'
            
            raise ValueError(msg.format(mean_function, inferred_mean_function))

        self.Ns = Ns
        return(self.Ns)
    
    def _calc_rate(self, delta_function, Ns):
        S = Ns * delta_function
        return(S / (1 - np.exp(-S)))

    def _calc_rate_vector(self, delta_function, Ns):
        rate = np.ones(delta_function.shape[0])
        idxs = np.isclose(delta_function, 0) == False
        rate[idxs] = self._calc_rate(delta_function[idxs], Ns)
        return(rate)
    
    def calc_rate_matrix(self, tol=1e-8):
        self.report('Calculating rate matrix with Ns={}'.format(self.Ns))
        i, j = self.space.get_neighbor_pairs()
        delta_function = self._calc_delta_function(i, j)
        size = (self.space.n_states, self.space.n_states)
        rate_ij = self._calc_rate_vector(delta_function, self.Ns)
        rate_matrix = csr_matrix((rate_ij, (i, j)), shape=size)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate_matrix.setdiag(-rate_matrix.sum(1).A1)
    
        rate_matrix, sandwich_rate_m = self._ensure_time_reversibility(rate_matrix, tol=tol)
        self.rate_matrix = rate_matrix
        self.sandwich_rate_matrix = sandwich_rate_m
