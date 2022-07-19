#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp

from gpmap.src.utils import (check_symmetric, get_sparse_diag_matrix, check_error,
                             write_log, check_eigendecomposition)
from scipy.special._basic import comb
from itertools import combinations
from scipy.linalg.decomp import eig


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
        
    def calc_relaxation_times(self):
        decay_rates = -self.eigenvalues[1:]
        relaxation_times = 1 / decay_rates 
        k = np.arange(1, decay_rates.shape[0] + 1)
        self.decay_rates_df = pd.DataFrame({'k': k, 'decay_rates': decay_rates,
                                            'relaxation_time': relaxation_times})
    
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
        self.calc_relaxation_times()
    
    def write_tables(self, prefix, write_edges=False, edges_format='npz'):
        self.nodes_df.to_csv('{}.nodes.csv'.format(prefix))
        self.decay_rates_df.to_csv('{}.decay_rates.csv'.format(prefix),
                                   index=False)
        if write_edges:
            if edges_format == 'npz':
                fpath = '{}.edges.npz'.format(prefix)
                self.space.write_edges_npz(fpath)
            elif edges_format == 'csv':
                fpath = '{}.edges.csv'.format(prefix)
                self.space.write_edges_csv(fpath)
            else:
                msg = 'edges_format can only take values ["npz", "csv"]'
                raise ValueError(msg)
    

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
    def calc_neutral_rates(self, neutral_stat_freqs=None,
                           exchange_rates=None, site_mut_rates=None):
        neutral_mixing_rate = np.inf
        
        if neutral_stat_freqs is None:
            neutral_stat_freqs = [np.full(alpha, 1/alpha)
                                  for alpha in self.space.n_alleles]
        
        if exchange_rates is None:
            exchange_rates = [np.full(int(comb(alpha, 2)), 1/comb(alpha, 2))
                              for alpha in  self.space.n_alleles]
        
        if site_mut_rates is None:
            site_mut_rates = np.ones(self.space.seq_length)
        
        neutral_site_Qs = []
        for freqs, ex_rates, site_mu in zip(neutral_stat_freqs, exchange_rates, site_mut_rates):
            
            msg = 'Ensure that the exchangeability rates matrices are symmetric'
            msg += 'to ensure neutral time reversible dynamics'
            check_error(np.all(ex_rates == ex_rates.T), msg=msg)
            
            msg = 'Make sure that the neutral stationary rates sum to 1'
            check_error(freqs.sum() == 1, msg=msg)
            
            msg = 'Make sure that all the exchangeability rates sum to 1: {}'.format(ex_rates)
            check_error(np.allclose(ex_rates.sum(), 1), msg=msg)
            
            # Create site matrix from GTR model
            D = get_sparse_diag_matrix(freqs).todense()
            ex_rates_m = np.zeros((freqs.shape[0], freqs.shape[0]))
            idxs = np.arange(freqs.shape[0])
            for x, (i, j) in zip(ex_rates, combinations(idxs, 2)):
                ex_rates_m[i, j] = x
                ex_rates_m[j, i] = x
            site_Q = np.dot(ex_rates_m, D)
            site_Q[idxs, idxs] = -site_Q.sum(1).A1
            
            msg = 'Rows in the site rate matrix do not add up to 0: {}'.format(site_Q)
            check_error(np.allclose(site_Q.sum(1), 0), msg=msg)
            
            msg = 'The neutral rates matrix is not time reversible'
            eq_Q = D.dot(site_Q)
            check_error(np.allclose(eq_Q, eq_Q.T), msg=msg)
            scaling_factor = - 1 / np.diag(eq_Q).sum()
            site_Q = scaling_factor * site_Q
            
            neutral_site_Qs.append(site_Q)
            
            # Create symmetrix matrix to decompose and estimate mixing rates
            r = np.sqrt(freqs)
            m = get_sparse_diag_matrix(r).dot(csr_matrix(site_Q)).dot(get_sparse_diag_matrix(1/r))
            m = (m + m.T) / 2
            lambdas = eigsh(m, 2, v0=freqs, which='SM', tol=1e-5)[0]
            if -lambdas[0] < neutral_mixing_rate:
                neutral_mixing_rate = -lambdas[0]
            
        
        if len(neutral_site_Qs) == 1 and self.space.seq_length > 1:
            neutral_site_Qs = neutral_site_Qs * self.space.seq_length
        
        self.neutral_mixing_rate = neutral_mixing_rate
        print(neutral_mixing_rate)
        return(neutral_site_Qs)
    
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
