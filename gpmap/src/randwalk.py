#!/usr/bin/env python
import numpy as np
import pandas as pd
import networkx as nx

from itertools import combinations
from scipy.sparse import csr_matrix, coo_matrix, identity
from scipy.sparse.linalg import bicgstab, cg, eigsh
from scipy.optimize import minimize
from scipy.special import logsumexp, comb

from gpmap.src.settings import DNA_ALPHABET
from gpmap.src.utils import check_error, write_log
from gpmap.src.matrix import (get_sparse_diag_matrix, calc_cartesian_product,
                              calc_tensor_product)
from gpmap.src.graph import calc_bottleneck, calc_pathway, has_path


class RandomWalk(object):
    def __init__(self, space, log=None):
        self.space = space
        self.log = log
    
    @property
    def is_time_reversible(self):
        return(False)
    
    @property
    def shape(self):
        return(self.space.n_states, self.space.n_states)
    
    def calc_jump_matrix(self):
        self.leaving_rates = -self.rate_matrix.diagonal()
        self.jump_matrix = get_sparse_diag_matrix(1/self.leaving_rates).dot(self.rate_matrix)
        self.jump_matrix.setdiag(0)
    
    def run_forward(self, time, state_idx=None):
        if state_idx is None:
            p = self.stationary_freqs
            path = [np.random.choice(self.space.state_idxs, p=p)]
        else:
            path = [state_idx]
            
        times = [0]
        remaining_time = time
        while True:
            t = np.random.exponential(1/self.leaving_rates[path[-1]])
            if t > remaining_time:
                times[-1] += remaining_time
                break
            else:
                p = self.jump_matrix[path[-1], :].todense().A1.flatten()
                new_state_idx = np.random.choice(self.space.state_idxs, p=p)
                path.append(new_state_idx)
                times.append(t)
                remaining_time = remaining_time - t
        return(times, path)
    
    def get_reactive_paths(self, start_labels, end_labels):
        start = self.space.get_state_idxs(start_labels).values
        end = self.space.get_state_idxs(end_labels).values
        paths = ReactivePaths(self.rate_matrix, self.stationary_freqs,
                              start, end, time_reversible=self.is_time_reversible)
        return(paths)
    
    def report(self, msg):
        write_log(self.log, msg)


class TimeReversibleRandomWalk(RandomWalk):
    @property
    def is_time_reversible(self):
        return(True)
    
    def set_stationary_freqs(self, log_freqs):
        self.stationary_freqs = np.exp(log_freqs)
        self.diag_freq = get_sparse_diag_matrix(np.exp(0.5 * log_freqs)) 
        self.diag_freq_inv = get_sparse_diag_matrix(np.exp(-0.5 * log_freqs))
    
    def calc_eigendecomposition(self, n_components=10, tol=1e-14):
        self.n_components = min(n_components + 1, self.space.n_states - 1)
        
        # Transform matrix shifting eigenvalues close to 0 to avoid numerical problems
        upper_bound = np.abs(self.sandwich_rate_matrix).sum(1).max()
        sandwich_aux_matrix = identity(self.space.n_states) + 1 / upper_bound * self.sandwich_rate_matrix
        
        self.report('Calculating {} eigenvalue-eigenvector pairs'.format(self.n_components))
        v0 = self.diag_freq.dot(self.stationary_freqs)
        lambdas, q = eigsh(sandwich_aux_matrix, self.n_components,
                           v0=v0, which='LM', tol=tol)
        
        # Reverse order
        lambdas = lambdas[::-1]
        q = np.fliplr(q)
        
        # Undo eigenvalue shifting
        self.eigenvalues = upper_bound * (lambdas - 1)
        
        # Store right eigenvectors of the rate matrix
        self.right_eigenvectors = self.diag_freq_inv.dot(q)

    def calc_diffusion_axis(self):
        self.report('Scaling projection axis')
        scaling_factors = get_sparse_diag_matrix(1 / np.sqrt(-self.eigenvalues[1:]))
        self.nodes_df = pd.DataFrame(scaling_factors.dot(self.right_eigenvectors[:, 1:].T).T,
                                     index=self.space.state_labels,
                                     columns=np.arange(1, self.n_components).astype(str))
        for col in self.nodes_df.columns:
            if self.nodes_df[col].mean() < 0:
                self.nodes_df[col] = -self.nodes_df[col]
        
        self.nodes_df['function'] = self.space.y
        self.nodes_df['stationary_freq'] = self.stationary_freqs
        if hasattr(self.space, 'protein_seqs'):
            self.nodes_df['protein'] = self.space.protein_seqs
        
    def calc_relaxation_times(self):
        decay_rates = -self.eigenvalues[1:]
        relaxation_times = 1 / decay_rates 
        k = np.arange(1, decay_rates.shape[0] + 1)
        self.decay_rates_df = pd.DataFrame({'k': k, 'decay_rates': decay_rates,
                                            'relaxation_time': relaxation_times})
    
    def calc_visualization(self, Ns=None, mean_function=None, 
                           mean_function_perc=None, n_components=10,
                           neutral_exchange_rates=None, neutral_stat_freqs=None,
                           tol=1e-12):
        '''
        Calculates the state coordinates to use for visualization 
        of the provided discrete space under a given time-reversible
        random walk. The coordinates consist on the right eigenvectors 
        of the associate rate matrix `Q`, re-scaled by the corresponding
        quantity so that the embedding is in units of square root of
        time
        
        Parameters
        ----------
        Ns : float
            Scaled effective population size to use in the underlying
            evolutionary model
        
        mean_function : float
            Mean function at stationarity to derive the associated Ns
            
        mean_function_perc: float
            Percentile that the mean function at stationarity takes within
            the distribution of function values along sequence space e.g.
            if `mean_function_perc=98`, then the mean function at stationarity
            is set to be at the 98th percentile across all the function values
        
        n_components: int (10)
            Number of eigenvectors or diffusion axis to calculate
        
        neutral_stat_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies at neutrality to define the
            time reversible neutral dynamics
        
        neutral_exchange_rates: scipy.sparse.csr.csr_matrix of shape (n_states, n_states)
            Sparse matrix containing the neutral exchange rates for the 
            whole sequence space. If not provided, uniform mutational dynamics
            are assumed.
        '''
        
        # Set or find Ns value to use
        if Ns is None and mean_function is None and mean_function_perc is None:
            msg = 'One of [Ns,  mean_function, mean_function_perc]'
            msg += 'is required to calculate the rate matrix'
            raise ValueError(msg)
        
        self.set_Ns(Ns=Ns, mean_function=mean_function,
                    mean_function_perc=mean_function_perc,
                    neutral_stat_freqs=neutral_stat_freqs)
        
        # Calculate rate matrix and re-scaled visualization coordinates
        log_freqs = self.calc_log_stationary_frequencies(self.Ns, neutral_stat_freqs)
        self.set_stationary_freqs(log_freqs)
        self.calc_sandwich_rate_matrix(Ns=self.Ns,
                                       neutral_stat_freqs=neutral_stat_freqs,
                                       neutral_exchange_rates=neutral_exchange_rates,)
        self.calc_eigendecomposition(n_components, tol=tol)
        self.calc_diffusion_axis()
        self.calc_relaxation_times()
    
    def write_tables(self, prefix, write_edges=False,
                     nodes_format='parquet', edges_format='npz'):
        '''
        Write the output of the visualization in tables with a common prefix.
        The output can consist in 2 to 3 different tables, as one of them
        may not be always necessarily stored multiple times
        
            - nodes coordinates : contains the coordinates for each state and
            the associated function values and stationary frequencies.
            It is stored in CSV format with suffix "nodes.csv" or parquet
            with suffix "nodes.pq"
            - decay rates : contains the decay rates and relaxation times
            associated to each component or diffusion axis. It is stored
            in CSV format with suffix "decay_rates.csv"
            - edges : contains the adjacency relationship between states.
            It is not stored by default unless `write_edges=True`, as it will 
            remain unchanged for any visualization on the same SequenceSpace.
            Therefore, so it only needs to be stored once. It can be stored in 
            CSV format, or in the more efficent npz format for sparse matrices
            
        Parameters
        ----------
        
        prefix: str
            Prefix of the files to store the different tables
            
        write_edges: bool (False)
            Option to write also the information about the adjacency relationships
            between pairs for states for plotting the edges
        
        nodes_format: str {'parquet', 'csv'}
            Format to store the nodes information. parquet is more efficient but
            CSV can be used in smaller cases for plain text storage.
        
        edges_format: str {'npz', 'csv'}
            Format to store the edges information. npz is more efficient but CSV
            can be used in smaller cases for plain text storage.
        
        '''
        self.decay_rates_df.to_csv('{}.decay_rates.csv'.format(prefix), index=False)
        
        if nodes_format in ['parquet', 'pq']:
            self.nodes_df.to_parquet('{}.nodes.pq'.format(prefix))
        elif nodes_format == 'csv':
            self.nodes_df.to_csv('{}.nodes.csv'.format(prefix))
        else:
            msg = 'nodes_format can only take values ["parquet", "csv"]'
            raise ValueError(msg)
        
        if write_edges:
            fpath = '{}.edges.{}'.format(prefix, edges_format)
            self.space.write_edges(fpath)
    

class WMWalk(TimeReversibleRandomWalk):
    '''
    Class for Weak Mutation Weak Selection Random Walk on a SequenceSpace.
    It is a time-reversible continuous time Markov Chain where the transition
    rates depend on the differences in fitnesses between two states 
    scaled by the effective population size `Ns` . 
    

    Attributes
    ----------
    space : DiscreteSpace class
        Space on which the random walk takes place
    Ns : float 
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
        Calculates the stationary frequencies of the states under the random
        walk specified on the discrete space
    
    calc_rate_matrix():
        Calculates the rate matrix for the continuous time process given 
        the scaled effective population size (Ns) or average phenotype at
        stationarity.
    
    
    '''
    def ex_rates_vector_to_matrix(self, ex_rates, n_alleles):
        ex_rates_m = np.zeros((n_alleles, n_alleles))
        idxs = np.arange(n_alleles)
        for x, (i, j) in zip(ex_rates, combinations(idxs, 2)):
            ex_rates_m[i, j] = x
            ex_rates_m[j, i] = x
        return(ex_rates_m)
    
    def calc_exchange_rate_matrix(self, exchange_rates=None):
        if exchange_rates is None:
            exchange_rates = [np.ones(int(comb(alpha, 2)))
                              for alpha in self.space.n_alleles]
        matrices = [csr_matrix(self.ex_rates_vector_to_matrix(m, a))
                    for m, a in zip(exchange_rates, self.space.n_alleles)]
        ex_rates_m = calc_cartesian_product(matrices)
        return(ex_rates_m) 

    def calc_neutral_stat_freqs(self, sites_stat_freqs=None):
        '''
        Calculates the neutral stationary frequencies assuming site independence
        
        Parameters
        ----------
        sites_stat_freqs: list of array-like of shape (n_alleles,)
            Matrix containing the site stationary frequencies that are used to
            parameterize the neutral dynamics with mutational biases for each
            independent site. If `None`, uniform frequencies across alleles
            will be set
            
        Returns
        -------
        neutral_stat_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies resulting from the product of the
            site-level stationary frequencies at neutrality
        
        '''
        if sites_stat_freqs is None:
            sites_stat_freqs = [np.ones(a) / a for a in self.space.n_alleles]
            
        sites_stat_freqs = [np.array([freqs]).T for freqs in sites_stat_freqs]
        freqs = calc_tensor_product(sites_stat_freqs).flatten()
        return(freqs)
    
    def calc_gtr_rate_matrix(self, exchange_rates_matrix, stationary_freqs):
        D = get_sparse_diag_matrix(stationary_freqs)
        Q = exchange_rates_matrix.dot(D).tolil()
        Q.setdiag(-Q.sum(1))
        return(Q.tocsr())
    
    def calc_neutral_rate_matrix(self, sites_exchange_rates=None,
                                 sites_stat_freqs=None):
        self.neutral_exchange_rates = self.calc_exchange_rate_matrix(sites_exchange_rates)
        self.neutral_stat_freqs = self.calc_neutral_stat_freqs(sites_stat_freqs)
        return(self.calc_gtr_rate_matrix(self.neutral_exchange_rates,
                                         self.neutral_stat_freqs))
        
    def calc_neutral_mixing_rates(self, site_exchange_rates,
                                  neutral_site_freqs):
        '''
        Calculates the neutral mixing rates for a SequenceSpace
        In case no GTR mutation model is specified, then the neutral
        mixing rates is limited by the site with the least number of alleles.
        Otherwise, as we assume that mutations are site-independent, 
        the slowest neutral mixing rate is going to by limited by the slowest
        site, provided by the smallest of second eigenvalues in the site
        rate matrices
        
        Parameters
        ----------
        neutral_site_Qs : list of array-like of shape (n_alleles, n_alleles)
            List containing site-specific rate matrices to use for calculating
            the limiting mixing in the neutral case. If not provided, uniform
            mutation rates are assumed.
            
        neutral_site_freqs : list of array-like of shape (n_alleles,)
            List containing vectors with the stationary frequencies under
            neutrality for each site. They are used to calculate the eigenvalues
            of the time reversible site specific neutral chain. By default,
            they are assumed to be uniform across sites and alleles.
        
        site_weights : array-like of shape (seq_length,)
            Vector containing the relative weight associated to each site. This
            value is used to scale the individually normalized rates matrices
            to ensure this specific leaving rate. By default, all weights
            are equal
            
        Returns
        -------
        neutral_mixing_rate: float
            Neutral mixing rate as the smallest second largest eigenvalue across
            sites.
        
        TODO: Re-implement functionality             
        '''
        return()
    
    def calc_log_stationary_frequencies(self, Ns, neutral_stat_freqs=None):
        '''Calculates the state stationary frequencies using Ns stored in 
        the object and stores the corresponding diagonal matrices with the
        sqrt transformation and its inverse
        
        Parameters
        ----------
        Ns : real 
            Scaled effective population size for the evolutionary model
        
        neutral_stat_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies resulting from the product of the
            site-level stationary frequencies at neutrality
        
        Returns
        -------
        stationary_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies in the selective regime
        
        '''
        if neutral_stat_freqs is None and hasattr(self, 'neutral_stat_freqs'):
            neutral_stat_freqs = self.neutral_stat_freqs
        
        if Ns < 0:
            msg = 'Ns must be non-negative'
            raise ValueError(msg)
        elif Ns == 0:
            log_phi = np.ones(self.space.n_states)
        else:
            log_phi = Ns * self.space.y
        
        if neutral_stat_freqs is not None:
            log_phi += np.log(neutral_stat_freqs)
            
        log_stationary_freqs = log_phi - logsumexp(log_phi)
        return(log_stationary_freqs)
    
    def calc_stationary_frequencies(self, Ns, neutral_stat_freqs=None):
        return(np.exp(self.calc_log_stationary_frequencies(Ns, neutral_stat_freqs)))
    
    def calc_stationary_mean_function(self, freqs=None):
        if freqs is None:
            check_error(hasattr(self, 'stationary_freqs'),
                        'Calculate the stationary frequencies first')
            freqs = self.stationary_freqs
        return(np.sum(self.space.y * freqs))
    
    def calc_neutral_mean_function(self, neutral_stat_freqs=None):
        if neutral_stat_freqs is None:
            return(self.space.y.mean())
        return(self.calc_stationary_mean_function(neutral_stat_freqs))
    
    def mean_function_to_Ns(self, mean_function, neutral_stat_freqs=None,
                            max_attempts=10, maxiter=100, tol=1e-4):
        
        def calc_stationary_function_error(logNs):
            Ns = np.exp(logNs)
            log_freqs = self.calc_log_stationary_frequencies(Ns, neutral_stat_freqs=neutral_stat_freqs)
            x = self.calc_stationary_mean_function(np.exp(log_freqs))
            sq_error = (mean_function - x) ** 2
            return(sq_error)
        
        for _ in range(max_attempts):
            result = minimize(calc_stationary_function_error, x0=0, tol=tol,
                              options={'maxiter': maxiter})
            Ns = np.exp(result.x[0])
            log_freqs = self.calc_log_stationary_frequencies(Ns, neutral_stat_freqs=neutral_stat_freqs)
            inferred_mean_function = np.sum(self.space.y * np.exp(log_freqs))
            if calc_stationary_function_error(result.x[0]) < tol:
                break
        else:
            msg = 'Could not find the Ns that yields the desired mean function '
            msg += '= {:.2f}. Best guess is {:.2f}'
            
            raise ValueError(msg.format(mean_function, inferred_mean_function))
    
        return(Ns)
    
    def set_Ns(self, Ns=None, mean_function=None, mean_function_perc=None,
               neutral_stat_freqs=None, tol=1e-4, maxiter=100, max_attempts=10):
        
        if Ns is not None:
            check_error(Ns >= 0, msg='Ns must be non-negative')
            self.Ns = Ns
        
        else:
            if mean_function_perc is not None:
                msg = 'mean_function_perc must be between 0 and 100'
                check_error(mean_function_perc > 0 or mean_function_perc < 100, msg=msg)
                mean_function = np.percentile(self.space.y, mean_function_perc)
                
            elif mean_function is None:
                msg = 'Either stationary_function or percentile must be provided'
                raise ValueError(msg)
            
            if neutral_stat_freqs is None and hasattr(self, 'neutral_stat_freqs'):
                neutral_stat_freqs = self.neutral_stat_freqs
            
            min_mean_function = self.calc_neutral_mean_function(neutral_stat_freqs)
            max_mean_function = self.space.y.max()
            
            msg = 'mean_function must be between the function mean ({:.2f}) and the maximum function value (:.2f)'
            msg = msg.format(min_mean_function, max_mean_function)
            check_error(mean_function >= min_mean_function and mean_function < max_mean_function, msg=msg)
                
            msg = 'Optimizing Ns to reach a stationary state with mean(f)={}'
            self.report(msg.format(mean_function))
            
            if np.allclose(min_mean_function, mean_function, atol=tol):
                self.Ns = 0
            else:
                self.Ns = self.mean_function_to_Ns(mean_function,
                                                   neutral_stat_freqs=neutral_stat_freqs,
                                                   max_attempts=max_attempts,
                                                   maxiter=maxiter, tol=tol)
    
    def _calc_sandwich_rate_vector(self, i, j, Ns,
                                   neutral_stat_freqs=None,
                                   neutral_exchange_rates=None):
        # Initialize entries
        values = np.ones(i.shape[0])

        if Ns > 0:
            df = self.space.y[j] - self.space.y[i]
            idxs = np.isclose(df, 0) == False
            
            # Calculate selection driven part
            S = Ns * df[idxs]
            S_half = 0.5 * S
            values[idxs] = S / (np.exp(S_half) - np.exp(-S_half))

        # Adjust with neutral rates if provided
        if neutral_stat_freqs is not None:
            log_freqs = np.log(neutral_stat_freqs)
            values = values * np.exp(0.5 * log_freqs[i] + 0.5 * log_freqs[j] + np.log(self.space.n_states))
        
        if neutral_exchange_rates is not None:
            values = values * neutral_exchange_rates.data
            
        return(values)

    def calc_sandwich_rate_matrix(self, Ns, neutral_stat_freqs=None,
                                  neutral_exchange_rates=None):
        '''
        Calculates the sandwich rate matrix for the random walk in the
        discrete space D^{1/2} Q D^{-1/2}
        
        Parameters
        ----------
        Ns : real 
            Scaled effective population size for the evolutionary model

        neutral_stat_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies at neutrality to define the
            time reversible neutral dynamics
        
        neutral_exchange_rates: scipy.sparse.csr.csr_matrix of shape (n_states, n_states)
            Sparse matrix containing the neutral exchange rates for the 
            whole sequence space. If not provided, uniform mutational dynamics
            are assumed.
        '''
        self.report('Calculating D^(1/2) Q D^(-1/2) matrix with Ns={}'.format(Ns))
        
        if neutral_stat_freqs is None and hasattr(self, 'neutral_stat_freqs'):
            neutral_stat_freqs = self.neutral_stat_freqs
        if neutral_exchange_rates is None and hasattr(self, 'neutral_exchange_rates'):
            neutral_exchange_rates = self.neutral_exchange_rates
        
        i, j = self.space.get_neighbor_pairs()
        values = self._calc_sandwich_rate_vector(i, j, Ns, neutral_stat_freqs,
                                                 neutral_exchange_rates)
        m = csr_matrix((values, (i, j)), shape=self.shape).tolil()
        
        # Fill diagonal entries
        log_freqs = self.calc_log_stationary_frequencies(Ns, neutral_stat_freqs)
        self.set_stationary_freqs(log_freqs)
        sqrt_freqs = np.exp(0.5 * (log_freqs + np.log(self.space.n_states)))
        m.setdiag(-m.dot(sqrt_freqs) / sqrt_freqs)
        self.sandwich_rate_matrix = m.tocsr()
    
    def calc_rate_matrix(self, Ns, neutral_stat_freqs=None,
                         neutral_exchange_rates=None):
        '''
        Calculates the rate matrix for the random walk in the discrete space
        and stores it in the attribute `rate_matrix`
        
        Parameters
        ----------
        Ns : real 
            Scaled effective population size for the evolutionary model
        
        neutral_stat_freqs : array-like of shape (n_states,)
            Genotype stationary frequencies at neutrality to define the
            time reversible neutral dynamics
        
        neutral_exchange_rates: scipy.sparse.csr.csr_matrix of shape (n_states, n_states)
            Sparse matrix containing the neutral exchange rates for the 
            whole sequence space. If not provided, uniform mutational dynamics
            are assumed.
            
        '''
        self.report('Calculating rate matrix with Ns={}'.format(Ns))
        self.calc_sandwich_rate_matrix(Ns=Ns,
                                       neutral_stat_freqs=neutral_stat_freqs,
                                       neutral_exchange_rates=neutral_exchange_rates)
        self.rate_matrix = self.diag_freq_inv.dot(self.sandwich_rate_matrix).dot(self.diag_freq)
    
    def calc_neutral_model(self, model, stat_freqs={}, exchange_rates={}):
        '''
        Calculate the neutral rate matrix for classic nucleotide substitution
        rates parameterized as in 
        
        https://en.wikipedia.org/wiki/Substitution_model
        
        Parameters
        ----------
        model : str {'F81', 'K80', 'HKY85', 'K81', 'TN93', 'SYM', 'GTR'}
            Specific nucleotide substitution model to use for every site in 
            the nucleotide sequence
            
        stat_freqs : dict with keys {'A', 'C', 'G', 'T'}
            Dictionary containing the allele stationary frequencies to use
            in the models that allow them to be different
        
        exchange_rates : dict with keys {'a', 'b', 'c', 'd', 'e', 'f'}
            Parameter values to use for each of the models. Only some of them 
            need to be specified for each of the models
            
        Returns
        -------
        neutral_rate_matrix: scipy.sparse.csr.csr_matrix of shape (n_states, n_states)
            Sparse matrix containing the neutral transition rates for the 
            whole sequence space
        
        '''
        
        msg = 'Ensure the space is a "dna" space for using nucleotide '
        msg += 'substitution models for the neutral dynamics'
        check_error(self.space.alphabet_type == 'dna', msg)
        
        if not stat_freqs:
            stat_freqs = {a: 0.25 for a in DNA_ALPHABET}
        
        ex_rates_def = {v : 1 for v in 'abcdef'}
        ex_rates_def.update(exchange_rates)
        exchange_rates = ex_rates_def
        
        if model == 'F81':
            exchange_rates = np.ones(6)
            stat_freqs = [stat_freqs[a] for a in DNA_ALPHABET]
        elif model == 'K80':
            exchange_rates = [exchange_rates['a'], exchange_rates['b'],
                              exchange_rates['a'], exchange_rates['a'],
                              exchange_rates['b'], exchange_rates['a']]
            stat_freqs = np.full(4, 1/4)
        elif model == 'HKY85':
            exchange_rates = [exchange_rates['a'], exchange_rates['b'],
                              exchange_rates['a'], exchange_rates['a'],
                              exchange_rates['b'], exchange_rates['a']]
            stat_freqs = [stat_freqs[a] for a in DNA_ALPHABET]
        elif model == 'K81':
            exchange_rates = [exchange_rates['a'], exchange_rates['b'],
                              exchange_rates['c'], exchange_rates['c'],
                              exchange_rates['b'], exchange_rates['a']]
            stat_freqs = np.full(4, 1/4)
        elif model == 'TN93':
            exchange_rates = [exchange_rates['a'], exchange_rates['b'],
                              exchange_rates['a'], exchange_rates['a'],
                              exchange_rates['e'], exchange_rates['a']]
            stat_freqs = [stat_freqs[a] for a in DNA_ALPHABET]
        elif model == 'SYM':
            exchange_rates = [exchange_rates[x]
                              for x in ['a', 'b', 'c', 'd', 'e', 'f']]
            stat_freqs = np.full(4, 1/4)
        elif model == 'GTR': 
            exchange_rates = [exchange_rates[x]
                              for x in ['a', 'b', 'c', 'd', 'e', 'f']]
            stat_freqs = [stat_freqs[a] for a in DNA_ALPHABET]
        else:
            msg = 'Model not supported: {}. Try one of the '.format(model)
            msg += 'following: [F81, K80, HKY85, K81, TN93, GTR]'
            raise ValueError(msg)

        msg = 'Ensure that the provided stationary frequencies add up to 1'
        check_error(np.sum(stat_freqs) == 1, msg=msg)

        sites_stat_freqs = [np.array(stat_freqs)] * self.space.seq_length
        self.neutral_stat_freqs = self.calc_neutral_stat_freqs(sites_stat_freqs)
        
        exchange_rates = [np.array(exchange_rates)] * self.space.seq_length
        self.neutral_exchange_rates = self.calc_exchange_rate_matrix(exchange_rates)


class ReactivePaths(object):
    '''
    Class for calculation of transition path theory objects and quantities
    
    Parameters
    ----------
    rate_matrix: csr_matrix
        csr_matrix containing the instantaneous rates between pairs of states
        
    start: array-like
        array-like object of indexes at which reactive paths start
        
    end: array-like
        array-like object of indexes at which reactive paths end
    '''
    
    def __init__(self, rate_matrix, stat_freqs, start, end, time_reversible=False):
        Q = rate_matrix.tocoo()
        Q.setdiag(0)
        Q.eliminate_zeros()
        self.i = Q.row
        self.j = Q.col
        self.Q_ij = Q.data
        self.time_reversible = time_reversible
        
        self.rate_matrix = rate_matrix
        self.n = rate_matrix.shape[0]
        self.stat_freqs = stat_freqs
        self.set_start_end(start, end)
        self.q_forward = self.calc_forward_p_time_reversible() if time_reversible else self.calc_forward_p()
        self.q_backward = 1 - self.q_forward if time_reversible else self.calc_backwards_p()
        
        self.start_id = self.n + 1
        self.end_id = self.n + 2
    
    def get_backwards_rate_matrix(self):
        D = get_sparse_diag_matrix(self.stat_freqs)
        D_inv = get_sparse_diag_matrix(1/self.stat_freqs)
        Q = D_inv @ self.rate_matrix.T @ D
        return(Q)
    
    def calc_jump_transition_matrix(self, a, b):
        Q = self.rate_matrix
        Q.setdiag(0)
        P = Q / Q.sum(1).A1
        P[self.end] = 0
        D_q = get_sparse_diag_matrix(self.q_forward[self.not_start])
        P[self.not_start] = P[self.not_start] 
                
        not_start_at_a = np.array([x not in a for x in i])
        start_at_b = np.array([x in b for x in i])
        p_ij[start_at_b] = 0
        p_ij[not_start_at_a] = p_ij[not_start_at_a] * q[j][not_start_at_a] / q[i][not_start_at_a]
        
        # Add absorbing probabilities at b
        i = np.append(i, b)
        j = np.append(j, b)
        p_ij = np.append(p_ij, np.ones(b.shape[0]))
        
        # Make sparse jump matrix
        jump_matrix = csr_matrix((p_ij, (i, j)), shape=size)
        return(jump_matrix)
        
    def set_start_end(self, start, end):
        msg = 'The two sets of start and end states cannot overlap'
        check_error(np.intersect1d(start, end).shape[0] == 0, msg)
        
        self.start = np.array(start)
        self.not_start = np.full(self.n, True)
        self.not_start[self.start] = False
        self.start_n = self.start.shape[0]
        self.end = np.array(end)
        self.end_n = self.end.shape[0]
        self.other = self.get_other_idxs(np.append(start, end))
    
    def get_other_idxs(self, idxs_rm):
        idx = np.full(self.n, True)
        idx[idxs_rm] = False
        return(np.where(idx)[0])
    
    def calc_forward_p(self):
        Q = self.rate_matrix
        partial_rate_matrix = Q[self.other, :]
        U = partial_rate_matrix[:, self.other]
        v = -partial_rate_matrix[:, self.end].sum(1)
        q_partial, res = bicgstab(U, v, atol=1e-16)
        
        if res != 0:
            mse = np.mean((U @ q_partial - v) ** 2)
            print('Warning: BICGSTAB exitCode: {}. MSE={}'.format(res, mse))
            
        q = np.zeros(self.n)
        q[self.other] = q_partial
        q[self.end] = 1
        return(q)
    
    def calc_backwards_p(self):
        Q = self.get_backwards_rate_matrix()
        partial_rate_matrix = Q[self.other, :]
        U = partial_rate_matrix[:, self.other]
        v = -partial_rate_matrix[:, self.start].sum(1)
        q_partial, res = bicgstab(U, v, atol=1e-16)

        if res != 0:
            mse = np.mean((U @ q_partial - v) ** 2)
            print('Warning: BICGSTAB exitCode: {}. MSE={}'.format(res, mse))
            
        q = np.zeros(self.n)
        q[self.other] = q_partial
        q[self.start] = 1
        return(q)
    
    def calc_forward_p_time_reversible(self):
        D = get_sparse_diag_matrix(self.stat_freqs)
        DQ = D @ self.rate_matrix
        partial_rate_matrix = DQ[self.other, :]
        U = partial_rate_matrix[:, self.other]
        v = -partial_rate_matrix[:, self.end].sum(1)
        q_partial, res = cg(U, v, atol=1e-16)
        
        if res != 0:
            mse = np.mean((U @ q_partial - v) ** 2)
            print('Warning: CG exitCode: {}. MSE={}'.format(res, mse))
            
        q = np.zeros(self.n)
        q[self.other] = q_partial
        q[self.end] = 1
        return(q)

    def calc_reactive_p(self):
        return(self.q_forward * self.q_backwards * self.stat_freqs)
    
    def calc_reactive_log_enrichment(self):
        return(np.log10(self.q_backward * self.q_forward))
    
    def calc_flow(self):
        flow = self.stat_freqs[self.i]
        flow *= self.q_backward[self.i] * self.Q_ij * self.q_forward[self.j]
        return(flow)
    
    def flow_to_eff_flow(self, flow):
        m = coo_matrix((flow, (self.i, self.j)), shape=(self.n, self.n))
        eff_flow = (m - m.T).data
        eff_flow[eff_flow < 0] = 0
        return(eff_flow)
    
    def calc_reactive_rate(self, flow):
        if self.start_n < self.end_n:
            sel_idxs = np.isin(self.i, self.start)
        else:
            sel_idxs = np.isin(self.j, self.end)
        return(flow[sel_idxs].sum())
    
    def get_ij_eff_flow(self, eff_flow=None, ij=None, is_sorted=False):
        ij = ij if ij is not None else np.vstack([self.i, self.j]).T
        if eff_flow is None:
            eff_flow = self.flow_to_eff_flow(self.calc_flow())
        
        msg = 'Ensure the shape of `eff_flow` is the same as that of `ij`'
        n_edges = eff_flow.shape[0]
        check_error(ij.shape[0] == n_edges, msg)
        positive = eff_flow > 0
        eff_flow, ij = eff_flow[positive], ij[positive, :]
        return(ij, eff_flow)
    
    def calc_graph(self, ij, eff_flow=None):
        graph = nx.DiGraph()
        if eff_flow is None:
            graph.add_edges_from(ij)
        else:
            graph.add_weighted_edges_from([(i, j, f) for (i, j), f in zip(ij, eff_flow)])
        return(graph)
    
    def calc_bottleneck(self, eff_flow=None, ij=None, is_sorted=False):
        ij, eff_flow = self.get_ij_eff_flow(eff_flow, ij, is_sorted)
        graph = self.calc_graph(ij, eff_flow)
        bottleneck = calc_bottleneck(graph, self.start, self.end)
        return(bottleneck)
    
    def calc_pathway(self, eff_flow=None, ij=None, is_sorted=False):
        ij, eff_flow = self.get_ij_eff_flow(eff_flow, ij, is_sorted)
        graph = self.calc_graph(ij, eff_flow)
        path, eff_flow = calc_pathway(graph, self.start, self.end)
        return(path, eff_flow)
    
    def calc_pathways(self, n=20, eff_flow=None, ij=None, is_sorted=False):
        ij, eff_flow = self.get_ij_eff_flow(eff_flow, ij, is_sorted)
        graph = self.calc_graph(ij, eff_flow)
        
        for _ in range(n):
            path, max_min_flow = calc_pathway(graph, self.start, self.end)
            yield(path, max_min_flow)
            
            for edge in zip(path, path[1:]):
                graph.edges[edge]['weight'] -= max_min_flow
                if graph.edges[edge]['weight'] == 0:
                    graph.remove_edge(*edge)
            
            if not has_path(graph, self.start, self.end):
                break
