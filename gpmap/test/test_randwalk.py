#!/usr/bin/env python
import unittest
import sys

import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from itertools import product
from tempfile import NamedTemporaryFile
from scipy.sparse._matrix_io import load_npz

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.space import CodonSpace
from gpmap.src.randwalk import WMWSWalk
from gpmap.src.utils import get_sparse_diag_matrix


class RandomWalkTests(unittest.TestCase):
    def test_set_Ns(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        freqs = mc.calc_stationary_frequencies(Ns=0)
        assert(np.unique(freqs).shape[0] == 1)

        # Specify Ns directly
        mc.set_Ns(Ns=1)
        assert(mc.Ns == 1)
        
        try:
            mc.set_Ns(Ns=-1)
            self.fail()
        except ValueError:
            pass
        
        # Specify through the mean function at stationarity
        try: 
            mc.set_Ns(mean_function=0)
            self.fail()
        except ValueError:
            pass
        
        try: 
            mc.set_Ns(mean_function=3)
            self.fail()
        except ValueError:
            pass
        
        mc.set_Ns(mean_function=1.22743)
        assert(np.isclose(mc.Ns, 1))
        
        # Specify through the percentile at stationarity
        try: 
            mc.set_Ns(mean_function_perc=-1)
            self.fail()
        except ValueError:
            pass
        
        try: 
            mc.set_Ns(mean_function_perc=102)
            self.fail()
        except ValueError:
            pass
        
        mc.set_Ns(mean_function_perc=80)
        assert(np.isclose(mc.Ns, 0.59174))
    
    def test_calc_jump_matrix(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        assert(np.allclose(mc.jump_matrix.sum(1), 1))
    
    def test_run_forward(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        times, path = mc.run_forward(time=1)
        assert(np.sum(times) == 1)
        assert(len(path) == len(times))
        
    def test_run_forward_tree(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        times, path = mc.run_forward(time=1)
        assert(np.sum(times) == 1)
        assert(len(path) == len(times))
        
    def calc_neutral_stat_freqs(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        
        sites_stat_freqs = [np.array([0.4, 0.6]), np.array([0.3, 0.7])]
        freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(freqs, [0.12, 0.28, 0.18, 0.42]))
        
        sites_stat_freqs = np.array([[0.4, 0.6], [0.3, 0.7]])
        freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(freqs, [0.12, 0.28, 0.18, 0.42]))
        
    def test_calc_sites_mut_rates(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_Ns(1)
        
        # Ensure uniform neutral dynamics by default
        sites_neutral_Q = mc.calc_sites_GTR_mut_matrices()
        assert(len(sites_neutral_Q) == 3)
        
        for Q in sites_neutral_Q:
            for i, j in product(np.arange(Q.shape[0]), repeat=2):
                if i == j:
                    assert(Q[i, j] == -Q.shape[0]+1)
                else:
                    assert(Q[i, j] == 1)
            
        # Test with a single stationary frequency vector
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])]
        D = get_sparse_diag_matrix(sites_stat_freqs[0])
        sites_neutral_Q = mc.calc_sites_GTR_mut_matrices(sites_stat_freqs=sites_stat_freqs)
        for Q in sites_neutral_Q:
            eq_Q = D.dot(Q)
            
            # Test that rows sum to 0
            assert(np.allclose(Q.sum(1), 0))
            
            # Test time-reversibility
            assert(np.allclose((eq_Q-eq_Q.T).todense().A1, 0))
            
            # Test time units in expected number of mutations per site
            assert(np.allclose(eq_Q.diagonal().sum(), -3))
        
        # Force sites to have the same leaving rate: 1
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])]
        D = get_sparse_diag_matrix(sites_stat_freqs[0])
        sites_neutral_Q = mc.calc_sites_GTR_mut_matrices(sites_stat_freqs=sites_stat_freqs,
                                                         force_constant_leaving_rate=True)
        for Q in sites_neutral_Q:
            eq_Q = D.dot(Q)
            
            # Test time units in expected number of mutations per site
            assert(np.allclose(eq_Q.diagonal().sum(), -1))
        
        # Test with variable exchangeability rates
        exchange_rates = [np.array([1, 2, 1, 2, 1, 2])]
        sites_neutral_Q = mc.calc_sites_GTR_mut_matrices(exchange_rates=exchange_rates)
        for Q in sites_neutral_Q:
            
            # Ensure Q is symmetric
            assert(np.allclose((Q - Q.T).todense(), 0))
            
            # Ensure normalized leaving rate at stationarity
            assert(np.allclose(Q.diagonal().mean(), -3))
            
            # Ensure heterogeneous leaving rates from alleles
            assert(np.unique(Q.diagonal()).shape[0] > 1)
            
        # Test with site-specific mutation rates
        site_rates = np.array([1, 1, 2])
        sites_neutral_Q = mc.calc_sites_GTR_mut_matrices(site_mut_rates=site_rates)
        for Q, mu in zip(sites_neutral_Q, site_rates):
            
            # Ensure Q is symmetric
            assert(np.allclose((Q - Q.T).todense(), 0))
            
            # Ensure normalized leaving rate at stationarity
            assert(np.allclose(Q.diagonal().mean(), -3 * mu))
        
    def test_calc_neutral_rate_matrix(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        # Uniform mutation rates
        neutral_rate_matrix = mc.calc_neutral_rate_matrix()
        assert(np.allclose(neutral_rate_matrix.diagonal(), -9/64))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
        
        neutral_rate_matrix = mc.calc_neutral_rate_matrix(force_constant_leaving_rate=True)
        assert(np.allclose(neutral_rate_matrix.diagonal(), -1/64))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))

        # Variable rates
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])]
        neutral_rate_matrix = mc.calc_neutral_rate_matrix(sites_stat_freqs=sites_stat_freqs)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -9))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
        
        # Variable rates across sites
        sites_mu = np.array([1, 1, 2])
        neutral_rate_matrix = mc.calc_neutral_rate_matrix(site_mut_rates=sites_mu)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -9))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
        assert(np.unique(neutral_rate_matrix.data).shape[0] > 2)
    
    def test_calc_model_neutral_rate_matrix(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        # Ensure uniform stationary frequencies for K80 model
        exchange_rates = {'a': 1, 'b': 2}        
        mc.calc_model_neutral_rate_matrix(model='K80', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(np.allclose(np.unique(mc.neutral_rate_matrix.data),
                           [-0.140625, 0.01171875, 0.0234375 ]))
        assert(np.allclose(mc.neutral_rate_matrix.diagonal().sum(), -9))
        
        # F81 model
        stat_freqs = {'A': 0.4, 'C': 0.2, 'G': 0.1, 'T': 0.3}        
        mc.calc_model_neutral_rate_matrix(model='F81', stat_freqs=stat_freqs)
        assert(np.unique(mc.neutral_rate_matrix.data).shape[0] > 3)
        assert(not np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(np.allclose(mc.neutral_rate_matrix.diagonal().sum(), -9))
        
        # SYM model
        exchange_rates = {'a': 1, 'b': 2, 'c': 1, 'd': 1, 'e': 3, 'f': 2.5}        
        mc.calc_model_neutral_rate_matrix(model='SYM', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(np.unique(mc.neutral_rate_matrix.data).shape[0] > 3)
        assert(np.allclose(mc.neutral_rate_matrix.diagonal().sum(), -9))
        
        # Taking in default parameters if not specified
        exchange_rates = {'a': 1, 'b': 2}
        mc.calc_model_neutral_rate_matrix(model='HKY85', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(np.unique(mc.neutral_rate_matrix.data).shape[0] == 3)
        assert(np.allclose(mc.neutral_rate_matrix.diagonal().sum(), -9))
        
        # Check variable rates across sites
        site_mut_rates = np.array([1, 1, 2])
        mc.calc_model_neutral_rate_matrix(model='K80', site_mut_rates=site_mut_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(np.allclose(mc.neutral_rate_matrix.diagonal().sum(), -9))
        assert(np.unique(mc.neutral_rate_matrix.data).shape[0] > 2)
        
    def test_stationary_frequencies(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        codon_idxs = mc.space.get_state_idxs(codons)
        stop_codons = ['TGA', 'TAG', 'TAA']
        stop_codon_idxs = mc.space.get_state_idxs(stop_codons)
        
        # Check serine codons have high frequencies
        stat_freqs = mc.calc_stationary_frequencies(Ns=1)
        codon_freqs1 = stat_freqs[codon_idxs]
        assert(np.all(codon_freqs1 > 0.03))
        
        # Check stop codons have low frequencies
        assert(np.all(stat_freqs[stop_codon_idxs] < 0.01))
        
        # Check with biased neutral dynamics
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])] * 3
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(neutral_freqs.sum(), 1))
        
        freqs2 = mc.calc_stationary_frequencies(Ns=1,
                                                neutral_stat_freqs=neutral_freqs)
        assert(np.allclose(freqs2.sum(), 1))
        
        # Ensure frequencies have changed
        assert(not np.allclose(freqs2, stat_freqs))
        
        # Check with biases that should increase the frequency of high fitness
        # genotypes
        sites_stat_freqs = [np.array([0.4, 0.05, 0.05, 0.5]),
                            np.array([0.05, 0.5, 0.4, 0.05]),
                            np.array([0.4, 0.1, 0.1, 0.4])]
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        freqs2 = mc.calc_stationary_frequencies(Ns=1,
                                                neutral_stat_freqs=neutral_freqs)
        assert(freqs2[codon_idxs].sum() > codon_freqs1.sum())
    
    def test_stationary_function(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        
        # Ensure failure when stationary frequencies are not calculated
        try:
            mc.calc_stationary_mean_function()
            self.fail()
        except ValueError:
            pass
        
        # Ensure calculation with uniform frequencies
        mc.stationary_freqs = np.ones(mc.space.n_states) / mc.space.n_states
        mean_function = mc.calc_stationary_mean_function()
        assert(np.allclose(mean_function, mc.space.y.mean()))
        
        # Calculation with non uniform frequencies
        Ns = mc.set_Ns(mean_function=1.5)
        mc.calc_stationary_frequencies(Ns=Ns)
        mean_function = mc.calc_stationary_mean_function()
        assert(np.allclose(mean_function, 1.5))
        
        # See changes with modified neutral rates
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])] * 3
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        mc.calc_stationary_frequencies(Ns=Ns, neutral_stat_freqs=neutral_freqs)
        mean_function = mc.calc_stationary_mean_function()
        assert(mean_function < 1.5)
        
        # Check increased mean function by mutational biases in neutrality
        sites_stat_freqs = [np.array([0.4, 0.05, 0.05, 0.5]),
                            np.array([0.05, 0.5, 0.4, 0.05]),
                            np.array([0.4, 0.1, 0.1, 0.4])]
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        mc.stationary_freqs = neutral_freqs
        mean_function = mc.calc_stationary_mean_function()
        assert(mean_function > mc.space.y.mean())
        
        # Check increased mean function by mutational biases with selection
        mc.calc_stationary_frequencies(Ns=1)
        f1 = mc.calc_stationary_mean_function()
        
        mc.calc_stationary_frequencies(Ns=1, neutral_stat_freqs=neutral_freqs)
        f2 = mc.calc_stationary_mean_function()
        assert(f2 > f1)
    
    def test_calc_visualization(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        mc.calc_visualization(Ns=1, n_components=20)
        nd1 = mc.nodes_df
        assert(np.allclose(mc.decay_rates_df['relaxation_time'][0], 0.3914628))

        # Ensure extreme genotypes in axis 1 have high function       
        df = mc.nodes_df
        assert(df.iloc[np.argmax(df['1']), :]['function'] > 1.5)
        assert(df.iloc[np.argmin(df['1']), :]['function'] > 1.5)
        
        # Calculate visualization with biased mutation rates
        mc.calc_model_neutral_rate_matrix(model='HKY85',
                                          exchange_rates={'a': 1, 'b': 2},
                                          stat_freqs={'A': 0.2, 'T': 0.2,
                                                      'C': 0.3, 'G': 0.3})
        mc.calc_visualization(Ns=1, n_components=20)
        nd2 = mc.nodes_df
        assert(not np.allclose(nd2['1'], nd1['1']))
    
    def test_write_visualization(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.calc_visualization(Ns=1, n_components=20)
        
        with NamedTemporaryFile() as fhand:
            prefix = fhand.name
            mc.write_tables(prefix, write_edges=True, nodes_format='csv')
            
            nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
            assert(np.allclose(nodes_df.values, mc.nodes_df.values))
        
    def test_calc_visualization_bin_help(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
    
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
    
    def test_calc_visualization_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'serine.csv')
        
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath,
                   '-p', '90', '-e', '-nf', 'csv']
            check_call(cmd)
            
            df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert(df.iloc[np.argmax(df['1']), :]['function'] > 1.5)
            assert(df.iloc[np.argmin(df['1']), :]['function'] > 1.5)
            
            edges = load_npz('{}.edges.npz'.format(out_fpath))
            assert(np.all(edges.shape == (64, 64)))
    
    def test_calc_visualization_bin_guess_config(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'test.csv')
        
        with NamedTemporaryFile() as fhand:
            cmd = [sys.executable, bin_fpath, fpath, '-o', fhand.name,
                   '-m', '0.5', '-e']
            check_call(cmd)
    
    def test_calc_visualization_codon_restricted(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'test.csv')
        data = pd.read_csv(fpath, index_col=0)
        cmd = [sys.executable, bin_fpath, fpath, '-m', '0.65', '-e', '-nf', 'csv']
        
        # run with standard genetic code
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            check_call(cmd + ['-o', out_fpath])
            edges1 = load_npz('{}.edges.npz'.format(out_fpath))
        
        # run with bacterial genetic code 11
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            check_call(cmd + ['-o', out_fpath, '-c', '11']) 
            df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert(df.shape[0] == data.shape[0])

            # Ensure we have less edges when using codon restricted transitions        
            edges2 = load_npz('{}.edges.npz'.format(out_fpath))
            assert(edges1.sum() > edges2.sum())

    def test_calc_visualization_codon_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'serine.protein.csv')
        cmd = [sys.executable, bin_fpath, fpath, '-Ns', '1', 
               '-e', '-C', '-nf', 'csv']
        
        # standard genetic code
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            check_call(cmd + ['-o', out_fpath, '-c', 'Standard'])
            
            nodes = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert('protein' in nodes.columns)
            assert(nodes.shape[0] == 64)
        
        # custom genetic code
        with NamedTemporaryFile() as fhand:
            out_fpath = fhand.name
            codon_fpath = join(TEST_DATA_DIR, 'code_6037.csv')
            check_call(cmd + ['-o', out_fpath, '-c', codon_fpath])
            nodes = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert('protein' in nodes.columns)
            assert(nodes.shape[0] == 64)
        
        
if __name__ == '__main__':
    sys.argv = ['', 'RandomWalkTests.test_calc_visualization']
    unittest.main()
