#!/usr/bin/env python
import unittest
import sys

import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile
from scipy.sparse._matrix_io import load_npz
from scipy.sparse.csr import csr_matrix

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.space import CodonSpace, SequenceSpace
from gpmap.src.randwalk import WMWalk, ReactivePaths


class RandomWalkTests(unittest.TestCase):
    def test_set_Ns(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))

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
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        assert(np.allclose(mc.jump_matrix.sum(1), 1))
    
    def test_run_forward(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        times, path = mc.run_forward(time=1)
        assert(np.sum(times) == 1)
        assert(len(path) == len(times))
        
    def test_run_forward_tree(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        times, path = mc.run_forward(time=1)
        assert(np.sum(times) == 1)
        assert(len(path) == len(times))
        
    def calc_neutral_stat_freqs(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        
        sites_stat_freqs = [np.array([0.4, 0.6]), np.array([0.3, 0.7])]
        freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(freqs, [0.12, 0.28, 0.18, 0.42]))
        
        sites_stat_freqs = np.array([[0.4, 0.6], [0.3, 0.7]])
        freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(freqs, [0.12, 0.28, 0.18, 0.42]))
    
    def calc_neutral_exchange_rates(self):
        rw = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
    
        # Calculate default exchange rates matrix
        m = rw.calc_exchange_rate_matrix(exchange_rates=None)
        i, j = rw.space.get_neighbor_pairs()
        expected = csr_matrix((np.ones(i.shape[0]), (i, j)), shape=rw.shape) 
        assert(np.allclose(m.todense(),  expected.todense()))
        
        # Simpler space with different exchange rates
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = np.ones(X.shape[0])
        rw = WMWalk(SequenceSpace(X=X, y=y))
        m = rw.calc_exchange_rate_matrix([[1], [2]]).todense()
        expected = np.array([[0, 2, 1, 0], [2, 0, 0, 1], 
                             [1, 0, 0, 2], [0, 1, 2, 0]])
        assert(np.allclose(m, expected))
        
        # 3 alleles 1 site
        X = np.array(['A', 'B', 'C'])
        y = np.ones(X.shape[0])
        rw = WMWalk(SequenceSpace(X=X, y=y))
        m = rw.calc_exchange_rate_matrix([[1, 2, 3]]).todense()
        expected = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        assert(np.allclose(m, expected))
        
        # 3 alleles 2 sites
        X = np.array(['AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC'])
        y = np.ones(X.shape[0])
        rw = WMWalk(SequenceSpace(X=X, y=y))
        m = rw.calc_exchange_rate_matrix([[1, 2, 3]] * 2).todense()
        assert(np.allclose(m[:3, :][:, :3], expected))
        assert(np.allclose(m[3:6, :][:, 3:6], expected))
        assert(np.allclose(m[6:, :][:, 6:], expected))
        assert(np.allclose(m[:3, :][:, 3:6], 1 * np.eye(3)))
        assert(np.allclose(m[:3, :][:, 6:], 2 * np.eye(3)))
        assert(np.allclose(m[3:6, :][:, 6:], 3 * np.eye(3)))
    
    def test_calc_neutral_rate_matrix(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))

        # Uniform mutation rates
        neutral_rate_matrix = mc.calc_neutral_rate_matrix()
        assert(np.allclose(neutral_rate_matrix.diagonal(), -9/64))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
        
        # Variable stationary frequencies
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])] * 3
        neutral_rate_matrix = mc.calc_neutral_rate_matrix(sites_stat_freqs=sites_stat_freqs)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -9))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
        
        # Variable exchange rates
        sites_exchange_rates = [np.array([1, 2, 1, 1, 2, 1])] * 3
        neutral_rate_matrix = mc.calc_neutral_rate_matrix(sites_exchange_rates=sites_exchange_rates)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -12))
        assert(np.allclose(neutral_rate_matrix.sum(1), 0))
    
    def test_calc_neutral_model(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))

        # Ensure uniform stationary frequencies for K80 model
        exchange_rates = {'a': 1, 'b': 2}        
        mc.calc_neutral_model(model='K80', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        
        neutral_rate_matrix = mc.calc_gtr_rate_matrix(mc.neutral_exchange_rates,
                                                      mc.neutral_stat_freqs)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -12))
        assert(np.allclose(np.unique(neutral_rate_matrix.data),
                           [-12./64, 1./64, 2./64]))
        
        # F81 model
        stat_freqs = {'A': 0.4, 'C': 0.2, 'G': 0.1, 'T': 0.3}        
        mc.calc_neutral_model(model='F81', stat_freqs=stat_freqs)
        assert(np.allclose(mc.neutral_stat_freqs.sum(), 1))
        assert(not np.allclose(mc.neutral_stat_freqs, 1. / 64))
        assert(mc.neutral_stat_freqs[0] == 0.4**3)
        
        neutral_rate_matrix = mc.calc_gtr_rate_matrix(mc.neutral_exchange_rates,
                                                      mc.neutral_stat_freqs)
        assert(np.unique(neutral_rate_matrix.data).shape[0] > 3)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -9))
        
        # SYM model
        exchange_rates = {'a': 1, 'b': 2, 'c': 1, 'd': 1, 'e': 3, 'f': 2.5}        
        mc.calc_neutral_model(model='SYM', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        
        neutral_rate_matrix = mc.calc_gtr_rate_matrix(mc.neutral_exchange_rates,
                                                      mc.neutral_stat_freqs)
        assert(np.unique(neutral_rate_matrix.data).shape[0] > 3)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -15.75))
        
        # Taking in default parameters if not specified: 
        # HKY85 with uniform freqs is like K80
        exchange_rates = {'a': 1, 'b': 2}
        mc.calc_neutral_model(model='HKY85', exchange_rates=exchange_rates)
        assert(np.allclose(mc.neutral_stat_freqs, 1. / 64))
        
        neutral_rate_matrix = mc.calc_gtr_rate_matrix(mc.neutral_exchange_rates,
                                                      mc.neutral_stat_freqs)
        assert(np.allclose(neutral_rate_matrix.diagonal().sum(), -12))
        assert(np.allclose(np.unique(neutral_rate_matrix.data),
                           [-12./64, 1./64, 2./64]))
        
    def test_stationary_frequencies(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        codon_idxs = mc.space.get_state_idxs(codons)
        stop_codons = ['TGA', 'TAG', 'TAA']
        stop_codon_idxs = mc.space.get_state_idxs(stop_codons)
        
        # Check serine codons have high frequencies
        stat_freqs = np.exp(mc.calc_log_stationary_frequencies(Ns=1))
        codon_freqs1 = stat_freqs[codon_idxs]
        assert(np.all(codon_freqs1 > 0.03))
        
        # Check stop codons have low frequencies
        assert(np.all(stat_freqs[stop_codon_idxs] < 0.01))
        
        # Check with biased neutral dynamics
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])] * 3
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        assert(np.allclose(neutral_freqs.sum(), 1))
        
        freqs2 = np.exp(mc.calc_log_stationary_frequencies(1, neutral_freqs))
        assert(np.allclose(freqs2.sum(), 1))
        
        # Ensure frequencies have changed
        assert(not np.allclose(freqs2, stat_freqs))
        
        # Check with biases that should increase the frequency of high fitness
        # genotypes
        sites_stat_freqs = [np.array([0.4, 0.05, 0.05, 0.5]),
                            np.array([0.05, 0.5, 0.4, 0.05]),
                            np.array([0.4, 0.1, 0.1, 0.4])]
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        freqs2 = np.exp(mc.calc_log_stationary_frequencies(1, neutral_freqs))
        assert(freqs2[codon_idxs].sum() > codon_freqs1.sum())
    
    def test_stationary_function(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        
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
        mc.set_Ns(mean_function=1.5)
        freqs = mc.calc_stationary_frequencies(Ns=mc.Ns)
        mean_function = mc.calc_stationary_mean_function(freqs=freqs)
        assert(np.allclose(mean_function, 1.5))
        
        # See changes with modified neutral rates
        sites_stat_freqs = [np.array([0.4, 0.2, 0.1, 0.3])] * 3
        neutral_freqs = mc.calc_neutral_stat_freqs(sites_stat_freqs)
        mc.calc_stationary_frequencies(Ns=mc.Ns, neutral_stat_freqs=neutral_freqs)
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
        freqs1 = mc.calc_stationary_frequencies(Ns=1)
        freqs2 = mc.calc_stationary_frequencies(Ns=1,
                                                neutral_stat_freqs=neutral_freqs)
        f1 = mc.calc_stationary_mean_function(freqs=freqs1)
        f2 = mc.calc_stationary_mean_function(freqs=freqs2)
        assert(f2 > f1)
    
    def test_calc_sandwich_rate_matrix(self):
        # Simple space with neutral uniform dynamics 
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = np.ones(X.shape[0])
        rw = WMWalk(SequenceSpace(X=X, y=y))
        rw.calc_sandwich_rate_matrix(Ns=1)
        expected = np.array([[-2, 1, 1, 0], [1, -2, 0, 1], 
                             [1, 0, -2, 1], [0, 1, 1, -2]])
        assert(np.allclose(rw.sandwich_rate_matrix.todense(), expected))
        
        rw.calc_rate_matrix(Ns=1)
        assert(np.allclose(rw.rate_matrix.todense(), expected))
        
        # Introduce differences in fitness
        y = np.array([0, 1, 0, 1])
        rw = WMWalk(SequenceSpace(X=X, y=y))
        rw.calc_sandwich_rate_matrix(Ns=1)
        rw.calc_rate_matrix(Ns=1)
        Q = rw.rate_matrix
        assert(np.allclose(Q.sum(1), 0))
        
        rate1 = 1 / (1 - np.exp(-1))
        rate2 = -1 / (1 - np.exp(1))
        assert(np.allclose(Q[0, 1], rate1))
        assert(np.allclose(Q[1, 0], rate2))
        assert(np.allclose(Q[0, 2], 1))
        assert(np.allclose(Q[2, 0], 1))
        
    def test_calc_visualization(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))

        mc.calc_visualization(Ns=1, n_components=20)
        nd1 = mc.nodes_df
        assert(np.allclose(mc.decay_rates_df['relaxation_time'][0], 0.3914628))

        # Ensure extreme genotypes in axis 1 have high function
        assert(nd1.iloc[np.argmax(nd1['1']), :]['function'] > 1.5)
        assert(nd1.iloc[np.argmin(nd1['1']), :]['function'] > 1.5)
        
        # Calculate visualization with biased mutation rates
        mc.calc_neutral_model(model='HKY85', 
                              exchange_rates={'a': 1, 'b': 2},
                              stat_freqs={'A': 0.2, 'T': 0.2,
                                          'C': 0.3, 'G': 0.3})
        mc.calc_visualization(Ns=1, n_components=20)
        nd2 = mc.nodes_df
        assert(not np.allclose(nd2['1'], nd1['1']))
    
    def test_write_visualization(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.calc_visualization(Ns=1, n_components=20)
        
        with NamedTemporaryFile() as fhand:
            prefix = fhand.name
            mc.write_tables(prefix, write_edges=True, nodes_format='csv')
            
            nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
            assert(np.allclose(nodes_df.iloc[:, :-1].values,
                               mc.nodes_df.iloc[:, :-1].values))
        
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


class ReactivePathsTests(unittest.TestCase):
    def test_calc_committors(self):
        Q = csr_matrix([[-2, 1, 1, 0, 0, 0],
                        [1, -2, 0, 1, 0, 0],
                        [1, 0, -2, 0, 1, 0],
                        [0, 1, 0, -2, 0, 1],
                        [0, 0, 1, 0, -2, 1],
                        [0, 0, 0, 1, 1, -2]])
        n = Q.shape[0]
        stat_freqs = np.ones(n) / n
        start, end = np.array([0]), np.array([5])
        paths = ReactivePaths(Q, stat_freqs, start, end)
        q = paths.calc_forward_p()
        assert(np.allclose(q[0], 0))
        assert(np.allclose(q[1], 1./3))
        assert(np.allclose(q[-2], 2./3))
        assert(np.allclose(q[-1], 1))
        
        p = paths.calc_backwards_p()
        assert(np.allclose(q,  1 - p))
        
    def test_calc_flows(self):
        Q = csr_matrix([[-2, 1, 1, 0, 0, 0],
                        [1, -2, 0, 1, 0, 0],
                        [1, 0, -2, 0, 1, 0],
                        [0, 1, 0, -2, 0, 1],
                        [0, 0, 1, 0, -2, 1],
                        [0, 0, 0, 1, 1, -2]])
        n = Q.shape[0]
        stat_freqs = np.ones(n) / n
        start, end = np.array([0]), np.array([5])
        paths = ReactivePaths(Q, stat_freqs, start, end)
        flow = paths.calc_flow()
        assert(np.allclose(flow[0], 1/18))
        
        rate = paths.calc_reactive_rate(flow)
        assert(np.allclose(rate, 2/18))
        
        
if __name__ == '__main__':
    sys.argv = ['', 'ReactivePathsTests']
    unittest.main()
