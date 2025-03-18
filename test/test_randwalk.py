#!/usr/bin/env python
import unittest
import sys

import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile
from scipy.sparse import load_npz, csr_matrix

from gpmap.datasets import DataSet
from gpmap.settings import BIN_DIR
from gpmap.space import CodonSpace, SequenceSpace, DiscreteSpace
from gpmap.randwalk import WMWalk, ReactivePaths, PopulationSizeModel
from gpmap.seq import get_seqs_from_alleles, translate_seqs


class RandomWalkTests(unittest.TestCase):
    def test_estimate_Ns(self):
        y = np.array([1, 0, 0.5, 0.75])
        model = PopulationSizeModel(y)
        assert(np.isclose(model.calc_p(Ns=0).sum(), 1))
        assert(np.isclose(model.calc_p(Ns=1).sum(), 1))
        m = model.predict(Ns=1)
        
        # Evaluate loss and gradient at the right value
        loss, grad = model.loss(logNs=0, exp_m=m, return_grad=True)
        assert(np.isclose(loss, 0))
        assert(np.isclose(grad, 0))
        
        _, grad = model.loss(logNs=-1, exp_m=m, return_grad=True)
        assert(grad > 0)
        
        _, grad = model.loss(logNs=1, exp_m=m, return_grad=True)
        assert(grad < 0)
        
        # Fit model
        Ns = model.fit(m=m)
        pred = model.predict(Ns=Ns)
        assert(np.isclose(Ns, 1, atol=1e-4))
        assert(np.isclose(pred, m, atol=1e-4))
        
        # Run in a larger set of y values
        y = np.random.normal(size=10000)
        model = PopulationSizeModel(y)
        m = model.predict(Ns=1)
        Ns = model.fit(m=m)
        pred = model.predict(Ns=Ns)
        assert(np.isclose(Ns, 1))
        assert(np.isclose(pred, m))
        
        # With neutral biases
        y = np.random.normal(size=10000)
        p_neutral = np.exp(np.random.normal(scale=0.2, size=y.shape[0]))
        model = PopulationSizeModel(y, p_neutral=p_neutral)
        m = model.predict(Ns=1)
        Ns = model.fit(m=m)
        pred = model.predict(Ns=Ns)
        assert(np.isclose(Ns, 1, atol=1e-4))
        assert(np.isclose(pred, m, atol=1e-4))
        
        # Test in the serine landscape
        space = CodonSpace(['S'], add_variation=True, seed=0)
        y = space.y
        true_Ns = 2
        model = PopulationSizeModel(y)
        m = model.predict(Ns=true_Ns)
        Ns = model.fit(m=m)
        pred = model.predict(Ns=Ns)
        assert(np.isclose(Ns, true_Ns, atol=1e-4))
        assert(np.isclose(pred, m, atol=1e-4))
        
    def test_set_Ns(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        model = PopulationSizeModel(mc.space.y)
        true_Ns = 3
        true_mf = model.predict(true_Ns)

        # Direct setting
        mc.set_Ns(Ns=true_Ns)
        assert(mc.Ns == true_Ns)
        
        # Verify methods of mean function calculation
        freqs = mc.calc_stationary_frequencies(Ns=true_Ns)
        true_mf2 = mc.calc_stationary_mean_function(freqs=freqs)
        assert(true_mf2 == true_mf)
        
        # Set through mean function
        mc.set_Ns(mean_function=true_mf)
        freqs = mc.calc_stationary_frequencies(Ns=mc.Ns)
        mf = mc.calc_stationary_mean_function(freqs=freqs)
        assert(np.isclose(mf, true_mf, atol=1e-4))
        assert(np.isclose(mc.Ns, true_Ns, atol=1e-4))
        
        # Set through mean function percentile
        perc = np.mean(mc.space.y <= true_mf) * 100
        true_mf = np.percentile(mc.space.y, perc)
        mc.set_Ns(mean_function_perc=perc)
        mf = mc.calc_stationary_mean_function(mc.calc_stationary_frequencies(Ns=mc.Ns))
        assert(np.isclose(mf, true_mf, atol=1e-4))
        
        # Check errors
        try:
            mc.set_Ns(Ns=-1)
            self.fail()
        except ValueError:
            pass
        
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
        
    def test_set_Ns_range(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        for Ns in np.geomspace(0.01, 10, 20):
            freqs = mc.calc_stationary_frequencies(Ns)
            true_mf = mc.calc_stationary_mean_function(freqs=freqs)
            mc.set_Ns(mean_function=true_mf)
            mf = mc.calc_stationary_mean_function(freqs=mc.calc_stationary_frequencies(mc.Ns))
            assert(np.allclose(true_mf, mf, atol=1e-4))
    
    def test_calc_jump_matrix(self):
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_stationary_freqs(mc.calc_stationary_frequencies(Ns=1))
        mc.calc_rate_matrix(Ns=1)
        mc.calc_jump_matrix()
        assert(np.allclose(mc.jump_matrix.sum(1), 1))
    
    def test_calc_hitting_prob_through(self):
        # Ensure right results when the probability is clearly 0.5 in a small
        # case with only 2 possible paths with the same probability
        A = csr_matrix([[0, 1, 1, 0],
                        [1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [0, 1, 1, 0]])
        y = np.array([1, 1, 1, 1])
        space = DiscreteSpace(A, y=y, state_labels=['A', 'B', 'C', 'D'])
        mc = WMWalk(space, Ns=1)
        mc.calc_rate_matrix()
        q = mc.calc_hitting_prob_through(['D'], ['C'])
        assert(np.allclose(q[0], 0.5))
        
        # With an additional neutral path
        A = csr_matrix([[0, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0],
                        [1, 0, 0, 1, 0],
                        [0, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0]])
        y = np.array([1, 1, 1, 1, 1])
        space = DiscreteSpace(A, y=y, state_labels=['A', 'B', 'C', 'D', 'E'])
        mc = WMWalk(space, Ns=1)
        mc.calc_rate_matrix()
        q = mc.calc_hitting_prob_through(['D'], ['C'])
        assert(np.allclose(q[0], 1/3.))
        
        # With the Ser codon landscape
        ser2 = ['AGT', 'AGC']
        ser4 = ['TCT', 'TCC', 'TCA', 'TCG']
        intermediates = ['ACT', 'ACC', 'TGT', 'TGC']
        mc = WMWalk(CodonSpace(['S'], add_variation=True, seed=0), Ns=1)
        mc.calc_rate_matrix()
        q = mc.calc_hitting_prob_through(ser2, intermediates)
        q = pd.Series(q, index=mc.space.genotypes)
        assert(np.allclose(q.loc[ser4].mean(), 0.29483, atol=1e-4))
        
        # Compare with TPT derived rates into the Ser2 block
        paths = mc.get_reactive_paths(ser4, ser2)
        idx1 = mc.space.get_state_idxs(intermediates)
        idx2 = mc.space.get_state_idxs(ser2)
        rate1 = paths.eff_flow_matrix[idx1, :][:, idx2].todense().sum()
        assert(np.allclose(q.loc[ser4].mean(), rate1 / paths.reactive_rate, atol=1e-2))
    
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
        data = DataSet('serine').landscape
        
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.input.csv'.format(fhand.name)
            data.to_csv(input_fpath)
             
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, input_fpath, '-o', out_fpath,
                   '-p', '90', '-e', '-nf', 'csv']
            check_call(cmd)
            
            df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert(df.iloc[np.argmax(df['1']), :]['function'] > 1.5)
            assert(df.iloc[np.argmin(df['1']), :]['function'] > 1.5)
            
            edges = load_npz('{}.edges.npz'.format(out_fpath))
            assert(np.all(edges.shape == (64, 64)))
    
    def test_calc_visualization_bin_guess_config(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        data = DataSet('serine').landscape
        
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.input.csv'.format(fhand.name)
            data.to_csv(input_fpath)
            
            cmd = [sys.executable, bin_fpath, input_fpath, '-o', fhand.name,
                   '-m', '1.5', '-e']
            check_call(cmd)
    
    def test_calc_visualization_codon_restricted(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        data = DataSet('serine').landscape
        data['protein'] = translate_seqs(data.index.values)
        data = data.groupby(['protein']).mean().drop('*', axis=0)
        
        # run with standard genetic code
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.input.csv'.format(fhand.name)
            data.to_csv(input_fpath)
            
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, input_fpath,
                   '-m', '1.5', '-e', '-nf', 'csv', '-o', out_fpath]
            check_call(cmd)
            edges1 = load_npz('{}.edges.npz'.format(out_fpath))
        
        # run with bacterial genetic code 11
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.input.csv'.format(fhand.name)
            data.to_csv(input_fpath)
            
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, input_fpath, '-m', '1.5', '-e',
                   '-nf', 'csv', '-o', out_fpath, '-c', '11']
            check_call(cmd) 
            df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert(df.shape[0] == data.shape[0])

            # Ensure we have less edges when using codon restricted transitions        
            edges2 = load_npz('{}.edges.npz'.format(out_fpath))
            assert(edges1.sum() > edges2.sum())

    def test_calc_visualization_codon_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        data = DataSet('serine').landscape
        data['protein'] = translate_seqs(data.index.values)
        data = data.groupby(['protein']).mean().drop('*', axis=0)
        
        # standard genetic code
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.input.csv'.format(fhand.name)
            data.to_csv(input_fpath)
            
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, input_fpath, '-Ns', '1', 
               '-e', '-C', '-nf', 'csv', '-o', out_fpath, '-c', 'Standard']
            check_call(cmd)
            
            nodes = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
            assert('protein' in nodes.columns)
            assert(nodes.shape[0] == 64)
        
        # custom genetic code
        aa = ['W', 'W', 'K', 'K', 'C', 'C', 'C', 'C',
              'H', 'H', '*', '*', 'I', 'I', '*', 'M',
              'K', 'K', 'K', 'K', 'E', 'E', 'E', 'E',
              'L', 'L', 'Q', 'Q', 'P', 'P', 'P', 'P',
              'D', 'D', 'D', 'S', 'A', 'A', 'A', 'A',
              'T', 'T', 'F', 'F', 'C', 'C', 'P', 'P',
              'V', 'V', 'V', 'V', 'G', 'G', 'G', 'G',
              'N', 'N', 'R', 'R', 'Y', 'Y', 'Y', 'Y']
        codons = ['UUU', 'UUC', 'UUA', 'UUG', 'UCU', 'UCC', 'UCA', 'UCG',
                  'UAU', 'UAC', 'UAA', 'UAG', 'UGU', 'UGC', 'UGA', 'UGG',
                  'CUU', 'CUC', 'CUA', 'CUG', 'CCU', 'CCC', 'CCA', 'CCG',
                  'CAU', 'CAC', 'CAA', 'CAG', 'CGU', 'CGC', 'CGA', 'CGG',
                  'AUU', 'AUC', 'AUA', 'AUG', 'ACU', 'ACC', 'ACA', 'ACG',
                  'AAU', 'AAC', 'AAA', 'AAG', 'AGU', 'AGC', 'AGA', 'AGG',
                  'GUU', 'GUC', 'GUA', 'GUG', 'GCU', 'GCC', 'GCA', 'GCG',
                  'GAU', 'GAC', 'GAA', 'GAG', 'GGU', 'GGC', 'GGA', 'GGG']
        aa_mapping = pd.DataFrame({'Letter': aa, 'Codon': codons})
        
        with NamedTemporaryFile() as fhand:
            codon_fpath = '{}.codon_table.csv'.format(fhand.name)
            aa_mapping.to_csv(codon_fpath)
            
            out_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, input_fpath, '-Ns', '1', 
                   '-e', '-C', '-nf', 'csv', '-o', out_fpath, '-c', codon_fpath]
            check_call(cmd)
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
        
        # Committor probabilities
        assert(np.allclose(paths.q_forward, [0, 1/3, 1/3, 2/3, 2/3, 1]))
        assert(np.allclose(paths.q_backward,  1 - paths.q_forward))
        
        # Effective flows
        flow = paths.eff_flow_matrix.todense()
        assert(np.allclose(flow[0, 1], 1/18))
        assert(np.allclose(flow[flow > 0], flow[0, 1]))
        
        # Overall rate
        assert(np.allclose(paths.reactive_rate, 2/18))
    
    def test_calc_committors_avoid(self):
        Q = csr_matrix([[-2, 1, 1, 0, 0, 0],
                        [1, -2, 0, 1, 0, 0],
                        [1, 0, -2, 0, 1, 0],
                        [0, 1, 0, -2, 0, 1],
                        [0, 0, 1, 0, -2, 1],
                        [0, 0, 0, 1, 1, -2]])
        n = Q.shape[0]
        stat_freqs = np.ones(n) / n
        start, end, avoid = np.array([0]), np.array([5]), np.array([4])
        paths = ReactivePaths(Q, stat_freqs, start, end, avoid=avoid)
        
        # Committor probabilities
        assert(np.allclose(paths.q_forward, [0, 1/3, 0, 2/3, 0, 1]))
        assert(np.allclose(paths.q_backward,  [1, 2/3, 0.5, 1/3, 0, 0]))
        assert(np.allclose(paths.conditional_reactive_p, [0, 1/2, 0, 1/2, 0, 0]))
        
        # Effective flows
        flow = paths.eff_flow_matrix.todense()
        assert(np.allclose(flow[0, 1], 1/18))
        assert(np.allclose(flow[flow > 0], flow[0, 1]))
        
        # Overall rate
        assert(np.allclose(paths.reactive_rate, 1/18))
        
    def test_calc_bottleneck(self):
        Q = csr_matrix([[-1.5, 1, 0.5, 0, 0, 0],
                        [1, -2, 0, 1, 0, 0],
                        [1, 0, -2, 0, 1, 0],
                        [0, 1, 0, -2, 0, 1],
                        [0, 0, 1, 0, -2, 1],
                        [0, 0, 0, 1, 1, -2]])
        n = Q.shape[0]
        stat_freqs = np.ones(n) / n
        start, end = np.array([0]), np.array([5])
        paths = ReactivePaths(Q, stat_freqs, start, end)
        bottleneck, min_flow, _ = paths.calc_bottleneck()
        assert(np.allclose(min_flow, 0.05555556))
        assert(bottleneck in [(0, 1), (1, 3), (3, 5)])
    
    def test_calc_pathway(self):
        Q = csr_matrix([[-1.5, 1, 0.5, 0, 0, 0],
                        [1, -2, 0, 1, 0, 0],
                        [1, 0, -2, 0, 1, 0],
                        [0, 1, 0, -2, 0, 1],
                        [0, 0, 1, 0, -2, 1],
                        [0, 0, 0, 1, 1, -2]])
        n = Q.shape[0]
        stat_freqs = np.ones(n) / n
        start, end = np.array([0]), np.array([5])
        paths = ReactivePaths(Q, stat_freqs, start, end)
        path, min_flow = paths.calc_pathway()
        assert(np.allclose(min_flow, 0.05555556))
        assert(np.allclose(path, [0, 1, 3, 5]))
        
    def test_randomwalk_calc_bottleneck(self):
        X = np.array(list(get_seqs_from_alleles([['A', 'B']] * 3)))
        y = np.array([2, 0, 2, 2,
                      2, 2, 0, 2])
        space = SequenceSpace(X=X, y=y)
        Ns = 0.628
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=Ns)
        paths = rw.get_reactive_paths(['BBB'], ['AAA'])
        bottleneck, eff_flow, _ = paths.calc_bottleneck()
        assert(bottleneck in [(3, 2), (5, 4)])
        assert(np.allclose(eff_flow, 0.040054396))
    
    def test_randomwalk_calc_pathway(self):
        X = np.array(list(get_seqs_from_alleles([['A', 'B']] * 3)))
        y = np.array([2, 0, 2, 2,
                      2, 2, 0, 2])
        space = SequenceSpace(X=X, y=y)
        Ns = 0.628
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=Ns)
        paths = rw.get_reactive_paths(['BBB'], ['AAA'])
        path, min_flow = paths.calc_pathway()
        assert(1 not in path)
        assert(6 not in path)
        assert(np.allclose(min_flow, 0.040054396))
    
    def test_randomwalk_calc_pathways(self):
        X = np.array(list(get_seqs_from_alleles([['A', 'B']] * 3)))
        y = np.array([2, 0, 2, 2,
                      2, 2, 0, 2])
        space = SequenceSpace(X=X, y=y)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.628)
        paths = rw.get_reactive_paths(['BBB'], ['AAA'])
        
        total_flows = np.sum([x[1] for x in paths.calc_pathways()])
        assert(np.allclose(total_flows, paths.reactive_rate))
        
        pathways = paths.calc_pathways()
        df = paths.pathways_to_df(pathways)
        assert(df.shape[0] == 18)
        
        y = np.array([2, 0, 0, 0,
                      2, 2, 0, 2])
        space = SequenceSpace(X=X, y=y)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.628)
        paths = rw.get_reactive_paths(['BBB'], ['AAA'])
        
        rate = paths.reactive_rate
        total_flows = np.sum([x[1] for x in paths.calc_pathways()])
        assert(np.allclose(total_flows, rate))
        
        pathways = paths.calc_pathways()
        df = paths.pathways_to_df(pathways)
        assert(df.shape[0] == 18)
        
    
    def test_calc_jump_matrix(self):
        X = np.array(list(get_seqs_from_alleles([['A', 'B']] * 2)))
        y = np.array([2, 0, 2, 2])
        space = SequenceSpace(X=X, y=y)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.628)
        paths = rw.get_reactive_paths(['AA'], ['BB'])
        P = paths.calc_jump_matrix().todense()
        
        # Ensure valid transition matrix
        assert(np.allclose(P.sum(1), 1))
        
        # Ensure absorbing state
        assert(np.allclose(P[-1, -1], 1))
        
        # Ensure starting transition probabilities are unchanged
        row1 = np.array(rw.rate_matrix.todense())[0, :]
        row1[0] = 0
        row1 = row1 / row1.sum()
        assert(np.allclose(P[0, :], row1))
        
        ##### Test on a larger space #####
        space = CodonSpace(['S'], add_variation=True, seed=0)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.628)
        paths = rw.get_reactive_paths(['AAA'], ['TTT'])
        P = paths.calc_jump_matrix().todense()
        assert(np.allclose(P.sum(1), 1))
        assert(np.allclose(P[-1, -1], 1))
    
    def xtest_sample_reactive_paths(self):
        X = np.array(list(get_seqs_from_alleles([['A', 'B']] * 2)))
        y = np.array([2, 0, 2, 2])
        space = SequenceSpace(X=X, y=y)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.628)
        paths = rw.get_reactive_paths(['AA'], ['BB'])
        p = np.mean([p[1] == 1 for p in paths.sample(n=100)])
        assert(np.abs(p - 1/3) < 0.05)
        
        ##### Test on a larger space #####
        space = CodonSpace(['S'], add_variation=True, seed=0)
        rw = WMWalk(space)
        rw.calc_rate_matrix(Ns=0.1)
        paths = rw.get_reactive_paths(['AGT', 'AGC'],
                                      ['TCT', 'TCC', 'TCA', 'TCG'])
        lengths = [len(p) for p in paths.sample(n=100)]
        assert(np.mean(lengths) == 24.7)
        assert(len(lengths) == 100)
    
        
if __name__ == '__main__':
    sys.argv = ['', 'RandomWalkTests']
    unittest.main()
