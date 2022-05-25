#!/usr/bin/env python
import unittest
import sys

from os.path import join
from subprocess import check_call

import numpy as np
import pandas as pd

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.space import CodonSpace
from gpmap.src.randwalk import WMWSWalk
from scipy.sparse._matrix_io import load_npz


class RandomWalkTests(unittest.TestCase):
    def test_set_Ns(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        mc.set_Ns(Ns=1)
        assert(mc.Ns == 1)
        
        mc.set_Ns(mean_function=1.22743)
        assert(np.isclose(mc.Ns, 1))
        
        mc.set_Ns(mean_function_perc=80)
        assert(np.isclose(mc.Ns, 0.59174))
    
    def test_stationary_frequencies(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_Ns(1)
        mc.calc_stationary_frequencies()
        
        # Check serine codons have high frequencies
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        assert(np.all(mc.stationary_freqs[mc.space.get_state_idxs(codons)] > 0.03))
        
        # Check stop codons have low frequencies
        codons = ['TGA', 'TAG', 'TAA']
        assert(np.all(mc.stationary_freqs[mc.space.get_state_idxs(codons)] < 0.01))
    
    def test_stationary_function(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.set_Ns(Ns=1)
        
        # Ensure failure when stationary frequencies are not calculated
        try:
            mc.calc_stationary_mean_function()
            self.fail()
        except ValueError:
            pass
        
        # Ensure calculation with uniform frequencies
        mc.stationary_freqs = np.ones(mc.space.n_states) / mc.space.n_states
        mean_function = mc.calc_stationary_mean_function()
        assert(np.allclose(mean_function, mc.space.function.mean()))
    
    def test_calc_visualization(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))

        mc.calc_visualization(Ns=1, n_components=20)
        assert(np.allclose(mc.decay_rates_df['decay_rates'][0], 0.3914628))

        # Ensure extreme genotypes in axis 1 have high function       
        df = mc.nodes_df
        assert(df.iloc[np.argmax(df['1']), :]['function'] > 1.5)
        assert(df.iloc[np.argmin(df['1']), :]['function'] > 1.5)
    
    def test_write_visualization(self):
        mc = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        mc.calc_visualization(Ns=1, n_components=20)
        prefix = join(TEST_DATA_DIR, 'serine')
        mc.write_tables(prefix, write_edges=True)
        
        nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
        assert(np.allclose(nodes_df.values, mc.nodes_df.values))
        
    def test_calc_visualization_bin_help(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
    
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
    
    def test_calc_visualization_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'serine.csv')
        
        out_fpath = join(TEST_DATA_DIR, 'serine') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath, '-p', '90',
               '-A', 'rna', '-e']
        check_call(cmd)
        
        df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
        assert(df.iloc[np.argmax(df['1']), :]['function'] > 1.5)
        assert(df.iloc[np.argmin(df['1']), :]['function'] > 1.5)
        
        edges = load_npz('{}.edges.npz'.format(out_fpath))
        assert(np.all(edges.shape == (64, 64)))
        
    def test_calc_visualization_bin_guess_config(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'test.csv')
        
        out_fpath = join(TEST_DATA_DIR, 'test') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath, '-m', '0.5',
               '-A', 'guess', '-e']
        check_call(cmd)
    
    def test_calc_visualization_codon_restricted(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
        fpath = join(TEST_DATA_DIR, 'test.csv')
        
        out_fpath = join(TEST_DATA_DIR, 'test') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath, '-m', '0.65', '-e',
               '-A', 'guess']
        check_call(cmd)
        
        edges1 = load_npz('{}.edges.npz'.format(out_fpath))
        
        out_fpath = join(TEST_DATA_DIR, 'test.codon') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath, '-m', '0.65',
               '-e', '-c', '11', '-A', 'guess']
        check_call(cmd)
        
        df = pd.read_csv('{}.nodes.csv'.format(out_fpath), index_col=0)
        data = pd.read_csv(fpath, index_col=0)
        assert(df.shape[0] == data.shape[0])

        # Ensure we have less edges when using codon restricted transitions        
        edges2 = load_npz('{}.edges.npz'.format(out_fpath))
        assert(edges1.sum() > edges2.sum())
        
        
if __name__ == '__main__':
    sys.argv = ['', 'RandomWalkTests']
    unittest.main()
