#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd

from gpmap.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.inference import SeqDEFT
from subprocess import check_call


class SeqDEFTTests(unittest.TestCase):
    def test_seq_deft_simulate(self):
        a = 1e6
        seqdeft = SeqDEFT(4, 6, P=2)
        data = seqdeft.simulate(N=10000, a_true=a, random_seed=None)
    
    def test_seq_deft_inference(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        
        seqdeft = SeqDEFT(P=2)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_seq_deft_cv_plot(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        
        seqdeft = SeqDEFT(4, P=2)
        seqdeft.fit(data['counts'], resolution=0.1, max_a_max=1e6, num_a=50)
        seqdeft.plot_summary(fname='data/test_seqdeft')
    
    def test_seq_deft_bin(self):
        counts_fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        out_fpath = join(TEST_DATA_DIR, 'seqdeft.output.csv')
        bin_fpath = join(BIN_DIR, 'fit_seqdeft.py')
        cmd = ['/home/martigo/miniconda3/envs/rna/bin/python', bin_fpath, counts_fpath, '-A', 'rna',
               '-o', out_fpath]
        # cmd = ['which', 'python']
        print(' '.join(cmd))
        check_call(cmd)
    
    def test_handle_counts(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        seqdeft = SeqDEFT(4, 4, P=2)
        seqdeft.load_data(data['counts'])
        obs = seqdeft.expand_counts()
        counts = seqdeft.count_obs(obs)
        assert(np.all(counts == data.values))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqDEFTTests.test_seq_deft_inference']
    unittest.main()
