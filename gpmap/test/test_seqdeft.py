#!/usr/bin/env python
import unittest

from os.path import join
from subprocess import check_call

import numpy as np
import pandas as pd

from gpmap.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.inference import SeqDEFT
from gpmap.src.utils import get_sparse_diag_matrix


class SeqDEFTTests(unittest.TestCase):
    def test_seq_deft_simulate(self):
        a = 1e6
        seqdeft = SeqDEFT(4, 6, P=2)
        data = seqdeft.simulate(N=10000, a_true=a, random_seed=None)
    
    def test_construct_D_kernel_basis(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, n_alleles=4, alphabet_type='custom')
        basis1 = seqdeft.construct_D_kernel_basis()
        basis2 = seqdeft.construct_D_kernel_basis2()
        
        # Ensure basis is orthonormal
        prod = basis1.T.dot(basis1)
        identity = get_sparse_diag_matrix(np.ones(prod.shape[0]))
        assert(np.allclose((prod - identity).todense(), 0))
        
        prod = basis2.T.dot(basis2)
        identity = get_sparse_diag_matrix(np.ones(prod.shape[0]))
        assert(np.allclose((prod - identity).todense(), 0))
        
        # Ensure basis is sparse
        max_values = basis1.shape[0] * basis1.shape[1]
        assert(basis1.data.shape[0] < max_values) 
        
        max_values = basis2.shape[0] * basis2.shape[1]
        assert(basis2.data.shape[0] < max_values) 
        
        # Ensure they generate good projection matrices
        u = np.dot(basis1, basis1.T).dot(basis1)
        error = (basis1 - u).mean()
        assert(np.allclose(error, 0))
        
        u = np.dot(basis2, basis2.T).dot(basis2)
        error = (basis2 - u).mean()
        assert(np.allclose(error, 0))
        
        # Ensure both basis represent the same subspace
        u = np.dot(basis1, basis1.T).dot(basis2)
        error = (basis2 - u).mean()
        assert(np.allclose(error, 0))
        
        u = np.dot(basis2, basis2.T).dot(basis1)
        error = (basis1 - u).mean()
        assert(np.allclose(error, 0))
    
    def test_seq_deft_inference(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        seqdeft = SeqDEFT(P=2)
        
        # Infer hyperparameter using CV
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Infer with the provided a value
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values,
                                    a_value=500)
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
