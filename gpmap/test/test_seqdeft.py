#!/usr/bin/env python
import unittest

from os.path import join
from subprocess import check_call

import numpy as np
import pandas as pd

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.inference import SeqDEFT
from gpmap.src.utils import get_sparse_diag_matrix
from gpmap.src.plot import plot_SeqDEFT_summary, savefig
from scipy.stats.stats import pearsonr


class SeqDEFTTests(unittest.TestCase):
    def xtest_seq_deft_simulate(self):
        a = 1e6
        seqdeft = SeqDEFT(4, 6, P=2)
        data = seqdeft.simulate(N=10000, a_true=a, random_seed=None)
    
    def test_construct_D_kernel_basis(self):
        seqdeft = SeqDEFT(P=3)
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
        seqdeft = SeqDEFT(P=2)
        
        # Infer hyperparameter using CV
        data = pd.read_csv(fpath, index_col=0)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        Q = seq_densities['Q_star']
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Infer with the provided a value
        data = pd.read_csv(fpath, index_col=0)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values,
                                    a_value=500)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Inference with disordered and missing sequences for the 0
        data = pd.read_csv(fpath, index_col=0)
        data = data.loc[data['counts'] > 0, :]
        rownames = data.index.values.copy()
        np.random.shuffle(rownames)
        data = data.loc[rownames, :]
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values,
                                    a_value=500)
        r = pearsonr(np.log(seq_densities['Q_star']), np.log(Q))[0]
        assert(r > 0.98)
        
        # Save results
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.csv')
        seq_densities.to_csv(fpath)
        
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.log_Ls.csv')
        seqdeft.log_Ls.to_csv(fpath)
    
    def test_seq_deft_cv_plot(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.csv')
        seq_densities = pd.read_csv(fpath, index_col=0)
        
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.log_Ls.csv')
        log_Ls = pd.read_csv(fpath, index_col=0)
        
        fig = plot_SeqDEFT_summary(log_Ls, seq_densities)
        fpath = join(TEST_DATA_DIR, 'seqdeft_output')
        savefig(fig, fpath)
    
    def test_seq_deft_bin(self):
        counts_fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        out_fpath = join(TEST_DATA_DIR, 'seqdeft.output.csv')
        bin_fpath = join(BIN_DIR, 'fit_seqdeft.py')
        
        cmd = [sys.executable, bin_fpath, counts_fpath, '-o', out_fpath]
        check_call(cmd)
        
        cmd = [sys.executable, bin_fpath, counts_fpath, '-o', out_fpath, '-P', '3']
        check_call(cmd)
    
    def test_handle_counts(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        seqdeft = SeqDEFT(2)
        seqdeft.init(genotypes=data.index.values)
        seqdeft.set_data(data.index.values, data['counts'].values)
        obs = seqdeft.expand_counts()
        counts = seqdeft.count_obs(obs)
        assert(np.all(counts == data['counts'].values))
    
    def test_missing_alleles(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        seqdeft = SeqDEFT(P=2)
        
        # Load data and remove two alleles in one site
        data = pd.read_csv(fpath, index_col=0)
        data = data.loc[[x[0] not in ['A', 'C'] for x in data.index], :]
        
        # Ensure that it runs even if the allele was not observed at all
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Ensure that the alleles were never observed and that the estimated
        # densities are very small
        missing = seq_densities.loc[[x[0] == 'A' for x in seq_densities.index], :]
        assert(missing['frequency'].sum() == 0)
        assert(missing['Q_star'].sum() < 1e-8)
        
        # Load data and set counts to 0
        data = pd.read_csv(fpath, index_col=0)
        selected = [x[0] not in ['A', 'C'] for x in data.index]
        data.loc[selected, 'counts'] = 0
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_2_consecutive_fits(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        seqdeft = SeqDEFT(P=2)
        
        # Load data and select a few sequences to be observed only
        data = pd.read_csv(fpath, index_col=0)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        seqs = np.random.choice(data.index, size=200)
        data.loc[seqs, 'counts'] = 0
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
    def test_very_few_sequences(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        seqdeft = SeqDEFT(P=2)
        
        # Load data and select a few sequences to be observed only
        data = pd.read_csv(fpath, index_col=0)
        seqs = np.random.choice(data.index, size=240)
        data.loc[seqs, 'counts'] = 0
        
        # Ensure that it runs
        seq_densities = seqdeft.fit(X=data.index.values,
                                    counts=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqDEFTTests']
    unittest.main()
