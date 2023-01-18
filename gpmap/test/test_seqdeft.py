#!/usr/bin/env python
import unittest

from os.path import join
from subprocess import check_call

import numpy as np
import pandas as pd

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.inference import SeqDEFT
from gpmap.src.utils import get_sparse_diag_matrix
from gpmap.src.plot import (plot_SeqDEFT_summary, savefig, init_fig,
                            plot_a_optimization)
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
        data = pd.read_csv(fpath, index_col=0)
        
        # Try invalid a
        try:
            SeqDEFT(P=2, a=-500)
            self.fail()
        except ValueError:
            pass
         
        # Infer with the provided a value
        seqdeft = SeqDEFT(P=2, a=500)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        Q = seq_densities['Q_star']
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
         
        # Inference with disordered and missing sequences for the 0
        data = pd.read_csv(fpath, index_col=0)
        data = data.loc[data['counts'] > 0, :]
        rownames = data.index.values.copy()
        np.random.shuffle(rownames)
        data = data.loc[rownames, :]
        seqdeft = SeqDEFT(P=2, a=500)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        seq_densities.sort_index(inplace=True)
        r = pearsonr(np.log(seq_densities['Q_star']), np.log(Q))[0]
        assert(r > 0.98)#             self.calc_A_triu()
        
        # Infer hyperparameter using CV
        seqdeft = SeqDEFT(P=2)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        seqdeft.logL_df.to_csv(fpath)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_plot_a_optimization(self):
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        log_Ls = pd.read_csv(fpath, index_col=0)
        
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_a_optimization(log_Ls, axes, err_bars='stderr', x='log_sd')
        fpath = join(TEST_DATA_DIR, 'seqdeft_a')
        savefig(fig, fpath)
    
    def test_seq_deft_cv_plot(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.csv')
        seq_densities = pd.read_csv(fpath, index_col=0)
        
        fpath = join(TEST_DATA_DIR, 'logL.csv')
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
        
        cmd = [sys.executable, bin_fpath, counts_fpath, '-o', out_fpath,
               '--get_a_values']
        check_call(cmd)
        
    def test_missing_alleles(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        seqdeft = SeqDEFT(P=2)
        
        # Load data and remove two alleles in one site
        data = pd.read_csv(fpath, index_col=0)
        data = data.loc[[x[0] not in ['A', 'C'] for x in data.index], :]
        
        # Ensure that it runs even if the allele was not observed at all
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
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
                                    y=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_2_consecutive_fits(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        seqdeft = SeqDEFT(P=2)
        
        # Load data and select a few sequences to be observed only
        data = pd.read_csv(fpath, index_col=0)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        seqs = np.random.choice(data.index, size=200)
        data.loc[seqs, 'counts'] = 0
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
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
                                    y=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
    def test_tk_gloop(self):
        data = pd.read_csv(join(TEST_DATA_DIR, 'tk_gloop5.3.counts.csv'))
        print(data, data['counts'].sum())
        X, y = data['seq'].values, data['counts'].values
        
        seqdeft = SeqDEFT(P=2, num_a=10, nfolds=10)
        result = seqdeft.fit(X, y)
        print(seqdeft.logL_df)
        
        fig = plot_SeqDEFT_summary(seqdeft.logL_df, result)
        out_fpath = join(TEST_DATA_DIR, 'tk.fit2')
        savefig(fig, out_fpath)

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqDEFTTests.test_tk_gloop']
    unittest.main()
