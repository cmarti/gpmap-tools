#!/usr/bin/env python
import unittest
import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from scipy.stats.stats import pearsonr

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.inference import SeqDEFT
from gpmap.src.plot import plot_SeqDEFT_summary, savefig
from tempfile import NamedTemporaryFile


class SeqDEFTTests(unittest.TestCase):
    def xtest_seq_deft_simulate(self):
        a = 1e6
        seqdeft = SeqDEFT(4, 6, P=2)
        data = seqdeft.simulate(N=10000, a_true=a, random_seed=None)
    
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
        assert(r > 0.98)
        
        # Infer hyperparameter using CV
        seqdeft = SeqDEFT(P=2)
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        seqdeft.logL_df.to_csv(fpath)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_weighted_inference(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        data = pd.read_csv(fpath, index_col=0)
        w = np.exp(np.random.normal(0, 0.5, size=data.shape[0]))
        data['counts'] = data['counts'] * w
        
        # Test that data splitting works and returns float weights
        seqdeft = SeqDEFT(P=2)
        seqdeft.set_data(X=data.index.values, y=data['counts'].values)
        for _, _, train, test in seqdeft.get_cv_iter([1]):
            assert(train[1].dtype == float)
            assert(test[1].dtype == float)
            
            # Ensure that set_data is not really transforming the data
            seqdeft._set_data(X=train[0], y=train[1])
            assert(seqdeft.y.dtype == float)
            assert(np.allclose(seqdeft.y, train[1]))
        
        # Test inference
        seq_densities = seqdeft.fit(X=data.index.values,
                                    y=data['counts'].values)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
    
    def test_seq_deft_cv_plot(self):
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.csv')
        seq_densities = pd.read_csv(fpath, index_col=0)
        
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        log_Ls = pd.read_csv(fpath, index_col=0)
        
        fig = plot_SeqDEFT_summary(log_Ls, seq_densities)
        fpath = join(TEST_DATA_DIR, 'seqdeft_output')
        savefig(fig, fpath)
    
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

    def test_seq_deft_bin(self):
        counts_fpath = join(TEST_DATA_DIR, 'seqdeft_counts.csv')
        bin_fpath = join(BIN_DIR, 'fit_seqdeft.py')
        
        with NamedTemporaryFile() as fhand:
            cmd = [sys.executable, bin_fpath, counts_fpath, '-o', fhand.name]
            check_call(cmd + ['--get_a_values'])
            check_call(cmd)
        

if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqDEFTTests']
    unittest.main()
