#!/usr/bin/env python
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join
from tempfile import NamedTemporaryFile
from subprocess import check_call
from scipy.stats import pearsonr

from gpmap.src.settings import BIN_DIR
from gpmap.src.inference import SeqDEFT
from gpmap.src.plot.mpl import plot_SeqDEFT_summary, savefig


class SeqDEFTTests(unittest.TestCase):
    def test_seqdeft_init(self):
        seqdeft = SeqDEFT(P=2)
        X = np.array(['AAA', 'ACA', 'BAA', 'BCA',
                      'AAD', 'ACD', 'BAD', 'BCD'])
        seqdeft.set_data(X=X)
        
        assert(seqdeft.seq_length == 3)
        assert(seqdeft.n_alleles == 2)
        
    def test_seq_deft_simulate(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        
        # Ensure right scale of phi
        np.random.seed(2)
        a = 5e2
        phi = seqdeft.simulate_phi(a=a)
        ss = seqdeft.DP.quad(phi) / seqdeft.n_genotypes
        ss = ss * a / seqdeft.DP.n_p_faces
        assert(np.abs(ss - 1) < 0.1)
        
        # Sample sequences directly
        X = seqdeft.simulate(N=100, a=a)
        assert(X.shape[0] == 100)
        
        # Sample sequences indirectly
        X = seqdeft.simulate(N=2000, phi=phi)
        assert(X.shape[0] == 2000)
        
        # Ensure frequencies correlate with probabilities
        x, y = np.unique(X, return_counts=True)
        merged = pd.DataFrame({'phi': phi}, index=seqdeft.genotypes)
        merged = merged.join(pd.DataFrame({'logy': np.log(y)}, index=x)).dropna()
        r = pearsonr(-merged['phi'], merged['logy'])[0]
        assert(r > 0.6)
    
    def test_seq_deft_invalid_a(self):
        try:
            SeqDEFT(P=2, a=-500)
            self.fail()
        except ValueError:
            pass
    
    def test_seq_deft_inference(self):
        a = 500
        seqdeft = SeqDEFT(P=2, a=a)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=a)
        X = seqdeft.simulate(N=1000, phi=phi)

        # Ensure it is a probability distribution        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Ensure it is similar the true probabilities
        r = pearsonr(-phi, np.log(seq_densities['Q_star']))[0]
        assert(r > 0.6)
    
    def test_seq_deft_inference_cv(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        # Ensure approximate inference of a
        logfc = np.log2(seqdeft.a / 500)
        assert(np.abs(logfc) < 1)
        
        # Ensure it is similar the true probabilities
        r = pearsonr(-phi, np.log(seq_densities['Q_star']))[0]
        assert(r > 0.6)
    
    def test_seq_deft_inference_weigths(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        
        y = np.exp(np.random.normal(size=X.shape[0]))
        seq_densities = seqdeft.fit(X=X, y=y)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        logfc = np.log2(seqdeft.a / 500)
        assert(np.abs(logfc) < 5)
        
        r = pearsonr(-phi, np.log(seq_densities['Q_star']))[0]
        assert(r > 0.4)
    
    def test_seq_deft_inference_phylo_correction(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        X = np.array([x + np.random.choice(X) for x in X])
        positions = np.arange(5)
        
        seq_densities = seqdeft.fit(X=X, phylo_correction=True, positions=positions)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        assert(len(seq_densities.index[0]) == 5)
        
        logfc = np.log2(seqdeft.a / 500)
        assert(np.abs(logfc) < 1)
        
        r = pearsonr(-phi, np.log(seq_densities['Q_star']))[0]
        assert(r > 0.6)
    
    def test_seqdeft_missing_alleles(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        X = X[np.array([x[0] not in ['A', 'C'] for x in X])]
        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        missing = seq_densities.loc[[x[0] == 'A' for x in seq_densities.index], :]
        assert(missing['frequency'].sum() == 0)
        assert(missing['Q_star'].sum() < 1e-8)
    
    def test_seqdeft_very_few_sequences(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=50, phi=phi)
        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
    def test_seqdeft_bin(self):
        bin_fpath = join(BIN_DIR, 'fit_seqdeft.py')
        
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        X = seqdeft.simulate(N=1000, a=500)
        
        with NamedTemporaryFile() as fhand:

            x_fpath = '{}.x.txt'.format(fhand.name)
            with open(x_fpath, 'w') as input_fhand:
                for x in X:
                    input_fhand.write(x + '\n')
            
            cmd = [sys.executable, bin_fpath, x_fpath, '-o', fhand.name + '.pq']
            check_call(cmd + ['--get_a_values'])
            check_call(cmd)
    
    def test_seq_deft_cv_plot(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        X = seqdeft.simulate(N=1000, a=500)

        seq_densities = seqdeft.fit(X=X)
        log_Ls = seqdeft.logL_df

        with NamedTemporaryFile(mode='w') as fhand:        
            fig = plot_SeqDEFT_summary(log_Ls, seq_densities, legend_loc=2)
            savefig(fig, fhand.name)
        

if __name__ == '__main__':
    sys.argv = ['', 'SeqDEFTTests']
    unittest.main()
