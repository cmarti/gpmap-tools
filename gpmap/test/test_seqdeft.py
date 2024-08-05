#!/usr/bin/env python
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from tempfile import NamedTemporaryFile
from subprocess import check_call
from scipy.stats import pearsonr, multinomial
from scipy.special import logsumexp

from gpmap.src.settings import BIN_DIR
from gpmap.src.seq import generate_possible_sequences
from gpmap.src.matrix import quad
from gpmap.src.linop import ProjectionOperator
from gpmap.src.inference import SeqDEFT
from gpmap.src.plot.mpl import plot_SeqDEFT_summary, savefig, plot_density_vs_frequency


class SeqDEFTTests(unittest.TestCase):
    def test_init(self):
        seqdeft = SeqDEFT(P=2)
        X = np.array(['AAA', 'ACA', 'BAA', 'BCA',
                      'AAD', 'ACD', 'BAD', 'BCD'])
        seqdeft.set_data(X=X)
        
        assert(seqdeft.seq_length == 3)
        assert(seqdeft.n_alleles == 2)

        # Test giving incompatible sequences
        X = np.array(['AA', 'AC', 'BA', 'BC'])
        try:
            seqdeft.set_data(X=X)
        except KeyError:
            pass

        # Test giving incompatible sequences in predefined RNA landscape
        seqdeft = SeqDEFT(P=2, seq_length=2, alphabet_type='rna')
        X = np.array(['AA', 'AC', 'BA', 'BC'])
        try:
            seqdeft.set_data(X=X)
        except KeyError:
            pass

        # Test incomplete definition of sequence space
        try:
            seqdeft = SeqDEFT(P=2, alphabet_type='rna')
        except ValueError:
            pass
    
    def test_log_likelihood(self):
        seqdeft = SeqDEFT(P=2, seq_length=2, n_alleles=2)
        X = np.array(['00', '01', '10', '11'])
        phi = np.zeros(4)
        dist = multinomial(4, p=seqdeft.phi_to_Q(phi))
        exp_ll = dist.logpmf([1, 1, 1, 1])
        seqdeft.set_data(X=X)
        ll = seqdeft.calc_logL(phi)
        ll2 = seqdeft.calc_logL(phi + 1)        
        assert(ll == exp_ll)
        assert(ll2 == exp_ll)

        # Compute likelihood with inf phi
        X = np.array(['00', '01', '10'])
        seqdeft.set_data(X=X)
        phi[3] = np.inf
        ll = seqdeft.calc_logL(phi)
        dist = multinomial(3, p=seqdeft.phi_to_Q(phi))
        exp_ll = dist.logpmf([1, 1, 1, 0])
        assert(np.isfinite(ll))
        assert(np.allclose(ll, exp_ll))
        
        phi[1] = np.inf
        ll = seqdeft.calc_logL(phi)
        assert(np.isinf(ll))
    
    def test_loss(self):
        # With specific a
        seqdeft = SeqDEFT(a=10, P=2, seq_length=2, n_alleles=2)
        b = np.array([2, 1, 1.])
        phi = seqdeft._b_to_phi(b)
        X = np.array(['00', '00', '01', '10'])
        
        # Calculate regular loss function
        seqdeft.set_data(X=X)
        loss, grad = seqdeft.calc_loss(phi, return_grad=True)
        hess = seqdeft.calc_loss_finite_hessian(phi) @ np.eye(4)
        assert(np.allclose(loss, 9.484376662317993))
        assert(np.allclose(grad,  [-2.,          -0.47151776,  -0.47151776,  -0.54134113]))
        assert(np.allclose(hess, [[ 14,          -10.,         -10.,          10.        ],
                                  [-10.,          11.47151776,  10.,         -10.        ],
                                  [-10.,          10.,          11.47151776, -10.        ],
                                  [ 10.,         -10.,         -10.,          10.54134113]]))
        
        # With a=inf in additive landscape should have the same loss
        loss, grad = seqdeft.calc_loss_inf(b, return_grad=True)
        assert(np.allclose(loss, 9.48437666231799))
        assert(np.allclose(grad, [-1.74218833,  0.72932943,  0.72932943]))
        
    def test_simulate(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        
        # Ensure right scale of phi
        np.random.seed(2)
        a = 5e2
        phi = seqdeft.simulate_phi(a=a)
        ss = quad(seqdeft.DP, phi) / seqdeft.n_genotypes
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
    
    def test_invalid_a(self):
        try:
            SeqDEFT(P=2, a=-500)
            self.fail()
        except ValueError:
            pass
    
    def test_inference(self):
        np.random.seed(0)
        a = 500
        l = 5
        seqdeft = SeqDEFT(P=2, a=a, seq_length=l, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=a)
        X = seqdeft.simulate(N=1000, phi=phi)

        # Infer with a=inf first
        seqdeft = SeqDEFT(P=2, a=np.inf, seq_length=l, alphabet_type='dna')
        probs = seqdeft.fit(X)
        assert(np.allclose(probs['Q_star'].sum(), 1))

        # Ensure it can't learn a function with only k>P interactions
        r1 = pearsonr(-phi, np.log(probs['Q_star']))[0]
        assert(r1 < 0.05)

        # Verify uniform distribution
        assert(np.allclose(probs['Q_star'], 1. / probs.shape[0], atol=1e-3))

        # Ensure it is a probability distribution        
        seqdeft = SeqDEFT(P=2, a=a, seq_length=l, alphabet_type='dna')
        probs = seqdeft.fit(X=X)
        assert(np.allclose(probs['Q_star'].sum(), 1))

        # Ensure convergence
        _, grad = seqdeft.calc_loss(probs['phi'])
        assert(np.allclose(grad, 0, atol=2e-2))
        
        # Ensure it is similar the true probabilities
        r2 = pearsonr(-phi, np.log(probs['Q_star']))[0]
        assert(r2 > 0.6)

        # Test inference of maximum entropy model
        seqdeft = SeqDEFT(P=2, a=np.inf, seq_length=l, alphabet_type='dna')
        b = 2 * np.random.normal(size=seqdeft.kernel_basis.shape[1])
        phi = seqdeft._b_to_phi(b)
        X = seqdeft.simulate(N=10000, phi=phi)
        probs = seqdeft.fit(X=X)
        r3 = pearsonr(-phi, np.log(probs['Q_star']))[0]
        assert(r3 > 0.95)

        # Ensure convergence
        b = seqdeft._phi_to_b(probs['phi'])
        _, grad = seqdeft.calc_loss(b)
        assert(np.allclose(grad, 0, atol=1e-3))
    
    def test_inference_baseline(self):
        np.random.seed(3)
        seqdeft = SeqDEFT(P=2, a=500, seq_length=5, alphabet_type='dna')
        baseline_phi = seqdeft.simulate_phi(a=1000)
        target_phi = seqdeft.simulate_phi(a=500)
        obs_phi = baseline_phi + target_phi
        X = seqdeft.simulate(N=1000, phi=obs_phi)
        probs1 = seqdeft.fit(X, baseline_phi=baseline_phi,
                             baseline_X=seqdeft.genotypes)
        probs2 = seqdeft.fit(X)

        # Ensure adjusting improves prediction of the target phi
        r1 = pearsonr(probs1['phi'], target_phi)[0]
        r2 = pearsonr(probs2['phi'], target_phi)[0]
        assert(r1 > r2)
        
        # Ensure poor prediction of the baseline phi
        r3 = pearsonr(probs1['phi'], baseline_phi)[0]
        assert(np.abs(r3) < 0.2)

        # Ensure better prediction of the observed phi
        r4 = pearsonr(probs2['phi'], obs_phi)[0]
        r5 = pearsonr(probs1['phi'] + baseline_phi, obs_phi)[0]
        assert(r5 > r4)
    
    def test_inference_outside_interactions(self):
        np.random.seed(3)
        l, a = 7, 4
        seqdeft_a = 50
        out = 3

        # Simulate a pairwise function on l=5
        x = np.random.normal(size=a ** l)
        lambdas = np.zeros(l+1)
        lambdas[1:3] = [300, 75]
        P = ProjectionOperator(a, l, lambdas=np.sqrt(lambdas))
        phi = P @ x

        # Ensure lack of higher order components
        P3 = ProjectionOperator(a, l, k=3)
        k3 = quad(P3, phi)
        assert(k3 < 1e-8)

        # Average out last 2 positions
        seqs = np.array(list(generate_possible_sequences(l)))
        baseline = pd.DataFrame({'seqs': seqs, 'phi': phi, 'subseq': [x[:-out] for x in seqs],
                                 'Q': np.exp(phi - logsumexp(phi))})
        baseline_phi = -np.log(baseline.groupby('subseq')['Q'].sum())

        # Ensure some higher order component induced by missing sites
        P3 = ProjectionOperator(a, l-out, k=3)
        k3_short = quad(P3, baseline_phi)
        assert(k3_short > 1e3 * k3)

        # Simulate from prior at l=4 with baseline phi
        seqdeft = SeqDEFT(P=3, a=seqdeft_a)
        seqdeft.init(seq_length=l-out, alphabet_type='rna')
        phi = seqdeft.simulate_phi(a=seqdeft_a)
        X = seqdeft.simulate(N=2000, phi=phi + baseline_phi.values)

        # Fit model without baseline
        seq_densities = seqdeft.fit(X=X)
        phi1 = seq_densities['phi']
        r1 = pearsonr(phi, phi1)[0]

        # Fit model with baseline should increase correlation with true phi
        seq_densities = seqdeft.fit(X=X,
                                    baseline_phi=baseline_phi.values,
                                    baseline_X=baseline_phi.index.values)
        phi2 = seq_densities['phi']
        r2 = pearsonr(phi, phi2)[0]
        assert(r2 > r1)

        # Calculate variance components
        lambdas, lambdas2 = [], []
        for k in np.arange(l - out + 1):
            W = ProjectionOperator(a, l-out, k=k)
            lambdas.append(quad(W, phi) / W.m_k[k])
            lambdas2.append(quad(W, phi2) / W.m_k[k])

        fig, subplots = plt.subplots(1, 3, figsize=(9, 3))

        axes = subplots[0]
        axes.hist(baseline_phi - baseline_phi.mean(), alpha=0.5, label='$\phi_{baseline}$')
        axes.hist(phi - phi.mean(), alpha=0.5, label='$\phi_{target}$')
        axes.legend(loc=2)
        axes.set(xlabel='$\phi$', ylabel='# sequences')
        axes.grid(alpha=0.2)

        axes = subplots[1]
        axes.scatter(-phi2, -phi, s=5, c='black')
        axes.set(xlabel='$-\phi_{inferred}$', ylabel='$-\phi_{target}$')
        axes.grid(alpha=0.2)

        axes = subplots[2]
        plot_density_vs_frequency(seq_densities, axes)
        axes.grid(alpha=0.2)
    
    def test_inference_cv(self):
        seqdeft = SeqDEFT(P=2, opt_method='trust-krylov')
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
    
    def test_inference_weigths(self):
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
    
    def test_inference_phylo_correction(self):
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
        
    def test_inference_adjusted_logq(self):
        seqdeft = SeqDEFT(P=2, a=500)
        seqdeft.init(seq_length=4, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        X = np.array([x + np.random.choice(X) for x in X])
        positions = np.arange(4)
        
        allele_freqs = {'A': 0.3, 'G': 0.3, 'C': 0.2, 'T': 0.2}
        seq_densities = seqdeft.fit(X=X, positions=positions,
                                    adjust_freqs=True,
                                    allele_freqs=allele_freqs)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        assert(np.allclose(seq_densities['adjusted_Q_star'].sum(), 1))
        assert(len(seq_densities.index[0]) == 4)
        
        # Ensure adjustmnet was done in the right direction
        assert(seq_densities.loc['AAAA', 'Q_star'] > seq_densities.loc['AAAA', 'adjusted_Q_star'])
        assert(seq_densities.loc['TTTT', 'Q_star'] < seq_densities.loc['TTTT', 'adjusted_Q_star'])
        
        r = pearsonr(-phi, np.log(seq_densities['Q_star']))[0]
        assert(r > 0.6)
    
    def test_missing_alleles(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=1000, phi=phi)
        X = X[np.array([x[0] not in ['A', 'C'] for x in X])]
        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
        missing = seq_densities.loc[[x[0] == 'A' for x in seq_densities.index], :]
        assert(missing['frequency'].sum() == 0)
        assert(missing['Q_star'].sum() < 1e-6)
    
    def test_very_few_sequences(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        phi = seqdeft.simulate_phi(a=500)
        X = seqdeft.simulate(N=50, phi=phi)
        
        seq_densities = seqdeft.fit(X=X)
        assert(np.allclose(seq_densities['Q_star'].sum(), 1))
        
    def test_bin(self):
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
    
    def test_cv_plot(self):
        seqdeft = SeqDEFT(P=2)
        seqdeft.init(seq_length=5, alphabet_type='dna')
        X = seqdeft.simulate(N=1000, a=500)

        seq_densities = seqdeft.fit(X=X)
        log_Ls = seqdeft.logL_df

        with NamedTemporaryFile(mode='w') as fhand:        
            fig = plot_SeqDEFT_summary(log_Ls, seq_densities, legend_loc=2)
            fpath = fhand.name
            savefig(fig, fpath)
        

if __name__ == '__main__':
    sys.argv = ['', 'SeqDEFTTests']
    unittest.main()
