#!/usr/bin/env python
import unittest
from os.path import join, exists

import numpy as np
from scipy.stats.stats import pearsonr

from gpmap.td import AdditiveConvolutionalModel, BPStacksConvolutionalModel
from gpmap.plot_utils import init_fig, savefig
from gpmap.settings import TEST_DATA_DIR
from gpmap.utils import write_pickle


class TDTests(unittest.TestCase):
    def test_sim_seqs(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel('AGGA')
        
        seqs = m.simulate_random_seqs(length=5, n_seqs=10)
        assert(len(seqs) == 10)
        for seq in seqs:
            assert(len(seq) == 5)
        
        wt = 'AGGAGG'
        seqs = list(m.simulate_random_mutants(seq=wt, n_seqs=10, p_mut=0.1))
        assert(len(seqs) == 10)
        for seq in seqs:
            assert(seq != wt)
        
        singles = list(m.get_single_mutants(seq))
        assert(len(singles) == 18)
        
        doubles = list(m.get_single_and_double_mutants(seq))
        assert(len(doubles) == 154)
        
        mutants = list(m.simulate_sequences(length=5, seq='AGGAG', p_mut=0.2,
                                            mode='error_pcr', n_seqs=10))
        assert(len(mutants) == 10)
        assert(len(mutants[0]) == 5)
        
        mutants = m.simulate_sequences(length=4, mode='combinatorial')
        assert(len(mutants) == 4**4)
        assert(len(mutants[0]) == 4)
        
        mutants = m.add_flanking_seqs(mutants, n_backgrounds=2)
        assert(len(mutants) == 2*4**4)
        assert(len(mutants[0]) == 10)
    
    def test_allele_features(self):
        m = AdditiveConvolutionalModel(ref_seq='AGGA', positional_effects=True)
        features = m.seq_to_encoding('AGCA')
        assert(features == {'G3C': 1})
        
        encoding = m.get_conv_encoding(['AGGAGG'])
        assert(encoding['n_positions'] == 3)
        assert(np.all(encoding['positions'] == np.arange(3)))
    
    def test_stacks_features(self):
        m = BPStacksConvolutionalModel(template='UCCU')
        counts = m.seq_to_encoding('AGGAGG')
        expected = {'UC|AG': 1, 'UC|GG': 0, 'CC|GG': 1, 'CU|GA': 1, 'CU|GG': 0}
        assert(counts == expected)
        
        encoding = m.get_conv_encoding(['AGAGAG'])
        assert(len(encoding['X']) == 3)
        assert(encoding['n_positions'] == 3)

        # With bulges now        
        m = BPStacksConvolutionalModel(template='UCCU', allow_bulges=True)
        counts = m.seq_to_encoding('AGAGAG', bulge_pos=2)
        expected = {'UC|AG': 1, 'UC|GG': 0, 'CC|GG': 1, 'CU|GA': 1, 'CU|GG': 0,
                    'bulge': 1}
        assert(counts == expected)
        
        encoding = m.get_conv_encoding(['AGAGAG'])
        assert(len(encoding['X']) == 5)
        assert(encoding['n_positions'] == 3)
        
        # With base bulges
        m = BPStacksConvolutionalModel(template='UCCU', allow_bulges=True,
                                       base_bulges=True)
        counts = m.seq_to_encoding('AGAGAG', bulge_pos=2)
        expected = {'UC|AG': 1, 'UC|GG': 0, 'CC|GG': 1, 'CU|GA': 1, 'CU|GG': 0,
                    'bA': 1, 'bC': 0, 'bG': 0, 'bU': 0}
        assert(counts == expected)
        
        encoding = m.get_conv_encoding(['AGAGAG'])
        assert(len(encoding['X']) == 5)
        assert(encoding['n_positions'] == 3)
    
    def test_sim_data(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel('AGGA')
        seqs = m.simulate_random_seqs(length=5, n_seqs=1500)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=2)
        data = m.simulate_data(seqs)
        assert(data['L'] == 11)
        assert(data['F'] == 12)
        assert(data['C'] == 4)
        
        m = BPStacksConvolutionalModel(template='AGGA')
        seqs = m.simulate_combinatorial_mutants(length=5)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs)
        
        fpath = join(TEST_DATA_DIR, 'data.staks.pickle')
        if not exists(fpath):
            write_pickle(data, fpath)
    
    def test_fit_additive(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel('AGGA')
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, background=1, sigma=0.1)
        fit = m.fit(data)
        r = pearsonr(data['theta'], fit['theta'])[0]
        assert(r > 0.9)
        assert(np.abs(fit['mu'] - 2) < 0.2)
        
        # With bulges
        m = AdditiveConvolutionalModel('AGGA', allow_bulges=True)
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, background=1, sigma=0.1)
        fit = m.fit(data)
        r = pearsonr(data['theta'], fit['theta'])[0]
        assert(r > 0.9)
        assert(np.abs(fit['mu'] - 2) < 0.2)
    
    def test_fit_total_constant(self):
        np.random.seed(0)
        mu_0, theta_0 = 8, 2
        m = AdditiveConvolutionalModel('AGGA', total_is_constant=True)
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, theta_0=theta_0, mu_0=mu_0,
                               background=1, sigma=0.1)
        fit = m.fit(data)
        r = pearsonr(data['theta'], fit['theta'])[0]
        
        assert(r > 0.9)
        assert(np.abs(fit['mu'] - mu_0) < 0.6)
        assert(np.abs(fit['theta_0'] - theta_0) < 0.6)
        
    def test_fit_additive_pos_eff(self):
        m = AdditiveConvolutionalModel('AGGA', positional_effects=True)
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, background=1, sigma=0.1)
        fit = m.fit(data)
        r = pearsonr(data['theta'], fit['theta'])[0]
        
        fig, axes = init_fig(2, 3)
        axes = axes.flatten()
        m.plot_y_distribution(data, axes[0])
        m.plot_predictions(fit, axes[1], hist=False)
        m.plot_mut_eff(data, fit, axes[2])
        m.plot_mu(fit, axes[3])
        m.plot_theta_logo(fit, axes[4])
        m.plot_theta_heatmap(fit, axes[5])
        savefig(fig, 'test_td')
        assert(r > 0.9)
    
    def test_fit_stacks(self):    
        np.random.seed(0)
        m = BPStacksConvolutionalModel(template='AGGA')
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, background=1, sigma=0.1)
        fit = m.fit(data)
        
        r = pearsonr(data['theta'], fit['theta'])[0]
        assert(r > 0.9)
        
        m = BPStacksConvolutionalModel(template='AGGA', positional_effects=True)
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1, flank_size=2)
        data = m.simulate_data(seqs, background=1, sigma=0.1)
        fit = m.fit(data)
        
        r = pearsonr(data['theta'], fit['theta'])[0]
        assert(r > 0.9)
        
        r = pearsonr(data['mu'], fit['mu'])[0]
        assert(r > 0.9)
    
    def test_predict(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel('AGGA')
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, background=1)
        fit = m.fit(data)
        
        ypred = m.predict(seqs, fit)
        assert(np.allclose(ypred, fit['yhat']))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'TDTests']
    unittest.main()
