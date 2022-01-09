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
        m = AdditiveConvolutionalModel(filter_size=4)
        
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
    
    def test_seq_features(self):
        m = BPStacksConvolutionalModel(template='UCCU')
        counts = m.seq_to_encoding('AGGAGG')
        expected = {'UC|AG': 1, 'UC|GG': 0, 'CC|GG': 1, 'CU|GA': 1, 'CU|GG': 0}
        assert(counts == expected)
        
        counts = m.get_conv_encoding(['AGAGAG'])
        assert(len(counts) == 3)

        # With bulges now        
        m = BPStacksConvolutionalModel(template='UCCU', allow_bulges=True)
        
        counts = m.seq_to_encoding('AGAGAG', bulge_pos=2)
        expected = {'UC|AG': 1, 'UC|GG': 0, 'CC|GG': 1, 'CU|GA': 1, 'CU|GG': 0,
                    'bA': 1}
        assert(counts == expected)
        
        counts = m.get_conv_encoding(['AGAGAG'])
        assert(len(counts) == 5)
    
    def test_sim_data(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4)
        seqs = m.simulate_random_seqs(length=5, n_seqs=1500)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=2)
        data = m.simulate_data(seqs)
        assert(data['L'] == 11)
        assert(data['F'] == 12)
        assert(data['C'] == 4)
        
        m = BPStacksConvolutionalModel(template='AGGA')
        seqs = m.simulate_combinatorial_mutants(length=5)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, theta0=-5)
        
        fpath = join(TEST_DATA_DIR, 'data.staks.pickle')
        if not exists(fpath):
            write_pickle(data, fpath)
    
    def test_fit(self):
        # We need to get some range of variation in the phenotype for this to work
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4, recompile=False,
                                       model_label='conv_sd')
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, background=1)
        fit = m.fit(data)
        print(fit['mu'])
        r = pearsonr(data['theta'].mean(1), fit['theta'])[0]
         
        fig, axes = init_fig(2, 3)
        axes = axes.flatten()
        m.plot_y_distribution(data, axes[0])
        m.plot_predictions(fit, axes[1], hist=False)
        m.plot_mut_eff(data, fit, axes[2])
        m.plot_mu(fit, axes[3])
        axes[3].set_ylim((1.5, 2.5))
        m.plot_theta_logo(fit, axes[4])
        m.plot_theta_heatmap(fit, axes[5])
        savefig(fig, 'test_td')
        assert(r > 0.9)
        
        # Test BP stacking energies model
        np.random.seed(0)
        m = BPStacksConvolutionalModel(template='AGGA')
        data = m.simulate_data(seqs)
        fit = m.fit(data)
        r = pearsonr(data['theta'].mean(1), fit['theta'])[0]
        # TODO: review why test does not work
        assert(r > 0.9)
    
    def test_predict(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4, 
                                       ref_seq='AGGA',
                                       model_label='conv_sd')
        seqs = m.simulate_random_seqs(length=5, n_seqs=2000)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, background=1)
        fit = m.fit(data)
        
        seqs_pred = seqs[:2]
        y = m.predict(seqs_pred, fit)
        assert(y.shape[0] == 2)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'TDTests']
    unittest.main()
