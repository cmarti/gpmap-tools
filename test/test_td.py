#!/usr/bin/env python
import unittest
from os.path import join, exists

import numpy as np
from scipy.stats.stats import pearsonr

from gpmap.td import AdditiveConvolutionalModel, BPStacksConvolutionalModel
from gpmap.plot_utils import init_fig, savefig
from gpmap.settings import TEST_DATA_DIR
from gpmap.utils import write_pickle, load_pickle


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
    
    def test_sim_data(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4)
        seqs = m.simulate_random_seqs(length=5, n_seqs=10)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=2)
        data = m.simulate_data(seqs)
        assert(data['L'] == 11)
        assert(data['F'] == 13)
        assert(data['C'] == 4)
    
    def test_fit(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4)
        seqs = m.simulate_combinatorial_mutants(length=5)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, theta0=-5)
        fit = m.fit(data)
        r = pearsonr(data['theta'].mean(1)[1:], fit['theta'][1:])[0]
        assert(r > 0.9)
        
        fpath = join(TEST_DATA_DIR, 'fit.pickle')
        if not exists(fpath):
            write_pickle(fit, fpath)
            
        fpath = join(TEST_DATA_DIR, 'data.pickle')
        if not exists(fpath):
            write_pickle(data, fpath)
        
        # Test BP stacking energies model
        m = BPStacksConvolutionalModel(filter_size=4, template='AGGA')
        seqs = m.simulate_combinatorial_mutants(length=5)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=1)
        data = m.simulate_data(seqs, theta0=-5)

        fit = m.fit(data)
        r = pearsonr(data['theta'].mean(1)[1:], fit['theta'][1:])[0]
        assert(r > 0.9)

    def test_plots(self):
        m = AdditiveConvolutionalModel(filter_size=4)
        data = load_pickle(join(TEST_DATA_DIR, 'data.pickle'))
        fit = load_pickle(join(TEST_DATA_DIR, 'fit.pickle'))
        
        fig, axes = init_fig(1, 4)
        m.plot_y_distribution(data, axes[0])
        m.plot_predictions(data, fit, axes[1], hist=True)
        m.plot_mut_eff(data, fit, axes[2])
        m.plot_theta_heatmap(fit, axes[3])
        savefig(fig, 'test_td')
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'TDTests']
    unittest.main()
