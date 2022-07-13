#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd

from gpmap.visualization import (Visualization, CodonFitnessLandscape,
                                 filter_genotypes)
from gpmap.utils import LogTrack, guess_configuration
from gpmap.inference import VCregression
from gpmap.settings import TEST_DATA_DIR, BIN_DIR
from subprocess import check_call
from gpmap.plot import (plot_nodes, plot_edges, figure_visualization,
                        plot_decay_rates, figure_Ns_grid,
                        init_fig, savefig, figure_allele_grid)
from gpmap.base import extend_ambigous_seq


class VisualizationTests(unittest.TestCase):
    def test_extend_seq(self):
        seq = 'XX'
        mapping = [{'X': 'AB', 'A': 'A', 'B': 'B'}] * 2
        
        seqs = list(extend_ambigous_seq(seq, mapping))
        assert(seqs == ['AA', 'AB', 'BA', 'BB'])
        
    def test_n_alleles(self):
        v = Visualization(2, alphabet_type='dna')
        assert(v.n_alleles == [4, 4])
        
        v = Visualization(2, alphabet_type='protein')
        assert(v.n_alleles == [20, 20])
        
        v = Visualization(2, n_alleles=2, alphabet_type='custom')
        assert(v.n_alleles == [2, 2])

        # Raise error when n_alleles is specified for non custom alphabets        
        try:
            v = Visualization(2, n_alleles=4)
            self.fail()
        except ValueError:
            pass
        
        # Try variable number of alleles per site
        v = Visualization(4, n_alleles=[2, 4, 2, 2], alphabet_type='custom')
        assert(v.n_genotypes == 32)
        assert(v.genotypes.shape[0] == 32)
    
    def test_guess_configuration(self):
        fpath = join(TEST_DATA_DIR, 'gfp.short.csv')
        data = pd.read_csv(fpath).sort_values('pseudo_prot').set_index('pseudo_prot')
        config = guess_configuration(data.index.values)
        assert(config['length'] == 13)
        assert(config['n_alleles'] == [2, 1, 8, 1, 2, 2, 6, 1, 1, 1, 2, 1, 2])
        
    def test_translate(self):
        v = Visualization(6, alphabet_type='dna')
        prot = v.get_protein_seq()
        assert(prot.shape[0] == v.n_genotypes)
    
    def test_adjacency_matrix(self):
        v = Visualization(1, alphabet_type='dna')
        A = v.get_adjacency_matrix().todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) == 1))
        
        v = Visualization(2, 2, alphabet_type='custom')
        A = v.get_adjacency_matrix().todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) + np.fliplr(np.eye(4)) == 1))
    
    def test_get_neighbors(self):
        v = Visualization(3, alphabet_type='rna')
        seq = 'AAA'
        
        idxs = v.get_neighborhood_idxs(seq, max_distance=1)
        seqs = v.genotypes[idxs]
        assert(np.all(seqs == ['AAA', 'AAC', 'AAG', 'AAU', 'ACA',
                               'AGA', 'AUA', 'CAA', 'GAA', 'UAA']))
        
        idxs = v.get_neighborhood_idxs(seq, max_distance=2)
        seqs = v.genotypes[idxs]
        for seq in seqs:
            assert('A' in seq)
        
    def test_calc_stationary_frequencies(self):
        v = Visualization(2, 2, alphabet_type='custom')
        v.set_function([2, 1, 1, 1])
        Ns = v.calc_Ns(stationary_function=1.5)
        v.calc_stationary_frequencies(Ns)
        fmean = v.calc_stationary_function()
        
        # Ensure optimization works well
        assert(np.abs(fmean - 1.5) < 1e-4)
        
        # Try in a bigger v
        v = Visualization(8)
        v.set_random_function()
        Ns = v.calc_Ns(stationary_function=1.5)
        v.calc_stationary_frequencies(Ns)
        fmean = v.calc_stationary_function()
        assert(np.abs(fmean - 1.5) < 1e-4)
        assert(np.all(v.genotypes_stationary_frequencies > 0))
        assert(np.allclose(np.sum(v.genotypes_stationary_frequencies), 1))
    
    def test_filter_genotypes(self):
        prefix = join(TEST_DATA_DIR, 'codon_v')
        nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
        edges_df = pd.read_csv('{}.edges.csv'.format(prefix))
        
        nodes_df, edges_df = filter_genotypes(nodes_df, nodes_df['f'] > 0.2,
                                              edges_df=edges_df)
        
        fig_fpath = join(TEST_DATA_DIR, 'codon_v.filtered')
        figure_visualization(nodes_df, edges_df, nodes_size=50,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')
        
    def test_2codon_v(self):
        fpath = join(TEST_DATA_DIR, '2codon.v.csv')
        data = pd.read_csv(fpath, index_col=0)
        
        fig_fpath = join(TEST_DATA_DIR, '2aa.v')
        v = Visualization(2, alphabet_type='protein')
        v.set_function(data['log_binding'])
        v.calc_visualization(meanf=1, n_components=5)
        figure_visualization(v.nodes_df, v.edges_df,
                             fpath=fig_fpath, z='3', interactive=True)
        figure_visualization(v.nodes_df, v.edges_df, fpath=fig_fpath,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein')
        
        fig_fpath = join(TEST_DATA_DIR, '2codon.v')
        v = Visualization(6)
        v.set_function(data['log_binding'], codon_table='11')
        v.calc_visualization(meanf=1, n_components=5)
        figure_visualization(v.nodes_df, v.edges_df,
                             fpath=fig_fpath, z='3', interactive=True, is_prot=True,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein')
        figure_visualization(v.nodes_df, v.edges_df, fpath=fig_fpath,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein', is_prot=True)
        
    def xtest_big_v(self):
        log = LogTrack()
        np.random.seed(1)
        length = 10
        lambdas = np.array([0, 1e6, 1e5, 1e4,
                            1e3, 1e2, 1e1, 1e0,
                            1e-1, 0, 0])
    
        log.write('Simulate data')
        vc = VCregression(length, n_alleles=4, log=log)
        v = Visualization(length, log=log)
        v.set_function(vc.simulate(lambdas))
        v.calc_stationary_frequencies(Ns=1)
        v.calc_rate_matrix(Ns=1)
        v.calc_visualization(meanf=2, n_components=10)
    
    def test_calc_visualization_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_visualization.py')
    
        # Test help
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
    
        # Calc visualization
        fpath = join(TEST_DATA_DIR, 'small_landscape.csv')
        out_fpath = join(TEST_DATA_DIR, 'small_landscape') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath, '-p', '90']
        check_call(cmd)


class TPTTests(unittest.TestCase):
    def test_calc_committor_p(self):
        np.random.seed(1)
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_stationary_frequencies(Ns=1)
        v.calc_rate_matrix(Ns=1)
        a, b = v.get_AB_genotypes_idxs(['AAA'], ['BBB'])
        q = v.calc_committor_probability(a, b)
        assert(np.all(q >= 0))
        assert(np.all(q <= 1))
    
    def test_calc_dynamic_bottleneck(self):
        v = Visualization(3, n_alleles=2, alphabet_type='custom')
        
        # Normal conditions easy path
        i = np.array([0, 0, 1, 1, 2, 4, 5, 5])
        j = np.array([1, 5, 2, 4, 3, 3, 2, 4])
        flow = np.array([1, 1, 0.9, 0.1, 1.1, 0.9, 0.2, 0.8])
        a = [0]
        b = [3]
        
        bottleneck, flow, m = v._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (1, 2))
        assert(flow == 0.9)
        assert(m == 4)
        
        # Change the order of the edges with the same flow now
        i = np.array([0, 0, 1, 2, 4, 5, 5, 1])
        j = np.array([1, 5, 4, 3, 3, 2, 4, 2])
        flow = np.array([1, 1, 0.1, 1.1, 0.9, 0.2, 0.8, 0.9])
        a = [0]
        b = [3]
        
        bottleneck, flow, m = v._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (1, 2))
        assert(flow == 0.9)
        assert(m == 5)
        
        # Simple graph
        i = np.array([0, 0, 1])
        j = np.array([1, 2, 2])
        flow = np.array([2, 1, 2.5])
        a = [0]
        b = [2]
        
        bottleneck, flow, m = v._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (0, 1))
        assert(flow == 2)
        assert(m == 2)
        
        # Single edge graph
        i = np.array([0])
        j = np.array([1])
        flow = np.array([1])
        a = [0]
        b = [1]
        
        bottleneck, flow, m = v._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (0, 1))
        assert(flow == 1)
        assert(m == 2)
        
        # Specific case when bottleneck is the last edge
        i = np.array([4, 2])
        j = np.array([3, 3])
        flow = np.array([0.9, 1.1])
        a = [2]
        b = [3]
        
        bottleneck, flow, m = v._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (2, 3))
        assert(flow == 1.1)
        assert(m == 3)
        
    def test_calc_representative_path(self):
        v = Visualization(3, n_alleles=2, alphabet_type='custom')
        
        # Trivial path graph
        i = np.array([0, 1, 2])
        j = np.array([1, 2, 3])
        flow = np.array([1, 1, 1])
        a = [0]
        b = [3]
        
        flow_dict = v.get_flows_dict(i, j, a, b, flow)
        path = v._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2, 3])
        
        # Normal conditions easy path
        i = np.array([0, 0, 1, 1, 2, 4, 5, 5])
        j = np.array([1, 5, 2, 4, 3, 3, 2, 4])
        flow = np.array([1, 1, 0.9, 0.1, 1.1, 0.9, 0.2, 0.8])
        a = [0]
        b = [3]
        
        flow_dict = v.get_flows_dict(i, j, a, b, flow)
        path = v._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2, 3])
        
        # Simple graph
        i = np.array([0, 0, 1])
        j = np.array([1, 2, 2])
        flow = np.array([2, 1, 2.5])
        a = [0]
        b = [2]
        
        flow_dict = v.get_flows_dict(i, j, a, b, flow)
        path = v._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2])
    
    def test_calc_representative_paths(self):
        np.random.seed(1)
        alpha = 2
        length = 3
    
        v = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([100, 10, 1, 0])
        v.set_function(f)
        
        Ns = v.calc_Ns(stationary_function=1)
        v.calc_stationary_frequencies(Ns)
        v.calc_rate_matrix(Ns=Ns)
        idx1, idx2 = v.get_AB_genotypes_idxs(['AAA'], ['BBB'])
        
        bottleneck = v.calc_dynamic_bottleneck(idx1, idx2)[0]
        assert(bottleneck == (4, 6))
        
        path = v.calc_representative_pathway(idx1, idx2)[0]
        assert(np.all(path == [0, 4, 6, 7]))
        
        paths = list(v.calc_representative_pathways(idx1, idx2))
        assert(len(paths) == 6)
    
    def test_calc_representative_path_big(self):
        np.random.seed(0)
        alpha = 3
        length = 4
    
        v = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([0, 1000, 100, 10, 1])
        v.set_function(f)
        
        v.calc_stationary_frequencies(Ns=1)
        v.calc_rate_matrix(Ns=1)
        a, b = v.get_AB_genotypes_idxs(['ABAA'], ['BBBB'])
        
        for _, path, _, p in v.calc_representative_pathways(a, b,
                                                            max_missing_flow_p=0.1):
            assert(len(path) == 4)
            assert(p > 0.01)
    
    def test_calc_transition_path_stats_big(self):
        np.random.seed(1)
        alpha = 4
        length = 9
    
        v = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([0, 10000, 1000, 100, 10, 1, 0, 0, 0, 0])
        v.set_function(f)
        v.calc_stationary_frequencies(Ns=1)
        v.calc_rate_matrix(Ns=1)
        a, b = v.get_AB_genotypes_idxs(['AAAAAAAAA'], ['BCDABABCC'])
        q = v.calc_committor_probability(a, b)
        assert(q.shape[0] == 4**9)
        
    def test_calc_jump_matrix(self):
        Ns = 1
    
        v = Visualization(length=1)
        f = np.array([0, 1, 0.5, 1.5])
        v.set_function(f)
        a, b = v.get_AB_genotypes_idxs(['A'], ['T'])
        v.calc_stationary_frequencies(Ns)
        v.calc_rate_matrix(Ns)
        jump_matrix = v.calc_jump_transition_matrix(a, b).todense()
        
        m = np.array([[0, 0.33071192, 0.26564942, 0.40363866],
                      [0, 0, 0.29082531, 0.70917469],
                      [0, 0.35427277, 0, 0.64572723],
                      [0, 0, 0, 1]])
        assert(np.allclose(jump_matrix, m))
    
    def test_calc_p_return(self):
        np.random.seed(1)
        Ns = 5
    
        v = Visualization(length=1)
        f = np.array([0, 1, 1, 0])
        v.set_function(f)
        a, b = v.get_AB_genotypes_idxs(['A'], ['T'])
        v.calc_stationary_frequencies(Ns)
        v.calc_rate_matrix(Ns)
        m = v.calc_p_return(a, b)
        assert(np.allclose(m, [0, 0.87698151, 0.87698151, 0]))
    
    def test_calc_p_return_big(self):
        np.random.seed(1)
        Ns = 1
    
        v = Visualization(7, n_alleles=4, alphabet_type='custom')
        vc = VCregression(7, n_alleles=4, alphabet_type='custom')
        f = vc.simulate([0, 10000, 1000, 100, 10, 1, 0, 0])
        v.set_function(f)
        a, b = v.get_AB_genotypes_idxs(['AAAAAAA'], ['ABBABBB'])
        v.calc_stationary_frequencies(Ns)
        v.calc_rate_matrix(Ns)
        m = v.calc_p_return(a, b, tol=1e-6, inverse=False)
        print(m)
    
    def test_neighbors(self):
        v = Visualization(4, 2)
        
        for i in range(v.n_genotypes):
            seq = np.array(v.seqs[i])
            
            js = np.array(list(v.get_neighbors(i, only_j_higher_than_i=False)))
            assert(js.shape[0] == 4)
            
            for j in js:
                seq2 = np.array(v.seqs[j])
                assert(np.sum(seq != seq2) == 1)
            
            js = np.array(list(v.get_neighbors(i, only_j_higher_than_i=True)))
            assert(np.all(js > i) or len(js) == 0)
    
    def test_transition_path_objects(self):
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        gt1, gt2 = ['UCU', 'UCA', 'UCC', 'UCG'], ['AGU', 'AGC']
        
        Ns = v.calc_Ns(stationary_function=1.3)
        v.calc_stationary_frequencies(Ns)
        v.calc_rate_matrix(Ns)
        
        tpt = v.calc_transition_path_objects(gt1, gt2)
        print(tpt['bottleneck'])
        print(tpt['dom_paths_edges'])
        tpt['bottleneck']['n'] = 1
        print(tpt['bottleneck'].groupby('mutation')['n', 'flow_p'].sum().sort_values('flow_p'))
        print(tpt['bottleneck']['flow_p'].sum())
        
        max_entropy = np.log(1/tpt['bottleneck'].shape[0])
        print(-max_entropy + np.sum(tpt['bottleneck']['flow_p'] * np.log(tpt['bottleneck']['flow_p'])))
        print(tpt['dom_paths_edges'].groupby('flow_p').count().reset_index().groupby('i')['flow_p'].sum())
    
    def test_transition_path_objects_bin(self):
        bin_fpath = join(BIN_DIR, 'calc_tpt.py')
    
        # Test help
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
    
        # Calculate objects for small landscape
        fpath = join(TEST_DATA_DIR, 'small_landscape.csv')
        out_fpath = join(TEST_DATA_DIR, 'small_landscape.tpt') 
        cmd = [sys.executable, bin_fpath, fpath, '-o', out_fpath,
               '-gt1', 'AGCT', '-gt2', 'TCGA', '-Ns', '1', '-K', '10']
        check_call(cmd)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VisualizationTests.test_guess_configuration']
    unittest.main()
