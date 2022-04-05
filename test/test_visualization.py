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


class VisualizationTests(unittest.TestCase):
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
        space = Visualization(1, alphabet_type='dna')
        A = space.get_adjacency_matrix().todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) == 1))
        
        space = Visualization(2, 2, alphabet_type='custom')
        A = space.get_adjacency_matrix().todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) + np.fliplr(np.eye(4)) == 1))
    
    def test_get_neighbors(self):
        space = Visualization(3, alphabet_type='rna')
        seq = 'AAA'
        
        idxs = space.get_neighborhood_idxs(seq, max_distance=1)
        seqs = space.genotypes[idxs]
        assert(np.all(seqs == ['AAA', 'AAC', 'AAG', 'AAU', 'ACA',
                               'AGA', 'AUA', 'CAA', 'GAA', 'UAA']))
        
        idxs = space.get_neighborhood_idxs(seq, max_distance=2)
        seqs = space.genotypes[idxs]
        for seq in seqs:
            assert('A' in seq)
        
    def test_calc_stationary_frequencies(self):
        gpmap = Visualization(2, 2, alphabet_type='custom')
        gpmap.set_function([2, 1, 1, 1])
        Ns = gpmap.calc_Ns(stationary_function=1.5)
        gpmap.calc_stationary_frequencies(Ns)
        fmean = gpmap.calc_stationary_function()
        
        # Ensure optimization works well
        assert(np.abs(fmean - 1.5) < 1e-4)
        
        # Try in a bigger landscape
        gpmap = Visualization(8)
        gpmap.set_random_function()
        Ns = gpmap.calc_Ns(stationary_function=1.5)
        gpmap.calc_stationary_frequencies(Ns)
        fmean = gpmap.calc_stationary_function()
        assert(np.abs(fmean - 1.5) < 1e-4)
        assert(np.all(gpmap.genotypes_stationary_frequencies > 0))
        assert(np.allclose(np.sum(gpmap.genotypes_stationary_frequencies), 1))
    
    def test_codon_landscape(self):
        fig_fpath = join(TEST_DATA_DIR, 'codon_landscape')
        landscape = CodonFitnessLandscape(add_variation=True, seed=0)
        landscape.calc_visualization(Ns=1, n_components=25)
        figure_visualization(landscape.nodes_df, landscape.edges_df,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')

        # Test whether saving the projections works        
        prefix = join(TEST_DATA_DIR, 'codon_landscape')
        landscape.write_tables(prefix)
        
        nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
        edges_df = pd.read_csv('{}.edges.csv'.format(prefix))
        decay_df = pd.read_csv('{}.decay_rates.csv'.format(prefix))
        
        figure_visualization(nodes_df, edges_df, nodes_size=50,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')
        
        fpath = join(TEST_DATA_DIR, 'codon_landscape.decay_rates')
        plot_decay_rates(decay_df, fpath=fpath)
    
    def test_filter_genotypes(self):
        prefix = join(TEST_DATA_DIR, 'codon_landscape')
        nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
        edges_df = pd.read_csv('{}.edges.csv'.format(prefix))
        
        nodes_df, edges_df = filter_genotypes(nodes_df, nodes_df['f'] > 0.2,
                                              edges_df=edges_df)
        
        fig_fpath = join(TEST_DATA_DIR, 'codon_landscape.filtered')
        figure_visualization(nodes_df, edges_df, nodes_size=50,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')
        
    def test_2codon_landscape(self):
        fpath = join(TEST_DATA_DIR, '2codon.landscape.csv')
        data = pd.read_csv(fpath, index_col=0)
        
        fig_fpath = join(TEST_DATA_DIR, '2aa.landscape')
        landscape = Visualization(2, alphabet_type='protein')
        landscape.set_function(data['log_binding'])
        landscape.calc_visualization(meanf=1, n_components=5)
        figure_visualization(landscape.nodes_df, landscape.edges_df,
                             fpath=fig_fpath, z='3', interactive=True)
        figure_visualization(landscape.nodes_df, landscape.edges_df, fpath=fig_fpath,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein')
        
        fig_fpath = join(TEST_DATA_DIR, '2codon.landscape')
        landscape = Visualization(6)
        landscape.set_function(data['log_binding'], codon_table='11')
        landscape.calc_visualization(meanf=1, n_components=5)
        figure_visualization(landscape.nodes_df, landscape.edges_df,
                             fpath=fig_fpath, z='3', interactive=True, is_prot=True,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein')
        figure_visualization(landscape.nodes_df, landscape.edges_df, fpath=fig_fpath,
                             highlight_genotypes=['LG', 'CA', 'AA', 'GM', 'FG'],
                             alphabet_type='protein', is_prot=True)
        
    def xtest_big_landscape(self):
        log = LogTrack()
        np.random.seed(1)
        length = 10
        lambdas = np.array([0, 1e6, 1e5, 1e4,
                            1e3, 1e2, 1e1, 1e0,
                            1e-1, 0, 0])
    
        log.write('Simulate data')
        vc = VCregression(length, n_alleles=4, log=log)
        gpmap = Visualization(length, log=log)
        gpmap.set_function(vc.simulate(lambdas))
        gpmap.calc_stationary_frequencies(Ns=1)
        gpmap.calc_rate_matrix(Ns=1)
        gpmap.calc_visualization(meanf=2, n_components=10)
    
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
    def test_calc_transition_path_stats(self):
        np.random.seed(1)
        alpha = 2
        length = 3
    
        space = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([100, 10, 1, 0])
        space.load_function(f)
        
        space.calc_stationary_frequencies()
        space.calc_reweighting_diag_matrices()
        space.get_sparse_reweighted_rate_matrix()
        q = space.calc_committor_probability('000', '111')
        
        assert(np.all(q >= 0))
        assert(np.all(q <= 1))
    
        p_reactive, gt_p = space.calc_genotypes_reactive_p('000', '010')
        assert(np.allclose(p_reactive, 0.1330978))
        assert(np.allclose(gt_p.sum(), 1))
        
        edges_flow = space.calc_edges_flow('000', '111')
    
    def test_calc_dynamic_bottleneck(self):
        space = Visualization(3, n_alleles=2, alphabet_type='custom')
        
        # Normal conditions easy path
        i = np.array([0, 0, 1, 1, 2, 4, 5, 5])
        j = np.array([1, 5, 2, 4, 3, 3, 2, 4])
        flow = np.array([1, 1, 0.9, 0.1, 1.1, 0.9, 0.2, 0.8])
        a = [0]
        b = [3]
        
        bottleneck, flow, m = space._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (1, 2))
        assert(flow == 0.9)
        assert(m == 4)
        
        # Change the order of the edges with the same flow now
        i = np.array([0, 0, 1, 2, 4, 5, 5, 1])
        j = np.array([1, 5, 4, 3, 3, 2, 4, 2])
        flow = np.array([1, 1, 0.1, 1.1, 0.9, 0.2, 0.8, 0.9])
        a = [0]
        b = [3]
        
        bottleneck, flow, m = space._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (1, 2))
        assert(flow == 0.9)
        assert(m == 5)
        
        # Simple graph
        i = np.array([0, 0, 1])
        j = np.array([1, 2, 2])
        flow = np.array([2, 1, 2.5])
        a = [0]
        b = [2]
        
        bottleneck, flow, m = space._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (0, 1))
        assert(flow == 2)
        assert(m == 2)
        
        # Single edge graph
        i = np.array([0])
        j = np.array([1])
        flow = np.array([1])
        a = [0]
        b = [1]
        
        bottleneck, flow, m = space._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (0, 1))
        assert(flow == 1)
        assert(m == 2)
        
        # Specific case when bottleneck is the last edge
        i = np.array([4, 2])
        j = np.array([3, 3])
        flow = np.array([0.9, 1.1])
        a = [2]
        b = [3]
        
        bottleneck, flow, m = space._calc_dynamic_bottleneck(i, j, a, b, flow)
        assert(bottleneck == (2, 3))
        assert(flow == 1.1)
        assert(m == 3)
        
    def test_calc_representative_path(self):
        space = Visualization(3, n_alleles=2, alphabet_type='custom')
        
        # Trivial path graph
        i = np.array([0, 1, 2])
        j = np.array([1, 2, 3])
        flow = np.array([1, 1, 1])
        a = [0]
        b = [3]
        
        flow_dict = space.get_flows_dict(i, j, a, b, flow)
        path = space._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2, 3])
        
        # Normal conditions easy path
        i = np.array([0, 0, 1, 1, 2, 4, 5, 5])
        j = np.array([1, 5, 2, 4, 3, 3, 2, 4])
        flow = np.array([1, 1, 0.9, 0.1, 1.1, 0.9, 0.2, 0.8])
        a = [0]
        b = [3]
        
        flow_dict = space.get_flows_dict(i, j, a, b, flow)
        path = space._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2, 3])
        
        # Simple graph
        i = np.array([0, 0, 1])
        j = np.array([1, 2, 2])
        flow = np.array([2, 1, 2.5])
        a = [0]
        b = [2]
        
        flow_dict = space.get_flows_dict(i, j, a, b, flow)
        path = space._calc_representative_pathway(i, j, a, b, flow_dict)
        assert(path == [0, 1, 2])
    
    def test_calc_representative_paths(self):
        np.random.seed(1)
        alpha = 2
        length = 3
    
        space = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([100, 10, 1, 0])
        space.load_function(f)
        
        space.calc_stationary_frequencies()
        space.calc_reweighting_diag_matrices()
        space.get_sparse_reweighted_rate_matrix()
        idx1, idx2 = space.get_AB_genotypes_idxs(['000'], ['111'])
        
        bottleneck, min_flow = space.calc_dynamic_bottleneck(idx1, idx2)
        assert(bottleneck == (4, 6))
        assert(np.allclose(min_flow, 0.0020637))
        
        path, flow = space.calc_representative_pathway(idx1, idx2)
        assert(np.all(path == [0, 4, 6, 7]))
        assert(np.allclose(flow, 0.0020637))
        
        paths = list(space.calc_representative_pathways(idx1, idx2))
        assert(len(paths) == 6)
        assert(np.allclose(sum([p[2] for p in paths]), 1))
    
    def test_calc_representative_path_big(self):
        np.random.seed(1)
        alpha = 3
        length = 4
    
        space = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([0, 1000, 100, 10, 1])
        space.load_function(f)
        
        space.calc_stationary_frequencies()
        space.calc_reweighting_diag_matrices()
        space.get_sparse_reweighted_rate_matrix()
        space.calc_visualization()
        idx1 = np.array([np.argmin(space.projection[1])])
        idx2 = np.array([np.argmax(space.projection[1])]) 
        
        print(idx1, idx2)
        idx1, idx2 = space.get_AB_genotypes_idxs(['0100'], ['1111'])
        
        # path, flow = space.calc_representative_pathway(idx1, idx2)
        for path, flow, p in space.calc_representative_pathways(idx1, idx2,
                                                                max_missing_flow_p=0.1):
            print(path, flow, p)
    
    def test_calc_transition_path_stats_big(self):
        np.random.seed(1)
        alpha = 4
        length = 9
    
        space = Visualization(length, n_alleles=alpha, alphabet_type='custom')
        vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
        f = vc.simulate([0, 10000, 1000, 100, 10, 1, 0, 0, 0, 0])
        space.load_function(f)
        space.calc_stationary_frequencies()
        space.calc_reweighting_diag_matrices()
        space.get_sparse_reweighted_rate_matrix()
        space.calc_genotypes_reactive_p(['000000000'], ['123010122'])
        
    def test_calc_jump_matrix(self):
        Ns = 1
    
        space = Visualization(length=1)
        f = np.array([0, 1, 0.5, 1.5])
        space.set_function(f)
        a, b = space.get_AB_genotypes_idxs(['A'], ['T'])
        space.calc_stationary_frequencies(Ns)
        space.calc_rate_matrix(Ns)
        jump_matrix = space.calc_jump_transition_matrix(a, b).todense()
        
        m = np.array([[0, 0.33071192, 0.26564942, 0.40363866],
                      [0, 0, 0.29082531, 0.70917469],
                      [0, 0.35427277, 0, 0.64572723],
                      [0, 0, 0, 1]])
        assert(np.allclose(jump_matrix, m))
    
    def test_calc_exp_n_returns(self):
        np.random.seed(1)
        Ns = 5
    
        space = Visualization(length=1)
        f = np.array([0, 1, 1, 0])
        space.set_function(f)
        a, b = space.get_AB_genotypes_idxs(['A'], ['T'])
        space.calc_stationary_frequencies(Ns)
        space.calc_rate_matrix(Ns)
        exp_return_times = space.calc_exp_n_returns(a, b)
        m = exp_return_times / (1 + exp_return_times)
        assert(np.allclose(m, [0, 0.87698151, 0.87698151, 0]))
    
    def test_calc_rate_p(self):
        gpmap = Visualization(2, 2, ns=1)
        gpmap.load_function([1.05, 1, 1, 1])
        gpmap.calc_stationary_frequencies()
        
        # Calculate re-weighted rate matrix  
        gpmap.calc_reweighting_diag_matrices()
        t = gpmap.calc_sparse_reweighted_rate_matrix()
        assert(t[3, 3] == -2)
        assert(t[0, 3] == 0)
        assert(np.all(gpmap.T.diagonal() < 0))
        
    def test_neighbors(self):
        gpmap = Visualization(4, 2, ns=1)
        
        for i in range(gpmap.n_genotypes):
            seq = np.array(gpmap.seqs[i])
            
            js = np.array(list(gpmap.get_neighbors(i, only_j_higher_than_i=False)))
            assert(js.shape[0] == 4)
            
            for j in js:
                seq2 = np.array(gpmap.seqs[j])
                assert(np.sum(seq != seq2) == 1)
            
            js = np.array(list(gpmap.get_neighbors(i, only_j_higher_than_i=True)))
            assert(np.all(js > i) or len(js) == 0)
    
    def test_transition_path_objects(self):
        landscape = CodonFitnessLandscape(add_variation=True, seed=0)
        gt1, gt2 = ['UCU', 'UCA', 'UCC', 'UCG'], ['AGU', 'AGC']
        
        Ns = landscape.calc_Ns(stationary_function=1.3)
        landscape.calc_stationary_frequencies(Ns)
        landscape.calc_rate_matrix(Ns)
        
        tpt = landscape.calc_transition_path_objects(gt1, gt2)
        print(tpt['bottleneck'])
        print(tpt['dom_paths_edges'])
        tpt['bottleneck']['n'] = 1
        print(tpt['bottleneck'].groupby('mutation')['n', 'flow_p'].sum().sort_values('flow_p'))
        print(tpt['bottleneck']['flow_p'].sum())
        
        max_entropy = np.log(1/tpt['bottleneck'].shape[0])
        print(-max_entropy + np.sum(tpt['bottleneck']['flow_p'] * np.log(tpt['bottleneck']['flow_p'])))
        print(tpt['dom_paths_edges'].groupby('flow_p').count().reset_index().groupby('i')['flow_p'].sum())
        

class PlottingTests(unittest.TestCase):
    def test_variable_allele_number(self):
        fpath = join(TEST_DATA_DIR, 'gfp.short.csv')
        data = pd.read_csv(fpath).sort_values('pseudo_prot').set_index('pseudo_prot')
        config = guess_configuration(data.index.values)
        
        v = Visualization(config['length'], n_alleles=config['n_alleles'],
                          alphabet_type=config['alphabet'])
        v.set_function(data['G'], label='GFP')
        assert(np.all(v.genotypes == data.index.values))
        v.calc_visualization(meanf=0.85)

        fpath = join(TEST_DATA_DIR, 'gfp_core')
        figure_visualization(v.nodes_df, v.edges_df, fpath=fpath)
        
        fpath = join(TEST_DATA_DIR, 'gfp_core.Ns')
        figure_Ns_grid(v, fpath=fpath, fmin=0.5, fmax=0.85,
                       ncol=3, nrow=3, show_edges=True, nodes_cmap_label='GFP')
        
        v = Visualization(config['length'], n_alleles=config['n_alleles'],
                          alphabet_type=config['alphabet'])
        v.set_function(data['C'], label='CFP')
        fpath = join(TEST_DATA_DIR, 'cfp_core.Ns')
        figure_Ns_grid(v, fpath=fpath, fmin=0.2, fmax=0.60,
                       ncol=3, nrow=3, show_edges=True, nodes_cmap_label='CFP')
        
    def test_plotting(self):
        fpath = join(TEST_DATA_DIR, 'codon_landscape')
        landscape = CodonFitnessLandscape(add_variation=True, seed=0)
        landscape.calc_visualization(Ns=1, n_components=5)
        nodes_df = landscape.nodes_df
        edges_df = landscape.edges_df
        
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_nodes(axes, nodes_df, size='f', color='white', lw=0.2)
        plot_edges(axes, nodes_df, edges_df)
        savefig(fig, fpath)
        
    def test_plot_visualization_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        nodes_fpath = join(TEST_DATA_DIR, 'codon_landscape.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'codon_landscape.edges.csv')
        decay_fpath = join(TEST_DATA_DIR, 'codon_landscape.decay_rates.csv')
        plot_fpath = join(TEST_DATA_DIR, 'codon_landscape') 
        
        # Test bin
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
        
        # Test visualization
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath]
        check_call(cmd)
        
        # Highlighting peaks in nucleotide sequence
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-g', 'UCN,AGY', '-A', 'rna']
        check_call(cmd)
        
        # Highlighting coding sequence
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath,
               '-g', 'S,L', '--protein_seq', '-l', 'log(binding)',
               '-A', 'protein']
        check_call(cmd)
        
        # Screeplot for decay rates
        bin_fpath = join(BIN_DIR, 'plot_decay_rates.py')
        plot_fpath = join(TEST_DATA_DIR, 'codon_landscape.decay_rates') 
        cmd = [sys.executable, bin_fpath, decay_fpath, '-o', plot_fpath]
        check_call(cmd)
        
    def test_interactive_plot(self):
        landscape = Visualization(length=3, n_alleles=4)
        landscape.set_random_function(0)
        landscape.calc_visualization(n_components=20)
        landscape.plot_interactive_2d(fname='test_interactive_2d', show_edges=True)
        landscape.plot_interactive_3d(fname='test_interactive_3d', show_edges=True,
                                      force_coords=True)
    
    def test_visualize_reactive_paths(self):
        np.random.seed(0)
        landscape = CodonFitnessLandscape(add_variation=True)
        Ns = landscape.calc_Ns(stationary_function=1.3)
        landscape.calc_stationary_frequencies(Ns)
        landscape.calc_visualization(Ns, n_components=5)
        
        gt1, gt2 = ['UCU', 'UCA', 'UCC', 'UCG'], ['AGU', 'AGC']
        fpath = join(TEST_DATA_DIR, 'reactive_path')
        # landscape.figure(fpath=fpath, size=40, cmap='coolwarm', 
        #                  genotypes1=gt1, genotypes2=gt2, figsize=(8, 6),
        #                  dominant_paths=False, p_reactive_paths=True)
        
        landscape.figure(fpath=fpath, size=40, cmap='coolwarm', 
                         genotypes1=gt1, genotypes2=gt2, figsize=(8, 6),
                         dominant_paths=True, edge_widths=2,
                         edges_cmap='Greens', p_reactive_paths=True)
    
    def test_laplacian(self):
        gpmap = Visualization(4, 2)
        gpmap.calc_laplacian()
        gpmap.T = gpmap.L
        assert(np.all(gpmap.T.diagonal() == 4))
        
        gpmap.set_random_function(seed=0)
        gpmap.filter_genotypes(np.where(gpmap.f > 0.5)[0])
        
        assert(gpmap.T.shape[0] < 16)
        assert(np.all(gpmap.f > 0.5))
    
    def test_laplacian_visualization(self):
        gpmap = Visualization(5, 4)
        gpmap.set_random_function(0)
        gpmap.calc_laplacian_visualization(selected_genotypes=gpmap.f > 0)
        gpmap.plot_visual(fname='laplacian_test', show_labels=False)
    
    def test_visualization_colors(self):
        gpmap = Visualization(3, 4)
        gpmap.set_random_function(0)
        gpmap.calc_visualization()
        gpmap.plot_visual(fname='test_colors', size=25,
                          color_key=lambda x: 'orange' if x.startswith('A') else 'black')
    
    def test_visualization_grid_allele(self):
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_visualization(Ns=1)
        
        fpath = join(TEST_DATA_DIR, 'codon_landscape.alleles')
        figure_allele_grid(v.nodes_df, edges_df=v.edges_df, fpath=fpath)
    
    def test_figure_Ns_grid(self):
        log = LogTrack()
        np.random.seed(5)
        length = 7
        lambdas = np.array([0, 1e6, 1e5, 1e4,
                            1e3, 1e2, 1e1, 1e0,])
    
        log.write('Simulate data')
        vc = VCregression(length, n_alleles=4, log=log)
        landscape = Visualization(length, log=log)
        landscape.set_function(vc.simulate(lambdas))
        
        fig_fpath = join(TEST_DATA_DIR, 'fgrid')
        figure_Ns_grid(landscape, fig_fpath, ncol=3)
    
    def test_get_edges(self):
        landscape = Visualization(length=3, n_alleles=4)
        landscape.set_random_function(0)
        landscape.calc_visualization(n_components=20)
        x, y, z = landscape.get_nodes_coord(z=4)
        edges = landscape.get_edges_coord(x, y, z)
        assert(np.all(edges.shape == (576, 2, 3)))
    
    def test_rotate_coords(self):
        np.random.seed(0)
        
        landscape = CodonFitnessLandscape(add_variation=True)
        landscape.calc_visualization()
        c = np.vstack([landscape.axis['Axis 2'],
                       landscape.axis['Axis 3'],
                       landscape.axis['Axis 4']])
        c2 = landscape.rotate_coords(c, theta=np.pi / 2, axis='x')
        assert(np.allclose(c[0], c2[0]))
        assert(np.allclose(c[1], c2[2]))
        assert(np.allclose(c[2], -c2[1]))
        
        c2 = landscape.rotate_coords(c, theta=0, axis='x')
        assert(np.allclose(c, c2))
        
    def test_smn1_visualization(self):
        fpath = join(DATA_DIR, 'smn1.csv')
        data = pd.read_csv(fpath)
        data.loc[data['phenotype'] > 100, 'phenotype'] = 100
        landscape = Visualization(length=8, n_alleles=4, label='smn1',
                                  ns=1, cache_prefix='smn1')
        landscape.load_function(data['phenotype'])
        landscape.calc_stationary_frequencies()
        landscape.tune_ns(stationary_function=80)
        landscape.calc_visualization(n_components=10, recalculate=True)
        landscape.plot_visual()
    
    def test_rotation_movie(self):
        np.random.seed(0)
    
        landscape = CodonFitnessLandscape(add_variation=True)
        landscape.calc_visualization(Ns=1)
        dpath = join(TEST_DATA_DIR, 'rotation_movie')
        landscape.plot_rotation_movie(dpath=dpath, nframes=60,
                                      lims=(-2, 2), force=True)
    
    def test_ns_movie(self):
        np.random.seed(0)
        landscape = CodonFitnessLandscape(add_variation=True)
        dpath = join(TEST_DATA_DIR, 'ns_movie')
        landscape.plot_ns_movie(dpath, nframes=60, fmax=1.64,
                                n_components=20, force=True)

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VisualizationTests']
    unittest.main()
