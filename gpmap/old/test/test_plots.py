#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd

from gpmap.visualization import Visualization, CodonFitnessLandscape
from gpmap.utils import LogTrack, guess_configuration
from gpmap.inference import VCregression
from gpmap.settings import TEST_DATA_DIR, BIN_DIR
from subprocess import check_call
from gpmap.plot import (plot_nodes, plot_edges, figure_visualization,
                        plot_decay_rates, figure_Ns_grid,
                        init_fig, savefig, figure_allele_grid,
                        figure_shifts_grid)
        

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
        fpath = join(TEST_DATA_DIR, 'codon_v')
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_visualization(Ns=1, n_components=5)
        nodes_df = v.nodes_df
        edges_df = v.edges_df
        
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_nodes(axes, nodes_df, size='f', color='white', lw=0.2)
        plot_edges(axes, nodes_df, edges_df)
        savefig(fig, fpath)
        
        # Test with centering of function in color scale
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_nodes(axes, nodes_df, size='f', lw=0.2, vcenter=0)
        plot_edges(axes, nodes_df, edges_df)
        savefig(fig, fpath)
    
    def test_codon_visualization(self):
        fig_fpath = join(TEST_DATA_DIR, 'codon_landscape')
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_visualization(Ns=1, n_components=25)
        figure_visualization(v.nodes_df, v.edges_df,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')

        # Test whether saving the projections works        
        prefix = join(TEST_DATA_DIR, 'codon_landscape')
        v.write_tables(prefix)
        
        nodes_df = pd.read_csv('{}.nodes.csv'.format(prefix), index_col=0)
        edges_df = pd.read_csv('{}.edges.csv'.format(prefix))
        decay_df = pd.read_csv('{}.decay_rates.csv'.format(prefix))
        
        figure_visualization(nodes_df, edges_df, nodes_size=50,
                             fpath=fig_fpath, highlight_genotypes=['UCN', 'AGY'],
                             palette='Set1', alphabet_type='rna')
        
        fpath = join(TEST_DATA_DIR, 'codon_landscape.decay_rates')
        plot_decay_rates(decay_df, fpath=fpath)
        
    def test_plot_visualization_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        nodes_fpath = join(TEST_DATA_DIR, 'codon_v.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'codon_v.edges.csv')
        decay_fpath = join(TEST_DATA_DIR, 'codon_v.decay_rates.csv')
        plot_fpath = join(TEST_DATA_DIR, 'codon_v') 
        
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
        plot_fpath = join(TEST_DATA_DIR, 'codon_v.decay_rates') 
        cmd = [sys.executable, bin_fpath, decay_fpath, '-o', plot_fpath]
        check_call(cmd)
        
    def test_interactive_plot(self):
        v = Visualization(length=3, n_alleles=4)
        v.set_random_function(0)
        v.calc_visualization(n_components=20)
        v.plot_interactive_2d(fname='test_interactive_2d', show_edges=True)
        v.plot_interactive_3d(fname='test_interactive_3d', show_edges=True,
                                      force_coords=True)
    
    def test_visualize_reactive_paths(self):
        np.random.seed(0)
        v = CodonFitnessLandscape(add_variation=True)
        Ns = v.calc_Ns(stationary_function=1.3)
        v.calc_stationary_frequencies(Ns)
        v.calc_visualization(Ns, n_components=5)
        
        gt1, gt2 = ['UCU', 'UCA', 'UCC', 'UCG'], ['AGU', 'AGC']
        fpath = join(TEST_DATA_DIR, 'reactive_path')
        # v.figure(fpath=fpath, size=40, cmap='coolwarm', 
        #                  genotypes1=gt1, genotypes2=gt2, figsize=(8, 6),
        #                  dominant_paths=False, p_reactive_paths=True)
        
        v.figure(fpath=fpath, size=40, cmap='coolwarm', 
                         genotypes1=gt1, genotypes2=gt2, figsize=(8, 6),
                         dominant_paths=True, edge_widths=2,
                         edges_cmap='Greens', p_reactive_paths=True)
    
    def test_laplacian(self):
        v = Visualization(4, 2)
        v.calc_laplacian()
        v.T = v.L
        assert(np.all(v.T.diagonal() == 4))
        
        v.set_random_function(seed=0)
        v.filter_genotypes(np.where(v.f > 0.5)[0])
        
        assert(v.T.shape[0] < 16)
        assert(np.all(v.f > 0.5))
    
    def test_laplacian_visualization(self):
        v = Visualization(5, 4)
        v.set_random_function(0)
        v.calc_laplacian_visualization(selected_genotypes=v.f > 0)
        v.plot_visual(fname='laplacian_test', show_labels=False)
    
    def test_visualization_colors(self):
        v = Visualization(3, 4)
        v.set_random_function(0)
        v.calc_visualization()
        v.plot_visual(fname='test_colors', size=25,
                          color_key=lambda x: 'orange' if x.startswith('A') else 'black')
    
    def test_visualization_grid_allele(self):
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_visualization(Ns=1)
        
        fpath = join(TEST_DATA_DIR, 'codon_v.alleles')
        figure_allele_grid(v.nodes_df, edges_df=v.edges_df, fpath=fpath)
    
    def test_visualization_grid_shifts(self):
        v = CodonFitnessLandscape(add_variation=True, seed=0)
        v.calc_visualization(Ns=1)
        
        fpath = join(TEST_DATA_DIR, 'codon_v.shifts')
        figure_shifts_grid(v.nodes_df, seq='AU', edges_df=v.edges_df,
                           fpath=fpath, alphabet_type='rna',
                           labels_full_seq=True)
    
    def test_figure_Ns_grid(self):
        log = LogTrack()
        np.random.seed(1)
        length = 8
        lambdas = np.array([0, 1e6, 1e5, 1e4,
                            1e3, 1e2, 1e1, 1e0, 0])
    
        log.write('Simulate data')
        vc = VCregression(length, n_alleles=4, log=log, alphabet_type='custom')
        v = Visualization(length, log=log)
        v.set_function(vc.simulate(lambdas))
        
        fig_fpath = join(TEST_DATA_DIR, 'fgrid')
        figure_Ns_grid(v, fig_fpath, ncol=3)
    
    def test_get_edges(self):
        v = Visualization(length=3, n_alleles=4)
        v.set_random_function(0)
        v.calc_visualization(n_components=20)
        x, y, z = v.get_nodes_coord(z=4)
        edges = v.get_edges_coord(x, y, z)
        assert(np.all(edges.shape == (576, 2, 3)))
    
    def test_rotate_coords(self):
        np.random.seed(0)
        
        v = CodonFitnessLandscape(add_variation=True)
        v.calc_visualization()
        c = np.vstack([v.axis['Axis 2'],
                       v.axis['Axis 3'],
                       v.axis['Axis 4']])
        c2 = v.rotate_coords(c, theta=np.pi / 2, axis='x')
        assert(np.allclose(c[0], c2[0]))
        assert(np.allclose(c[1], c2[2]))
        assert(np.allclose(c[2], -c2[1]))
        
        c2 = v.rotate_coords(c, theta=0, axis='x')
        assert(np.allclose(c, c2))
        
    def test_smn1_visualization(self):
        fpath = join(DATA_DIR, 'smn1.csv')
        data = pd.read_csv(fpath)
        data.loc[data['phenotype'] > 100, 'phenotype'] = 100
        v = Visualization(length=8, n_alleles=4, label='smn1',
                                  ns=1, cache_prefix='smn1')
        v.load_function(data['phenotype'])
        v.calc_stationary_frequencies()
        v.tune_ns(stationary_function=80)
        v.calc_visualization(n_components=10, recalculate=True)
        v.plot_visual()
    
    def test_rotation_movie(self):
        np.random.seed(0)
    
        v = CodonFitnessLandscape(add_variation=True)
        v.calc_visualization(Ns=1)
        dpath = join(TEST_DATA_DIR, 'rotation_movie')
        v.plot_rotation_movie(dpath=dpath, nframes=60,
                                      lims=(-2, 2), force=True)
    
    def test_ns_movie(self):
        np.random.seed(0)
        v = CodonFitnessLandscape(add_variation=True)
        dpath = join(TEST_DATA_DIR, 'ns_movie')
        v.plot_ns_movie(dpath, nframes=60, fmax=1.64,
                                n_components=20, force=True)

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'PlottingTests.test_visualization_grid_shifts']
    unittest.main()
