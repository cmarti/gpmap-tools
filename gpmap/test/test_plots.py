#!/usr/bin/env python
import unittest
import sys

from os.path import join
from subprocess import check_call

import numpy as np
import pandas as pd

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.plot import (plot_holoview, get_lines_from_edges_df,
                            figure_allele_grid_datashader, plot_nodes,
                            plot_edges, savefig, init_fig, figure_visualization,
                            figure_allele_grid, save_holoviews,
                            plot_relaxation_times, plot_interactive,
                            figure_Ns_grid)
from gpmap.src.genotypes import select_genotypes
from gpmap.src.randwalk import WMWSWalk
from gpmap.src.space import CodonSpace
        

class PlottingTests(unittest.TestCase):
    def test_get_lines_from_edges_df(self):
        nodes_df = pd.DataFrame({'1': [0, 1, 2],
                                 '2': [1, 2, 3],
                                 '3': [2, 3, 4]})
        edges_df = pd.DataFrame({'i': [0, 1],
                                 'j': [1, 2]})
        
        # Test with two axis
        line_coords = get_lines_from_edges_df(nodes_df, edges_df, x='1', y='2')
        exp_x = [0, 1, np.nan, 1, 2, np.nan]
        exp_y = [1, 2, np.nan, 2, 3, np.nan]
        for a, b, c, d in zip(line_coords[:, 0], exp_x, line_coords[:, 1], exp_y):
            assert(a == b or (np.isnan(a) and np.isnan(b)))
            assert(c == d or (np.isnan(c) and np.isnan(d)))
            
        # Test with 3 axis
        line_coords = get_lines_from_edges_df(nodes_df, edges_df, x='1', y='2', z='3')
        exp_x = [0, 1, np.nan, 1, 2, np.nan]
        exp_y = [1, 2, np.nan, 2, 3, np.nan]
        exp_z = [2, 3, np.nan, 3, 4, np.nan]
        for a, b, c, d, e, f in zip(line_coords[:, 0], exp_x, line_coords[:, 1],
                                    exp_y, line_coords[:, 2], exp_z):
            assert(a == b or (np.isnan(a) and np.isnan(b)))
            assert(c == d or (np.isnan(c) and np.isnan(d)))
            assert(e == f or (np.isnan(e) and np.isnan(f)))
        
    def test_datashader_small(self):
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        # Test only with nodes
        dsg =  plot_holoview(nodes_df, nodes_color='function')
        save_holoviews(dsg, plot_fpath)
        
        # Test with edges
        dsg =  plot_holoview(nodes_df, edges_df=edges_df, nodes_color='function')
        save_holoviews(dsg, plot_fpath)
        
        # Test without shading
        dsg = plot_holoview(nodes_df, edges_df=edges_df,
                            nodes_color='function', linewidth=0,
                            nodes_size=20, nodes_vmin=-5,
                            shade_nodes=False, shade_edges=False)
        save_holoviews(dsg, plot_fpath)
        
    def test_datashader_big(self):  
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot.ds')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        dsg = plot_holoview(nodes_df, edges_df=edges_df, nodes_color='f',
                            nodes_cmap='viridis', edges_cmap='grey',
                            edges_resolution=1800, nodes_resolution=600)
        save_holoviews(dsg, plot_fpath)
    
    def test_datashader_big_vmax(self):  
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot.ds')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        dsg = plot_holoview(nodes_df, edges_df=edges_df, nodes_color='f',
                            nodes_cmap='viridis', edges_cmap='grey',
                            edges_resolution=1800, nodes_resolution=2000)
        save_holoviews(dsg, plot_fpath)
    
    def test_datashader_alleles(self):  
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot.alleles')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        figure_allele_grid_datashader(nodes_df, plot_fpath, edges_df=edges_df,
                                      x='1', y='2')
    
    def test_alleles_variable_sites(self):  
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot.alleles')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)

        # Test with all alleles per site
        figure_allele_grid(nodes_df, fpath=plot_fpath, edges_df=edges_df, x='1', y='2')
        
        # Test with different number of alleles per site
        genotypes = np.array([seq[-3] != 'C' for seq in nodes_df.index])
        nodes_df, edges_df = select_genotypes(nodes_df, genotypes, edges=edges_df)
        figure_allele_grid(nodes_df, fpath=plot_fpath, edges_df=edges_df, x='1', y='2')
        
    def test_datashader_alleles_variable_sites(self):  
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot.alleles')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        genotypes = np.array([seq.endswith('AA') and seq[-3] != 'C'
                              for seq in nodes_df.index])
        nodes_df, edges_df = select_genotypes(nodes_df, genotypes, edges=edges_df)
        
        figure_allele_grid_datashader(nodes_df, plot_fpath, edges_df=edges_df,
                                      x='1', y='2')
    
    def test_plotting(self):
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_nodes(axes, nodes_df, size='function', color='white', lw=0.2)
        plot_edges(axes, nodes_df, edges_df)
        savefig(fig, plot_fpath)
        
        # Test with centering of function in color scale
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
        plot_nodes(axes, nodes_df, color='white', size='function', lw=0.2, vcenter=0)
        plot_edges(axes, nodes_df, edges_df)
        savefig(fig, plot_fpath)
    
    def test_Ns_grid(self):
        rw = WMWSWalk(CodonSpace(['S'], add_variation=True, seed=0))
        fpath = join(TEST_DATA_DIR, 'serine.Ns')
        figure_Ns_grid(rw, fpath)
    
    def test_figure_visualization(self):
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        figure_visualization(nodes_df, edges_df, nodes_color='function',
                             fpath=plot_fpath, highlight_genotypes=['TCN', 'AGY'],
                             palette='Set1', alphabet_type='dna')
    
    def test_figure_visualization_big(self):
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.csv')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        figure_visualization(nodes_df, edges_df, nodes_color='f',
                             fpath=plot_fpath)
        
    def test_plot_visualization_bin_help(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
        
    def test_plot_visualization_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.npz')
        
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'function', '-s', 'function']
        check_call(cmd)
        
        # Highlighting peaks in nucleotide sequence
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot.2sets')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-g', 'TCN,AGY', '-A', 'rna',
               '-nc', 'function', '-s', 'function']
        check_call(cmd)
        
        # Highlighting coding sequence
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot.aa')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath,
               '-g', 'S,L', '--protein_seq', '-l', 'log(binding)',
               '-A', 'protein', '-nc', 'function', '-s', 'function']
        check_call(cmd)
        
        # Interactive
        plot_fpath = join(TEST_DATA_DIR, 'serine.plot')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'function', '-s', 'function',
               '--interactive']
        check_call(cmd)
    
    def test_plot_visualization_bin_datashader(self):
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.npz')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot')
        
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'f', '--datashader', '-nr', '800',
               '-er', '1800']
        check_call(cmd)
        
    def test_plot_visualization_bin_alleles(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.npz')
        
        plot_fpath = join(TEST_DATA_DIR, 'serine.alleles')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'function', '-s', 'function', '--alleles']
        check_call(cmd)
    
    def test_plot_visualization_bin_datashader_alleles(self):
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.npz')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot')
        
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'f', '--datashader', '-nr', '800',
               '-er', '1800', '--alleles']
        check_call(cmd)
    
    def test_plot_relaxation_times(self):    
        decay_fpath = join(TEST_DATA_DIR, 'serine.decay_rates.csv')
        fpath = join(TEST_DATA_DIR, 'serine.decay_rates')
        df = pd.read_csv(decay_fpath)
        plot_relaxation_times(df, fpath=fpath, neutral_time=1/4)
    
    def test_plot_relaxation_times_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_relaxation_times.py')
        decay_fpath = join(TEST_DATA_DIR, 'serine.decay_rates.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.decay_rates') 
        cmd = [sys.executable, bin_fpath, decay_fpath, '-o', plot_fpath]
        check_call(cmd)
        
    def test_interactive_plot(self):
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.csv')
        nodes_df = pd.read_csv(nodes_fpath, index_col=0)
        edges_df = pd.read_csv(edges_fpath)
        
        fpath = join(TEST_DATA_DIR, 'serine.interactive2d')
        plot_interactive(nodes_df, edges_df=edges_df, fpath=fpath,
                         nodes_color='function', nodes_size=10,
                         edges_width=1)
        
        fpath = join(TEST_DATA_DIR, 'serine.interactive3d')
        plot_interactive(nodes_df, edges_df=edges_df, fpath=fpath,
                         nodes_color='function', nodes_size=10,
                         edges_width=1, z='3')
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'PlottingTests']
    unittest.main()
