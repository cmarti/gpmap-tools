#!/usr/bin/env python
import unittest
import sys

import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

import gpmap.src.plot.mpl as pmpl
import gpmap.src.plot.ds as pds
import gpmap.src.plot.ply as ply

from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR, VIZ_DIR
from gpmap.src.genotypes import select_genotypes
from gpmap.src.randwalk import WMWalk
from gpmap.src.plot.utils import get_lines_from_edges_df
from gpmap.src.datasets import DataSet
        

class MatPlotsTests(unittest.TestCase):
    def test_get_lines_from_edges_df(self):
        nodes_df = pd.DataFrame({'1': [0, 1, 2],
                                 '2': [1, 2, 3],
                                 '3': [2, 3, 4]})
        edges_df = pd.DataFrame({'i': [0, 1],
                                 'j': [1, 2]})
        
        # Test with two axis
        line_coords = get_lines_from_edges_df(nodes_df, edges_df, x='1', y='2')
        exp_x = [1, 0, np.nan, 2, 1, np.nan]
        exp_y = [2, 1, np.nan, 3, 2, np.nan]
        for a, b, c, d in zip(line_coords[:, 0], exp_x, line_coords[:, 1], exp_y):
            assert(a == b or (np.isnan(a) and np.isnan(b)))
            assert(c == d or (np.isnan(c) and np.isnan(d)))
            
        # Test with 3 axis
        line_coords = get_lines_from_edges_df(nodes_df, edges_df, x='1', y='2', z='3')
        exp_x = [1, 0, np.nan, 2, 1, np.nan]
        exp_y = [2, 1, np.nan, 3, 2, np.nan]
        exp_z = [3, 2, np.nan, 4, 3, np.nan]
        for a, b, c, d, e, f in zip(line_coords[:, 0], exp_x, line_coords[:, 1],
                                    exp_y, line_coords[:, 2], exp_z):
            assert(a == b or (np.isnan(a) and np.isnan(b)))
            assert(c == d or (np.isnan(c) and np.isnan(d)))
            assert(e == f or (np.isnan(e) and np.isnan(f)))
    
    def test_plot_hist(self):
        ser = DataSet('serine')

        fig, axes = pmpl.init_fig(1, 1, colsize=3, rowsize=1.5)
        pmpl.plot_color_hist(axes, ser.nodes, bins=20)
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath)
    
    def test_draw_cbar(self):
        fig, axes = pmpl.init_fig(1, 1, figsize=(3, 3))
        pmpl.draw_cbar(axes, cmap='viridis', label='Function',
                       orientation='vertical')
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath, tight=False)
            
        fig, axes = pmpl.init_fig(1, 1, figsize=(3, 3))
        pmpl.draw_cbar(axes, cmap='viridis', label='Function',
                       orientation='horizontal')
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath, tight=False)
    
    def test_add_cbar_hist_inset(self):
        fig, axes = pmpl.init_fig(1, 1, figsize=(5, 4))
        values = np.random.normal(0, 1, size=1000)
        pmpl.add_cbar_hist_inset(axes, values,
                                 pos=(0.6, 0.1), fontsize=8,
                                 width=0.4, height=0.2, bins=20)
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath, tight=False)

    def test_plot_nodes(self):
        ser = DataSet('serine')

        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_nodes(axes, ser.nodes, size='function', color='white', lw=0.2)
        
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
            
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_nodes(axes, ser.nodes, color='white', size='function',
                        lw=0.2, vcenter=0)
        
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
    
    def test_plot_edges(self):
        ser = DataSet('serine')

        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_edges(axes, ser.nodes, ser.edges)
        
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
    
    def test_plot_visualization(self):
        ser = DataSet('serine')

        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_visualization(axes, ser.nodes, ser.edges)
        
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
            
        fig, axes = pmpl.init_fig(1, 1, colsize=5, rowsize=4)
        pmpl.plot_visualization(axes, ser.nodes, ser.edges, center_spines=True,
                                nodes_size=30, add_hist=True, inset_cbar=True)
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath, tight=False)
        
    def test_plot_genotypes_box(self):
        fig, axes = pmpl.init_fig(1, 1)
        pmpl.plot_genotypes_box(axes, (0, 1), (0, 1), c='black', title='bigbox',
                                title_pos='top')
        pmpl.plot_genotypes_box(axes, (0.2, 0.4), (0.1, 0.3), c='red',
                                title='inner box', title_pos='right')
        pmpl.plot_genotypes_box(axes, (-0.5, 0.3), (-0.1, 0.2), c='blue',
                           title='other box', title_pos='bottom')
        axes.set(xlim=(-1, 2), ylim=(-1, 2))
        
        with NamedTemporaryFile('w') as fhand:
            fig.savefig(fhand.name)
    
    def test_Ns_grid(self):
        ser = DataSet('serine')
        rw = WMWalk(ser.to_sequence_space())
        
        with NamedTemporaryFile() as fhand:
            fpath  = fhand.name
            pmpl.figure_Ns_grid(rw, fpath=fpath)
    
    def test_axis_grid(self):
        ser = DataSet('serine')

        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.figure_axis_grid(ser.nodes, max_axis=4, edges_df=ser.edges,
                                  fpath=fpath, fmt='png')
    
    def test_alleles_grid(self):  
        ser = DataSet('serine')

        # Test with all alleles per site
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.figure_allele_grid(ser.nodes, edges_df=ser.edges, 
                                   fpath=fpath, nodes_size=40)
        
        # Test with different number of alleles per site
        genotypes = np.array([seq[-3] != 'C' for seq in ser.nodes.index])
        ndf, edf = select_genotypes(ser.nodes, genotypes, edges=ser.edges)
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.figure_allele_grid(ndf, edges_df=edf, fpath=fpath, nodes_size=40)
    
    def test_plot_relaxation_times(self):    
        ser = DataSet('serine')
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.plot_relaxation_times(ser.relaxation_times, fpath=fpath)
            pmpl.plot_relaxation_times(ser.relaxation_times, neutral_time=1/4, 
                                       fpath=fpath)
    
    def test_plot_visualization_bin_help(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        cmd = [sys.executable, bin_fpath, '-h']
        check_call(cmd)
        
    def test_plot_visualization_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        
        nodes_fpath = join(VIZ_DIR, 'serine.nodes.pq')
        edges_fpath = join(VIZ_DIR, 'serine.edges.npz')
        
        with NamedTemporaryFile('w') as fhand:
            plot_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath, '-nc', 'function', '-s', 'function']
            check_call(cmd)
        
        # Highlighting peaks in nucleotide sequence
        with NamedTemporaryFile('w') as fhand:
            plot_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath, '-g', 'TCN,AGY', '-A', 'rna',
                   '-nc', 'function', '-s', 'function']
            check_call(cmd)
        
        # Highlighting coding sequence
        with NamedTemporaryFile('w') as fhand:
            plot_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath,
                   '-g', 'S,L', '--protein_seq', '-l', 'log(binding)',
                   '-A', 'protein', '-nc', 'function', '-s', 'function']
            check_call(cmd)
        
        # Interactive
        with NamedTemporaryFile('w') as fhand:
            plot_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath, '-nc', 'function', '-s', 'function',
                   '--interactive']
            check_call(cmd)
    
    def test_plot_visualization_bin_alleles(self):    
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        
        nodes_fpath = join(TEST_DATA_DIR, 'serine.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'serine.edges.npz')
        
        plot_fpath = join(TEST_DATA_DIR, 'serine.alleles')
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'function', '-s', 'function', '--alleles']
        check_call(cmd)
        
    def test_plot_relaxation_times_bin(self):    
        bin_fpath = join(BIN_DIR, 'plot_relaxation_times.py')
        decay_fpath = join(TEST_DATA_DIR, 'serine.decay_rates.csv')
        plot_fpath = join(TEST_DATA_DIR, 'serine.decay_rates') 
        cmd = [sys.executable, bin_fpath, decay_fpath, '-o', plot_fpath]
        check_call(cmd)
        
        
class DatashaderTests(unittest.TestCase):
    def test_plot_visualization(self):
        ser = DataSet('serine')
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
        
            # Only nodes
            dsg =  pds.plot_visualization(ser.nodes, shade_nodes=True)
            pds.savefig(dsg, fpath)
            
            # With edges
            dsg =  pds.plot_visualization(ser.nodes, edges_df=ser.edges,
                                          shade_nodes=True, shade_edges=True)
            pds.savefig(dsg, fpath)
            
            # Test without shading
            dsg =  pds.plot_visualization(ser.nodes, edges_df=ser.edges,
                                          shade_nodes=False, shade_edges=False)
            pds.savefig(dsg, fpath)
        
    def test_plot_visualization_big(self):  
        gb1 = DataSet('gb1')
        dsg =  pds.plot_visualization(gb1.nodes, edges_df=gb1.edges,
                                      shade_nodes=True, shade_edges=True)
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.savefig(dsg, fpath)
        
    def test_alleles_grid(self):  
        ser = DataSet('serine')

        # Test with all alleles per site
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.figure_allele_grid(ser.nodes, edges_df=ser.edges, 
                                   fpath=fpath, nodes_size=40)
        
        # Test with different number of alleles per site
        genotypes = np.array([seq[-3] != 'C' for seq in ser.nodes.index])
        ndf, edf = select_genotypes(ser.nodes, genotypes, edges=ser.edges)
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.figure_allele_grid(ndf, edges_df=edf, fpath=fpath, nodes_size=40)
        
    
    def test_plot_visualization_bin_datashader(self):
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        nodes_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.nodes.csv')
        edges_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.edges.npz')
        plot_fpath = join(TEST_DATA_DIR, 'dmsc.2.3.plot')
        
        cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
               '-o', plot_fpath, '-nc', 'f', '--datashader', '-nr', '800',
               '-er', '1800']
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
    

class PlotlyTests(unittest.TestCase):
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


class InferencePlotsTests(unittest.TestCase):
    def test_plot_a_optimization(self):
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        log_Ls = pd.read_csv(fpath, index_col=0)
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_hyperparam_cv(log_Ls, axes, err_bars='stderr',
                           x='log_sd', xlabel=r'$\log_{10}(\sigma_P)$')
        fpath = join(TEST_DATA_DIR, 'seqdeft_a')
        pmpl.savefig(fig, fpath)
    
    def test_plot_beta_optimization(self):    
        fpath = join(TEST_DATA_DIR, 'vc.cv_loss.csv')
        mses = pd.read_csv(fpath, index_col=0)
        print(mses)
        
        fig, subplots = pmpl.init_fig(1, 3, colsize=4, rowsize=3.5)
        pmpl.plot_hyperparam_cv(mses, subplots[0], err_bars='stderr',
                           x='log_beta', xlabel=r'$\log_{10}(\beta)$',
                           y='mse', ylabel='MSE', highlight='min')
        pmpl.plot_hyperparam_cv(mses, subplots[1], err_bars='stderr',
                           x='log_beta', xlabel=r'$\log_{10}(\beta)$',
                           y='logL', ylabel='log(L)')
        pmpl.plot_hyperparam_cv(mses, subplots[2], err_bars='stderr',
                           x='log_beta', xlabel=r'$\log_{10}(\beta)$',
                           y='r2', ylabel=r'$R^2$')
               
        fpath = join(TEST_DATA_DIR, 'vc_beta')
        pmpl.savefig(fig, fpath)
    
    def test_plot_SeqDEFT_summary(self):
        fpath = join(TEST_DATA_DIR, 'logL.csv')
        logl = pd.read_csv(fpath)
        
        fig = pmpl.plot_SeqDEFT_summary(logl)
        fpath = join(TEST_DATA_DIR, 'seqdeft_output.log_Ls.png')
        pmpl.savefig(fig, fpath)

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'MatPlotsTests']
    unittest.main()

