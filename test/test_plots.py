#!/usr/bin/env python
import unittest
import sys

import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

import gpmap.plot.mpl as pmpl
import gpmap.plot.ds as pds
import gpmap.plot.ply as ply

from gpmap.settings import BIN_DIR, VIZ_DIR
from gpmap.genotypes import select_genotypes
from gpmap.randwalk import WMWalk
from gpmap.plot.utils import get_lines_from_edges_df
from gpmap.datasets import DataSet
from gpmap.plot.mpl import raster_edges, raster_nodes, calc_raster
from gpmap.utils import write_dataframe, write_edges
        

class MatPlotsTests(unittest.TestCase):
    def xtest_raster_nodes(self):
        nodes_df = pd.DataFrame({'1': [0, 1, 3],
                                 '2': [0, 3, 1],
                                 'function': [1, 1, 2]})
        z = raster_nodes(nodes_df, resolution=4, color=None)
        assert(np.all(z == np.array([[0., 1., 0., 0.], 
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 1.],
                                     [1., 0., 0., 0.]])))
        
        z = raster_nodes(nodes_df, resolution=4)
        assert(np.all(z == np.array([[0., 1., 0., 0.], 
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 2.],
                                     [1., 0., 0., 0.]])))
        
        z = raster_nodes(nodes_df, resolution=4, only_first=True)
        assert(np.all(z == np.array([[0., 1., 0., 0.], 
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 2.],
                                     [1., 0., 0., 0.]])))
        
    
    def xtest_raster_edges(self):
        nodes_df = pd.DataFrame({'1': [0, 1, 3],
                                 '2': [0, 3, 1]})
        edges_df = pd.DataFrame({'i': [0, 0],
                                 'j': [1, 2]})
        z = raster_edges(nodes_df, edges_df, aa=False, resolution=4)
        assert(np.all(z == np.array([[0., 1., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [1., 0., 1., 1.],
                                     [2., 1., 0., 0.]])))
        
        z = raster_edges(nodes_df, edges_df, aa=False, resolution=4,
                         only_first=True)
        assert(np.all(z == np.array([[0., 1., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [1., 0., 1., 1.],
                                     [1., 1., 0., 0.]])))
    
    def xtest_plot_rasterized_visualization(self):
        ser = DataSet('serine')

        nodes_raster, edges_raster, extent = calc_raster(ser.nodes, edges_df=ser.edges, 
                                                         nodes_resolution=100,
                                                         edges_resolution=200)

        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_visualization_raster(axes, nodes_raster, extent,
                                       edges_raster=edges_raster,
                                       inset_cbar=False)
        
        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath)
        
        
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

        # Plot with a fixed color
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_nodes(axes, ser.nodes, size='function', color='white', lw=0.2)
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
            
        # Plot with a colormap
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_nodes(axes, ser.nodes, color='white', size='function',
                        lw=0.2, vcenter=0)
        with NamedTemporaryFile('w') as fhand:
            pmpl.savefig(fig, fhand.name)
            
        # Plot with a qualitative palette
        ser.nodes['group'] = [x.startswith('A') for x in ser.nodes.index]
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_nodes(axes, ser.nodes, color='group', size='function',
                        lw=0.2, vcenter=0, palette='Set1')
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath)
    
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
            pmpl.figure_Ns_grid(rw, fpath=fpath, nrow=2, ncol=5)
    
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
        
        # Interactive
        with NamedTemporaryFile('w') as fhand:
            plot_fpath = fhand.name
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath, '-nc', 'function', '-s', 'function',
                   '--interactive']
            check_call(cmd)
            
        # Alleles
        with NamedTemporaryFile('w') as fhand:
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', plot_fpath, '-nc', 'function', '-s', 'function',
                   '--alleles']
            check_call(cmd)
    
    def test_plot_relaxation_times_bin(self):
        bin_fpath = join(BIN_DIR, 'plot_relaxation_times.py')
        ser = DataSet('serine')
        
        with NamedTemporaryFile() as fhand:
            input_fpath = '{}.decay_rates.csv'.format(fhand.name)
            ser.relaxation_times.to_csv(input_fpath, index=False)
            
            output_fpath = '{}.decay_rates.png'.format(fhand.name)
            cmd = [sys.executable, bin_fpath, input_fpath, '-o', output_fpath]
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
        
    def test_figure_allele_grid(self):  
        ser = DataSet('serine')

        # Test with all alleles per site
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.figure_allele_grid(ser.nodes, edges_df=ser.edges, fpath=fpath)
            
        # Sorting by function value
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.figure_allele_grid(ser.nodes, edges_df=ser.edges, fpath=fpath,
                                   sort_by='function', sort_ascending=True)
        
        # Test with different number of alleles per site
        genotypes = np.array([seq[-3] != 'C' for seq in ser.nodes.index])
        ndf, edf = select_genotypes(ser.nodes, genotypes, edges=ser.edges)
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.figure_allele_grid(ndf.copy(), edges_df=edf.copy(), fpath=fpath)
    
    def xtest_plot_visualization_big(self):  
        gb1 = DataSet('gb1')
        dsg =  pds.plot_visualization(gb1.nodes, edges_df=gb1.edges,
                                      shade_nodes=True, shade_edges=True,
                                      sort_by='3', sort_ascending=True,
                                      nodes_resolution=200)
        
        with NamedTemporaryFile('w') as fhand:
            fpath = fhand.name
            pds.savefig(dsg, fpath)    
    
    def test_plot_visualization_bin_datashader(self):
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        serine = DataSet('serine')
        
        with NamedTemporaryFile('w') as fhand:
            nodes_fpath = '{}.nodes.pq'.format(fhand.name)
            write_dataframe(serine.nodes, nodes_fpath)
            
            edges_fpath = '{}.edges.npz'.format(fhand.name)
            write_edges(serine.edges, edges_fpath)

            output_fpath = '{}.visualization.png'.format(fhand.name)
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', output_fpath, '--datashader',
                   '-nr', '800', '-er', '800']
            check_call(cmd)
    
    def test_plot_visualization_bin_datashader_alleles(self):
        bin_fpath = join(BIN_DIR, 'plot_visualization.py')
        serine = DataSet("serine")
        
        with NamedTemporaryFile('w') as fhand:
            nodes_fpath = '{}.nodes.pq'.format(fhand.name)
            write_dataframe(serine.nodes, nodes_fpath)
            
            edges_fpath = '{}.edges.npz'.format(fhand.name)
            write_edges(serine.edges, edges_fpath)

            output_fpath = '{}.visualization.png'.format(fhand.name)
            cmd = [sys.executable, bin_fpath, nodes_fpath, '-e', edges_fpath,
                   '-o', output_fpath, '--datashader',
                   '-nr', '800', '-er', '800', '--alleles']
            check_call(cmd)
    

class PlotlyTests(unittest.TestCase):
    def test_interactive_plot(self):
        ser = DataSet('serine')
        
        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            ply.plot_visualization(ser.nodes, edges_df=ser.edges, fpath=fpath,
                                   nodes_color='function', nodes_size=10,
                                   edges_width=1)
        
        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            ply.plot_visualization(ser.nodes, edges_df=ser.edges, fpath=fpath,
                                   nodes_color='function', nodes_size=10,
                                   edges_width=1, z='3')


class InferencePlotsTests(unittest.TestCase):
    def test_plot_a_optimization(self):
        log_a = np.linspace(-2, 2, 11)
        logL = np.exp(-log_a ** 2) - 5
        log_a = np.array([-np.inf] + list(log_a) + [np.inf])
        logL = np.array([-5] + list(logL) + [-5])
        df1 = pd.DataFrame({'log_a': log_a, 'logL': logL, 'fold': np.zeros(logL.shape)})
        df2 = pd.DataFrame({'log_a': log_a, 'logL': logL-1, 'fold': np.ones(logL.shape)})
        df3 = pd.DataFrame({'log_a': log_a, 'logL': logL+0.5, 'fold': np.ones(logL.shape) * 2})
        df = pd.concat([df1, df2, df3])
        
        fig, axes = pmpl.init_fig(1, 1, colsize=4, rowsize=3.5)
        pmpl.plot_hyperparam_cv(df, axes, err_bars='stderr',
                                x='log_a', xlabel=r'$\log_{10}(a)$')
        
        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            pmpl.savefig(fig, fpath)
    
    def test_figure_SeqDEFT_summary(self):
        log_a = np.linspace(-2, 2, 11)
        logL = np.exp(-log_a ** 2) - 5
        log_a = np.array([-np.inf] + list(log_a) + [np.inf])
        logL = np.array([-5] + list(logL) + [-5])
        df1 = pd.DataFrame({'log_a': log_a, 'logL': logL, 'fold': np.zeros(logL.shape)})
        df2 = pd.DataFrame({'log_a': log_a, 'logL': logL-1, 'fold': np.ones(logL.shape)})
        df3 = pd.DataFrame({'log_a': log_a, 'logL': logL+0.5, 'fold': np.ones(logL.shape) * 2})
        df = pd.concat([df1, df2, df3])

        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            fig = pmpl.figure_SeqDEFT_summary(df)
            pmpl.savefig(fig, fpath)

        with NamedTemporaryFile() as fhand:
            fpath = fhand.name
            fig = pmpl.figure_SeqDEFT_summary(df, normalize_logL=False)
            pmpl.savefig(fig, fpath)

        
if __name__ == '__main__':
    import sys
    sys.argv = ['', 'MatPlotsTests']
    unittest.main()
