#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd

from settings import DATA_DIR
from visualization import Visualization, CodonFitnessLandscape
from utils import LogTrack


class VisualizationTests(unittest.TestCase):
    def test_get_neighbors(self):
        space = Visualization(3, 4, alphabet_type='rna')
        seq = 'AAA'
        
        idxs = space.get_neighborhood_idxs(seq, max_distance=1)
        seqs = space.get_seq_from_idx(idxs)
        assert(np.all(seqs == ['AAA', 'AAC', 'AAG', 'AAU', 'ACA',
                               'AGA', 'AUA', 'CAA', 'GAA', 'UAA']))
        
        idxs = space.get_neighborhood_idxs(seq, max_distance=2)
        seqs = space.get_seq_from_idx(idxs)
        for seq in seqs:
            assert('A' in seq)
        
    def test_calc_stationary_frequencies(self):
        gpmap = Visualization(2, 2, ns=2)
        gpmap.load_function([2, 1, 1, 1])
        gpmap.calc_stationary_frequencies()
        fmean = gpmap.tune_ns(stationary_function=1.5)
        
        # Ensure optimization works well
        assert(np.abs(fmean - 1.5) < 1e-4)
        
        # Ensure Ns has been updated in the object
        assert(gpmap.ns != 2)
        
        # Try in a bigger landscape
        gpmap = Visualization(8, 4, ns=2)
        gpmap.set_random_function()
        gpmap.calc_stationary_frequencies()
        fmean = gpmap.tune_ns(stationary_function=1.5)
        assert(np.abs(fmean - 1.5) < 1e-4)
    
    def test_calc_transition_p(self):
        gpmap = Visualization(2, 2, ns=1)
        gpmap.load_function([1.05, 1, 1, 1])
        gpmap.calc_stationary_frequencies()
        
        # Calculate re-weighted transition matrix  
        gpmap.calc_reweighting_diag_matrices()
        t = gpmap.calc_sparse_reweighted_transition_matrix()
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
        
    def test_interactive_plot(self):
        landscape = Visualization(length=3, n_alleles=4)
        landscape.set_random_function(0)
        landscape.calc_visualization(n_components=20)
        landscape.plot_interactive_2d(fname='test_interactive_2d', show_edges=True)
        landscape.plot_interactive_3d(fname='test_interactive_3d', show_edges=True,
                                      force_coords=True)
    
    def test_codon_landscape(self):
        np.random.seed(0)
        landscape = CodonFitnessLandscape(add_variation=False)
        landscape.calc_stationary_frequencies()
        landscape.tune_ns(stationary_function=1.93)
        assert(np.all(landscape.genotypes_stationary_frequencies > 0))
        assert(np.allclose(np.sum(landscape.genotypes_stationary_frequencies), 1))

        landscape.calc_visualization(n_components=5, recalculate=True)
        landscape.figure(fname='codon_landscape', size=40)
        landscape.figure(fname='codon_landscape_3d', size=40, z=3)
    
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
        gpmap = Visualization(3, 4, label='test')
        gpmap.set_random_function(0)
        gpmap.calc_visualization()
        gpmap.plot_grid_allele(size=25)
    
    def test_eq_grid(self):
        gpmap = Visualization(3, 4)
        gpmap.set_random_function(0)
        gpmap.plot_grid_eq_f('test_eq_f_grid', fmin=0, fmax=2,
                             ncol=3, nrow=3,  size=25)
        
        np.random.seed(0)
        landscape = CodonFitnessLandscape(add_variation=True)
        landscape.plot_grid_eq_f(fname='codon_landscape.ns', size=40,
                                 n_components=50, fmin=1)
    
    def test_gpmap_cache(self):
        # Store transition matrix in cache file
        gpmap = Visualization(2, 2, ns=1, cache_prefix='test')
        gpmap.load_function([1.01, 1, 1, 1.01])
        gpmap.calc_visualization(recalculate=True)
        
        # Load now cached transition matrix
        log = LogTrack()
        gpmap = Visualization(2, 2, ns=1, cache_prefix='test', log=log)
        gpmap.load_function([1.01, 1, 1, 1.01])
        gpmap.calc_visualization(recalculate=False)
        
        # Ensure it T was loaded from file
        assert(gpmap.cached_T)
        assert(gpmap.cached_eigenvectors)
        assert(gpmap.cached_eigenvalues)
        
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
        landscape.calc_visualization()
        landscape.plot_rotation_movie(fdir='codon_rotation',
                                      lims=(-2, 2), force=True)
    
    def test_ns_movie(self):
        np.random.seed(0)
        landscape = CodonFitnessLandscape(add_variation=False)
        landscape.plot_ns_movie(fdir='codon_ns', nframes=120, fmax=1.64,
                                n_components=50, force=True)
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VisualizationTests']
    unittest.main()
