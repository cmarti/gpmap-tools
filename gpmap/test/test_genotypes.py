#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

from scipy.sparse.csr import csr_matrix

from gpmap.src.genotypes import (select_genotypes, dataframe_to_csr_matrix,
                                 select_d_neighbors, select_genotypes_re,
                                 select_genotypes_ambiguous_seqs,
                                 select_closest_genotypes, select_local_optima,
    marginalize_landscape_positions)


class GenotypeTests(unittest.TestCase):
    def test_dataframe_to_csr_matrix(self):
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2],
                                 'j': [1, 2, 0, 2, 0, 1]})
        m = dataframe_to_csr_matrix(edges_df).tocoo()
        assert(np.all(m.row == [0, 0, 1, 1, 2, 2]))
        assert(np.all(m.col == [1, 2, 0, 2, 0, 1]))
        assert(np.all(m.data == np.arange(6)))
        
    def test_select_genotypes_nodes(self):
        nodes_df = pd.DataFrame({'index': np.arange(3)}, index=['A', 'B', 'C'])
        genotypes = ['A', 'B']
        
        nodes_df = select_genotypes(nodes_df, genotypes)
        assert(np.all(nodes_df.index == ['A', 'B']))
        
    def test_select_genotypes_df(self):
        nodes_df = pd.DataFrame({'index': np.arange(3)}, index=['A', 'B', 'C'])
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2],
                                 'j': [1, 2, 0, 2, 0, 1],
                                 'id': np.arange(6)})
        genotypes = ['A', 'B']
        
        nodes_df, edges_df = select_genotypes(nodes_df, genotypes, edges=edges_df)
        assert(np.all(nodes_df.index == ['A', 'B']))
        assert(np.all(edges_df['i'] == [0, 1]))
        assert(np.all(edges_df['j'] == [1, 0]))
        assert(np.all(edges_df['id'] == [0, 2]))
    
    def test_select_genotypes_csr(self):
        nodes_df = pd.DataFrame({'index': np.arange(3)}, index=['A', 'B', 'C'])
        edges = csr_matrix((np.arange(6), ([0, 0, 1, 1, 2, 2],
                                           [1, 2, 0, 2, 0, 1])), shape=(3, 3))
        genotypes = ['A', 'B']
        
        nodes_df, edges = select_genotypes(nodes_df, genotypes, edges=edges)
        assert(np.all(nodes_df.index == ['A', 'B']))
        assert(np.all(edges.row == [0, 1]))
        assert(np.all(edges.col == [1, 0]))
        assert(np.all(edges.data == [0, 2]))
    
    def test_select_d_neighbors(self):
        nodes_df = pd.DataFrame({'index': np.arange(4)},
                                index=['AA', 'AB', 'BA', 'BB'])
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2, 3, 3],
                                 'j': [1, 2, 0, 3, 0, 3, 1, 2],
                                 'id': np.arange(8)})
        genotypes = ['AA']
        nodes_df, edges = select_d_neighbors(nodes_df, genotypes, d=1, edges=edges_df)
        assert(np.all(nodes_df.index == ['AA', 'AB', 'BA']))
        assert(np.all(edges['i'] == [0, 0, 1, 2]))
        assert(np.all(edges['j'] == [1, 2, 0, 0]))
        assert(np.all(edges['id'] == [0, 1, 2, 4]))
        
    def test_select_genotypes_re(self):
        nodes_df = pd.DataFrame({'index': np.arange(4)},
                                index=['AA', 'AB', 'BA', 'BB'])
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2, 3, 3],
                                 'j': [1, 2, 0, 3, 0, 3, 1, 2],
                                 'id': np.arange(8)})
        
        nodes_df, edges = select_genotypes_re(nodes_df, 'A', edges=edges_df)
        assert(np.all(nodes_df.index == ['AA', 'AB', 'BA']))
        assert(np.all(edges['i'] == [0, 0, 1, 2]))
        assert(np.all(edges['j'] == [1, 2, 0, 0]))
        assert(np.all(edges['id'] == [0, 1, 2, 4]))
        
        nodes_df, edges = select_genotypes_re(nodes_df, 'A[AB]', edges=edges_df)
        assert(np.all(nodes_df.index == ['AA', 'AB']))
        assert(np.all(edges['i'] == [0, 1]))
        assert(np.all(edges['j'] == [1, 0]))
        assert(np.all(edges['id'] == [0, 2]))
    
    def test_select_genotypes_ambiguous_seqs(self):
        nodes_df = pd.DataFrame({'index': np.arange(4)},
                                index=['A', 'C', 'G', 'T'])
        edges_df = pd.DataFrame({'i': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 'j': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
                                 'id': np.arange(12)})
        
        nodes_df, edges = select_genotypes_ambiguous_seqs(nodes_df, seqs='Y',
                                                          alphabet_type='dna',
                                                          edges=edges_df)
        assert(np.all(nodes_df.index == ['C', 'T']))
        assert(np.all(edges['i'] == [0, 1]))
        assert(np.all(edges['j'] == [1, 0]))
        assert(np.all(edges['id'] == [5, 10]))
    
    def test_select_closest_genotypes(self):
        nodes_df = pd.DataFrame({'index': np.arange(4),
                                 '1': [-1, 0, 1, 2],
                                 '2': [1, 0, -1, 0]},
                                index=['A', 'C', 'G', 'T'])
        
        edges_df = pd.DataFrame({'i': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 'j': [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
                                 'id': np.arange(12)})
        
        nodes_df, edges = select_closest_genotypes(nodes_df, genotype='A',
                                                   n_genotypes=2, edges=edges_df)
        assert(np.all(nodes_df.index == ['A', 'C']))
        assert(np.all(edges['i'] == [0, 1]))
        assert(np.all(edges['j'] == [1, 0]))
        assert(np.all(edges['id'] == [0, 3]))
    
    def test_select_local_optima(self):
        nodes_df = pd.DataFrame({'index': np.arange(4),
                                 'function': [3, 1, 1, 3]},
                                index=['AA', 'AB', 'BA', 'BB'])
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2, 3, 3],
                                 'j': [1, 2, 0, 3, 0, 3, 1, 2],
                                 'id': np.arange(8)})
        
        local_optima = select_local_optima(nodes_df, edges_df)
        assert(np.all(local_optima.index == ['AA', 'BB']))
        assert(np.all(local_optima['function'] == 3))
    
    def test_marginalize_positions(self):
        nodes_df = pd.DataFrame({'index': np.arange(4)},
                                index=['AA', 'AB', 'BA', 'BB'])
        ndf = marginalize_landscape_positions(nodes_df, keep_pos=[0])
        assert(np.allclose(ndf['index'],  [0.5, 2.5]))
        
        ndf, edf = marginalize_landscape_positions(nodes_df, keep_pos=[0],
                                                   return_edges=True)
        assert(np.allclose(edf,  [[0, 1], [1, 0]]))
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'GenotypeTests']
    unittest.main()
