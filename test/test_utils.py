#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

from tempfile import NamedTemporaryFile

from gpmap.utils import (get_CV_splits, edges_df_to_csr_matrix, read_edges,
                             write_edges, counts_to_seqs,
                             evaluate_predictions, get_training_p_splits,
                             generate_p_training_config)

class UtilsTests(unittest.TestCase):
    def test_dataframe_to_csr_matrix(self):
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2],
                                 'j': [1, 2, 0, 2, 0, 1]})
        m = edges_df_to_csr_matrix(edges_df).tocoo()
        assert(np.all(m.row == [0, 0, 1, 1, 2, 2]))
        assert(np.all(m.col == [1, 2, 0, 2, 0, 1]))
        assert(np.all(m.data == np.arange(6)))
    
    def test_edges_io(self):
        edges_df = pd.DataFrame({'i': [0, 0, 1, 1, 2, 2],
                                 'j': [1, 2, 0, 2, 0, 1]})
        
        with NamedTemporaryFile() as fhand:
            fpath = '{}.npz'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath)
            assert(np.all(edges_df == edf))
            
            fpath = '{}.pq'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath)
            assert(np.all(edges_df == edf))
            
            fpath = '{}.csv'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath)
            assert(np.all(edges_df == edf))
        
        edges_df = edges_df_to_csr_matrix(edges_df)
        with NamedTemporaryFile() as fhand:
            fpath = '{}.npz'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath, return_df=False)
            assert(np.all(edges_df.todense() == edf.todense()))
            
            fpath = '{}.pq'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath, return_df=False)
            assert(np.all(edges_df.todense() == edf.todense()))
            
            fpath = '{}.csv'.format(fhand.name)
            write_edges(edges_df, fpath, triangular=False)
            edf = read_edges(fpath, return_df=False)
            assert(np.all(edges_df.todense() == edf.todense()))
    
    def test_counts_to_seqs(self):
        X = ['AA', 'AB', 'BA', 'BB']
        y = [2, 3, 2, 1]
        X_seqs = ['AA', 'AA', 'AB', 'AB', 'AB', 'BA', 'BA', 'BB']
        
        # With integer counts
        seqs = counts_to_seqs(X, y)
        assert(np.all(seqs == X_seqs))
        
        # Fail with float counts
        y = [2, 3, 2, 1.2]
        try:
            seqs = counts_to_seqs(X, y)
            self.fail()
        except TypeError:
            pass

    def test_generate_p_training_config(self):
        config = generate_p_training_config(n_ps=3, n_reps=2)
        assert(np.all(config['id'] == np.arange(1, 7))) 
        assert(np.allclose(config['p'], [0.05, 0.05, 0.5, 0.5, 0.95, 0.95])) 
        assert(np.all(config['rep'] == [0, 1, 0, 1, 0, 1]))    

        config = generate_p_training_config(ps=np.array([0.2, 0.5, 0.8]), n_reps=2)
        assert(np.all(config['id'] == np.arange(1, 7))) 
        assert(np.allclose(config['p'], [0.2, 0.2, 0.5, 0.5, 0.8, 0.8])) 
        assert(np.all(config['rep'] == [0, 1, 0, 1, 0, 1]))    

    def test_get_CV_splits(self):
        np.random.seed(0)
        X = np.array(['A', 'B', 'C'])
        y = np.array([1, 2, 2])
        
        # Test with real values data
        splits = get_CV_splits(X, y, nfolds=3)
        for _, (x_train, y_train, _), (x_test, y_test, _) in splits:
            assert(x_train.shape[0] == 2)
            assert(y_train.shape[0] == 2)
            assert(x_test.shape[0] == 1)
            assert(y_test.shape[0] == 1)
             
    def test_get_training_p_splits(self):
        config = pd.DataFrame({'id': [0, 1], 'p': [0.5, 1], 'rep': [1, 1]})
        
        X = np.array(['AAG', 'AGG', 'ATG'])
        y = np.array([0, 1, 0])
        
        # Random sampling
        splits = list(get_training_p_splits(config, X=X, y=y))
        assert(len(splits) == 2)
        
        for _, train, test in splits:
            seqs = np.sort(np.append(train[0], test[0]))
            assert(np.all(X == seqs))
            
        # Limit amount of test data
        splits = list(get_training_p_splits(config, X=X, y=y, max_pred=1))
        for _, train, test in splits:
            assert(test[0].shape[0] <= 1)
            
            seqs = np.append(train[0], test[0])
            assert(np.all(np.isin(seqs, X)))
        
        # Fix test for the same replicate
        tests = [s[2] for s in get_training_p_splits(config, X=X, y=y, max_pred=1,
                                                     fixed_test=True)]
        assert(tests[0] == tests[1])
        
        # Capture error in arguments
        try:
            splits = list(get_training_p_splits(config, X=X, y=y, fixed_test=True))
            self.fail()
        except ValueError:
            pass
    
    def test_evaluate_predictions(self):
        seqs = ['AGG', 'ATG', 'AGG']
        test_pred_sets = [(1, pd.DataFrame({'ypred': [1.05]}, index=[seqs[0]])),
                          (2, pd.DataFrame({'ypred': [2]}, index=[seqs[1]])),
                          (3, pd.DataFrame({'ypred': [1.1, 2.2]}, index=seqs[:2]))]

        # Without experimental variance
        data = pd.DataFrame({'y': [1, 2, 1.1], 'seq': seqs}).set_index('seq')        
        df = evaluate_predictions(test_pred_sets, data)
        assert(np.all(df.columns == ['label', 'n', 'mse', 'r2']))
        assert(np.allclose(df['mse'].values, [0.0025, 0, 0.05 / 3]))
        assert(np.all(np.isnan(df['r2'].values[:2])))
        assert(np.allclose(df['r2'].values[2], 0.991758))
        
        # With known variance
        data = pd.DataFrame({'y': [1, 2, 1.1],
                             'y_var': [0.1, 0.2, 0.1],
                             'seq': seqs}).set_index('seq')        
        df = evaluate_predictions(test_pred_sets, data)
        assert(np.allclose(df['loglikelihood'].values,
                           [0.219854, -0.114220, 0.066829]))

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
