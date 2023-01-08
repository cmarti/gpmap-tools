#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

from os.path import join

from gpmap.src.utils import calc_cartesian_product, get_CV_splits
from gpmap.src.settings import TEST_DATA_DIR


class UtilsTests(unittest.TestCase):
    def test_cartesian_product(self):
        # With adjacency matrices
        matrix = np.array([[0, 1],
                           [1, 0]])
        expected = np.array([[0, 1, 1, 0],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]])
        
        result = calc_cartesian_product([matrix, matrix])
        assert(np.all(result == expected))
        
        # With more than 2 matrices
        matrix = np.array([[0, 1],
                           [1, 0]])
        expected = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0, 0, 1, 0],
                             [0, 1, 1, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 1, 1, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1],
                             [0, 0, 1, 0, 1, 0, 0, 1],
                             [0, 0, 0, 1, 0, 1, 1, 0]])
        
        result = calc_cartesian_product([matrix, matrix, matrix])
        assert(np.all(result == expected))
        
        # With pseudo-rate matrices (without diagonals)
        matrix = np.array([[0, 0.6],
                           [0.4, 0]])
        expected = np.array([[0, 0.6, 0.6, 0],
                             [0.4, 0, 0, 0.6],
                             [0.4, 0, 0, 0.6],
                             [0, 0.4, 0.4, 0]])
        
        result = calc_cartesian_product([matrix, matrix])
        assert(np.all(result == expected))
        
        # With different pseudo-rate matrices (without diagonals)
        matrix1 = np.array([[0, 0.6],
                            [0.4, 0]])
        matrix2 = np.array([[0, 0.7],
                            [0.3, 0]])
        expected = np.array([[0, 0.7, 0.6, 0],
                             [0.3, 0, 0, 0.6],
                             [0.4, 0, 0, 0.7],
                             [0, 0.4, 0.3, 0]])
        result = calc_cartesian_product([matrix1, matrix2])
        assert(np.all(result == expected))
    
    def test_get_CV_splits(self):
        np.random.seed(0)
        X = np.array(['A', 'B', 'C'])
        y = np.array([1, 2, 2])
        
        splits = get_CV_splits(X, y, nfolds=3)
        for _, (x_train, y_train, _), (x_test, y_test, _) in splits:
            assert(x_train.shape[0] == 1)
            assert(y_train.shape[0] == 1)
            assert(x_test.shape[0] == 2)
            assert(y_test.shape[0] == 2)
            
        splits = get_CV_splits(X, y, nfolds=3, count_data=True)
        for _, (x_train, y_train), (x_test, y_test) in splits:
            # Test total numbers are preserved
            assert(y_train.sum() + y_test.sum() == 5)
            
            # Test exact counts match total data
            counts = {seq: c for seq, c in zip(x_train, y_train)}
            for seq, c in zip(x_test, y_test):
                try:
                    counts[seq] += c
                except KeyError:
                    counts[seq] = c
            
            for seq, c in zip(X, y):
                assert(c == counts[seq])
                
        try:
            splits = list(get_CV_splits(X, y, nfolds=10, count_data=True))
            self.fail()
        except ValueError:
            pass
    
    def test_get_CV_splits_big_dataset(self):    
        data = pd.read_csv(join(TEST_DATA_DIR, 'seqdeft_counts.csv'),
                           index_col=0)
        X, y = data.index.values, data.iloc[:, 0].values
        splits = get_CV_splits(X, y, nfolds=5, count_data=True)
        test_counts = {}
        for _, (x_train, y_train), (x_test, y_test) in splits:
            assert(y_train.sum() + y_test.sum() == y.sum())
            
            counts = {seq: c for seq, c in zip(x_train, y_train)}
            for seq, c in zip(x_test, y_test):
                try:
                    counts[seq] += c
                except KeyError:
                    counts[seq] = c
                    
                try:
                    test_counts[seq] += c
                except KeyError:
                    test_counts[seq] = c
            
            # Test that each split has all the data
            for seq, c in zip(X, y):
                if seq in counts:
                    assert(c == counts[seq])
                else:
                    assert(c == 0)
        
        # Test whether test counts amount to the total counts
        for seq, c in zip(X, y):
            if seq in test_counts:
                assert(c == test_counts[seq])
            else:
                assert(c == 0)
            
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
