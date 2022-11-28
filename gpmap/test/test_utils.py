#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.src.utils import calc_cartesian_product, get_CV_splits


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
            assert(y_train.sum() + y_test.sum() == 5)
            
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
