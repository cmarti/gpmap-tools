#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

from os.path import join

from gpmap.src.utils import calc_cartesian_product, get_CV_splits,\
    calc_tensor_product, calc_cartesian_product_dot, calc_tensor_product_dot,\
    calc_tensor_product_quad, quad
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
    
    def test_cartesian_product_dot(self):
        m1 = np.array([[0, 1],
                      [1, 0]])
        ms = [m1, m1]
        m2 = calc_cartesian_product(ms)
        v = np.random.normal(size=m2.shape[0])
        u1 = m2.dot(v)
        u2 = calc_cartesian_product_dot([m1, m1], v)
        assert(np.allclose(u1, u2))
        
        ms = [m1, m1, m1, m1, m1, m1, m1]
        m2 = calc_cartesian_product(ms)
        v = np.random.normal(size=m2.shape[0])
        u1 = m2.dot(v)
        u2 = calc_cartesian_product_dot(ms, v)
        assert(np.allclose(u1, u2))
        
    def test_tensor_product(self):
        matrix1 = np.array([[0.6],
                            [0.4]])
        matrix2 = np.array([[0.7],
                            [0.3]])
        expected = np.array([[0.42],
                             [0.18],
                             [0.28],
                             [0.12]])
        result = calc_tensor_product([matrix1, matrix2])
        assert(np.allclose(result, expected))
    
    def test_tensor_product_dot(self):
        m1 = 0.5 * np.array([[1, 1],
                             [1, 1]])
        m2 = 0.5 * np.array([[1, -1],
                             [-1, 1]])
        ms = [m1, m2]
        m = calc_tensor_product(ms)
        
        v = np.ones(4)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot(ms, v)
        assert(np.allclose(u1, u2))
        
        # Test in larger scenario
        ms = [m1, m2, m2, m1, m2, m1, m1]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot(ms, v)
        assert(np.allclose(u1, u2))
    
    def test_tensor_product_quad(self):
        m1 = 0.5 * np.array([[1, 1],
                             [1, 1]])
        m2 = 0.5 * np.array([[1, -1],
                             [-1, 1]])
        
        # Test small case
        ms = [m1, m2]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = quad(m, v)
        u2 = calc_tensor_product_quad(ms, v, v)
        assert(np.allclose(u1, u2))

        # Test in larger scenario
        ms = [m1, m2, m2]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = quad(m, v)
        u2 = calc_tensor_product_quad(ms, v)
        assert(np.allclose(u1, u2))
    
    def test_get_CV_splits(self):
        np.random.seed(0)
        nfolds = 3 
        X = np.array(['A', 'B', 'C'])
        y = np.array([1, 2, 2])
        
        splits = get_CV_splits(X, y, nfolds=3)
        for _, (x_train, y_train, _), (x_test, y_test, _) in splits:
            assert(x_train.shape[0] == 1)
            assert(y_train.shape[0] == 1)
            assert(x_test.shape[0] == 2)
            assert(y_test.shape[0] == 2)
            
        splits = list(get_CV_splits(X, y, nfolds=nfolds, count_data=True))
        assert(len(splits) == nfolds)
        for _, (x_train, y_train), (x_test, y_test) in splits:
            # Test total numbers are preserved
            assert(y_train.sum() + y_test.sum() == y.sum())
            
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
        nfolds = 7
        data = pd.read_csv(join(TEST_DATA_DIR, 'seqdeft_counts.csv'),
                           index_col=0)
        X, y = data.index.values, data.iloc[:, 0].values
        print(y.sum())
        splits = get_CV_splits(X, y, nfolds=nfolds, count_data=True)
        test_counts = {}
        for i, (x_train, y_train), (x_test, y_test) in splits:
            print(y_train.sum())
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
        
        assert(i == nfolds - 1)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'UtilsTests']
    unittest.main()
