#!/usr/bin/env python
import unittest

from gpmap.src.datasets import DataSet
from gpmap.src.settings import DATASETS


class DatasetsTests(unittest.TestCase):
    def test_dataset_class(self):
        # Test dimensions on GB1 dataset
        dataset = DataSet('gb1')
        
        data = dataset.data
        assert(data.shape[0] < 20**4)
        
        landscape = dataset.landscape
        assert(landscape.shape[0] == 20**4)
        
        # Test that all datasets in global variable are available
        for dataset_name in DATASETS:
            dataset = DataSet(dataset_name)
            assert(dataset.landscape.shape[0] >= dataset.data.shape[0])
            
        # Test error with missing dataset
        try:
            dataset = DataSet('gb2')
            self.fail()
        except ValueError:
            pass
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'DatasetsTests']
    unittest.main()
