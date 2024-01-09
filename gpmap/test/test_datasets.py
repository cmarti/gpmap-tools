#!/usr/bin/env python
import unittest
import numpy as np

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
            
        # Test error with missing dataset
        try:
            dataset = DataSet('gb2')
            self.fail()
        except ValueError:
            pass
    
    def test_raw_data(self):
        dataset = DataSet('gb1')
        assert(dataset.raw_data.shape[0] < dataset.landscape.shape[0])
        assert(np.all(dataset.raw_data.columns == ['input', 'selected']))
        
        dataset = DataSet('pard')
        assert(dataset.raw_data.shape[0] <= dataset.landscape.shape[0])
        assert(dataset.raw_data.shape[1] == 8)
        
        # Test error with missing raw data
        try:
            dataset = DataSet('serine')
            dataset.raw_data
            self.fail()
        except ValueError:
            pass
    
    def test_visualization(self):
        serine = DataSet('serine')
        assert(serine.nodes.shape[0] == serine.data.shape[0])
        assert(serine.edges.shape[0] > serine.data.shape[0])
        assert(serine.relaxation_times.shape[0] == 19)
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'DatasetsTests']
    unittest.main()
