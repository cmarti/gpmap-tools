#!/usr/bin/env python
import unittest
import numpy as np
import pandas as pd

from gpmap.datasets import DataSet, list_available_datasets
from gpmap.inference import VCregression, SeqDEFT


class DatasetsTests(unittest.TestCase):
    def test_dataset_class(self):
        # Test dimensions on GB1 dataset
        dataset = DataSet("gb1")

        data = dataset.data
        assert data.shape[0] < 20**4

        landscape = dataset.landscape
        assert landscape.shape[0] == 20**4

        # Test that all datasets in global variable are available
        for dataset_name in list_available_datasets():
            dataset = DataSet(dataset_name)

        # Test error with missing dataset
        try:
            dataset = DataSet("gb2")
            self.fail()
        except ValueError:
            pass

    def test_raw_data(self):
        dataset = DataSet("gb1")
        assert dataset.raw_data.shape[0] < dataset.landscape.shape[0]
        assert np.all(dataset.raw_data.columns == ["input", "selected"])

        dataset = DataSet("pard")
        assert dataset.raw_data.shape[0] <= dataset.landscape.shape[0]
        assert dataset.raw_data.shape[1] == 8

        # Test error with missing raw data
        try:
            dataset = DataSet("serine")
            dataset.raw_data
            self.fail()
        except ValueError:
            pass

    def test_build_regression_dataset(self):
        np.random.seed(0)
        lambdas = np.array([10, 2, 0.5, 0.1, 0.02, 0])
        model = VCregression(
            seq_length=5, alphabet_type="dna", lambdas=lambdas
        )
        _, X, y, y_var = model.simulate(y_var=0.01, p_missing=0.2)
        data = pd.DataFrame({'y': y, 'y_var': y_var}, index=X)

        # Build dataset
        test = DataSet("test", data=data)
        test.build()

        # Load newly built dataset
        test = DataSet("test")
        assert test.landscape.shape[0] == 4**5
        assert test.nodes.shape[0] == 4**5

    def test_build_probability_dataset(self):
        np.random.seed(0)
        model = SeqDEFT(P=2, a=500, seq_length=5, alphabet_type="dna")
        _, X = model.simulate(N=1000)
        data = pd.DataFrame({"X": X})

        # Build dataset
        test = DataSet("test", data=data)
        test.build()

        # Load newly built dataset
        test = DataSet("test")
        assert test.landscape.shape[0] == 4**5
        assert test.nodes.shape[0] == 4**5

    def test_visualization(self):
        serine = DataSet("serine")
        assert serine.nodes.shape[0] == serine.data.shape[0]
        assert serine.edges.shape[0] > serine.data.shape[0]
        assert serine.relaxation_times.shape[0] == 19

    def test_build_dataset(self):
        dataset = DataSet("serine")


if __name__ == "__main__":
    import sys

    sys.argv = ["", "DatasetsTests"]
    unittest.main()
