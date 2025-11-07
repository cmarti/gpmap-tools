#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

from gpmap.datasets import DataSet
from gpmap.summary import GPmapSummarizer


class GPmapSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summarizer = GPmapSummarizer(2, 2)
        f = DataSet("gb1").landscape["y"].values
        V_U_vcs = {
            "U": [(0,), (1,), (0, 1)],
            "k": [1, 1, 2],
            "variance": [1, 2, 1],
        }
        self.V_U_vcs = pd.DataFrame(V_U_vcs)
        V_U_vcs = {
            "U": [(0,), (1,), (0, 1), (2,)],
            "k": [1, 1, 2, 1],
            "variance": [1, 2, 1, 1],
        }
        self.V_U_vcs_error = pd.DataFrame(V_U_vcs)
        self.gb1 = GPmapSummarizer(20, 4, f=f)
        return super().setUp()
    
    def test_root_mean_squared_epistatic_coeff_constant(self):
        f = np.array([1, 1, 1, 1])
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2, f=f)
        assert np.allclose(rmsec, 0)
    
    def test_root_mean_squared_epistatic_coeff_additive(self):
        f = np.array([1, 0, 0, -1])
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2, f=f)
        assert np.allclose(rmsec, 0)
    
    def test_root_mean_squared_epistatic_coeff_pairwise(self):
        f = np.array([1, -1, -1, 1])
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2, f=f)
        assert np.allclose(rmsec, 4.)

    def test_calc_V_k_variance_components_constant(self):
        f = np.array([1, 1, 1, 1])
        k_vcs = self.summarizer.calc_V_k_variance_components(f)
        assert k_vcs.shape[0] == 2
        assert np.allclose(k_vcs["variance"], 0)

    def test_calc_V_k_variance_components_additive(self):
        f = np.array([1, 0, 0, -1])
        k_vcs = self.summarizer.calc_V_k_variance_components(f)
        assert np.allclose(k_vcs["variance"], [2, 0])

    def test_calc_V_k_variance_components_pairwise(self):
        f = np.array([1, -1, -1, 1])
        k_vcs = self.summarizer.calc_V_k_variance_components(f)
        assert np.allclose(k_vcs["variance"], [0, 4])

    def test_calc_V_U_variance_components_constant(self):
        f = np.array([1, 1, 1, 1])
        V_U_vcs = self.summarizer.calc_V_U_variance_components(f)
        assert np.allclose(V_U_vcs["variance"], 0)

    def test_calc_V_U_variance_components_site1(self):
        f = np.array([1, 1, -1, -1])
        V_U_vcs = self.summarizer.calc_V_U_variance_components(f)
        assert np.allclose(V_U_vcs["variance"], [4, 0, 0])

    def test_calc_V_U_variance_components_site2(self):
        f = np.array([1, -1, 1, -1])
        V_U_vcs = self.summarizer.calc_V_U_variance_components(f)
        assert np.allclose(V_U_vcs["variance"], [0, 4, 0])

    def test_calc_V_U_variance_components_pairwise(self):
        f = np.array([1, -1, -1, 1])
        V_U_vcs = self.summarizer.calc_V_U_variance_components(f)
        assert np.allclose(V_U_vcs["variance"], [0, 0, 4])

    def test_calc_sites_not_in_U_error(self):
        with self.assertRaises(ValueError):
            self.summarizer.validate_U(self.V_U_vcs_error)

        with self.assertRaises(ValueError):
            self.summarizer.calc_sites_variance_perc(self.V_U_vcs_error)

        with self.assertRaises(ValueError):
            self.summarizer.calc_site_pairs_variance_perc(self.V_U_vcs_error)

    def test_calc_sites_variance_perc(self):
        sites_vcs = self.summarizer.calc_sites_variance_perc(self.V_U_vcs)
        assert np.allclose(sites_vcs[0], [25, 25])
        assert np.allclose(sites_vcs[1], [50, 25])

    def test_calc_site_pairs_variance_perc(self):
        pairs_vcs = self.summarizer.calc_site_pairs_variance_perc(self.V_U_vcs)
        assert pairs_vcs.shape[0] == 1
        assert np.allclose(pairs_vcs["variance"], 1)
        assert np.allclose(pairs_vcs["variance_perc"], 100)

    def test_summarize_gb1(self):
        rmsec = self.gb1.calc_root_mean_squared_epistatic_coeff(P=2)
        assert(rmsec > 0.)
        
        k_vcs = self.gb1.calc_V_k_variance_components()
        k_vcs = k_vcs.set_index("k")["variance"].to_dict()
        for k in range(1, 4):
            assert k_vcs[k] > k_vcs[k + 1]

        V_U_vcs = self.gb1.calc_V_U_variance_components()
        V_U_vcs_dict = {
            tuple(U): v for U, v in zip(V_U_vcs["U"], V_U_vcs["variance"])
        }
        assert V_U_vcs_dict[(2,)] > V_U_vcs_dict[(0,)]
        assert V_U_vcs_dict[(3,)] > V_U_vcs_dict[(1,)]
        assert V_U_vcs_dict[(2, 3)] > V_U_vcs_dict[(0, 1)]
        assert V_U_vcs_dict[(2, 3)] > V_U_vcs_dict[(0, 2)]

        sites_vcs = self.gb1.calc_sites_variance_perc(V_U_vcs)
        assert sites_vcs.shape == (4, 4)
        assert np.all(sites_vcs.loc[1, :] < sites_vcs.loc[2, :])
        assert np.all(sites_vcs.loc[3, :] < sites_vcs.loc[2, :])
        assert np.all(sites_vcs.loc[4, :] < sites_vcs.loc[2, :])

        pairs_vcs = self.gb1.calc_site_pairs_variance_perc(V_U_vcs)
        pairs_vcs_high_order = self.gb1.calc_site_pairs_variance_perc(
            V_U_vcs, min_k=3
        )
        assert not np.allclose(pairs_vcs, pairs_vcs_high_order)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "GPmapSummaryTests"]
    unittest.main()
