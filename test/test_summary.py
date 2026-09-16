import unittest
from itertools import product, combinations

import numpy as np
import pandas as pd
from scipy.special import comb

from gpmap.datasets import DataSet
from gpmap.inference import VCregression
from gpmap.summary import GPDataSummarizer, GPmapSummarizer


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
        self.summarizer.set_f(f)
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2)
        assert np.allclose(rmsec, 0)

    def test_root_mean_squared_epistatic_coeff_additive(self):
        f = np.array([1, 0, 0, -1])
        self.summarizer.set_f(f)
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2)
        assert np.allclose(rmsec, 0)

    def test_root_mean_squared_epistatic_coeff_pairwise(self):
        f = np.array([1, -1, -1, 1])
        self.summarizer.set_f(f)
        rmsec = self.summarizer.calc_root_mean_squared_epistatic_coeff(P=2)
        assert np.allclose(rmsec, 4.0)

    def test_root_U_mean_squared_epistatic_coeffs(self):
        summarizer = GPmapSummarizer(2, 3)
        f = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        summarizer.set_f(f)
        rmsecs = summarizer.calc_U_root_mean_squared_epistatic_coeffs(P=2)
        assert np.allclose(rmsecs["rmsec"], [0, 0, 4.0])

    def test_calc_V_k_variance_components_constant(self):
        f = np.array([1, 1, 1, 1])
        self.summarizer.set_f(f)
        k_vcs = self.summarizer.calc_V_k_variance_components()
        assert k_vcs.shape[0] == 2
        assert np.allclose(k_vcs["variance"], 0)

    def test_calc_V_k_variance_components_additive(self):
        f = np.array([1, 0, 0, -1])
        self.summarizer.set_f(f)
        k_vcs = self.summarizer.calc_V_k_variance_components()
        assert np.allclose(k_vcs["variance"], [2, 0])

    def test_calc_V_k_variance_components_pairwise(self):
        f = np.array([1, -1, -1, 1])
        self.summarizer.set_f(f)
        k_vcs = self.summarizer.calc_V_k_variance_components()
        assert np.allclose(k_vcs["variance"], [0, 4])

    def test_calc_V_U_variance_components_constant(self):
        f = np.array([1, 1, 1, 1.0])
        self.summarizer.set_f(f)
        V_U_vcs = self.summarizer.calc_V_U_variance_components()
        assert np.allclose(V_U_vcs["variance"], 0)

    def test_calc_V_U_variance_components_site1(self):
        f = np.array([1, 1, -1, -1.0])
        self.summarizer.set_f(f)
        V_U_vcs = self.summarizer.calc_V_U_variance_components()
        assert np.allclose(V_U_vcs["variance"], [4, 0, 0])

    def test_calc_V_U_variance_components_site2(self):
        f = np.array([1, -1, 1, -1.0])
        self.summarizer.set_f(f)
        V_U_vcs = self.summarizer.calc_V_U_variance_components()
        assert np.allclose(V_U_vcs["variance"], [0, 4, 0])

    def test_calc_V_U_variance_components_pairwise(self):
        f = np.array([1, -1, -1, 1.0])
        self.summarizer.set_f(f)
        V_U_vcs = self.summarizer.calc_V_U_variance_components()
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

    def test_calc_gamma_U_D_errors(self):
        f = np.array([1, -1, -1, 1])
        self.summarizer.set_f(f)

        with self.assertRaises(ValueError):
            self.summarizer.calc_gamma_U_D(U=[0, 1], D=[0, 1])

        with self.assertRaises(ValueError):
            self.summarizer.calc_gamma_U_D(U=[0, 1, 2], D=[])

    def test_calc_gamma_U_D_additive(self):
        f = np.array([1, 0, 0, -1])
        self.summarizer.set_f(f)

        gamma_1_to_2 = self.summarizer.calc_gamma_U_D(U=[1], D=[0])
        assert np.allclose(gamma_1_to_2, 1.0)

        gamma_2_to_1 = self.summarizer.calc_gamma_U_D(U=[0], D=[1])
        assert np.allclose(gamma_2_to_1, 1.0)

        alleles = [(0, 1)]
        gamma_1_to_2 = self.summarizer.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=alleles, D_alleles=alleles
        )
        assert np.allclose(gamma_1_to_2, 1.0)

        gamma_2_to_1 = self.summarizer.calc_gamma_U_D(
            U=[0], D=[1], U_alleles=alleles, D_alleles=alleles
        )
        assert np.allclose(gamma_2_to_1, 1.0)

    def test_calc_gamma_U_D_anticorrelated(self):
        f = np.array([1, -1, -1, 1])
        self.summarizer.set_f(f)

        gamma_1_to_2 = self.summarizer.calc_gamma_U_D(U=[1], D=[0])
        assert np.allclose(gamma_1_to_2, -1.0)

        gamma_2_to_1 = self.summarizer.calc_gamma_U_D(U=[0], D=[1])
        assert np.allclose(gamma_2_to_1, -1.0)

        alleles = [(0, 1)]
        gamma_1_to_2 = self.summarizer.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=alleles, D_alleles=alleles
        )
        assert np.allclose(gamma_1_to_2, -1.0)

        gamma_2_to_1 = self.summarizer.calc_gamma_U_D(
            U=[0], D=[1], U_alleles=alleles, D_alleles=alleles
        )
        assert np.allclose(gamma_2_to_1, -1.0)

    def test_calc_gamma_U_D_arbitrary(self):
        f = np.array([1, 0, 0, -2])
        self.summarizer.set_f(f)
        U, D = [1], [0]
        alleles = [(0, 1)]

        gamma_UD = self.summarizer.calc_gamma_U_D(U, D)
        asec_U = self.summarizer.calc_U_avg_squared_epistatic_coeff(U)
        asec_UD = self.summarizer.calc_U_avg_squared_epistatic_coeff(U + D)
        assert np.allclose(gamma_UD, 1 - (asec_UD / (2 * asec_U)))

        gamma_UD_alleles = self.summarizer.calc_gamma_U_D(
            U, D, U_alleles=alleles, D_alleles=alleles
        )
        assert np.allclose(gamma_UD_alleles, gamma_UD)

    def test_calc_gamma_U_D_random(self):
        f = np.random.normal(size=4)
        self.summarizer.set_f(f)

        U, D = [1], [0]
        gamma_UD = self.summarizer.calc_gamma_U_D(U, D)
        asec_U = self.summarizer.calc_U_avg_squared_epistatic_coeff(U)
        asec_UD = self.summarizer.calc_U_avg_squared_epistatic_coeff(U + D)
        assert np.allclose(gamma_UD, 1 - (asec_UD / (2 * asec_U)))

    def test_calc_gamma_i_to_j_alleles_additive(self):
        f = np.array([-1, 1, 0.0, -1, 1, 0.0, -1, 1, 0.0])
        s = GPmapSummarizer(3, 2, f=f)

        gamma_A1B_A2B = s.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=[(0, 1)], D_alleles=[(0, 1)]
        )
        assert np.allclose(gamma_A1B_A2B, 1)

        gamma_A1B_A2C = s.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=[(0, 2)], D_alleles=[(0, 1)]
        )
        assert np.allclose(gamma_A1B_A2C, 1)

    def test_calc_gamma_i_to_j_alleles_epistatic(self):
        f = np.array([-1, 1, 0.0, -1, 1, 0.0, 1, -1, 0.0])
        s = GPmapSummarizer(3, 2, f=f)

        gamma_A1C_A2C = s.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=[(0, 2)], D_alleles=[(0, 2)]
        )
        assert np.allclose(gamma_A1C_A2C, -1.0)

        gamma_A1C_A2B = s.calc_gamma_U_D(
            U=[1], D=[0], U_alleles=[(0, 1)], D_alleles=[(0, 2)]
        )
        assert np.allclose(gamma_A1C_A2B, -1.0)

    def test_calc_Gamma_i_to_j_alleles_random(self):
        f = np.random.normal(size=4**8)
        s = GPmapSummarizer(4, 8, f=f)
        U, D = [1], [0]

        Gamma_UD_1 = s.calc_Gamma_U_D(U, D)
        Gamma_UD_2 = []
        for c1, c2 in combinations(range(4), 2):
            Gamma_UD_2.append(s.calc_Gamma_U_D(U, D, U_alleles=[(c1, c2)]))
        Gamma_UD_2 = np.mean(Gamma_UD_2)
        assert np.allclose(Gamma_UD_1, Gamma_UD_2)

        Gamma_UD_3 = []
        for c1, c2 in combinations(range(4), 2):
            Gamma_UD_3.append(s.calc_Gamma_U_D(U, D, D_alleles=[(c1, c2)]))
        Gamma_UD_3 = np.mean(Gamma_UD_3)
        assert np.allclose(Gamma_UD_1, Gamma_UD_3)

    def test_calc_Gamma_U_D_alleles_random(self):
        alpha = 5
        f = np.random.normal(size=alpha**8)
        s = GPmapSummarizer(alpha, 8, f=f)
        U, D = [0, 1], [2, 3]

        Gamma_UD_1 = s.calc_Gamma_U_D(U, D)
        Gamma_UD_2 = []
        pairs = list(combinations(range(alpha), 2))
        for U_alleles in product(pairs, repeat=2):
            Gamma_UD_2.append(s.calc_Gamma_U_D(U, D, U_alleles=U_alleles))
        Gamma_UD_2 = np.mean(Gamma_UD_2)
        assert np.allclose(Gamma_UD_1, Gamma_UD_2)

        Gamma_UD_3 = []
        for D_alleles in product(pairs, repeat=2):
            Gamma_UD_3.append(s.calc_Gamma_U_D(U, D, D_alleles=D_alleles))
        Gamma_UD_3 = np.mean(Gamma_UD_3)
        assert np.allclose(Gamma_UD_1, Gamma_UD_3)

    def test_calc_gamma_U_D_random_epistatic_coeffs(self):
        f = np.random.normal(size=8)
        self.summarizer = GPmapSummarizer(2, 3, f=f)

        U, D = [1, 2], [0]
        gamma_UD = self.summarizer.calc_gamma_U_D(U, D)
        asec_U = self.summarizer.calc_U_avg_squared_epistatic_coeff(U)
        asec_UD = self.summarizer.calc_U_avg_squared_epistatic_coeff(U + D)
        assert np.allclose(gamma_UD, 1 - (asec_UD / (2 * asec_U)))

    def test_calc_gamma_i_to_j(self):
        alpha, ell = 4, 6
        f = np.random.normal(size=alpha**ell)
        s = GPmapSummarizer(alpha, ell, f=f)

        gamma_i_to_j = s.calc_gamma_i_to_j()
        assert gamma_i_to_j.shape == (30, 4)

    def test_calc_gamma_ij_D(self):
        alpha, ell = 4, 6
        f = np.random.normal(size=alpha**ell)
        s = GPmapSummarizer(alpha, ell, f=f)
        D = [0]

        gamma_ij_D = s.calc_gamma_ij_D(D)
        assert gamma_ij_D.shape == (10, 4)

    def test_calc_gamma_AiBi_to_AjBj(self):
        alpha, ell = 4, 6
        f = np.random.normal(size=alpha**ell)
        s = GPmapSummarizer(alpha, ell, f=f)

        gamma_AiBi_to_AjBj = s.calc_gamma_AiBi_to_AjBj()
        assert gamma_AiBi_to_AjBj.shape == (1080, 8)

    def test_calc_gamma_AiBiAjBj_D(self):
        alpha, ell = 4, 6
        f = np.random.normal(size=alpha**ell)
        s = GPmapSummarizer(alpha, ell, f=f)

        gamma_AiBiAjBj_D = s.calc_gamma_AiBiAjBj_D(D=[0], D_alleles=[(0, 1)])
        assert gamma_AiBiAjBj_D.shape == (360, 8)

    def test_calc_U_avg_squared_mean_epistatic_coeff(self):
        f = np.array([-1.5, -0.5, 0.0, -1.5, -0.5, 0.0, -1.5, -0.5, 0.0])
        s = GPmapSummarizer(3, 2, f=f)

        U = [0]
        asmec01 = s.calc_U_avg_squared_mean_epistatic_coeff(
            U, U_alleles=[(0, 1)]
        )
        assert np.allclose(asmec01, 0.0)

        asmec02 = s.calc_U_avg_squared_mean_epistatic_coeff(
            U, U_alleles=[(0, 2)]
        )
        assert np.allclose(asmec02, 0.0)

        U = [1]
        asmec01 = s.calc_U_avg_squared_mean_epistatic_coeff(
            U, U_alleles=[(0, 1)]
        )
        assert np.allclose(asmec01, 1.0)

        asmec02 = s.calc_U_avg_squared_mean_epistatic_coeff(
            U, U_alleles=[(0, 2)]
        )
        assert np.allclose(asmec02, 1.5**2)

    def test_calc_U_avg_squared_mean_epistatic_coeff_binary_random(self):
        f = np.random.normal(size=2**10)
        self.summarizer = GPmapSummarizer(2, 10, f=f)

        U = [1]
        U_alleles = [(0, 1)]
        asmec1 = self.summarizer.calc_U_avg_squared_mean_epistatic_coeff(U)
        asmec2 = self.summarizer.calc_U_avg_squared_mean_epistatic_coeff(
            U, U_alleles
        )
        assert np.allclose(asmec1, asmec2)

    def test_summarize_gb1(self):
        rmsec = self.gb1.calc_root_mean_squared_epistatic_coeff(P=2)
        assert rmsec > 0.0

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

        U, D = [2], [3]
        gamma_UD = self.gb1.calc_gamma_U_D(U=U, D=D)
        asec_U = self.gb1.calc_U_avg_squared_epistatic_coeff(U)
        asec_UD = self.gb1.calc_U_avg_squared_epistatic_coeff(U + D)
        assert np.allclose(gamma_UD, 1 - (asec_UD / (2 * asec_U)))

        gamma_i_to_j = self.gb1.calc_gamma_i_to_j()
        assert gamma_i_to_j.shape == (12, 4)


class GPDataSummaryTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.alphabet = list("ACGT")
        self.seq_length = 4
        s = product(self.alphabet, repeat=self.seq_length)
        self.genotypes = np.array(["".join(x) for x in s])
        idx = np.random.uniform(size=self.genotypes.shape[0]) < 0.9
        self.X = self.genotypes[idx]
        self.y = np.random.normal(size=self.X.shape[0])
        self.y_var = np.full_like(self.y, 0.1)

    def test_define_summarizer(self):
        configs = [
            {"seq_length": self.seq_length, "alphabet_type": "dna"},
            {"seq_length": self.seq_length, "alphabet": self.alphabet},
            {"genotypes": self.genotypes},
        ]

        for config in configs:
            # Using set_data
            data = GPDataSummarizer(**config)
            data.set_data(self.X, self.y, y_var=self.y_var)
            X_op = data.get_X_operator(self.X)
            assert X_op.shape[0] == self.X.shape[0]

            # Providing data in construction
            config.update({"X": self.X, "y": self.y, "y_var": self.y_var})
            data = GPDataSummarizer(**config)
            X_op = data.get_X_operator(self.X)
            assert X_op.shape[0] == self.X.shape[0]

    def test_calc_distance_covariance(self):
        # Test simple cases
        X = np.array(["AA", "AB", "BA", "BB"])
        s = GPDataSummarizer(alphabet=list("AB"), seq_length=2)

        # Constant function
        y = np.array([1, 1, 1, 1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_distance(centered=False)
        assert np.allclose(cov, 1)
        assert np.allclose(ns, [4, 8, 4])

        cov, ns = s.calc_covariance_distance(centered=True)
        assert np.allclose(cov, 0)

        # Additive function
        y = np.array([2, 0, 0, -2])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_distance(centered=False)
        assert np.allclose(cov, [2, 0, -2])
        assert np.allclose(ns, [4, 8, 4])

        cov, ns = s.calc_covariance_distance(centered=True)
        assert np.allclose(cov, [2, 0, -2])

        # Pairwise function
        y = np.array([1, -1, -1, 1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_distance(centered=False)
        assert np.allclose(cov, [1, -1, 1])
        assert np.allclose(ns, [4, 8, 4])

        cov, ns = s.calc_covariance_distance(centered=True)
        assert np.allclose(cov, [1, -1, 1])

        # Test with simulated data
        np.random.seed(1)
        lambdas = np.array([0, 200, 20, 2, 0.2])
        n_alleles, seq_length = 4, 4

        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )
        _, X, y, y_var = vc.simulate(y_var=0.01)

        s = GPDataSummarizer(genotypes=X)
        s.set_data(X=X, y=y, y_var=y_var)  # type: ignore
        cov1, ns = s.calc_covariance_distance(centered=True)
        cov1_uncentered, ns = s.calc_covariance_distance(centered=False)

        s.set_data(X=X, y=y + 5.0, y_var=y_var)  # type: ignore
        cov2, ns = s.calc_covariance_distance(centered=True)
        cov2_uncentered, ns = s.calc_covariance_distance(centered=False)
        assert np.allclose(cov2, cov1)
        assert not np.allclose(cov2_uncentered, cov1_uncentered, rtol=0.1)

        # Ensure we get the expected number of pairs per distance category
        for d in range(seq_length + 1):
            total_genotypes = n_alleles**seq_length
            d_combs = comb(seq_length, d)
            d_sites_genotypes = (n_alleles - 1) ** d
            ns[d] = total_genotypes * d_combs * d_sites_genotypes

        # Ensure anticorrelated distances
        assert cov1[3] < 0
        assert cov1[4] < 0

        # With missing data
        _, X, y, y_var = vc.simulate(y_var=0.01, p_missing=0.1)
        s.set_data(X=X, y=y, y_var=y_var)  # type: ignore
        cov, ns = s.calc_covariance_distance(centered=True)

        # Ensure anticorrelated distances
        assert cov[3] < 0
        assert cov[4] < 0

    def test_calc_covariance_U_sites(self):
        # Test simple cases
        X = np.array(["AA", "AB", "BA", "BB"])
        s = GPDataSummarizer(alphabet=list("AB"), seq_length=2)

        # Constant function
        y = np.array([1, 1, 1, 1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert np.allclose(cov, 1)
        assert np.allclose(ns, 4)

        cov, ns = s.calc_covariance_distance(centered=True)
        assert np.allclose(cov, 0)

        # Additive for site 1 function
        y = np.array([1, -1, 1, -1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert np.allclose(cov, [1, -1, 1, -1])
        assert np.allclose(ns, 4)

        cov, ns = s.calc_covariance_U_sites(centered=True)
        assert np.allclose(cov, [1, -1, 1, -1])

        # Additive for site 2 function
        y = np.array([1, 1, -1, -1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert np.allclose(cov, [1, 1, -1, -1])
        assert np.allclose(ns, 4)

        cov, ns = s.calc_covariance_U_sites(centered=True)
        assert np.allclose(cov, [1, 1, -1, -1])

        # Additive for site 1 and 2 function
        y = np.array([2, 0, 0, -2])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert np.allclose(cov, [2, 0, 0, -2])
        assert np.allclose(ns, 4)

        cov, ns = s.calc_covariance_U_sites(centered=True)
        assert np.allclose(cov, [2, 0, 0, -2])

        # Pairwise function
        y = np.array([1, -1, -1, 1])
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert np.allclose(cov, [1, -1, -1, 1])
        assert np.allclose(ns, 4)

        cov, ns = s.calc_covariance_U_sites(centered=True)
        assert np.allclose(cov, [1, -1, -1, 1])

    def test_calc_covariance_U_sites_longer_seqs(self):
        sl = 5
        s = GPDataSummarizer(alphabet=list("ABCD"), seq_length=sl)
        X = s.genotypes
        y = np.random.normal(size=s.n_genotypes)
        expected_cov = np.zeros(2**sl)
        expected_cov[0] = 1.0

        # Verify output shapes
        s.set_data(X, y)
        cov, ns = s.calc_covariance_U_sites(centered=False)
        assert cov.shape == (2**5,)
        assert ns.shape == (2**5,)
        assert np.allclose(cov, expected_cov, atol=0.05)

        # Ensure changes when seeing only part of the data
        idx = np.random.uniform(size=s.n_genotypes) < 0.9
        X, y = X[idx], y[idx]
        s.set_data(X, y)
        cov2, ns2 = s.calc_covariance_U_sites(centered=False)
        assert cov.shape == (2**sl,)
        assert ns.shape == (2**sl,)

        assert np.all(ns2 <= ns)
        assert np.all(cov2 != cov)
        assert np.allclose(cov, expected_cov, atol=0.05)

    def test_calc_avg_local_epistatic_coeff(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        s = GPDataSummarizer(alphabet=list("AB"), seq_length=2)
        y_constant = np.array([1, 1, 1, 1])
        s.set_data(X, y_constant)
        assert np.allclose(
            s.calc_avg_squared_local_epistatic_coeff(P=2),
            0.0,
        )

        y_pairwise = np.array([1, -1, -1, 1])
        s.set_data(X, y_pairwise)
        assert np.allclose(
            s.calc_avg_squared_local_epistatic_coeff(P=2),
            16.0,
        )


if __name__ == "__main__":
    import sys

    sys.argv = ["", "GPmapSummaryTests", "GPDataSummaryTests"]
    unittest.main()
