#!/usr/bin/env python
import sys
import unittest
from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import pearsonr

from gpmap.inference import (
    VCregression,
    calc_covariance_distance,
    calc_covariance_vjs,
)
from gpmap.linop import (
    LaplacianOperator,
    ProjectionOperator,
)
from gpmap.matrix import rayleigh_quotient
from gpmap.settings import BIN_DIR


class VCTests(unittest.TestCase):
    def test_lambdas_to_variance_p(self):
        lambdas = np.array([0, 1, 0.5, 0.1])
        vc = VCregression(n_alleles=2, seq_length=3, lambdas=lambdas)
        v = vc.lambdas_to_variance(lambdas)
        assert np.allclose(v, [3 / 4.6, 1.5 / 4.6, 0.1 / 4.6])

    def test_simulate_vc(self):
        np.random.seed(1)
        seq_length, n_alleles = 4, 4

        lambdas = np.array([0, 200, 20, 2, 0.2])
        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )

        # Test functioning and size
        y_true, X, y, y_var = vc.simulate(y_var=0.01)
        assert y_true.shape[0] == 256
        assert X.shape[0] == 256
        assert y.shape[0] == 256
        assert y_var.shape[0] == 256

        # Test missing genotypes
        y_true, X, y, y_var = vc.simulate(y_var=0.01, p_missing=0.1)
        assert y_true.shape[0] == 256
        assert X.shape[0] < 256
        assert y.shape[0] < 256
        assert y_var.shape[0] < 256

        # Test pure components
        for k1 in range(seq_length + 1):
            vc.set_lambdas(k=k1)
            f = vc.simulate()[0]

            for k2 in range(seq_length + 1):
                W = ProjectionOperator(n_alleles, seq_length, k=k2)
                f_k_rq = rayleigh_quotient(W, f)
                assert np.allclose(f_k_rq, k1 == k2)

    def test_calc_distance_covariance(self):
        np.random.seed(1)
        lambdas = np.array([0, 200, 20, 2, 0.2])
        n_alleles, seq_length = 4, 4

        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )
        y_true = vc.simulate(y_var=0.01)[0]
        rho, n = calc_covariance_distance(y_true, n_alleles, seq_length)

        # Ensure we get the expected number of pairs per distance category
        for d in range(seq_length + 1):
            total_genotypes = n_alleles**seq_length
            d_combs = comb(seq_length, d)
            d_sites_genotypes = (n_alleles - 1) ** d
            n[d] = total_genotypes * d_combs * d_sites_genotypes

        # Ensure anticorrelated distances
        assert rho[3] < 0
        assert rho[4] < 0

        # With missing data
        y_true, X, y, _ = vc.simulate(y_var=0.01, p_missing=0.1)
        idx = vc.get_obs_idx(X)
        rho, n = calc_covariance_distance(y, n_alleles, seq_length, idx=idx)

        # Ensure anticorrelated distances
        assert rho[3] < 0
        assert rho[4] < 0
    
    def test_calc_covariance_vjs(self):
        # Test simple cases
        a, sl = 2, 2
        y = np.array([1, 1, 1, 1])
        cov, ns, sites = calc_covariance_vjs(y, a, sl)
        assert np.allclose(cov, 1)
        assert np.allclose(ns, 4)

        y = np.array([1, -1, 1, -1])
        cov, ns, sites = calc_covariance_vjs(y, a, sl)
        assert np.allclose(cov, [1, 1, -1, -1])
        assert np.allclose(ns, 4)

        y = np.array([1, 1, -1, -1])
        cov, ns, sites = calc_covariance_vjs(y, a, sl)
        assert np.allclose(cov, [1, -1, 1, -1])
        assert np.allclose(ns, 4)

        y = np.array([1, 0, 0, -1])
        cov, ns, sites = calc_covariance_vjs(y, a, sl)
        assert np.allclose(cov, [0.5, 0, 0, -0.5])
        assert np.allclose(ns, 4)

        # Test in a bigger landscape
        a, sl = 4, 5
        n = a**sl
        y = np.random.normal(size=n)

        # Verify output shapes
        cov, ns, sites = calc_covariance_vjs(y, a, sl)
        assert cov.shape == (2**sl,)
        assert ns.shape == (2**sl,)
        assert sites.shape == (2**sl, sl)

        # Ensure changes when seeing only part of the data
        idx = np.arange(n)[np.random.uniform(size=n) < 0.9]
        cov2, ns2, sites2 = calc_covariance_vjs(y[idx], a, sl, idx=idx)
        assert cov.shape == (2**sl,)
        assert ns.shape == (2**sl,)
        assert sites.shape == (2**sl, sl)

        assert np.all(ns2 <= ns)
        assert np.all(cov2 != cov)

    def test_vc_fit(self):
        # Simulate data
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        print(lambdas)
        n_alleles, seq_length = 4, lambdas.shape[0] - 1
        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )
        f, X, y, y_var = vc.simulate(y_var=0.01)

        # Ensure MSE is within a small range
        vc = VCregression(n_alleles=n_alleles, seq_length=seq_length)
        vc.fit(X, y)
        sd1 = np.log2((vc.lambdas[1:] + 1e-6) / (lambdas[1:] + 1e-6)).std()
        assert sd1 < 2

        # Try with regularization and CV
        vc = VCregression(
            n_alleles=n_alleles,
            seq_length=seq_length,
            cross_validation=True,
        )
        vc.fit(X, y)
        sd2 = np.log2((vc.lambdas[1:] + 1e-6) / (lambdas[1:] + 1e-6)).std()
        assert sd2 < 1
        assert vc.beta > 0
        assert sd2 < sd1

        # Ensure taking into account experimental variance improves results
        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )
        vc.fit(X, y, y_var)
        sd3 = np.log2((vc.lambdas[1:] + 1e-6) / (lambdas[1:] + 1e-6)).std()
        assert sd3 < sd2

    def test_vc_process_data(self):
        # Simulate data
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        n_alleles, seq_length = 4, lambdas.shape[0] - 1
        vc = VCregression(
            n_alleles=n_alleles, seq_length=seq_length, lambdas=lambdas
        )
        _, X, y, y_var = vc.simulate(y_var=0.01)

        vc = VCregression(n_alleles=n_alleles, seq_length=seq_length)
        X, _, _, _, ns = vc.process_data((X, y, y_var))
        assert np.isclose(X.shape[0] ** 2, ns.sum())

    def test_vc_docs(self):
        # Simulate data
        np.random.seed(0)
        lambdas_true = np.array([1e3, 1e3, 2e2, 1e0, 1e-1, 3e-3, 1e-5])
        model = VCregression(
            seq_length=6, alphabet_type="dna", lambdas=lambdas_true
        )
        _, X, y, y_var = model.simulate(y_var=0.01, p_missing=0.1)

        # Without regularization
        model.fit(X, y, y_var)

        # Try with regularization and CV
        cvmodel = VCregression(
            seq_length=6,
            alphabet_type="dna",
            lambdas=lambdas_true,
            cross_validation=True,
        )
        cvmodel.fit(X, y, y_var)

        # Try with different CV metric
        cvmodel = VCregression(
            seq_length=6,
            alphabet_type="dna",
            lambdas=lambdas_true,
            cross_validation=True,
            cv_loss_function="logL",
        )
        cvmodel.fit(X, y, y_var)

    def test_vc_calc_posterior(self):
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2])
        n_alleles, seq_length = 4, lambdas.shape[0] - 1
        vc = VCregression(
            seq_length=seq_length,
            n_alleles=n_alleles,
            lambdas=lambdas,
            progress=False,
        )
        f, X, y, y_var = vc.simulate(y_var=0.01, p_missing=0.05)
        idx = vc.likelihood.idx.loc[X]
        X_test = np.delete(vc.genotypes, idx)
        f_test = np.delete(f, idx)
        n = X_test.shape[0]

        # Ensure that the posterior has the right form
        vc.set_data(X, y, y_var)
        m, S = vc.calc_posterior(X_pred=X_test)
        assert m.shape == (n,)
        assert S.shape == (n, n)

        # Ensure good performance on test data
        r = pearsonr(m, f_test)[0]
        assert r > 0.9

        # Test that the diagonal correspond to the posterior variances calculated individually
        y_var1 = np.diag(S @ np.eye(S.shape[1]))
        y_var2 = vc.predict(X_pred=X_test, calc_variance=True)["f_var"]
        assert np.allclose(y_var1, y_var2)

        # Test posterior of linear combination
        B = np.array([[0, 1, -1], [1, -1, 0], [1, 0, -1]])
        m, S = vc.calc_posterior(X_pred=X_test[:3], B=B)
        S = S @ np.eye(S.shape[1])
        idx = vc.likelihood.idx.loc[X_test[:3]]
        r = pearsonr(m, B @ f[idx])[0]
        assert r > 0.9

    def test_vc_contrasts(self):
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2])
        seq_length = lambdas.shape[0] - 1
        vc = VCregression(
            seq_length=seq_length,
            alphabet_type="dna",
            lambdas=lambdas,
            progress=False,
        )
        _, X, y, y_var = vc.simulate(y_var=0.01, p_missing=0.05)
        vc.set_data(X, y, y_var)
        seqs = ["AGCT", "AGCC", "TGCT"]

        # Test a single contrast
        contrast_matrix = pd.DataFrame({"c1": [0, 1, -1]}, index=seqs)
        results = vc.make_contrasts(contrast_matrix)
        assert results.shape == (1, 5)

        # Make contrasts between random test points
        contrast_matrix = pd.DataFrame(
            {"c1": [0, 1, -1], "c2": [0.5, 0.5, 0]}, index=seqs
        )
        results = vc.make_contrasts(contrast_matrix)
        assert results.shape == (2, 5)

        # Make contrasts for mutational effect and epistatic coefficient
        seqs = ["AGCT", "AGCC", "TGCT", "TGCC"]
        contrast_matrix = pd.DataFrame(
            {"T4C": [0, 0, 1, -1], "T4C:A1T": [1, -1, -1, 1]}, index=seqs
        )
        results = vc.make_contrasts(contrast_matrix)
        assert results.shape == (2, 5)

    def test_vc_predict(self):
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        n_alleles, seq_length = 4, lambdas.shape[0] - 1
        vc = VCregression(
            seq_length=seq_length, n_alleles=n_alleles, lambdas=lambdas
        )
        f, X, y, y_var = vc.simulate(y_var=0.01, p_missing=0.05)
        idx = vc.likelihood.idx.loc[X]
        X_test = np.delete(vc.genotypes, idx)
        f_test = np.delete(f, idx)

        # Using the a priori known variance components
        vc.set_data(X, y, y_var)
        pred = vc.predict()
        mse = np.mean((pred["f"] - f) ** 2)
        rho = pearsonr(pred["f"], f)[0]
        assert rho > 0.95
        assert mse < 0.05

        # Estimate posterior variances
        pred = vc.predict(X_pred=X_test, calc_variance=True)
        r = pearsonr(pred['f'], f_test)[0]
        p = np.mean((pred['ci_95_lower'] < f_test) & (f_test < pred['ci_95_upper']))
        assert "f_var" in pred.columns
        assert np.all(pred["f_var"] > 0)
        assert r > 0.9
        assert p > 0.9

        # Capture error with missing lambdas
        vc = VCregression(seq_length=seq_length, n_alleles=n_alleles)
        try:
            vc.set_data(X, y, y_var)
            pred = vc.predict()
            self.fail()
        except AttributeError:
            pass

    def test_vc_regression_bin(self):
        bin_fpath = join(BIN_DIR, "vc_regression.py")

        with NamedTemporaryFile() as fhand:
            data_fpath = "{}.data.csv".format(fhand.name)
            lambdas_fpath = "{}.lambdas.csv".format(fhand.name)
            out_fpath = "{}.out.csv".format(fhand.name)
            xpred_fpath = "{}.xpred.txt".format(fhand.name)

            # Simulate data
            seq_length = 5
            lambdas = np.exp(-np.arange(0, seq_length + 1))

            vc = VCregression(
                alphabet_type="dna", seq_length=seq_length, lambdas=lambdas
            )
            _, X, y, y_var = vc.simulate(y_var=0.0025, p_missing=0.05)
            X_test = np.delete(vc.genotypes, vc.genotype_idxs.loc[X])
            data = pd.DataFrame({'y': y, 'y_var': y_var}, index=X)

            # Save simulated data in temporary files
            data.to_csv(data_fpath)
            with open(xpred_fpath, "w") as fhand:
                for seq in X_test:
                    fhand.write(seq + "\n")

            with open(lambdas_fpath, "w") as fhand:
                for seq_length in lambdas:
                    fhand.write("{}\n".format(seq_length))

            # Direct kernel alignment
            cmd = [sys.executable, bin_fpath, data_fpath, "-o", out_fpath]
            check_call(cmd)

            # With known lambdas
            cmd = [
                sys.executable,
                bin_fpath,
                data_fpath,
                "-o",
                out_fpath,
                "-r",
                "--lambdas",
                lambdas_fpath,
            ]
            check_call(cmd)

            # Run with regularization
            cmd = [
                sys.executable,
                bin_fpath,
                data_fpath,
                "-o",
                out_fpath,
                "-r",
            ]
            check_call(cmd)

            # Predict few sequences and their variances under known lambdas
            cmd = [
                sys.executable,
                bin_fpath,
                data_fpath,
                "-o",
                out_fpath,
                "-r",
                "--var",
                "-p",
                xpred_fpath,
                "--lambdas",
                lambdas_fpath,
            ]
            check_call(cmd)


class SkewedVCTests(unittest.TestCase):
    def xtest_simulate_skewed_vc(self):
        np.random.seed(1)
        seq_length, n_alleles = 2, 2
        vc = VCregression()

        # With p=1
        ps = 1 * np.ones((seq_length, n_alleles))
        L = LaplacianOperator(n_alleles, seq_length)
        vc.init(seq_length, n_alleles, ps=ps)
        W = ProjectionOperator(L=L)
        for k1 in range(seq_length + 1):
            lambdas = np.zeros(seq_length + 1)
            lambdas[k1] = 1

            data = vc.simulate(lambdas)
            f = data["y_true"].values

            for k2 in range(seq_length + 1):
                W.set_lambdas(k=k2)
                f_k_rq = rayleigh_quotient(W, f)
                assert np.allclose(f_k_rq, k1 == k2)

        # with variable ps
        ps = (
            np.random.dirichlet(np.ones(n_alleles), size=seq_length)
            * n_alleles
        )
        L = LaplacianOperator(n_alleles, seq_length, ps=ps)
        vc.init(seq_length, n_alleles, ps=ps)
        W = ProjectionOperator(L=L)

        for k1 in range(seq_length + 1):
            lambdas = np.zeros(seq_length + 1)
            lambdas[k1] = 1

            data = vc.simulate(lambdas)
            f = data["y_true"].values

            for k2 in range(seq_length + 1):
                W.set_lambdas(k=k2)
                f_k_rq = rayleigh_quotient(W, f)
                assert np.allclose(f_k_rq, k1 == k2)


if __name__ == "__main__":
    sys.argv = ["", "VCTests"]
    unittest.main()
