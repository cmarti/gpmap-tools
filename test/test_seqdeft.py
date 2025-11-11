#!/usr/bin/env python
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join
from tempfile import NamedTemporaryFile
from subprocess import check_call
from scipy.stats import pearsonr, multinomial
from scipy.special import logsumexp

from gpmap.settings import BIN_DIR
from gpmap.seq import generate_possible_sequences
from gpmap.matrix import quad
from gpmap.linop import ProjectionOperator, DeltaKernelRegularizerOperator
from gpmap.inference import SeqDEFT
from gpmap.plot.mpl import figure_SeqDEFT_summary, savefig


class SeqDEFTTests(unittest.TestCase):
    def test_init(self):
        X = np.array(["AAA", "ACA", "BAA", "BCA", "AAD", "ACD", "BAD", "BCD"])
        model = SeqDEFT(genotypes=X, P=2)
        model.set_data(X=X)
        assert model.seq_length == 3
        assert model.n_alleles == 2

        # Test giving incompatible sequences
        X = np.array(["AA", "AC", "BA", "BC"])
        try:
            model.set_data(X=X)
        except KeyError:
            pass

        # Test giving incompatible sequences in predefined RNA landscape
        model = SeqDEFT(P=2, seq_length=2, alphabet_type="rna")
        X = np.array(["AA", "AC", "BA", "BC"])
        try:
            model.set_data(X=X)
        except KeyError:
            pass

        # Test incomplete definition of sequence space
        try:
            model = SeqDEFT(P=2, alphabet_type="rna")
        except ValueError:
            pass

    def test_log_likelihood(self):
        model = SeqDEFT(P=2, seq_length=2, n_alleles=2)
        X = np.array(["00", "01", "10", "11"])
        phi = np.zeros(4)
        dist = multinomial(4, p=model.likelihood.phi_to_Q(phi))
        exp_ll = dist.logpmf([1, 1, 1, 1])
        model.set_data(X=X)
        ll = model.likelihood.calc_logL(phi)
        ll2 = model.likelihood.calc_logL(phi + 1)
        assert ll == exp_ll
        assert ll2 == exp_ll

        # Compute likelihood with inf phi
        X = np.array(["00", "01", "10"])
        model.set_data(X=X)
        phi[3] = np.inf
        ll = model.likelihood.calc_logL(phi)
        dist = multinomial(3, p=model.likelihood.phi_to_Q(phi))
        exp_ll = dist.logpmf([1, 1, 1, 0])
        assert np.isfinite(ll)
        assert np.allclose(ll, exp_ll)

        phi[1] = np.inf
        ll = model.likelihood.calc_logL(phi)
        assert np.isinf(ll)

    def test_loss(self):
        # With specific a
        model = SeqDEFT(a=10, P=2, seq_length=2, n_alleles=2)
        b = np.array([2, 1, 1.0])
        phi = model.b_to_phi(b)
        X = np.array(["00", "00", "01", "10"])

        # Calculate regular loss function
        model.set_data(X=X)
        loss, grad = model.calc_loss(phi, return_grad=True)
        hess = model.hess @ np.eye(4)
        assert np.allclose(loss, 9.484376662317993)
        assert np.allclose(grad, [-2.0, -0.47151776, -0.47151776, -0.54134113])
        assert np.all(np.linalg.eigvalsh(hess) > 0)
        assert np.allclose(
            hess,
            [
                [14, -10.0, -10.0, 10.0],
                [-10.0, 11.47151776, 10.0, -10.0],
                [-10.0, 10.0, 11.47151776, -10.0],
                [10.0, -10.0, -10.0, 10.54134113],
            ],
        )

        # With a=inf in additive landscape should have the same loss
        loss, grad = model.calc_loss_inf_a(b, return_grad=True)
        assert np.allclose(loss, 9.48437666231799)
        assert np.allclose(grad, [-1.74218833, 0.72932943, 0.72932943])

    def test_simulate(self):
        np.random.seed(2)
        a = 5e2
        model = SeqDEFT(P=2, a=a, seq_length=5, alphabet_type="dna")

        # Ensure right scale of phi
        phi = model.sample_prior()
        ss = quad(model.DP, phi) / model.n_genotypes
        ss = ss * a / model.DP.n_p_faces
        assert np.abs(ss - 1) < 0.1

        # Sample sequences directly
        phi, X = model.simulate(N=100)
        assert X.shape[0] == 100

        # Sample sequences indirectly
        X = model.likelihood.sample(phi, N=2000)
        assert X.shape[0] == 2000

        # Ensure frequencies correlate with probabilities
        x, y = np.unique(X, return_counts=True)
        merged = pd.DataFrame({"phi": phi}, index=model.genotypes)
        merged = merged.join(
            pd.DataFrame({"logy": np.log(y)}, index=x)
        ).dropna()
        r = pearsonr(-merged["phi"], merged["logy"])[0]
        assert r > 0.6

        # Verify zero penalty for the null space
        lambdas_P_inv = np.array([1e-6, 1])
        m = SeqDEFT(
            P=2,
            a=a,
            seq_length=5,
            alphabet_type="dna",
            lambdas_P_inv=lambdas_P_inv,
        )
        kernel_regularizer = DeltaKernelRegularizerOperator(
            m.kernel_basis, lambdas_P_inv
        )
        c = np.dot(phi, kernel_regularizer @ phi)
        assert np.allclose(c, 0)

        # Simulate from regularized model
        phi = m.sample_prior()
        c = np.dot(phi, kernel_regularizer @ phi)
        assert c > 0

    def test_invalid_a(self):
        try:
            SeqDEFT(P=2, a=-500)
            self.fail()
        except ValueError:
            pass

    def test_inference(self):
        np.random.seed(0)
        a = 500
        seq_length = 5
        model = SeqDEFT(seq_length=seq_length, alphabet_type="dna", P=2, a=a)
        phi, X = model.simulate(N=1000)

        # Infer with a=inf first
        model = SeqDEFT(
            seq_length=seq_length, alphabet_type="dna", P=2, a=np.inf
        )
        model.fit(X)
        probs = model.predict()
        assert np.allclose(probs["Q_star"].sum(), 1)

        # Ensure it can't learn a function with only k>P interactions
        r1 = pearsonr(-phi, np.log(probs["Q_star"]))[0]
        assert r1 < 0.05

        # Verify uniform distribution
        assert np.allclose(probs["Q_star"], 1.0 / probs.shape[0], atol=1e-3)

        # Ensure it is a probability distribution
        model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
        model.fit(X)
        probs = model.predict()
        assert np.allclose(probs["Q_star"].sum(), 1)

        # Ensure convergence
        _, grad = model.calc_loss(probs["phi"])
        assert np.allclose(grad, 0, atol=2e-2)

        # Ensure it is similar the true probabilities
        r2 = pearsonr(-phi, np.log(probs["Q_star"]))[0]
        assert r2 > 0.6

        # Test inference of maximum entropy model
        model = SeqDEFT(
            P=2, a=np.inf, seq_length=seq_length, alphabet_type="dna"
        )
        b = 2 * np.random.normal(size=model.kernel_basis.shape[1])
        phi = model.b_to_phi(b)
        X = model.likelihood.sample(phi, N=10000)
        model.fit(X)
        probs = model.predict()
        r3 = pearsonr(-phi, np.log(probs["Q_star"]))[0]
        assert r3 > 0.95

        # Ensure convergence
        b = model.phi_to_b(probs["phi"])
        _, grad = model.calc_loss(b)
        assert np.allclose(grad, 0, atol=1e-2)

    def test_inference_reg_null_space(self):
        np.random.seed(0)
        a = 500
        seq_length = 4
        lambdas_P_inv = np.array([1e-16, 50])

        model1 = SeqDEFT(seq_length=seq_length, alphabet_type="dna", P=2, a=a)
        model2 = SeqDEFT(
            P=2,
            a=a,
            seq_length=seq_length,
            alphabet_type="dna",
            lambdas_P_inv=lambdas_P_inv,
        )
        phi, X = model1.simulate(N=1000)

        # Fit models w/o regularizing the null space
        model1.fit(X)
        res1 = model1.predict()
        model2.fit(X)
        res2 = model2.predict()

        # Ensure regularizing works better
        r1 = pearsonr(res1["phi"], phi)[0]
        r2 = pearsonr(res2["phi"], phi)[0]
        assert r1 > 0.4
        assert r2 > r1

        # Fit a=inf models
        seq_length = 4
        r1, r2 = [], []
        for i in range(10):
            model = SeqDEFT(
                P=2,
                a=1e4,
                seq_length=seq_length,
                alphabet_type="dna",
                lambdas_P_inv=lambdas_P_inv,
            )
            phi, X = model.simulate(N=50)

            a = np.inf
            model1 = SeqDEFT(
                P=2, a=a, seq_length=seq_length, alphabet_type="dna"
            )
            model2 = SeqDEFT(
                P=2,
                a=a,
                seq_length=seq_length,
                alphabet_type="dna",
                lambdas_P_inv=lambdas_P_inv,
            )
            model1.fit(X)
            res1 = model1.predict()
            model2.fit(X)
            res2 = model2.predict()
            r1.append(pearsonr(res1["phi"], phi)[0])
            r2.append(pearsonr(res2["phi"], phi)[0])
        r1, r2 = np.mean(r1), np.mean(r2)
        assert r1 > 0.2
        assert r2 > r1

    def test_inference_baseline(self):
        np.random.seed(3)

        # With baseline in null space
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi_target = model.sample_prior()
        b = np.random.normal(size=model.kernel_basis.shape[1])
        phi_baseline = model.b_to_phi(b)
        phi_obs = phi_baseline + phi_target
        X = model.likelihood.sample(phi_obs, N=1000)

        model = SeqDEFT(P=2, a=np.inf, seq_length=5, alphabet_type="dna")
        model.fit(X)
        res1 = model.predict()
        model.fit(X, baseline_phi=phi_baseline, baseline_X=model.genotypes)
        res2 = model.predict()
        w1, w2 = np.log(res1["Q_star"]), np.log(res2["Q_star"])

        # Ensure solutions remain in null space and match
        assert np.allclose(np.dot(w1, model.DP @ w1), 0, atol=1e-10)
        assert np.allclose(
            np.dot(res2["phi"], model.DP @ res2["phi"]), 0, atol=1e-10
        )
        assert np.allclose(w1, w2, atol=1e-3)
        assert not np.allclose(res1["phi"], res2["phi"], atol=0.1)

        # Ensure baseline yields uncorrelated phi estimates
        r1 = pearsonr(res1["phi"], phi_baseline)[0]
        assert r1 > 0.4

        r2 = pearsonr(res2["phi"], phi_baseline)[0]
        assert r2 < 0.1

        # Adding high order interactions improves target inference
        model = SeqDEFT(P=2, a=500, seq_length=5, alphabet_type="dna")
        model.fit(X)
        res1 = model.predict()
        model.fit(X, baseline_phi=phi_baseline, baseline_X=model.genotypes)
        res2 = model.predict()

        w1, w2 = np.log(res1["Q_star"]), np.log(res2["Q_star"])
        assert np.allclose(w1, w2, atol=1e-2)

        r1 = pearsonr(res1["phi"], phi_target)[0]
        r2 = pearsonr(res2["phi"], phi_target)[0]
        assert r2 > r1

        # With baseline in column space
        model = SeqDEFT(P=2, a=1000, seq_length=5, alphabet_type="dna")
        baseline_phi = model.sample_prior()
        model.set_a(a=500)
        target_phi = model.sample_prior()
        obs_phi = baseline_phi + target_phi
        X = model.likelihood.sample(obs_phi, N=1000)
        model.fit(X, baseline_phi=baseline_phi, baseline_X=model.genotypes)
        probs1 = model.predict()
        model.fit(X)
        probs2 = model.predict()

        # Ensure adjusting improves prediction of the target phi
        r1 = pearsonr(probs1["phi"], target_phi)[0]
        r2 = pearsonr(probs2["phi"], target_phi)[0]
        assert r1 > r2

        # Ensure poor prediction of the baseline phi
        r3 = pearsonr(probs1["phi"], baseline_phi)[0]
        assert np.abs(r3) < 0.2

        # Ensure better prediction of the observed phi
        r4 = pearsonr(probs2["phi"], obs_phi)[0]
        r5 = pearsonr(probs1["phi"] + baseline_phi, obs_phi)[0]
        assert r5 > r4

    def test_inference_outside_interactions(self):
        np.random.seed(3)
        seq_length, a = 7, 4
        seqdeft_a = 50
        out = 3

        # Simulate a pairwise function on seq_length=5
        x = np.random.normal(size=a**seq_length)
        lambdas = np.zeros(seq_length + 1)
        lambdas[1:3] = [300, 75]
        P = ProjectionOperator(a, seq_length, lambdas=np.sqrt(lambdas))
        phi = P @ x

        # Ensure lack of higher order components
        P3 = ProjectionOperator(a, seq_length, k=3)
        k3 = quad(P3, phi)
        assert k3 < 1e-8

        # Average out last 2 positions
        seqs = np.array(list(generate_possible_sequences(seq_length)))
        baseline = pd.DataFrame(
            {
                "seqs": seqs,
                "phi": phi,
                "subseq": [x[:-out] for x in seqs],
                "Q_star": np.exp(phi - logsumexp(phi)),
            }
        )
        baseline_phi = -np.log(baseline.groupby("subseq")["Q_star"].sum())

        # Ensure some higher order component induced by missing sites
        P3 = ProjectionOperator(a, seq_length - out, k=3)
        k3_short = quad(P3, baseline_phi)
        assert k3_short > 1e3 * k3

        # Simulate from prior at seq_length=4 with baseline phi
        model = SeqDEFT(
            P=3, a=seqdeft_a, seq_length=seq_length - out, alphabet_type="rna"
        )
        phi = model.sample_prior()
        X = model.likelihood.sample(phi + baseline_phi.values, N=2000)

        # Fit model without baseline
        model.fit(X=X)
        pred = model.predict()
        phi1 = pred["phi"]
        r1 = pearsonr(phi, phi1)[0]

        # Fit model with baseline should increase correlation with true phi
        model.fit(
            X=X,
            baseline_phi=baseline_phi.values,
            baseline_X=baseline_phi.index.values,
        )
        pred = model.predict()
        phi2 = pred["phi"]
        r2 = pearsonr(phi, phi2)[0]
        assert r2 > r1

    def test_predict(self):
        np.random.seed(0)
        a = 200
        seq_length = 5
        model = SeqDEFT(seq_length=seq_length, alphabet_type="dna", P=2, a=a)
        phi, X = model.simulate(N=1000)
        X_pred = np.random.choice(model.genotypes, 100, replace=False)
        idx = model.get_obs_idx(X_pred)
        phi_test = phi[idx]

        # Infer
        model = SeqDEFT(seq_length=seq_length, alphabet_type="dna", P=2, a=a)
        model.set_data(X)
        result = model.predict(X_pred=X_pred, calc_variance=True)
        assert result.shape == (X_pred.shape[0], 5)
        assert np.all(result.index == X_pred)

        c = (phi_test - result["f"]).mean()
        result["phi_true"] = phi_test - c

        r = pearsonr(result["f"], result["phi_true"])[0]
        calibration = np.mean(
            (result["ci_95_lower"] < result["phi_true"])
            & (result["phi_true"] < result["ci_95_upper"])
        )
        assert r > 0.65
        assert calibration > 0.9

    def test_contrast(self):
        np.random.seed(0)
        a = 200
        seq_length = 5
        model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
        phi, X = model.simulate(N=1000)

        seqs = ["AGCTA", "AGCTG"]
        idx = model.get_obs_idx(seqs)
        mut_eff = phi[idx[1]] - phi[idx[0]]

        # Inference
        model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
        model.set_data(X)

        # Contrasts
        contrast = pd.DataFrame({seqs[0]: [-1], seqs[1]: [1]}, index=["G5A"]).T
        result = model.make_contrasts(contrast)
        assert result.shape == (1, 5)
        assert result.index[0] == "G5A"

        result = result.loc["G5A", :]
        assert result["ci_95_lower"] < mut_eff
        assert result["ci_95_upper"] > mut_eff

    def test_inference_cv(self):
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi, X = model.simulate(N=1000)

        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2)
        model.fit(X=X)
        pred = model.predict()
        assert np.allclose(pred["Q_star"].sum(), 1)

        # Ensure approximate inference of a
        logfc = np.log2(model.a / 500)
        assert np.abs(logfc) < 1

        # Ensure it is similar the true probabilities
        r = pearsonr(-phi, np.log(pred["Q_star"]))[0]
        assert r > 0.6

    def test_inference_weigths(self):
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi, X = model.simulate(N=1000)

        y = np.exp(np.random.normal(size=X.shape[0]))
        model.fit(X=X, y=y)
        pred = model.predict()
        assert np.allclose(pred["Q_star"].sum(), 1)

        logfc = np.log2(model.a / 500)
        assert np.abs(logfc) < 5

        r = pearsonr(-phi, np.log(pred["Q_star"]))[0]
        assert r > 0.4

    def test_inference_phylo_correction(self):
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi, X = model.simulate(N=1000)
        X = np.array([x + np.random.choice(X) for x in X])
        positions = np.arange(5)

        model.fit(X=X, phylo_correction=True, positions=positions)
        pred = model.predict()
        assert np.allclose(pred["Q_star"].sum(), 1)
        assert len(pred.index[0]) == 5

        logfc = np.log2(model.a / 500)
        assert np.abs(logfc) < 1

        r = pearsonr(-phi, np.log(pred["Q_star"]))[0]
        assert r > 0.6

    def test_inference_adjusted_logq(self):
        model = SeqDEFT(P=2, a=500, seq_length=4, alphabet_type="dna")
        phi, X = model.simulate(N=1000)
        X = np.array([x + np.random.choice(X) for x in X])
        positions = np.arange(4)

        allele_freqs = {"A": 0.3, "G": 0.3, "C": 0.2, "T": 0.2}
        model.fit(
            X=X,
            positions=positions,
            adjust_freqs=True,
            allele_freqs=allele_freqs,
        )
        pred = model.predict()
        assert np.allclose(pred["Q_star"].sum(), 1)
        assert np.allclose(pred["Q_adj"].sum(), 1)
        assert len(pred.index[0]) == 4

        # Ensure adjustmnet was done in the right direction
        assert pred.loc["AAAA", "Q_star"] > pred.loc["AAAA", "Q_adj"]
        assert pred.loc["TTTT", "Q_star"] < pred.loc["TTTT", "Q_adj"]

        r = pearsonr(-phi, np.log(pred["Q_star"]))[0]
        assert r > 0.5

    def test_missing_alleles(self):
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi, X = model.simulate(N=2000)
        X = X[np.array([x[0] not in ["A", "C"] for x in X])]

        model.fit(X=X)
        pred = model.predict()
        assert np.allclose(pred["Q_star"].sum(), 1)

        missing = pred.loc[[x[0] == "A" for x in pred.index], :]
        assert missing["frequency"].sum() == 0
        assert missing["Q_star"].sum() < 1e-6

    def test_very_few_sequences(self):
        np.random.seed(0)
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        phi, X = model.simulate(N=75)

        model.fit(X=X)
        pred = model.predict()
        r = pearsonr(-phi, np.log(pred["Q_star"]))[0]
        assert np.allclose(pred["Q_star"].sum(), 1)
        assert r > 0.1

    def test_bin(self):
        bin_fpath = join(BIN_DIR, "fit_seqdeft.py")

        model = SeqDEFT(P=2, seq_length=5, alphabet_type="dna", a=500)
        _, X = model.simulate(N=1000)

        with NamedTemporaryFile() as fhand:
            x_fpath = "{}.x.txt".format(fhand.name)
            with open(x_fpath, "w") as input_fhand:
                for x in X:
                    input_fhand.write(x + "\n")

            cmd = [
                sys.executable,
                bin_fpath,
                x_fpath,
                "-o",
                fhand.name + ".pq",
            ]
            check_call(cmd + ["--get_a_values"])
            check_call(cmd)

    def test_cv_plot(self):
        model = SeqDEFT(seq_length=5, alphabet_type="dna", P=2, a=500)
        _, X = model.simulate(N=1000)

        model = SeqDEFT(P=2, seq_length=5, alphabet_type="dna")
        pred = model.fit(X=X)
        log_Ls = model.logL_df

        with NamedTemporaryFile(mode="w") as fhand:
            fig = figure_SeqDEFT_summary(log_Ls, pred, legend_loc=2)
            fpath = fhand.name
            savefig(fig, fpath)

    # def test_predictive_distribution(self):
    #     np.random.seed(1)
    #     a = 1000
    #     seq_length = 5
    #     model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
    #     phi = model.sample_prior()
    #     X = model.simulate(N=250, phi=phi)
    #     X_test = model.simulate(N=10000, phi=phi)

    #     # Infer with a=inf first
    #     a_values = np.linspace(500, 1500, 11)
    #     for a in a_values:
    #         model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
    #         model.set_data(X=X)
    #         phi1, cov = model.calc_posterior()
    #         ll1 = model.cv_evaluate((X_test, None, None), phi1)

    #         # predictive distribution
    #         cov = cov @ np.eye(cov.shape[1])
    #         L = np.linalg.cholesky(cov)
    #         x = np.random.normal(size=(phi1.shape[0], 2000))
    #         samples = phi1[:, None] + L @ x
    #         p = np.exp(-samples) / np.exp(-samples).sum(0)
    #         phi2 = -np.log(p.mean(1))
    #         ll2 = model.cv_evaluate((X_test, None, None), phi2)
    #         print(a, ll1, ll2)

    # def test_hmc(self):
    #     def logp(x):
    #         return np.sum(0.5 * x**2), x

    #     def logp_grad(x):
    #         return x

    #     sampler = HMC(logp, logp_grad, step_size=1, path_length=1)
    #     n = 1000

    #     posterior = []
    #     for i in range(4):
    #         x0 = np.random.normal(size=1)
    #         samples = np.array([s for s in sampler.sample(x0, n)])
    #         print(sampler.num_acceptance / n)
    #         print(samples.mean(0), samples.std(0))
    #         plt.plot(samples.T[0])

    #         posterior.append(samples.T)
    #     posterior = np.array(posterior)
    #     print(posterior.shape)

    #     rhats = sampler.compute_R_hat(posterior)
    #     print(rhats)
    #     plt.show()

    # def test_mcmc(self):
    #     np.random.seed(0)
    #     a = 500
    #     seq_length = 4
    #     model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
    #     phi = model.sample_prior()
    #     X = model.simulate(N=1000, phi=phi)

    #     model = SeqDEFT(P=2, a=a, seq_length=seq_length, alphabet_type="dna")
    #     model.set_data(X=X)
    #     posterior = model.mcmc(n_samples=100, n_chains=4)
    #     assert posterior.shape == (400, model.n_genotypes)

    #     post_mean = posterior.mean(0)
    #     result = model.fit(X=X)
    #     r = pearsonr(post_mean, result["phi"])[0]
    #     print(r)
    #     assert r > 0.95


if __name__ == "__main__":
    sys.argv = ["", "SeqDEFTTests"]
    unittest.main()
