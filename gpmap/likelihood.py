#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.special import loggamma, logsumexp

from gpmap.seq import (
    calc_allele_frequencies,
    calc_expected_logp,
    calc_genetic_code_aa_freqs,
    calc_msa_weights,
    get_subsequences,
)
from gpmap.utils import (
    check_error,
    safe_exp,
)


class PhiRegularizer(object):
    def __init__(self, n_genotypes, phi_lower=0, phi_upper=100):
        self.phi_upper = phi_upper
        self.phi_lower = phi_lower
        self.n_genotypes = n_genotypes

    def calc_grad(self, phi):
        grad = np.zeros(self.n_genotypes)
        idx1 = phi > self.phi_upper
        idx2 = phi < self.phi_lower

        if idx1.any() > 0:
            dphi = phi[idx1] - self.phi_upper
            grad[idx1] += 2 * dphi

        if idx2.any() > 0:
            dphi = phi[idx2] - self.phi_lower
            grad[idx2] += 2 * dphi
        return grad

    def calc_loss_grad_hess(self, phi):
        loss = 0
        grad = np.zeros(self.n_genotypes)
        hess = np.zeros(self.n_genotypes)

        idx1 = phi > self.phi_upper
        idx2 = phi < self.phi_lower

        if idx1.any() > 0:
            dphi = phi[idx1] - self.phi_upper
            loss += np.sum(dphi**2)
            grad[idx1] += 2 * dphi
            hess[idx1] = 2

        if idx2.any() > 0:
            dphi = phi[idx2] - self.phi_lower
            loss += np.sum(dphi**2)
            grad[idx2] += 2 * dphi
            hess[idx2] = 2
        return (loss, grad, hess)


class SeqDEFTLikelihood(object):
    def __init__(self, genotypes):
        self.genotypes = genotypes
        self.n_genotypes = genotypes.shape[0]
        self.regularizer = PhiRegularizer(self.n_genotypes)

    def set_offset(self, offset=None):
        if offset is not None:
            msg = "Size of the offset ({}) is different the number of"
            msg = (msg + "genotypes ({})").format(
                offset.shape, self.n_genotypes
            )
            check_error(offset.shape[0] == self.n_genotypes, msg=msg)
        self.offset = offset

    def fill_zeros_counts(self, X, y):
        obs = (
            pd.DataFrame({"x": X, "y": y})
            .groupby(["x"])["y"]
            .sum()
            .reset_index()
        )
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        try:
            data.loc[obs["x"].values] = obs["y"].values
        except KeyError:
            msg = "Sequences outside of sequence space found"
            raise KeyError(msg)
        return data

    def set_data(
        self,
        X,
        y=None,
        offset=None,
        positions=None,
        phylo_correction=False,
        adjust_freqs=False,
        allele_freqs=None,
    ):
        self.positions = positions
        self.adjust_freqs = adjust_freqs
        self.phylo_correction = phylo_correction
        self.set_offset(offset)

        if y is None:
            y = calc_msa_weights(X, phylo_correction=self.phylo_correction)

        if self.adjust_freqs:
            if allele_freqs is None:
                allele_freqs = calc_allele_frequencies(X, y=y)
            elif isinstance(allele_freqs, dict):
                self.allele_freqs = allele_freqs
            else:
                self.allele_freqs = calc_genetic_code_aa_freqs(allele_freqs)

        self.X = get_subsequences(X, positions=self.positions)
        self.y = y
        self.y_var = None

        data = self.fill_zeros_counts(self.X, y).values
        self.N = data.sum()
        self.R = data / self.N
        self.counts = data
        self.multinomial_constant = (
            loggamma(self.counts.sum() + 1) - loggamma(self.counts + 1).sum()
        )
        self.obs_idx = data > 0.0

    def phi_to_phi_obs(self, phi):
        return phi + self.offset if self.offset is not None else phi

    def calc_grad(self, phi):
        obs_phi = self.phi_to_phi_obs(phi)
        reg_grad = self.regularizer.calc_grad(obs_phi)
        N_exp_phi = self.N * safe_exp(-obs_phi)
        grad = self.counts - N_exp_phi + reg_grad
        return grad

    def calc_loss_grad_hess(self, phi):
        obs_phi = self.phi_to_phi_obs(phi)
        reg_loss, reg_grad, reg_hess = self.regularizer.calc_loss_grad_hess(
            obs_phi
        )
        N_exp_phi = self.N * safe_exp(-obs_phi)
        loss = self.N * np.dot(self.R, obs_phi) + N_exp_phi.sum() + reg_loss
        grad = self.counts - N_exp_phi + reg_grad
        hess = N_exp_phi + reg_hess
        return (loss, grad, hess)

    def phi_to_logQ(self, phi):
        return -phi - logsumexp(-phi)

    def phi_to_Q(self, phi):
        return np.exp(self.phi_to_logQ(phi))

    def calc_logL(self, phi):
        c = self.multinomial_constant
        obs_phi = self.phi_to_phi_obs(phi)
        logq = self.phi_to_logQ(obs_phi)
        return c + np.dot(self.counts[self.obs_idx], logq[self.obs_idx])

    def phi_to_output(self, phi):
        obs_phi = self.phi_to_phi_obs(phi)
        Q = self.phi_to_Q(obs_phi)
        output = pd.DataFrame(
            {"frequency": self.R, "phi": phi, "Q_star": Q},
            index=self.genotypes
        )
        if self.adjust_freqs:
            exp_logp = calc_expected_logp(self.genotypes, self.allele_freqs)
            logp_adj = np.log(Q) - exp_logp
            output["Q_adj"] = self.phi_to_Q(-logp_adj)

        return output

    def sample(self, phi, N, seed=None):
        if seed is not None:
            np.random.seed(seed)
        Q = self.phi_to_Q(phi)
        X = np.random.choice(self.genotypes, size=N, replace=True, p=Q)
        return X
