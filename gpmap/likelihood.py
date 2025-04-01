#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.special import loggamma, logsumexp

from gpmap.linop import SelIdxOperator, DiagonalOperator, IdentityOperator
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
            index=self.genotypes,
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


class GaussianLikelihood(object):
    """
    GaussianLikelihood class for modeling Gaussian likelihoods.

    This class provides methods for setting data, calculating loss, gradients,
    and Hessians, as well as sampling from the Gaussian likelihood.

    Parameters
    ----------
    genotypes : array-like
        List of genotypes in the sequence space.

    """
    def __init__(self, genotypes):
        self.genotypes = genotypes
        self.n_genotypes = genotypes.shape[0]

    def set_data(self, X, y, y_var=None):
        '''
        Set the data for the Gaussian likelihood model.

        Parameters
        ----------
        X : array-like
            Observed genotypes or input data.
        y : array-like
            Observed values corresponding to the input data X.
        y_var : array-like, optional
            Variance of the observed values y. If not provided, it is assumed
            to be zero (no variance).

        Raises
        ------
        ValueError
            If the y vector contains NaN values.
        '''
        if y is None:
            y = np.zeros(X.shape[0])
        
        if np.any(np.isnan(y)):
            msg = "y vector contains nans"
            raise ValueError(msg)

        self.X = X
        self.y = y
        self.zero_var = y_var is None
        self.y_var = np.zeros_like(y) if y_var is None else y_var
        self.n_obs = self.X.shape[0]
        self.constant = 0.5 * self.n_obs * np.log(2 * np.pi)

        self.idx = pd.Series(np.arange(self.n_genotypes), index=self.genotypes)
        self.Xop = SelIdxOperator(self.n_genotypes, self.idx.loc[X])

        self._Zop = None
        self._D = None
        self._D_var = None
        self._D_var_inv = None
        self._D_var_inv_sqrt = None
        self._D_var_sqrt = None
        self._logdet = None

    @property
    def Zop(self):
        if not hasattr(self, "_Zop") or self._Zop is None:
            z = np.full(self.n_genotypes, True)
            z[self.idx.loc[self.X]] = False
            self.pred_idx = np.where(z)[0]
            self._Zop = SelIdxOperator(self.n_genotypes, self.pred_idx)
        return self._Zop

    @property
    def D_var(self):
        if not hasattr(self, "_D_var") or self._D_var is None:
            self._D_var = DiagonalOperator(self.y_var)
        return self._D_var

    @property
    def D_var_inv(self):
        if not hasattr(self, "_D_var_inv") or self._D_var_inv is None:
            if np.any(self.y_var == 0):
                raise ValueError("y_var cannot contain zero values")
            self._D_var_inv = DiagonalOperator(1.0 / self.y_var)
        return self._D_var_inv

    @property
    def D_var_inv_sqrt(self):
        if (
            not hasattr(self, "_D_var_inv_sqrt")
            or self._D_var_inv_sqrt is None
        ):
            if np.any(self.y_var == 0):
                raise ValueError("y_var cannot contain zero values")
            self._D_var_inv_sqrt = DiagonalOperator(1.0 / np.sqrt(self.y_var))
        return self._D_var_inv_sqrt

    @property
    def D_var_sqrt(self):
        if not hasattr(self, "_D_var_sqrt") or self._D_var_sqrt is None:
            self._D_var_sqrt = DiagonalOperator(np.sqrt(self.y_var))
        return self._D_var_sqrt

    @property
    def D(self):
        if not hasattr(self, "_D") or self._D is None:
            self._D = self.Xop.transpose() @ self.D_var_inv @ self.Xop
        return self._D

    @property
    def logdet(self):
        if not hasattr(self, "_logdet") or self._logdet is None:
            self._logdet = self.D_var.logdet()
        return self._logdet

    def calc_loss_grad_hess(self, phi):
        # TODO: reivew when we need to express it as a function of complete phi
        # all unobserved entries are 0 anyway
        diff = self.Xop.transpose() @ (self.y - self.Xop @ phi)
        grad = self.D @ diff
        loss = 0.5 * np.dot(phi, grad)
        return (loss, grad, self.D)

    def calc_logL(self, phi):
        loss = self.calc_loss_grad_hess(phi)[0]
        logL = -0.5 * self.logdet - loss - self.constant
        return logL

    def sample(self, phi, seed=None):
        '''
        Generate samples from the Gaussian likelihood model.

        Parameters
        ----------
        phi : array-like
            The mean function used to sample data with added noise.
        seed : int, optional
            A random seed for reproducibility. If None, the random number generator
            will not be seeded.

        Returns
        -------
        array-like
            Samples generated from the Gaussian likelihood model.
        '''
        if seed is not None:
            np.random.seed(seed)

        z = np.random.normal(size=self.n_obs)
        D = DiagonalOperator(np.sqrt(self.y_var))
        y = self.Xop @ phi + D @ z
        return y
