from itertools import combinations, product

import numpy as np
from scipy.special import gammaln, logsumexp


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


class DeltaPtoVkTransform:
    def __init__(self, n_alleles, seq_length, P):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.P = P
        self.log_alphaP = P * np.log(self.n_alleles)

        self.DeltaP_log_lambdas = []
        for k in range(self.seq_length + 1):
            if k < P:
                self.DeltaP_log_lambdas.append(-16)
            else:
                self.DeltaP_log_lambdas.append(self.log_alphaP + log_comb(k, P))
        self.input_size = self.P + 1
        self.output_size = self.seq_length + 1

    def get_log_lambda_k(self, x):
        return x[: self.P]

    def get_log_a(self, x):
        return x[-1]
    
    def __call__(self, x, return_grad=True):
        if x.shape[0] != self.input_size:
            msg = f"input x should have size {self.input_size} but got {x.shape[0]}"
            raise ValueError(msg)

        log_lambda_k = self.get_log_lambda_k(x)
        log_a = self.get_log_a(x)

        log_lambda_m = np.zeros(self.seq_length + 1)
        log_lambda_m[:self.P] = log_lambda_k
        log_lambda_m[self.P:] = -log_a - self.DeltaP_log_lambdas[self.P:]

        if return_grad:
            grad = np.zeros((self.input_size, self.output_size))
            idx = np.arange(self.P)
            grad[idx, idx] = 1
            grad[-1, self.P:] = -1
            return log_lambda_m, grad
        else:
            return log_lambda_m


class ConnectednessToVUTransform:
    def __init__(self, n_alleles, seq_length):
        self.n_alleles = n_alleles
        self.seq_length = seq_length

        # All possible subsets of sites
        U = np.array(list(product([False, True], repeat=seq_length)))
        U0 = self.seq_length - U.sum(1)
        self.U = np.hstack([U0[:, None], U])
        self.input_size = self.seq_length + 1
        self.output_size = 2**seq_length

    def __call__(self, x, return_grad=True):
        if x.shape[0] != self.input_size:
            msg = f"input x should have size {self.input_size} but got {x.shape[0]}"
            raise ValueError(msg)

        log_lambda_U = self.U @ x
        if return_grad:
            grad = self.U.T
            return log_lambda_U, grad
        else:
            return log_lambda_U


class DeltaUtoVUTransform:
    def __init__(self, n_alleles, seq_length, P):
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.P = P
        self.log_alphaP = P * np.log(self.n_alleles)

        # All possible subsets of sites
        self.V = np.array(list(product([False, True], repeat=seq_length)))

        # All subsets of size P
        self.Us = list(combinations(range(self.seq_length), P))
        self.n_Us = len(self.Us)

        # Matrix storing whether all sites in U are in each V
        self.V_to_U = np.array(
            [[np.all([x[s] for s in U]) for U in self.Us] for x in self.V]
        )

        self.no_U_idx = self.V.sum(1) < P
        self.U_idx = np.where(~self.no_U_idx)[0]
        self.m = self.no_U_idx.sum()
        self.input_size = self.m + self.n_Us
        self.output_size = 2**seq_length

        # Precompute indices and masks used in __call__.
        self._m_idx = np.arange(self.m)
        self.mask_bool = self.V_to_U[self.U_idx]
        self.mask = self.mask_bool.astype(float, copy=False)

    def get_log_lambda_U(self, x):
        return x[: self.m]

    def get_log_a_U(self, x):
        return x[self.m :]

    def __call__(self, x, return_grad=True):
        if x.shape[0] != self.input_size:
            msg = f"input x should have size {self.input_size} but got {x.shape[0]}"
            raise ValueError(msg)

        log_lambda_U = self.get_log_lambda_U(x)
        log_a = self.get_log_a_U(x)
        U_idx = self.U_idx
        m = self.m
        log_alphaP = self.log_alphaP

        log_lambda_V = np.empty(self.output_size)
        log_lambda_V[self.no_U_idx] = log_lambda_U

        x_vec = logsumexp(log_a[None, :], b=self.mask, axis=1)
        log_lambda_V[U_idx] = -(log_alphaP + x_vec)

        if not return_grad:
            return log_lambda_V

        grad = np.zeros((self.input_size, self.output_size))
        grad[self._m_idx, self.no_U_idx] = 1

        # Compute exp only where entries contribute.
        probs = np.zeros_like(self.mask)
        np.exp(log_a[None, :] - x_vec[:, None], out=probs, where=self.mask_bool)
        grad[m:, U_idx] = -probs.T
        return log_lambda_V, grad
