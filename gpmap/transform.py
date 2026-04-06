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

        log_lambda_V = np.zeros(self.output_size)
        log_lambda_V[self.no_U_idx] = log_lambda_U

        if return_grad:
            grad = np.zeros((self.input_size, self.output_size))
            grad[np.arange(self.m), self.no_U_idx] = 1

        for i in self.U_idx:
            idx = np.where(self.V_to_U[i])[0]
            log_a_i = log_a[idx]

            x_i = logsumexp(log_a_i)
            log_lambda_V[i] = -(self.log_alphaP + x_i)
            if return_grad:
                grad[self.m + idx, i] = -np.exp(log_a_i - x_i)

        if return_grad:
            return log_lambda_V, grad
        else:
            return log_lambda_V
