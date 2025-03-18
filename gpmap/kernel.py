import numpy as np

from itertools import combinations
from scipy.special import comb, logsumexp

from gpmap.seq import seq_to_one_hot
from gpmap.utils import check_error
from gpmap.matrix import inner_product, get_sparse_diag_matrix


class SequenceKernel(object):
    def __init__(self, n_alleles, seq_length):
        self.alpha = n_alleles
        self.seq_length = seq_length
        self.lp1 = self.seq_length + 1
        self.n = self.alpha**seq_length

    def calc_hamming_distance(self, x1, x2):
        return self.seq_length - inner_product(x1, x2)

    def get_hamming_distance(self):
        if self._hamming is None:  # this happens when data is re-set
            self._hamming = self.calc_hamming_distance(self.x1, self.x2)
        return self._hamming

    def set_data(self, x1, x2=None, alleles=None, **kwargs):
        self.x1 = seq_to_one_hot(x1, alleles=alleles).astype(int)
        if x2 is None:
            self.x2 = self.x1
        else:
            self.x2 = seq_to_one_hot(x2, alleles=alleles).astype(int)

        if hasattr(self, "set_extra_data"):
            self.set_extra_data(**kwargs)
        self._hamming = None

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def grad(self, **kwargs):
        return self.backward(**kwargs)


class VarianceComponentKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles)
        self.calc_krawchouk_matrix()
        self.n_params = self.lp1

    def set_extra_data(self):
        self.same_seq = self.x1.dot(self.x2.T) == self.seq_length

    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.lp1):
            ss += (
                (-1.0) ** q
                * (self.alpha - 1.0) ** (k - q)
                * comb(d, q)
                * comb(self.seq_length - d, k - q)
            )
        return ss / self.n

    def calc_krawchouk_matrix(self):
        w_kd = np.zeros((self.lp1, self.lp1))
        for k in range(self.lp1):
            for d in range(self.lp1):
                w_kd[d, k] = self.calc_w_kd(k, d)
        self.W_kd = w_kd

    def forward(self, lambdas):
        hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
        cov = self.W_kd.dot(lambdas)[hamming_distance]
        return cov

    def backward(self, lambdas):
        hamming_distance = self.get_hamming_distance()
        for k, lambda_k in enumerate(lambdas):
            yield (
                self.transform_params_grad(lambda_k)
                * self.W_kd[:, k][hamming_distance]
            )

    def get_params0(self):
        params = np.append([-5], 2 - np.arange(self.seq_length))
        return params

    def transform_params(self, params):
        return np.exp(params)

    def transform_params_grad(self, f_params):
        return f_params

    def split_params(self, params):
        params_dict = {"lambdas": self.transform_params(params)}
        return params_dict


class ConnectednessKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, sites_equal=False):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles)
        self.n_params = self.lp1
        self.sites_equal = sites_equal

    def set_extra_data(self):
        self.same_seq = self.x1.dot(self.x2.T) == self.seq_length

    def calc_factor(self, rho):
        return np.log((1 + (self.alpha - 1) * rho) / (1 - rho))

    def get_rho(self, rho):
        rho = np.array(rho)
        if rho.shape[0] == 1:
            rho = np.full(self.seq_length, rho[0])
        elif rho.shape[0] != self.seq_length:
            msg = "Incorrect dimension of rho"
            raise ValueError(msg)
        return rho

    def calc_metric(self, rho):
        values = np.hstack([[self.calc_factor(r)] * self.alpha for r in rho])
        m = get_sparse_diag_matrix(values)
        return m

    def forward(self, rho):
        rho = self.get_rho(rho)
        if self.sites_equal:
            d = self.get_hamming_distance()
            s = np.log((1 + (self.alpha - 1) * rho[0]))
            cov = np.exp(
                d * (np.log(1 - rho[0]) - s)
                + self.seq_length * s
                - np.log(self.n)
            )
        else:
            metric = self.calc_metric(rho)
            c = np.prod(1 - rho)
            cov = c * np.exp(
                inner_product(self.x1, self.x2, metric=metric) - np.log(self.n)
            )

        self.cov = cov
        self.rho = rho
        return self.cov

    def backward(self, **kwargs):
        rho = self.rho
        sl = self.seq_length
        if self.sites_equal:
            rho = rho[0]
            d = self.get_hamming_distance()
            n = (self.alpha - 1) / (1 + (self.alpha - 1) * rho) * sl
            m = -self.alpha / ((1 - rho) * (1 + (self.alpha - 1) * rho))
            factor = m * d + n
            yield (factor * self.cov * self.transform_params_grad(rho))

        else:
            for p, rho_p in enumerate(rho):
                idx = np.arange(p * self.alpha, (p + 1) * self.alpha)
                s = inner_product(self.x1[:, idx], self.x2[:, idx])
                different_factor = -1 / (1 - rho_p)
                equal_factor = self.alpha - 1
                equal_factor /= 1 + (self.alpha - 1) * rho_p
                factor = equal_factor * s + (1 - s) * different_factor
                yield (factor * self.cov * self.transform_params_grad(rho_p))

    def get_params0(self):
        s = 1 if self.sites_equal else self.seq_length
        params = np.random.normal(size=s)
        return params

    def transform_params(self, params):
        return np.exp(params) / (1 + np.exp(params))

    def inv_transform_params(self, params):
        return np.log(params / (1 - params))

    def transform_params_grad(self, f_params):
        params = self.inv_transform_params(f_params)
        return np.exp(params) / (1 + np.exp(params)) ** 2

    def split_params(self, params):
        params_dict = {"rho": self.transform_params(params)}
        return params_dict


class SkewedVarianceComponentKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, q=None, use_p=False):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles)
        self.use_p = use_p

        if self.use_p:
            if q is None:
                q = (self.seq_length - 1) / self.seq_length
            self.q = q
            self.seq_lengthogq = np.log(q)
            self.B = self.calc_polynomial_coeffs()
            ks = np.arange(self.lp1)
            log_q_powers = self.seq_lengthogq * ks
            log_1mq_powers = np.append(
                [-np.inf], np.log(1 - np.exp(log_q_powers[1:]))
            )
            self.seq_lengthsf = self.seq_length * log_1mq_powers
            self.seq_lengthog_odds = log_q_powers - log_1mq_powers
        else:
            self.calc_krawchouk_matrix()

        self.n_params = self.lp1
        if use_p:
            self.n_params += self.seq_length * self.alpha

    def calc_polynomial_coeffs(self):
        k = np.arange(self.lp1)
        lambdas = np.exp(k * self.seq_lengthogq)

        B = np.zeros((self.lp1, self.lp1))
        idx = np.arange(self.lp1)
        for k in idx:
            k_idx = idx != k
            k_lambdas = lambdas[k_idx]
            norm_factor = 1 / np.prod(k_lambdas - lambdas[k])

            for power in idx:
                lambda_combs = list(
                    combinations(k_lambdas, self.seq_length - power)
                )
                p = np.sum([np.prod(v) for v in lambda_combs])
                B[power, k] = (-1) ** (power) * p * norm_factor

        return B

    def set_extra_data(self):
        self.same_seq = self.x1.dot(self.x2.T) == self.seq_length

    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.lp1):
            ss += (
                (-1.0) ** q
                * (self.alpha - 1.0) ** (k - q)
                * comb(d, q)
                * comb(self.seq_length - d, k - q)
            )
        return ss

    def calc_krawchouk_matrix(self):
        w_kd = np.zeros((self.lp1, self.lp1))
        for k in range(self.lp1):
            for d in range(self.lp1):
                w_kd[d, k] = self.calc_w_kd(k, d)
        self.W_kd = w_kd

    def _forward_ps(self, lambdas, log_p):
        coeffs = self.B.dot(lambdas)
        log_p_flat = log_p.flatten()
        M = np.diag(log_p_flat)

        cov = coeffs[0] * np.exp(-inner_product(self.x1, self.x2, M))
        cov *= self.same_seq

        for power in range(1, self.lp1):
            log_factors = np.stack(
                [
                    self.seq_lengthog_odds[power] - log_p_flat,
                    np.zeros(log_p_flat.shape),
                ],
                1,
            )
            log_factors = logsumexp(log_factors, 1)
            M = np.diag(log_factors)
            m = inner_product(self.x1, self.x2, M)
            cov += coeffs[power] * np.exp(self.seq_lengthsf[power] + m)
        return cov

    def forward(self, lambdas, log_p=None):
        if self.use_p:
            check_error(log_p is not None, msg="ps must be provided")
            cov = self._forward_ps(lambdas, log_p)
        else:
            hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
            cov = self.W_kd.dot(lambdas)[hamming_distance]
        return cov

    def backward(self, lambdas):
        if self.use_p:
            raise ValueError("Not implemented for variable p")
        else:
            hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
            for k, lambda_k in enumerate(lambdas):
                yield (lambda_k * self.W_kd[k, :][hamming_distance])

    def get_params0(self):
        params = np.append([-10], 2 - np.arange(self.seq_length))
        if self.use_p:
            params = np.append(params, np.zeros(self.seq_length * self.alpha))
        return params

    def split_params(self, params):
        params_dict = {}
        if self.use_p:
            params_dict["lambdas"] = np.exp(params[: self.lp1])
            log_ps = params[self.lp1:].reshape(
                self.seq_length, self.alpha
            )
            norm_factors = logsumexp(log_ps, axis=1)
            for i in range(log_ps.shape[1]):
                log_ps[:, i] -= log_ps[:, i] - norm_factors
            params_dict["ps"] = np.exp(log_ps)
        else:
            params_dict["lambdas"] = np.exp(params)
        return params_dict
