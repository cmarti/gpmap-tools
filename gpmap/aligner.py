from itertools import product

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import least_squares, lsq_linear, minimize
from scipy.special import comb, gammaln

from gpmap.linop import DeltaPOperator
from gpmap.matrix import kron
from gpmap.transform import (
    ConnectednessToVUTransform,
    DeltaPtoVkTransform,
    DeltaUtoVUTransform,
)
from gpmap.utils import check_error


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


class FrobeniusNorm:
    def __init__(self, covs, ns, W):
        covs = np.asarray(covs, dtype=float)
        ns = np.asarray(ns, dtype=float)
        W = np.asarray(W, dtype=float)

        if np.any(ns < 0):
            raise ValueError("ns must be nonnegative")

        sqrt_ns = np.sqrt(ns)
        self.A = sqrt_ns[:, None] * W.T
        self.b = sqrt_ns * covs
        self.b[np.isnan(covs)] = 0
    
    def calc_weighted_residuals(self, log_lambdas):
        lambdas = np.exp(log_lambdas)
        residual = self.A @ lambdas - self.b
        return(residual)

    def calc_weighted_residuals_grad(self, log_lambdas):
        lambdas = np.exp(log_lambdas)
        grad = lambdas[:, None] * self.A.T
        return(grad)

    def __call__(self, log_lambdas, return_grad=True):
        if return_grad:
            residual = self.calc_weighted_residuals(log_lambdas)
            grad = self.calc_weighted_residuals_grad(log_lambdas)
            Frob = residual @ residual
            grad = 2 * grad @ residual
            return Frob, grad
        else:
            residual = self.calc_weighted_residuals(log_lambdas)
            Frob = residual @ residual
            return Frob


class VCLogLambdaRegularizer:
    def __init__(self, seq_length, beta=0):
        self.seq_length = seq_length
        self.set_beta(beta)
        self.calc_second_order_diff_matrix()

    def set_beta(self, beta):
        check_error(beta >= 0, msg="beta must be >= 0")
        self.beta = beta

    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i : i + 3] = [-1, 2, -1]
        self.second_order_diff_matrix = Diff2.T.dot(Diff2)

    def __call__(self, log_lambdas, return_grad=True):
        if self.beta == 0:
            reg = 0
        else:
            reg_Av = self.second_order_diff_matrix @ log_lambdas[1:]
            reg = self.beta * np.dot(reg_Av, log_lambdas[1:])

        if return_grad:
            if self.beta == 0:
                grad = np.zeros_like(log_lambdas)
            else:
                grad = self.beta * np.append([0], 2 * reg_Av)
            return reg, grad
        else:
            return reg


class SitesVCLogLambdaRegularizer:
    def __init__(self, seq_length, beta=0):
        self.seq_length = seq_length
        self.set_beta(beta)
        A = DeltaPOperator(n_alleles=2, seq_length=seq_length, P=2) 
        A = 1.0 / A.n_p_faces * A @ np.eye(2**seq_length)
        A = 0.5 * (A + A.T)
        self.L = sqrtm(A)

    def set_beta(self, beta):
        check_error(beta >= 0, msg="beta must be >= 0")
        self.beta = beta

    def __call__(self, log_lambdas, return_grad=True):
        if self.beta == 0:
            reg = 0
        else:
            Lv = self.L @ log_lambdas
            reg = self.beta * np.dot(Lv, Lv)

        if return_grad:
            if self.beta == 0:
                grad = np.zeros_like(log_lambdas)
            else:
                grad = self.beta * 2 * self.L.T @ Lv
            return reg, grad
        else:
            return reg


class KernelAligner:
    def __init__(self, n_alleles, seq_length):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.eta = self.n_alleles - 1
        self.n_genotypes = n_alleles**seq_length

    def set_data(self, covs, ns, mean=0):
        self.mean = mean
        if covs.shape[0] != ns.shape[0]:
            msg = "covs and ns must be the same shape"
            raise ValueError(msg)
        if covs.shape[0] != self.n_covs:
            msg = f"covs size should match the number of distance classes: {self.n_covs}"
            raise ValueError(msg)

        self.covs = covs
        self.ns = ns
        self.frobenius_norm = FrobeniusNorm(covs, ns, self.W)
    
    def calc_residuals(self, x):
        return self.frobenius_norm.calc_weighted_residuals(x)

    def calc_residuals_grad(self, x):
        return self.frobenius_norm.calc_weighted_residuals_grad(x)
    
    def x_to_params(self, x):
        raise NotImplementedError("x_to_params must be implemented in subclasses")
    
    def get_x0(self):
        raise NotImplementedError("get_x0 must be implemented in subclasses")
    
    def fit(self, covs, ns, mean=0, x0=None):
        """
        Fits kernel parameters by minimizing the Frobenius Norm
        with the empirical covariance between sequences at different
        distance classes.

        Parameters
        ----------
        covs : array-like of shape (2 ** seq_length)
            Average empirical second moments at every possible
            combination of sites.
        ns : array-like of shape (2 ** seq_length)
            Number of pairs of sequences at every possible combination of sites.
        mean : float, optional
            Mean value to subtract from the covariances. Default is 0.
        x0 : array-like, optional
            Initial guess for the optimization. If None, it will be
            determined automatically. Default is None.

        Returns
        -------
        params : array-like or tuple of array-like
            Parameter values that best fit the empirical second moments.
        """
        self.set_data(covs, ns, mean=mean)
        if x0 is None:
            x0 = self.get_x0()
            
        bounds = [np.full_like(x0, -30.0), np.full_like(x0, 30.0)]
        res = least_squares(
            self.calc_residuals,
            jac=self.calc_residuals_grad,
            x0=x0,
            bounds=bounds,
            x_scale='jac',
            ftol=1e-12,
            method="trf",
        )
        
        if not res.success:
            msg = f"kernel alignment did not converge: {res}"
            raise ValueError(msg)

        self.res = res
        return self.x_to_params(res.x)


class RegularizedKernelAligner(KernelAligner):
    def __init__(self, n_alleles, seq_length, regularizer, beta=0):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.regularizer = regularizer(seq_length=seq_length, beta=beta)
    
    def get_x0(self):
        raise NotImplementedError("get_x0 must be implemented in subclasses")

    def calc_loss(self, x, return_grad=False):
        frob = self.frobenius_norm(x, return_grad=return_grad)
        reg = self.regularizer(x, return_grad=return_grad)

        if return_grad:
            loss = frob[0] + reg[0]
            grad = frob[1] + reg[1]
            return (loss, grad)
        else:
            loss = frob + reg
            return loss
    
    def fit(self, covs, ns, mean=0, x0=None, method="L-BFGS-B"):
        """
        Fits kernel parameters by minimizing the Frobenius Norm
        with the empirical covariance between sequences at different
        distance classes.

        Parameters
        ----------
        covs : array-like of shape (2 ** seq_length)
            Average empirical second moments at every possible
            combination of sites.
        ns : array-like of shape (2 ** seq_length)
            Number of pairs of sequences at every possible combination of sites.
        mean : float, optional
            Mean value to subtract from the covariances. Default is 0.
        x0 : array-like, optional
            Initial guess for the optimization. If None, it will be
            determined automatically. Default is None.
        method : str, optional
            Optimization method to use. Default is "L-BFGS-B".

        Returns
        -------
        params : array-like or tuple of array-like
            Parameter values that best fit the empirical second moments.
        """
        if self.regularizer.beta == 0.:
            return super().fit(covs, ns, mean=mean, x0=x0)

        else:
            self.set_data(covs, ns, mean=mean)
            if x0 is None:
                x0 = self.get_x0()
            options = {
                "ftol": 1e-12,
                "maxiter": 10000,
                "gtol": 1e-8,
                "maxls": 100,
                "maxcor": 20,
            }
            bounds = [(-30.0, 30.0)] * x0.shape[0]
            res = minimize(
                fun=self.calc_loss,
                jac=True,
                x0=x0,
                args=(True,),
                # bounds=bounds,
                method=method,
                options=options,
            )
            
            if not res.success:
                msg = f"kernel alignment did not converge: {res}"
                raise ValueError(msg)

            self.res = res
            return self.x_to_params(res.x)


class VkKernelAligner(KernelAligner):
    """
    Class to perform kernel alignment by matching the empirical
    covariances for pairs of sequences that differ at specific
    numbers of sites.

    Parameters
    ----------
    n_alleles : int
        The number of alleles per site.

    seq_length : int
        The number of sites in the sequence.
    """

    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.n_covs = seq_length + 1
        self.calc_W_kd_matrix()
        self.W = self.W_kd

    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        sl, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(sl + 1):
            value = (-1) ** q * (a - 1) ** (k - q)
            n_value = comb(d, q) * comb(sl - d, k - q)
            s += value * n_value
        return s / self.n_genotypes

    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)


class VCKernelAligner(RegularizedKernelAligner, VkKernelAligner):
    """
    Class to perform kernel alignment of empirical
    covariance-distance relationships with the Variance Components
    that generate them by minimizing the Frobenius norm
    of the resulting matrices.

    Parameters
    ----------
    n_alleles: int
        Number of alleles per site.

    seq_length: int
        Number of sites in the sequence.

    beta: float
        Regularization constant to penalize deviations from
        the linear decay of the log lambdas. By default, it does
        not perform regularization (beta=0).
    """

    def __init__(self, n_alleles, seq_length, beta=0):
        RegularizedKernelAligner.__init__(
            self,
            n_alleles=n_alleles,
            seq_length=seq_length,
            regularizer=VCLogLambdaRegularizer,
            beta=beta,
        )

    def get_x0(self):
        lambdas = lsq_linear(
            self.frobenius_norm.A,
            self.frobenius_norm.b,
            bounds=(0, np.inf),
            method="trf",
            lsmr_tol="auto",
            max_iter=10000,
            verbose=0,
        ).x
        scale = max(np.max(lambdas), 1.0)
        floor = 1e-10 * scale
        log_lambda0 = np.log(np.maximum(lambdas, floor))
        return log_lambda0

    def predict(self, lambdas):
        return self.W_kd.T.dot(lambdas)

    def x_to_params(self, x):
        lambdas = np.exp(x)
        return lambdas


class DeltaPKernelAligner(VkKernelAligner):
    """
    Class to perform kernel alignment of empirical
    covariance-distance relationships with the expected values
    under a prior distribution parametrized by `a` on the local
    epistatic coefficients by minimizing the Frobenius norm
    of the resulting matrices.

    Parameters
    ----------
    n_alleles: int
        Number of alleles per site.

    seq_length: int
        Number of sites in the sequence.

    P: float
        Order of local epistatic coefficients that are penalized.
    """

    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.P = P
        self.params_to_log_lambda_k = DeltaPtoVkTransform(
            n_alleles, seq_length, P
        )
        self.n_params = self.params_to_log_lambda_k.input_size

    def get_x0(self):
        return np.full(self.n_params, 0)

    def predict(self, x):
        log_lambda_k = self.params_to_log_lambda_k(x, return_grad=False)
        lambda_k = np.exp(log_lambda_k)
        return self.W_kd.T @ lambda_k
    
    def calc_residuals(self, x):
        log_lambda_k = self.params_to_log_lambda_k(x, return_grad=False)
        residuals = self.frobenius_norm.calc_weighted_residuals(log_lambda_k)
        return residuals

    def calc_residuals_grad(self, x):
        log_lambda_k, log_lambda_k_grad = self.params_to_log_lambda_k(x, return_grad=True)
        res_grad = self.frobenius_norm.calc_weighted_residuals_grad(log_lambda_k)
        grad = res_grad.T @ log_lambda_k_grad.T 
        return grad

    def calc_loss(self, x, return_grad=False):
        if return_grad:
            log_lambda_k, t_grad = self.params_to_log_lambda_k(
                x, return_grad=True
            )
            loss, loss_grad = self.frobenius_norm(
                log_lambda_k, return_grad=True
            )
            with np.errstate(invalid="ignore"):
                grad = t_grad @ loss_grad
            return (loss, grad)
        else:
            log_lambda_k = self.params_to_log_lambda_k(x, return_grad=False)
            loss = self.frobenius_norm(log_lambda_k, return_grad=False)
            return loss

    def x_to_params(self, x):
        return np.exp(x)


class VUKernelAligner(KernelAligner):
    """
    Class to perform kernel alignment by matching the empirical
    covariances for pairs of sequences that differ at specific
    subsets of sites.

    Parameters
    ----------
    n_alleles : int
        The number of alleles per site.

    seq_length : int
        The number of sites in the sequence.
    """

    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles, seq_length)
        self.n_covs = 2**seq_length
        self.calc_W_UD_matrix()
        self.W = self.W_UD

    def calc_W_UD_matrix(self):
        Padd = np.array([self.n_alleles - 1, -1.0])
        Pcon = np.array([1, 1.0])
        W_UD = [
            kron(Ps) for Ps in product([Pcon, Padd], repeat=self.seq_length)
        ]
        self.W_UD = np.vstack(W_UD) / self.n_genotypes


class SitesVCKernelAligner(RegularizedKernelAligner, VUKernelAligner):
    """
    Class to perform kernel alignment of empirical
    covariance-distance relationships with the Site(s) Variance Components
    that generate them by minimizing the Frobenius norm
    of the resulting matrices.

    Parameters
    ----------
    n_alleles: int
        Number of alleles per site.

    seq_length: int
        Number of sites in the sequence.

    beta: float
        Regularization constant to penalize deviations from
        the linear decay of the log lambdas. By default, it does
        not perform regularization (beta=0).
    """

    def __init__(self, n_alleles, seq_length, beta=0):
        RegularizedKernelAligner.__init__(
            self,
            n_alleles=n_alleles,
            seq_length=seq_length,
            regularizer=SitesVCLogLambdaRegularizer,
            beta=beta,
        )

    def get_x0(self):
        lambdas = lsq_linear(
            self.frobenius_norm.A,
            self.frobenius_norm.b,
            bounds=(0, np.inf),
            method="trf",
            lsmr_tol="auto",
            max_iter=10000,
            verbose=0,
        ).x
        scale = max(np.max(lambdas), 1.0)
        floor = 1e-10 * scale
        log_lambda0 = np.log(np.maximum(lambdas, floor))
        return log_lambda0

    def predict(self, lambdas):
        return self.W_UD.T.dot(lambdas)

    def x_to_params(self, x):
        lambdas = np.exp(x)
        return lambdas


class LowDimVUKernelAligner(VUKernelAligner):
    def __init__(self, n_alleles, seq_length, transform):
        super().__init__(n_alleles, seq_length)
        self.params_to_log_lambda_U = transform
        self.n_params = self.params_to_log_lambda_U.input_size

    def get_x0(self):
        x0 = np.zeros(self.n_params)
        return x0

    def predict(self, x):
        log_lambda_U = self.params_to_log_lambda_U(x, return_grad=False)
        lambda_U = np.exp(log_lambda_U)
        return self.W_UD.T @ lambda_U
    
    def calc_residuals(self, x):
        log_lambda_U = self.params_to_log_lambda_U(x, return_grad=False)
        residuals = self.frobenius_norm.calc_weighted_residuals(log_lambda_U)
        return residuals

    def calc_residuals_grad(self, x):
        log_lambda_U, log_lambda_U_grad = self.params_to_log_lambda_U(x, return_grad=True)
        res_grad = self.frobenius_norm.calc_weighted_residuals_grad(log_lambda_U)
        grad = res_grad.T @ log_lambda_U_grad.T 
        return grad

    def calc_loss(self, x, return_grad=False):
        if return_grad:
            log_lambda_U, t_grad = self.params_to_log_lambda_U(
                x, return_grad=True
            )
            loss, loss_grad = self.frobenius_norm(
                log_lambda_U, return_grad=True
            )
            with np.errstate(invalid="ignore"):
                grad = t_grad @ loss_grad
            return (loss, grad)
        else:
            log_lambda_U = self.params_to_log_lambda_U(x, return_grad=False)
            loss = self.frobenius_norm(log_lambda_U, return_grad=False)
            return loss

    def x_to_params(self, x):
        return np.exp(x)


class DeltaUKernelAligner(LowDimVUKernelAligner):
    """
    Class to determine the parameters of the DeltaU sum model
    that best align with the empirical covariances for sequences
    matching at all possible combinations of sites.

    Parameters
    ----------
    n_alleles : int
        The number of alleles per site.

    seq_length : int
        The number of sites in the sequence.

    P : int, optional
        Interaction order of local epistatic coefficients to
        be considered e.g. P=2 refers to the classical epistatic
        coefficients.

    """

    def __init__(self, n_alleles, seq_length, P):
        self.P = P
        transform = DeltaUtoVUTransform(n_alleles, seq_length, P)
        super().__init__(n_alleles, seq_length, transform=transform)

    def get_x0(self):
        return np.zeros(self.n_params)
        D = np.array(list(product([False, True], repeat=self.seq_length)))
        d = D.sum(1)
        idx = np.where(d <= self.P)[0]

        # Solve full system and extract up to P-th order lambdas
        lambda_U = lsq_linear(
            self.frobenius_norm.A,
            self.frobenius_norm.b,
            bounds=(0, np.inf),
            method="trf",
            lsmr_tol="auto",
            max_iter=10000,
            verbose=0,
        ).x[idx]

        # Get into right parametrization for the DeltaU models
        d = d[idx]
        log_lambda_U = np.log(lambda_U[d < self.P] + 1e-16)
        log_a = -self.P * np.log(self.n_alleles) - np.log(
            lambda_U[d == self.P] + 1e-16
        )
        x0 = np.append(log_lambda_U, log_a[::-1])
        return x0


class ConnectednessKernelAligner(LowDimVUKernelAligner):
    """
    Class to determine the parameters of the Connectedness model
    that best align with the empirical covariances for sequences
    matching at all possible combinations of sites.

    Parameters
    ----------
    n_alleles : int
        The number of alleles per site.

    seq_length : int
        The number of sites in the sequence.
    """
    def __init__(self, n_alleles, seq_length):
        transform = ConnectednessToVUTransform(n_alleles, seq_length)
        super().__init__(n_alleles, seq_length, transform=transform)
    
    def get_x0(self):
        # Solve full system and extract up to P-th order lambdas
        lambda_U = lsq_linear(
            self.frobenius_norm.A,
            self.frobenius_norm.b,
            bounds=(0, np.inf),
            method="bvls",
        ).x
        scale = max(np.max(lambda_U), 1.0)
        floor = 1e-10 * scale
        b = np.log(np.maximum(lambda_U, floor))

        # Initialize from least squares solution to the linear system
        A = self.params_to_log_lambda_U.U.astype(float)
        log_mu = np.linalg.lstsq(A, b, rcond=None)[0]
        log_mu_0 = log_mu[0]
        log_mu_i = log_mu[1:]
        
        x0 = np.append([log_mu_0], log_mu_i[::-1])
        return x0
