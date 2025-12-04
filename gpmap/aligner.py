import numpy as np

from itertools import combinations, product
from scipy.special import comb, factorial
from scipy.optimize import minimize, lsq_linear

from gpmap.matrix import kron
from gpmap.utils import check_error, safe_exp


class VkKernelAligner(object):
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
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.calc_W_kd_matrix()

    def set_data(self, covs, distances_n, sigma2=0):
        D_n = np.diag(distances_n)
        WD = self.W_kd @ D_n
        self.A = WD @ self.W_kd.T
        self.b = WD @ covs - (sigma2 * self.A).sum(1)
        self.c = np.dot(covs, D_n @ covs)
        self.sigma2 = sigma2

    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        sl, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(sl + 1):
            value = (-1) ** q * (a - 1) ** (k - q)
            n_value = comb(d, q) * comb(sl - d, k - q)
            s += value * n_value
        return s / a**sl

    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
    
    def predict(self, params):
        lambdas = self.params_to_lambdas(params)
        return self.W_kd.T.dot(lambdas)
    
    def calc_cov(self, x):
        params = self.x_to_params(x)
        return self.predict(params)
    
    def frobenius_norm(self, x):
        lambdas = self.params_to_lambdas(self.x_to_params(x))
        Av = self.A @ lambdas
        Frob = self.c + np.dot(lambdas, Av - 2 * self.b)
        return Frob

    def frobenius_norm_grad(self, params):
        raise ValueError("Gradient calculation not implemented")

    def fit(self, covs, ns):
        """
        Fits kernel parameters by minimizing the Frobenius Norm
        with the empirical covariance at sequences matching subsets
        of sites.

        Parameters
        ----------
        covs : array-like of shape (2 ** seq_length)
            Average empirical second moments at every possible
            combination of sites.
        ns : array-like of shape (2 ** seq_length)
            Number of pairs of sequences at every possible combination of sites.
        Returns
        -------
        params : array-like or tuple of array-like
            Parameter values that best fit the empirical second moments.
        """

        self.set_data(covs, ns)

        res = minimize(
            fun=self.frobenius_norm,
            x0=self.get_x0(),
            method="Powell",
            options={"ftol": 1e-16},
        )
        self.res = res
        return self.x_to_params(res.x)


class VCKernelAligner(VkKernelAligner):
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
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.set_beta(beta)
        self.calc_second_order_diff_matrix()
    
    def get_x0(self):
        return np.zeros(self.seq_length + 1)
    
    def x_to_params(self, x):
        lambdas = np.exp(x)
        return lambdas

    def params_to_x(self, lambdas):
        return np.log(lambdas)
    
    def params_to_lambdas(self, params):
        return(params)
    
    def set_beta(self, beta):
        check_error(beta >= 0, msg="beta must be >= 0")
        self.beta = beta

    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i : i + 3] = [-1, 2, -1]
        self.second_order_diff_matrix = Diff2.T.dot(Diff2)

    def calc_loss(self, log_lambdas, beta=None, return_grad=False):
        """Loss function is proportional to the frobenius norm of
        the difference between the empirical distance-covariance
        function and the expected under some lambdas"""
        if beta is None:
            beta = self.beta

        lambdas = safe_exp(log_lambdas)
        Av = self.A @ lambdas
        loss = self.c + np.dot(lambdas, Av - 2 * self.b)

        if beta > 0:
            reg_Av = self.second_order_diff_matrix @ log_lambdas[1:]
            reg = beta * np.dot(reg_Av, log_lambdas[1:])
            loss += reg

        if return_grad:
            with np.errstate(over="ignore"):
                grad = (2 * Av - 2 * self.b) * lambdas
            if beta > 0:
                grad += np.append([0], 2 * reg_Av)
            return (loss, grad)

        return loss

    def fit(self, covs, ns, sigma2=0):
        """
        Fit the Variance Component kernel by minimizing the Frobenius Norm
        with the covariance at each possible distance.

        Parameters
        ----------
        covs : array-like of shape (seq_length + 1)
            Average empirical second moments at every possible distance.
        ns : array-like of shape (seq_length + 1)
            Number of pairs of sequences at each possible distance.

        Returns
        -------
        lambdas : array-like of shape (seq_length + 1)
            Lambda values that best fit the empirical second moments.

        Example
        -------
        >>> aligner = VCKernelAligner(n_alleles=4, seq_length=4, beta=10)
        >>> lambdas = aligner.fit(covs, ns)
        """
        self.set_data(covs, ns, sigma2=sigma2)

        res = lsq_linear(self.A, self.b, bounds=(0, np.inf), method="bvls")
        lambdas = res.x

        if self.beta > 0:
            log_lambda0 = np.log(lambdas + 1e-16)
            res = minimize(
                fun=self.calc_loss,
                jac=True,
                x0=log_lambda0,  # method='powell',
                args=(self.beta, True),
                #    options={'maxiter': 1000, 'tol': 1e-16},
            )
            lambdas = np.exp(res.x)
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
        
        n_p_sites = comb(self.seq_length, self.P)
        n_p_faces_per_sites = comb(self.n_alleles, 2) ** self.P
        allelic_comb_remaining_sites = self.n_alleles ** (self.seq_length - self.P)
        self.n_p_faces = (
            n_p_sites * n_p_faces_per_sites * allelic_comb_remaining_sites
        )
        
        lambdas = []
        self.Pfactorial = factorial(self.P)
        for L_lambda_k in np.arange(self.seq_length + 1) * self.n_alleles:
            lambda_k = 1
            for p in range(self.P):
                lambda_k *= L_lambda_k - p * self.n_alleles
            lambdas.append(lambda_k / self.Pfactorial)
        self.lambdas = np.array(lambdas) / self.n_p_faces
        
    
    def get_x0(self):
        return 0.
    
    def x_to_params(self, log_a):
        return np.exp(log_a)
    
    def params_to_lambdas(self, a):
        lambdas = np.zeros_like(self.lambdas)
        lambdas[self.lambdas > 0] = 1. / (a * self.lambdas[self.lambdas > 0])
        return lambdas


class VUKernelAligner(object):
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
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.eta = self.n_alleles - 1
        self.n_covs = 2 ** seq_length
        self.n_Us = 2 ** seq_length
        self.n_seqs = n_alleles ** seq_length
        self.Padd = np.array([self.n_alleles - 1, -1.0]) / self.n_alleles
        self.Pcon = np.array([1, 1.0]) / self.n_alleles
        self.U_sites = list(product([False, True], repeat=self.seq_length))
        self.calc_W_sU_matrix()
    
    def calc_W_sU_matrix(self):
        W_Us = []
        for x in self.U_sites:
            W_Us.append(kron([self.Padd if x_i else self.Pcon for x_i in x]))
        self.W_sU = np.vstack(W_Us).T

    def set_data(self, covs, ns):
        if covs.shape[0] != ns.shape[0]:
            msg = 'covs and ns must be the same shape'
            raise ValueError(msg)
        
        self.covs = covs
        self.ns = ns

    def frobenius_norm(self, params):
        exp_cov = self.calc_cov(params)
        Frob = np.sum(self.ns * (self.covs - exp_cov) ** 2) / self.ns.sum()
        return Frob

    def frobenius_norm_grad(self, params):
        raise ValueError("Gradient calculation not implemented")

    def fit(self, covs, ns):
        """
        Fits kernel parameters by minimizing the Frobenius Norm
        with the empirical covariance at sequences matching subsets
        of sites.

        Parameters
        ----------
        covs : array-like of shape (2 ** seq_length)
            Average empirical second moments at every possible
            combination of sites.
        ns : array-like of shape (2 ** seq_length)
            Number of pairs of sequences at every possible combination of sites.
        Returns
        -------
        params : array-like or tuple of array-like
            Parameter values that best fit the empirical second moments.
        """

        self.set_data(covs, ns)

        res = minimize(
            fun=self.frobenius_norm,
            x0=self.get_x0(),
            method="Powell",
            options={"ftol": 1e-16},
        )
        self.res = res
        return self.x_to_params(res.x)

    def x_to_params(self, x):
        lambda_U = np.exp(x)
        return lambda_U

    def params_to_x(self, lambda_U):
        return np.log(lambda_U)

    def get_x0(self):
        return np.zeros(self.n_Us)

    def predict(self, lambda_U):
        cov = self.W_sU @ lambda_U
        return cov

    def calc_cov(self, x):
        lambda_U = self.x_to_params(x)
        return self.predict(lambda_U)
    


class DeltaUKernelAligner(VUKernelAligner):
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
        coefficients

    """

    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles, seq_length)
        self.P = P
        self.alphaP = self.n_alleles ** self.P
        self.n_a_values = int(comb(seq_length, P))
        self.Us = list(combinations(range(self.seq_length), self.P))
        self.calc_Us_matrix()

    def calc_Us_matrix(self):
        Us_matrix = []
        for x in self.U_sites:
            Us_matrix.append([np.all([x[s] for s in U]) for U in self.Us])
        self.Us_matrix = np.vstack(Us_matrix).astype(float)

    def x_to_params(self, x):
        a_values = np.exp(x)
        return a_values

    def params_to_x(self, a_values):
        return np.log(a_values)

    def get_x0(self):
        return np.zeros(self.n_a_values)

    def a_to_lambda_U(self, a_values):
        lambda_U_inv = self.alphaP * (self.Us_matrix @ a_values)
        lambda_U = np.zeros_like(lambda_U_inv)
        idx = lambda_U_inv > 0.0
        lambda_U[idx] = 1.0 / lambda_U_inv[idx]
        return lambda_U

    def predict(self, a_values):
        lambda_U = self.a_to_lambda_U(a_values)
        cov = self.W_sU @ lambda_U
        return cov
    
    def calc_cov(self, x):
        a_values = self.x_to_params(x)
        return(self.predict(a_values))


class ConnectednessKernelAligner(VUKernelAligner):
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

    def x_to_params(self, x):
        log_mu, logit_rho = x[0], x[1:]
        return (log_mu, logit_rho)

    def params_to_x(self, log_mu, logit_rho):
        return np.hstack([log_mu, logit_rho])

    def get_x0(self):
        return np.random.normal(size=self.seq_length + 1)

    def calc_cov(self, x):
        log_mu, logit_rho = self.x_to_params(x)
        log1mrho = -np.logaddexp(0.0, logit_rho)
        log_rho = logit_rho + log1mrho
        log_one_p_eta_rho = np.logaddexp(0.0, log_rho + np.log(self.eta))
        log_factors = log_one_p_eta_rho - log1mrho
        baseline = log1mrho.sum()
        cov = (
            np.exp(baseline + self.U_sites @ log_factors)
            - 1
            + np.exp(log_mu)
        )
        return cov

    def predict(self, logit_rho, log_mu=0):
        x = self.params_to_x(log_mu, logit_rho)
        cov = self.calc_cov(x)
        return cov



################################
# Full kernel alignment methds #
################################


class FullKernelAligner(object):
    def __init__(self, kernel, optimizer="BFGS"):
        self.kernel = kernel
        self.seq_length = kernel.l
        self.n_alleles = kernel.alpha
        self.optimizer = optimizer

    def set_data(self, X, y, y_var=None, alleles=None):
        self.X = X
        self.y = y
        self.y_var = y_var if y_var is not None else np.zeros(y.shape)
        self.n = y.shape[0]

        self.kernel.set_data(X, alleles=alleles)
        y_res = y.reshape((self.n, 1))
        self.target = y_res.dot(y_res.T)

    def frob2(self, **kwargs):
        cov = self.predict(**kwargs) + np.diag(self.y_var)
        self.residuals = cov - self.target
        return np.power(self.residuals, 2).sum()

    def loss(self, params):
        params_dict = self.kernel.split_params(params)
        frob = self.frob2(**params_dict)
        return frob

    def frob2_grad(self, **kwargs):
        grad = np.array(
            [
                np.sum(2 * self.residuals * grad_k)
                for grad_k in self.kernel.grad(**kwargs)
            ]
        )
        return grad

    def loss_grad(self, params):
        params_dict = self.kernel.split_params(params)
        grad = self.frob2_grad(**params_dict)
        return grad

    def fit(self, params0=None):
        if params0 is None:
            params0 = self.kernel.get_params0()
        jac = self.grad = (
            None if self.optimizer.lower() == "powell" else self.loss_grad
        )
        res = minimize(
            fun=self.loss,
            jac=jac,
            x0=params0,
            method=self.optimizer,
            options={"gtol": 1e-12, "maxiter": 1e5},
        )
        self.res = res
        params = self.kernel.transform_params(res.x)
        return params

    def predict(self, **kwargs):
        return self.kernel(**kwargs)
