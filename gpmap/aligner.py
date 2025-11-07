import numpy as np

from scipy.special import comb
from scipy.optimize import minimize, lsq_linear

from gpmap.utils import check_error, safe_exp


class VCKernelAligner(object):
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

    def __init__(self, n_alleles, seq_length, beta=0.):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.set_beta(beta)
        self.calc_W_kd_matrix()
        self.calc_second_order_diff_matrix()

    def set_data(self, covs, distances_n, sigma2=0.):
        D_n = np.diag(distances_n)
        WD = self.W_kd @ D_n
        self.A = WD @ self.W_kd.T
        self.b = WD @ covs - (sigma2 * self.A).sum(1)
        self.c = np.dot(covs, D_n @ covs)
        self.sigma2 = sigma2

    def set_beta(self, beta):
        check_error(beta >= 0, msg="beta must be >= 0")
        self.beta = beta

    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i: i + 3] = [-1, 2, -1]
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

    def fit(self, covs, ns, sigma2=0.):
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

    def predict(self, lambdas):
        return self.W_kd.T.dot(lambdas)


class VjKernelAligner(object):
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

    def set_data(self, covs, ns, sites_matrix):
        msg = "´sites_matrix´ should have a number of columns equal"
        msg += " to the sequence length"
        check_error(sites_matrix.shape[1] == self.seq_length, msg=msg)

        msg = "´sites_matrix´ should have a number of rows equal to"
        msg += " the the size of `covs`"
        check_error(sites_matrix.shape[0] == covs.shape[0], msg=msg)

        self.covs = covs
        self.ns = ns
        self.sites_matrix = sites_matrix

    def frobenius_norm(self, params):
        exp_cov = self.calc_cov(params)
        Frob = np.sum(self.ns * (self.covs - exp_cov) ** 2) / self.ns.sum()
        return Frob

    def frobenius_norm_grad(self, logit_rho):
        raise ValueError("Gradient calculation not implemented")

    def fit(self, covs, ns, sites_matrix):
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
        sites_matrix : array-like of shape (2 ** seq_length, seq_length)
            Matrix encoding the sites that are in common for every
            provided empirical second moment class.

        Returns
        -------
        params : array-like or tuple of array-like
            Parameter values that best fit the empirical second moments.
        """

        self.set_data(covs, ns, sites_matrix)
        res = minimize(
            fun=self.frobenius_norm,
            x0=self.get_x0(),
            method="Powell",
            options={"ftol": 1e-16},
        )
        self.res = res
        return self.x_to_params(res.x)

    def predict(self, logit_rho, log_mu=0):
        x = self.params_to_x(log_mu, logit_rho)
        cov = self.calc_cov(x)
        return cov


class RhoKernelAligner(VjKernelAligner):
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
            np.exp(baseline + self.sites_matrix @ log_factors)
            - 1
            + np.exp(log_mu)
        )
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
