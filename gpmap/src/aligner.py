import numpy as np

from scipy.special import comb
from scipy.optimize import minimize

from gpmap.src.utils import check_error, safe_exp
from gpmap.src.matrix import quad, dot_log


class VCKernelAligner(object):
    '''
    Class to perform kernel alignment of empirical
    covariance-distance relationships with the Variance Components
    that can generate them by minimizing the Frobenius norm
    of the resulting matrices

    Parameters
    ----------
    n_alleles: int
        Number of alleles per site

    seq_length: int
        Number of sites in sequence

    beta: float
        Regularization constant to penalize deviations from
        linear decay of the log lambdas. By default, it does
        not perform regularization (beta=0)
    '''
    def __init__(self, n_alleles, seq_length, beta=0):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.set_beta(beta)
        self.calc_W_kd_matrix()
        self.calc_second_order_diff_matrix()
    
    def set_data(self, covs, distances_n):
        self.covs = covs
        self.distances_n = distances_n
        self.construct_a(covs, distances_n)
        self.construct_M(distances_n)
        self.M_inv = np.linalg.inv(self.M)
    
    def set_beta(self, beta):
        check_error(beta >=0, msg='beta must be >= 0')
        self.beta = beta
    
    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i:i + 3] = [-1, 2, -1]
        self.second_order_diff_matrix = Diff2.T.dot(Diff2)
    
    def frobenius_norm(self, log_lambdas):
        """cost function for regularized least square method for inferring 
        lambdas"""
        lambdas = safe_exp(log_lambdas)
        Frob1 = lambdas.dot(self.M).dot(lambdas)
        Frob2 = 2 * lambdas.dot(self.a)
        Frob = Frob1 - Frob2
        if self.beta > 0:
            Frob += self.beta * quad(self.second_order_diff_matrix, log_lambdas[1:])
        return(Frob)
    
    def frobenius_norm2(self, log_lambdas):
        """cost function for regularized least square method for inferring 
        lambdas"""
        signlambdas = np.ones_like(log_lambdas)
        logMlambdas, signMlambdas = dot_log(self.logM, self.signM, log_lambdas, signlambdas)
        log_lambdas_T = np.expand_dims(log_lambdas, 0)
        signlambdas_T = np.expand_dims(signlambdas, 0)
        logFrob1, signFrob1 = dot_log(log_lambdas_T, signlambdas_T, logMlambdas, signMlambdas)
        logFrob2, signFrob2 = dot_log(log_lambdas_T, signlambdas_T, self.loga, self.signa)
        logFrob2 += np.log(2)
        Frob = signFrob2 * np.exp(logFrob2) * (signFrob1 * signFrob2 * np.exp(logFrob1 - logFrob2) - 1)
        Frob = Frob[0]
        if self.beta > 0:
            Frob += self.beta * quad(self.second_order_diff_matrix, log_lambdas[1:])
        return(Frob)
    
    def frobenius_norm_grad(self, log_lambdas):
        msg = 'gradient calculation only implemented for beta=0'
        check_error(self.beta == 0, msg=msg)
        lambdas = np.exp(log_lambdas)
        grad_Frob = (2 * self.M.dot(lambdas) - 2 * self.a)
        return(grad_Frob * lambdas)
    
    def construct_M(self, N_d):
        size = self.seq_length + 1
        M = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                for d in range(size):
                    M[i, j] += N_d[d] * self.W_kd[i, d] * self.W_kd[j, d]
        self.M = M
        self.logM = np.log(np.abs(M))
        self.signM = np.sign(M)
    
    def construct_a(self, rho_d, N_d):
        size = self.seq_length + 1
        a = np.zeros(size)
        for i in range(size):
            for d in range(size):
                a[i] += N_d[d] * self.W_kd[i, d] * rho_d[d]
        self.a = a
        self.loga = np.log(np.abs(a))
        self.signa = np.sign(a)
    
    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(s / a**l)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
    
    def fit(self, covs, ns):
        '''
        Fit Variance Component kernel by minizing the Frobenious Norm
        with the covariance at each possible distance

        Parameters
        ----------
        covs : array-like of shape (seq_length + 1)
            Average empirical second moments at every possible distance
        ns : array-like of shape (seq_length + 1)
            Number of pairs of sequences at each possible distance

        Returns
        -------
        lambdas: array-like of shape (seq_length + 1)
            lambdas that best fit the empirical second moments
        '''
        self.set_data(covs, ns)
        lambdas0 = np.dot(self.M_inv, self.a).flatten()
        lambdas0[lambdas0<0] = 1e-10
        log_lambda0 = np.log(lambdas0)
        # print('log_lambda0', log_lambda0)
        if self.beta == 0:
            res = minimize(fun=self.frobenius_norm,
                           jac=self.frobenius_norm_grad,
                           x0=log_lambda0, method='L-BFGS-B')
        elif self.beta > 0:
            res = minimize(fun=self.frobenius_norm,
                           x0=log_lambda0, method='Powell',
                           options={'xtol': 1e-8, 'ftol': 1e-8})
        lambdas = np.exp(res.x)
        return(lambdas)
    
    def predict(self, lambdas):
        return(self.W_kd.T.dot(lambdas))
    
    def calc_mse(self, lambdas):
        ss = (self.predict(lambdas) - self.covs) ** 2
        return(np.sum(ss * (1 / np.sum(self.distances_n)) * self.distances_n))
    

class VjKernelAligner(object):
    def __init__(self, n_alleles, seq_length):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.eta = self.n_alleles - 1
    
    def set_data(self, covs, ns, sites_matrix):
        msg = '´sites_matrix´ should have a number of columns equal to the sequence length'
        check_error(sites_matrix.shape[1] == self.seq_length, msg=msg)

        msg = '´sites_matrix´ should have a number of rows equal to the the size of `covs`'
        check_error(sites_matrix.shape[0] == covs.shape[0], msg=msg)

        self.covs = covs
        self.ns = ns
        self.sites_matrix = sites_matrix
    
    def frobenius_norm(self, params):
        exp_cov = self.calc_cov(params)
        Frob = np.sum(self.ns * (self.covs - exp_cov) ** 2) / self.ns.sum()
        return(Frob)
    
    def frobenius_norm_grad(self, logit_rho):
        raise ValueError('Gradient calculation not implemented')
    
    def fit(self, covs, ns, sites_matrix):
        '''
        Fit Connectedness kernel by minizing the Frobenious Norm
        with the covariance at sequences matching subsets of sites

        Parameters
        ----------
        covs : array-like of shape (2 ** seq_length)
            Average empirical second moments at every possible combination of sites
        ns : array-like of shape (2 ** seq_length)
            Number of pairs of sequences at every possible combination of sites
        sites_matrix : array-like of shape (2 ** seq_length, seq_length)
            Matrix encoding the sites that are in commong for every 
            provided empirical second moment class

        Returns
        -------
        params: array-like or tuple of array-like
            Parameter values that best fit the empirical second moments
        '''

        self.set_data(covs, ns, sites_matrix)
        res = minimize(fun=self.frobenius_norm,
                       x0=self.get_x0(), method='Powell',
                       options={'ftol': 1e-16})
        self.res = res
        return(self.x_to_params(res.x))
    
    def predict(self, logit_rho, log_mu=0):
        x = self.params_to_x(log_mu, logit_rho)
        cov = self.calc_cov(x)
        return(cov)


class RhoKernelAligner(VjKernelAligner):
    '''
    Class to perform kernel alignment of empirical
    covariance-distance relationships with the Variance Components
    that can generate them by minimizing the Frobenius norm
    of the resulting matrices

    Parameters
    ----------
    n_alleles: int
        Number of alleles per site

    seq_length: int
        Number of sites in sequence

    beta: float
        Regularization constant to penalize deviations from
        linear decay of the log lambdas. By default, it does
        not perform regularization (beta=0)
    '''
    def x_to_params(self, x):
        log_mu, logit_rho = x[0], x[1:]
        return(log_mu, logit_rho)
    
    def params_to_x(self, log_mu, logit_rho):
        return(np.hstack([log_mu, logit_rho]))
    
    def get_x0(self):
        return(np.random.normal(size=self.seq_length + 1))

    def calc_cov(self, x):
        log_mu, logit_rho = self.x_to_params(x)
        log1mrho = -np.logaddexp(0., logit_rho)
        log_rho = logit_rho + log1mrho
        log_one_p_eta_rho = np.logaddexp(0., log_rho + np.log(self.eta))
        log_factors = log_one_p_eta_rho - log1mrho
        baseline = log1mrho.sum()
        cov = np.exp(baseline + self.sites_matrix @ log_factors) - 1 + np.exp(log_mu)
        return(cov)
    

################################
# Full kernel alignment methds #
################################

class FullKernelAligner(object):
    def __init__(self, kernel, optimizer='BFGS'):
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
        return(np.power(self.residuals,  2).sum())
    
    def loss(self, params):
        params_dict = self.kernel.split_params(params)
        frob = self.frob2(**params_dict)
        return(frob)
    
    def frob2_grad(self, **kwargs):
        grad = np.array([np.sum(2 * self.residuals * grad_k)
                         for grad_k in self.kernel.grad(**kwargs)])
        return(grad)
    
    def loss_grad(self, params):
        params_dict = self.kernel.split_params(params)
        grad = self.frob2_grad(**params_dict)
        return(grad)
    
    def fit(self, params0=None):
        if params0 is None:
            params0 = self.kernel.get_params0()
        jac = self.grad = None if self.optimizer.lower() == 'powell' else self.loss_grad
        res = minimize(fun=self.loss, jac=jac, x0=params0,
                       method=self.optimizer,
                       options={'gtol': 1e-12, 'maxiter': 1e5})
        self.res = res
        params = self.kernel.transform_params(res.x)
        return(params)
    
    def predict(self, **kwargs):
        return(self.kernel(**kwargs))