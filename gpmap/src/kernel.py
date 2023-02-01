import numpy as np

from itertools import combinations
from scipy.special._basic import comb
from scipy.special._logsumexp import logsumexp
from scipy.optimize._minimize import minimize

from gpmap.src.seq import seq_to_one_hot
from gpmap.src.utils import check_error


class SequenceKernel(object):
    def __init__(self, seq_length, n_alleles):
        self.alpha = n_alleles
        self.l = seq_length
        self.lp1 = self.l + 1

    def inner_product(self, x1, x2, metric=None):
        if metric is None:
            return(x1.dot(x2.T))
        else:
            return(x1.dot(metric.dot(x2.T)))

    def calc_hamming_distance(self, x1, x2):
        return(self.l - self.inner_product(x1, x2))
    
    def set_data(self, x1, x2=None, alleles=None, **kwargs):
        self.x1 = seq_to_one_hot(x1, alleles=alleles).astype(int)
        if x2 is None:
            x2 = x1
        self.x2 = seq_to_one_hot(x2, alleles=alleles).astype(int)
        
        if hasattr(self, 'set_extra_data'):
            self.set_extra_data(**kwargs)
    
    def __call__(self, **kwargs):
        return(self.forward(**kwargs))
    
    def grad(self, **kwargs):
        return(self.backward(**kwargs))


class VarianceComponentKernel(SequenceKernel):
    def __init__(self, seq_length, n_alleles, q=None, use_p=False):
        super().__init__(seq_length=seq_length, n_alleles=n_alleles)
        self.use_p = use_p
        
        if self.use_p:
            if q is None:
                q = (self.l - 1) / self.l
            self.q = q
            self.logq = np.log(q)
            self.B = self.calc_polynomial_coeffs()
            ks = np.arange(self.lp1)
            log_q_powers = self.logq * ks
            log_1mq_powers = np.append([-np.inf], np.log(1 - np.exp(log_q_powers[1:])))
            self.lsf = self.l * log_1mq_powers
            self.log_odds = log_q_powers - log_1mq_powers
        else:
            self.calc_krawchouk_matrix()
            
        self.n_params = self.lp1
        if use_p:
            self.n_params += self.l * self.alpha
    
    def calc_polynomial_coeffs(self):
        k = np.arange(self.lp1)
        lambdas = np.exp(k * self.logq) 
        
        B = np.zeros((self.lp1, self.lp1))
        idx = np.arange(self.lp1)
        for k in idx:
            k_idx = idx != k
            k_lambdas = lambdas[k_idx]
            norm_factor = 1 / np.prod(k_lambdas - lambdas[k])
        
            for power in idx:
                lambda_combs = list(combinations(k_lambdas, self.l - power))
                p = np.sum([np.prod(v) for v in lambda_combs])
                B[power, k] = (-1) ** (power) * p * norm_factor

        return(B)
    
    def set_extra_data(self):
        self.same_seq = self.x1.dot(self.x2.T) == self.l
        
    def calc_w_kd(self, k, d):
        ss = 0
        for q in range(self.lp1):
            ss += (-1.)**q * (self.alpha-1.)**(k-q) * comb(d,q) * comb(self.l-d,k-q)
        return(ss)
    
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
        
        cov = coeffs[0] * np.exp(-self.inner_product(self.x1, self.x2, M))
        cov *= self.same_seq
        
        for power in range(1, self.lp1):
            log_factors = np.stack([self.log_odds[power] - log_p_flat,
                                    np.zeros(log_p_flat.shape)], 1)
            log_factors = logsumexp(log_factors, 1)
            M = np.diag(log_factors)
            m = self.inner_product(self.x1, self.x2, M)
            cov += coeffs[power] * np.exp(self.lsf[power] + m)
        return(cov)
    
    def forward(self, lambdas, log_p=None):
        if self.use_p:
            check_error(log_p is not None, msg='ps must be provided')
            cov = self._forward_ps(lambdas, log_p)
        else:
            hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
            cov = self.W_kd.dot(lambdas)[hamming_distance]
        return(cov)
    
    def backward(self):
        if self.use_p:
            raise ValueError('Not implemented for variable p')
        else:
            hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
            grad = [self.W_kd[k, :][hamming_distance] for k in range(self.lp1)]
        return(grad)
        

    def get_params0(self):
        params = np.append([-10], 2-np.arange(self.l))
        if self.use_p:
            params = np.append(params, np.zeros(self.l * self.alpha))
        return(params)
    
    def split_params(self, params):
        params_dict = {}
        if self.use_p:
            params_dict['lambdas'] = np.exp(params[:self.lp1])
            log_ps = params[self.lp1:].reshape(self.l, self.alpha)
            norm_factors = logsumexp(log_ps, axis=1)
            for i in range(log_ps.shape[1]):
                log_ps[:, i] -= log_ps[:, i] - norm_factors
            params_dict['ps'] = np.exp(log_ps)
        else:
            params_dict['lambdas'] = np.exp(params)
        return(params_dict) 

class FullKernelAligner(object):
    def __init__(self, kernel, beta=0):
        self.kernel = kernel
        self.seq_length = kernel.l
        self.n_alleles = kernel.alpha
        self.set_beta(beta)
    
    def set_beta(self, beta):
        check_error(beta >=0, msg='beta must be >= 0')
        self.beta = beta
    
    def set_data(self, X, y, alleles=None):
        self.X = X
        self.y = y
        self.n = y.shape[0]
        
        self.kernel.set_data(X, alleles=alleles)
        y_res = y.reshape((self.n, 1))
        self.target = y_res.dot(y_res.T)
    
    def squared_frobenius_norm(self, **kwargs):
        cov = self.predict(**kwargs)
        return((np.power(cov - self.target,  2).mean()))
    
    def squared_frobenius_norm_grad(self, **kwargs):
        cov = self.predict(**kwargs)
        diff2 = 2*(cov - self.target)
        grad = [np.sum(diff2 * grad_k for grad_k in self.kernel.grad())]
        return(grad)
    
    def calc_reg_lambdas(self, log_lambdas):
        if self.beta == 0:
            reg = 0
        else:
            reg = 0
        return(reg)
    
    def calc_reg_ps(self, ps):
        return(0)
    
    def loss(self, params):
        params_dict = self.kernel.split_params(params)
        frob = self.squared_frobenius_norm(**params_dict)
        
        # reg_lambdas = self.calc_reg_lambdas(params_dict['log_lambdas'])
        # reg_ps = self.calc_reg_ps(params_dict['log_ps'])
        
        
        return(frob)
    
    def fit(self, params0=None):
        if params0 is None:
            params0 = self.kernel.get_params0()
        res = minimize(fun=self.loss,
                       x0=params0, method='Powell',
                       options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 1e5})
        lambdas = np.exp(res.x)
        return(lambdas)
    
    def predict(self, **kwargs):
        return(self.kernel(**kwargs))


class KernelAligner(object):
    def __init__(self, seq_length, n_alleles, beta=0):
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
        lambdas = np.exp(log_lambdas)
        Frob1 = lambdas.dot(self.M).dot(lambdas)
        Frob2 = 2 * lambdas.dot(self.a)
        Frob = Frob1 - Frob2
        if self.beta > 0:
            Frob += self.beta * log_lambdas[1:].dot(self.second_order_diff_matrix).dot(log_lambdas[1:])
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
    
    def construct_a(self, rho_d, N_d):
        size = self.seq_length + 1
        a = np.zeros(size)
        for i in range(size):
            for d in range(size):
                a[i] += N_d[d] * self.W_kd[i, d] * rho_d[d]
        self.a = a
    
    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(1 / a**l * s)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
    
    def fit(self):
        lambdas0 = np.dot(self.M_inv, self.a).flatten()
        lambdas0[lambdas0<0] = 1e-5
        log_lambda0 = np.log(lambdas0)
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
              