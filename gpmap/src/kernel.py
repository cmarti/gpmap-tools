import numpy as np

from scipy.special._basic import comb
from scipy.special._logsumexp import logsumexp

from gpmap.src.seq import seq_to_one_hot
from gpmap.src.linop import ProjectionOperator, LaplacianOperator
from gpmap.src.utils import check_error


class SequenceKernel(object):
    def __init__(self, n_alleles, seq_length, q=None):
        self.alpha = n_alleles
        self.l = seq_length
        self.lp1 = self.l + 1
        self.t = self.l * self.alpha

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
        return(self.forward( **kwargs))


class VarianceComponentKernel(SequenceKernel):
    def __init__(self, n_alleles, seq_length, q=None, use_p=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.use_p = use_p
        
        if use_p:
            ps = 1. /n_alleles * np.ones((seq_length, n_alleles))
            L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length,
                                   max_size=1, ps=ps)
            W = ProjectionOperator(L=L)
            self.B = W.B
            
            if q is None:
                q = (self.l - 1) / self.l
            self.q = q
            self.logq = np.log(q)
            ks = np.arange(self.lp1)
            log_q_powers = self.logq * ks
            log_1mq_powers = np.append([-np.inf], np.log(1 - np.exp(log_q_powers[1:])))
            self.lsf = self.l * log_1mq_powers
            self.log_odds = log_q_powers - log_1mq_powers
        else:
            self.calc_krawchouk_matrix()
    
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
            cov += coeffs[power] * np.exp(self.log_scaling_factors[power] + m)
        return(cov)
    
    def forward(self, lambdas, ps=None):
        if self.use_p:
            check_error(ps is not None, msg='ps must be provided')
            hamming_distance = self.calc_hamming_distance(self.x1, self.x2)
            cov = self.W_kd.dot(lambdas)[hamming_distance]
        else:
            self._forward_ps(lambdas, ps)
        return(cov)
    
    