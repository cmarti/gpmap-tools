import numpy as np

from gpmap.linop import KronTriangularInverseOperator
from gpmap.utils import check_error


class Gaussian(object):
    def __init__(self, mean, K=None, C=None):
        self.mean = mean
        self.dim = mean.shape[0]
        
        if K is None and C is None:
            raise ValueError('Either K or C must be provided')
        
        elif K is not None:
            msg = "The size of the mean should match the covariance matrix"
            check_error(mean.shape[0] == K.shape[0], msg=msg)
            self.mode = 'covariance'
            self.K = K
            self.C = K.inverse() if hasattr(K, 'inverse') else None
                
        else:
            msg = "The size of the mean should match the precision matrix"
            check_error(mean.shape[0] == C.shape[0], msg=msg)
            self.mode = 'precision'
            self.C = C
            self.K = C.inverse() if hasattr(C, 'inverse') else None

        self.L = K.cholesky() if hasattr(K, 'cholesky') else None
        self.K_sqrt = K.matrix_sqrt() if hasattr(K, 'matrix_sqrt') else None
        self.L_inv = KronTriangularInverseOperator(self.L) if self.L is not None else None

    def logp(self, x):
        n = x.shape[0]
        logp = -0.5 * n * np.log(2 * np.pi)

        if self.mode == 'covariance' and self.L is not None:
            z = self.L_inv @ (x - self.mu)
            logp -= 0.5 * np.sum(np.square(z)) + self.L.logdet()
            
        else:
            msg = "Only covariance matrices with cholesky method are available"
            raise NotImplementedError(msg)
            
        return logp
    
    def cg_sample(self, n_samples, max_iter=100, tol=1e-8):
        r"""
        Adapted from https://pygauss-gaussian-sampling.readthedocs.io/en/latest/_modules/pygauss/direct_sampling.html#sampler_CG
        
        Algorithm dedicated to sample from a multivariate real-valued Gaussian 
        distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
        :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` based on the
        conjugate gradient algorithm.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        max_iter : int, optional
            Number of conjugate gradient iterations.
        tol : float, optional
            Tolerance threshold used to stop the conjugate gradient sampler.
            
        Returns
        -------
        x : ndarray
            Sample from the multivariate Gaussian distribution.
        """ 

        mu = self.mean.reshape((self.dim, 1))
        init = mu.flatten()
        loss_conj = False
        shape = (self.dim, n_samples)
        x = np.zeros(shape)
        iteration = 1
            
        if self.mode == "precision":
            C_init = (self.C @ init).reshape((self.dim, 1))
            r_old = np.random.normal(size=shape) - C_init
            p_old = r_old
            C_p_old = self.C @ p_old
            d_old = (p_old * C_p_old).sum(0)
            y = init.reshape((self.dim, 1))
            r_new = np.ones(shape)
            norms = np.array([np.linalg.norm(v) for v in r_new.T])
            
            while (norms >= tol).any() and iteration <= max_iter:
                gam = (r_old * r_old).sum(axis=0) / d_old
                z = np.random.normal(size=n_samples)
                y = y + z / np.sqrt(d_old) * p_old
                r_new = r_old - gam * C_p_old
                beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
                p_new = r_new - beta * p_old
                
                if (np.abs((p_new * C_p_old).sum(0)) >= 1e-4).any() and loss_conj:
                    print(f'Loss of conjugacy happened at iteration {iteration}')
                    self.loss_conj, self.iter_loss_conj = True, iteration
                
                C_p_new = self.C @ p_new
                d_new = (p_new * C_p_new).sum(0)
                r_old = r_new
                p_old = p_new
                d_old = d_new
                C_p_old = C_p_new
                norms = np.array([np.linalg.norm(v) for v in r_new.T])
                iteration += 1
                
        elif self.mode == "covariance":
            K_init = (self.K @ init).reshape((self.dim, 1))
            r_old = np.random.normal(size=shape) - K_init
            p_old = r_old
            K_p_old = self.K @ p_old
            d_old = (p_old * K_p_old).sum(0)
            y = init.reshape((self.dim, 1))
            r_new = np.ones(shape)
            
            norms = np.array([np.linalg.norm(v) for v in r_new.T])
            while (norms >= tol).any() and iteration <= max_iter:
                gam = (r_old * r_old).sum(axis=0) / d_old
                z = np.random.normal(size=n_samples)
                y = y + z / np.sqrt(d_old) * K_p_old
                r_new = r_old - gam * K_p_old
                beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
                p_new = r_new - beta * p_old
                
                if (np.abs((p_new * K_p_old).sum(0)) >= 1e-4).any() and loss_conj:
                    print(f'Loss of conjugacy happened at iteration {iteration}')
                    self.loss_conj, self.iter_loss_conj = True, iteration
                
                K_p_new = self.K @ p_new
                d_new = (p_new * K_p_new).sum(0)
                r_old = r_new
                p_old = p_new
                d_old = d_new
                K_p_old = K_p_new
                norms = np.array([np.linalg.norm(v) for v in r_new.T])
                iteration += 1
        
        else:
            msg = 'Sampling can only be done in precision or covariance modes'
            raise ValueError(msg)
        print(iteration)
        x = mu + y
        return x

    def sample(self, n_samples):
        
        if self.K is not None and isinstance(self.K, np.ndarray):
            L = np.linalg.cholesky(self.K)
            x = L @ np.random.normal(size=(self.dim, n_samples))
            
        elif self.L is not None:
            print('Cholesky')
            x = self.L @ np.random.normal(size=(self.dim, n_samples))
        
        elif self.K_sqrt is not None:
            print('Matrix sqrt')
            x = self.K_sqrt @ np.random.normal(size=(self.dim, n_samples))
            
        else:
            print('CG')
            x = self.cg_sample(n_samples=n_samples)
            
        return x