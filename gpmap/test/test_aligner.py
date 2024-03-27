#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.src.aligner import VCKernelAligner, RhoKernelAligner
from gpmap.src.linop import (RhoProjectionOperator, calc_covariance_vjs,
                             calc_covariance_distance, calc_variance_components)


class KernelAlignerTest(unittest.TestCase):
    def test_vc_kernel_alignment(self):
        # Simulate data
        np.random.seed(1)
        a, l, rho = 4, 5, 0.5
        P = 5 * RhoProjectionOperator(a, l, rho=rho).matrix_sqrt()
        y = P @ np.random.normal(size=P.shape[1])
        cov, ns = calc_covariance_distance(y, a, l)
        lambdas = calc_variance_components(y, a, l)

        # Define kernel aligner and fit unregularized model
        aligner = VCKernelAligner(a, l)
        lambdas_star = aligner.fit(cov, ns)
        pred = aligner.predict(lambdas_star)
        assert(np.allclose(cov, pred, rtol=0.01))
        assert(np.allclose(lambdas, lambdas_star, rtol=0.5))
        
        # Align with beta > 0
        aligner = VCKernelAligner(a, l, beta=0.01)
        lambdas_star = aligner.fit(cov, ns)
        pred = aligner.predict(lambdas_star)
        assert(np.allclose(cov, pred, atol=1e-5))
        assert(np.allclose(lambdas, lambdas_star, rtol=0.5))
        
    def test_rho_kernel_alignment(self):
        # Simulate data
        np.random.seed(1)
        log_mu0 = 0
        a, l = 4, 5
        logit_rho0 = np.array([-4, 0, 1., 1, -5])
        rho0 = np.exp(logit_rho0) / (1 + np.exp(logit_rho0))
        P = RhoProjectionOperator(a, l, rho=np.sqrt(rho0))
        y = P @ np.random.normal(size=P.shape[1])
        y = y / y.std()
        covs, ns, sites_matrix = calc_covariance_vjs(y, a, l)
        
        # Define kernel aligner and fit to empirical covariances
        aligner = RhoKernelAligner(a, l)
        log_mu, logit_rho = aligner.fit(covs, ns, sites_matrix)

        # Ensure big differences remain
        assert(np.all(logit_rho[0] < logit_rho[1:4] - 1))
        assert(np.all(logit_rho[-1] < logit_rho[1:4] - 1))

        # Ensure loss is lower than with the generating rhos
        loss0 = aligner.frobenius_norm(aligner.params_to_x(log_mu0, logit_rho0))
        loss = aligner.frobenius_norm(aligner.params_to_x(log_mu, logit_rho))
        assert(loss < loss0)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelAlignerTest']
    unittest.main()
