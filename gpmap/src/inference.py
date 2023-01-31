#!/usr/bin/env python
import numpy as np
import pandas as pd

from numpy.linalg.linalg import matrix_power
from tqdm import tqdm
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve import minres
from scipy.optimize._minimize import minimize
from scipy.special._basic import comb
from scipy.linalg.basic import solve
from scipy.sparse.linalg import cg
from scipy.special._logsumexp import logsumexp
from scipy.stats.stats import pearsonr
from scipy.stats._continuous_distns import norm

from gpmap.src.settings import U_MAX, PHI_LB, PHI_UB
from gpmap.src.utils import (get_sparse_diag_matrix, check_error,
                             reciprocal, get_CV_splits,
                             calc_matrix_polynomial_quad)
from gpmap.src.seq import (guess_space_configuration, get_alphabet,
                           get_seqs_from_alleles)
from gpmap.src.linop import (DeltaPOperator, ProjectionOperator,
                             LaplacianOperator)
from gpmap.src.kernel import KernelAligner


class LandscapeEstimator(object):
    def __init__(self, expand_alphabet=True, max_L_size=None):
        self.expand_alphabet = expand_alphabet
        self.max_L_size = max_L_size
    
    def get_regularization_constants(self):
        return(10**(np.linspace(self.min_log_reg, self.max_log_reg, self.num_reg)))
    
    def get_cv_iter(self, hyperparam_values):
        for fold, train, validation in get_CV_splits(X=self.X, y=self.y,
                                                     y_var=self.y_var,
                                                     nfolds=self.nfolds,
                                                     count_data=self.count_data):        
            for param in hyperparam_values:
                yield(param, fold, train, validation)
    
    def define_space(self, seq_length=None, n_alleles=None, genotypes=None,
                     alphabet_type='custom'):
        if genotypes is not None:
            configuration = guess_space_configuration(genotypes,
                                                      ensure_full_space=False,
                                                      force_regular=True)
            seq_length = configuration['length']
            alphabet = configuration['alphabet']
            n_alleles = configuration['n_alleles'][0]
            n_alleles = len(alphabet[0])
        else:
            msg = 'Either seq_length and alleles or genotypes must be provided'
            check_error(seq_length is not None and n_alleles is not None, msg=msg)
            alphabet = get_alphabet(n_alleles=n_alleles,
                                    alphabet_type=alphabet_type)
            alphabet = [alphabet] * seq_length
        
        self.set_config(seq_length, n_alleles, alphabet)
    
    def set_config(self, seq_length, n_alleles, alphabet):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.n_genotypes = n_alleles ** seq_length
        self.genotypes = np.array(list(get_seqs_from_alleles(alphabet)))
        self.genotype_idxs = pd.Series(np.arange(self.n_genotypes),
                                       index=self.genotypes)
    
    def calc_L_powers_unique_entries_matrix(self):
        """Construct entries of powers of L. 
        Column: powers of L. 
        Row: Hamming distance"""
        
        l, a, s = self.seq_length, self.n_alleles, self.seq_length + 1
    
        # Construct C
        C = np.zeros([s, s])
        for i in range(s):
            for j in range(s):
                if i == j:
                    C[i, j] = i * (a - 2)
                if i == j + 1:
                    C[i, j] = i
                if i == j - 1:
                    C[i, j] = (l - j + 1) * (a - 1)
    
        # Construct D
        D = np.array(np.diag(l * (a - 1) * np.ones(s), 0))
    
        # Construct B
        B = D - C
    
        # Construct u
        u = np.zeros(s)
        u[0], u[1] = l * (a - 1), -1
    
        # Construct MAT column by column
        MAT = np.zeros([s, s])
        MAT[0, 0] = 1
        for j in range(1, s):
            MAT[:, j] = matrix_power(B, j-1).dot(u)
    
        # Invert MAT
        self.L_powers_unique_entries_inv = np.linalg.inv(MAT)
        self.L_powers_unique_entries = MAT
        

class VCregression(LandscapeEstimator):
    '''
        Variance Component regression model that allows inference and prediction
        of a scalar function in sequence spaces under a Gaussian Process prior
        parametrized by the contribution of the different orders of interaction
        to the observed genetic variability of a continuous phenotype
        
        It requires the use of the same number of alleles per sites
        
        Parameters
        ----------
        p : np.array of shape (seq_length, n_alleles)
            Stationary frequencies of alleles under a simple random walk
            to use for the generalized or skewed variance component regression.
            If not provided, regular VC regression will be applied
            
    '''
    def __init__(self, beta=0, cross_validation=False, 
                 num_beta=20, nfolds=5, min_log_beta=-2,
                 max_log_beta=7, max_L_size=False):
        super().__init__(max_L_size=max_L_size)
        self.beta = beta
        self.nfolds = nfolds
        self.num_reg = num_beta
        self.total_folds = self.nfolds * self.num_reg
        
        self.min_log_reg = min_log_beta
        self.max_log_reg = max_log_beta
        self.count_data = False
        self.run_cv = cross_validation
        
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom', ps=None):
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.kernel_aligner = KernelAligner(self.seq_length, self.n_alleles)
        self.W = ProjectionOperator(L=LaplacianOperator(self.n_alleles, self.seq_length, ps=ps,
                                                        max_size=self.max_L_size))
        self.calc_L_powers_unique_entries_matrix()
        
    def get_obs_idx(self, seqs):
        obs_idx = self.genotype_idxs[seqs]
        return(obs_idx)
    
    def get_gt_to_data_matrix(self, idx=None, subset_idx=None):
        if idx is None:
            idx = self.genotype_idxs.values
        
        if subset_idx is None:
            obs_idx = idx.copy()
        else:
            obs_idx = idx[subset_idx]
            
        n_obs = obs_idx.shape[0]
        gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
                             shape=(self.n_genotypes, n_obs))
        return(gt2data)

    def set_data(self, X, y, y_var=None):
        self.init(genotypes=X)
        self.obs_idx = self.get_obs_idx(X)
        self.X = X
        self.y = y
        self.n_obs = y.shape[0]

        if y_var is None:
            y_var = np.zeros(y.shape[0])    
                
        self.y_var = y_var
        self.E_sparse = get_sparse_diag_matrix(y_var)
        self.gt2data = self.get_gt_to_data_matrix(self.obs_idx)
    
    def calc_cv_loss(self, cv_data):
        for beta, fold, train, test in tqdm(cv_data, total=self.total_folds):
            X_train, y_train, y_var_train = train 
            X_test, y_test, y_var_test = test

            # Fit on training data
            self.set_data(X=X_train, y=y_train, y_var=y_var_train)
            lambdas = self._fit(beta)

            # Calculate cv logL and r2 on test data
            self.set_lambdas(lambdas)
            ypred = self.predict(X_test)['ypred'].values
            r2 = pearsonr(ypred, y_test)[0] ** 2
            logL = norm.logpdf(y_test, loc=ypred, scale=np.sqrt(y_var_test)).sum()
            
            # Calculate loss in test data
            self.set_data(X=X_test, y=y_test, y_var=y_var_test)
            cov, ns = self.calc_emp_dist_cov()
            self.kernel_aligner.set_data(cov, ns)
            mse = self.kernel_aligner.calc_mse(lambdas)
            yield({'beta': beta, 'fold': fold, 'mse': mse, 'r2': r2,
                   'logL': logL})
            
    def fit_beta_cv(self):
        beta_values = self.get_regularization_constants()
        cv_data = self.get_cv_iter(beta_values)
        cv_loss = self.calc_cv_loss(cv_data)
         
        self.cv_loss_df = pd.DataFrame(cv_loss)
        self.cv_loss_df['log_beta'] = np.log10(self.cv_loss_df['beta'])
        logL = self.cv_loss_df.groupby('beta')['logL'].mean()
        self.beta = logL.index[np.argmax(logL)]
    
    def _fit(self, beta=None):
        if beta is None:
            beta = self.beta
            
        cov, ns = self.calc_emp_dist_cov()
        self.kernel_aligner.set_data(cov, ns)
        self.kernel_aligner.set_beta(beta)
        lambdas = self.kernel_aligner.fit()
        return(lambdas)
    
    def fit(self, X, y, y_var=None):
        """
        Infers the variance components from the provided data, this is, 
        the relative contribution of the different orders of interaction
        to the variability in the sequence-function relationships
        
        Stores learned `lambdas` in the attribute VCregression.lambdas
        to use internally for predictions and returns them as output
        
        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `y`
            
        y : array-like of shape (n_obs,)
            Vector containing the observed phenotypes corresponding to `X`
            sequences
        
        y_var : array-like of shape (n_obs,)
            Vector containing the empirical or experimental known variance for
            the measurements in `y`
        
        Returns
        -------
        lambdas: array-like of shape (seq_length + 1,)
            Variances for each order of interaction k inferred from the data
        
        """
        self.init(genotypes=X)
        self.set_data(X, y, y_var=y_var)
        
        if self.run_cv:
            self.fit_beta_cv()
            self.set_data(X, y, y_var=y_var)
        
        lambdas = self._fit()
        self.set_lambdas(lambdas) 
        return(lambdas)
    
    def calc_emp_dist_cov(self):
        seq_values = self.gt2data.dot(self.y)
        observed_seqs = self.gt2data.dot(np.ones(self.n_obs))

        # Compute rho_d and N_d
        size = self.seq_length + 1
        cov, distance_class_ns = np.zeros(size), np.zeros(size)
        for d in range(size):
            c_k = self.L_powers_unique_entries_inv[:, d]
            
            distance_class_ns[d] = calc_matrix_polynomial_quad(c_k, self.W.L,
                                                               observed_seqs)
            
            quad = calc_matrix_polynomial_quad(c_k, self.W.L, seq_values)
            cov[d] = reciprocal(quad, distance_class_ns[d])
            
        return(cov, distance_class_ns)
    
    def lambdas_to_variance(self, lambdas):
        variance_components = (lambdas * self.W.L.m_k)[1:]
        variance_components = variance_components / variance_components.sum()
        return(variance_components)
    
    def set_lambdas(self, lambdas):
        self.lambdas = lambdas
        self.W.set_lambdas(lambdas=lambdas)
    
    def predict(self, Xpred=None, estimate_variance=False):
        """
        Compute the Maximum a Posteriori (MAP) estimate of the phenotype at 
        the provided or all genotypes
        
        Parameters
        ----------
        Xpred : array-like of shape (n_genotypes,)
            Vector containing the genotypes for which we want to predict the
            phenotype. If `n_genotypes == None` then predictions are provided
            for the whole sequence space
        
        estimate_variance : bool (False)
            Option to also return the posterior variances for each individual
            genotype
        
        Returns
        -------
        function : pd.DataFrame of shape (n_genotypes, 1)
                   Returns the phenotypic predictions for each input genotype
                   in the column ``ypred`` and genotype labels as row names.
                   If ``estimate_variance=True``, then it has an additional
                   column with the posterior variances for each genotype
        """
        
        # Minimize error given lambdas
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=self.K_BB_E_dot)
        a_star = minres(Kop, self.y, tol=1e-9)[0]
        ypred = self.W.dot(self.gt2data.dot(a_star))
        pred = pd.DataFrame({'ypred': ypred}, index=self.genotypes)

        if Xpred is not None:
            pred = pred.loc[Xpred, :]

        if estimate_variance:
            pred['var'] = self.estimate_posterior_variance(Xpred=pred.index)
        
        return(pred)
    
    def K_BB_E_dot(self, v):
        """ multiply by the m by m matrix K_BB + E"""
        yhat = self.W.dot(self.gt2data.dot(v))
        ynoise = yhat + self.gt2data.dot(self.E_sparse.dot(v))
        return(self.gt2data.T.dot(ynoise))
    
    def estimate_posterior_variance(self, Xpred=None):
        """compute posterior variances for a list of sequences"""
        # TODO: fix response: cannot give negative variance estimates in test
        # TODO: replace rho by covariance matrix
        if Xpred is None:
            Xpred = self.space.state_labels
        n_pred, pred_idx = Xpred.shape[0], self.get_obs_idx(Xpred)
    
        Kop = LinearOperator((self.n_obs, self.n_obs), matvec=self.K_BB_E_dot)
        K_ii = self.W.calc_covariance_distance()[0]
        
        # Compute posterior variance for each target sequence
        post_vars = []
        for i in tqdm(pred_idx, total=n_pred):
            vec = np.zeros(self.n_genotypes)
            vec[i] = 1
            K_Bi = self.W.dot(vec)[self.obs_idx]
            # Posibility to optimize with a good preconditioner
            alph = cg(Kop, K_Bi)[0]
            post_vars.append(K_ii - np.sum(K_Bi * alph))
    
        return(np.array(post_vars))
    
    def simulate(self, lambdas, sigma=0, p_missing=0):
        '''
        Simulates data under the specified Variance component priors
        
        Parameters
        ----------
        lambdas : array-like of shape (seq_length + 1,)
            Vector containing the variance of each of the ``seq_length``
            components characterizing the prior
        
        sigma : real
            Standard deviation of the experimental noise additional to the
            variance components
        
        p_missing : float between 0 and 1
            Probability of randomly missing genotypes in the simulated output
            data
            
        Returns
        -------
        data : pd.DataFrame of shape (n_genotypes, 3)
            DataFrame with the columns ``y_true``, ``y``and ``var`` corresponding
            to the true function at each genotype, the observed values and the
            variance of the measurement respectively for each sequence or
             genotype indicated in the ``DataFrame.index`` 
        
        '''
        
        a = np.random.normal(size=self.n_genotypes)
        
        if hasattr(self, 'D_pi'):
            a = self.D_pi.dot(a)
        
        self.set_lambdas(np.sqrt(lambdas))
        yhat = self.W.dot(a)
        y = np.random.normal(yhat, sigma) if sigma > 0 else yhat
        variance = np.full(self.n_genotypes, sigma**2)
        
        if p_missing > 0:
            sel_idxs = np.random.uniform(self.n_genotypes) < p_missing
            y[sel_idxs], variance[sel_idxs] = np.nan, np.nan
        
        data = pd.DataFrame({'y_true': yhat, 'y': y, 'var': variance},
                            index=self.genotypes)
        return(data)


class DeltaPEstimator(LandscapeEstimator):
    def __init__(self, P, a=None, num_reg=20, nfolds=5,
                 a_resolution=0.1, max_a_max=1e12, fac_max=0.1, fac_min=1e-6,
                 opt_method='L-BFGS-B', optimization_opts={}, scale_by=1,
                 gtol=1e-3, max_L_size=False):
        super().__init__(max_L_size=max_L_size)
        self.P = P
        self.a = a
        self.a_is_fixed = a is not None
        self.nfolds = nfolds
        
        msg = '"a" can only be None or >= 0'
        check_error(a is None or a >= 0, msg=msg)
        
        # Attributes to generate a values
        self.num_reg = num_reg
        self.total_folds = self.nfolds * self.num_reg
        
        # Default bounds for a in absence of a more clever method
        self.min_log_reg = -4
        self.max_log_reg = 10
        
        # Parameters to generate a grid in SeqDEFT, but should be generalizable
        # by just defining a distance metric for phi
        self.a_resolution = a_resolution
        self.max_a_max = max_a_max
        self.scale_by = scale_by
        self.fac_max = fac_max
        self.fac_min = fac_min
        
        # Optimization attributes
        self.opt_method = opt_method
        optimization_opts['gtol'] = gtol
        optimization_opts['ftol'] = 0
        self.optimization_opts = optimization_opts
        
    def calc_n_p_faces(self, length, P, n_alleles):
        return(comb(length, P) * comb(n_alleles, 2) ** P * n_alleles ** (length - P))
        
    def init(self, seq_length=None, n_alleles=None, genotypes=None,
             alphabet_type='custom'):
        self.define_space(seq_length=seq_length, n_alleles=n_alleles,
                          genotypes=genotypes, alphabet_type=alphabet_type)
        self.n_p_faces = self.calc_n_p_faces(self.seq_length, self.P, self.n_alleles) 
        self.DP = DeltaPOperator(self.P, self.n_alleles, self.seq_length, max_L_size=self.max_L_size)
        self.DP.calc_kernel_basis()
    
    def get_a_values(self):
        return(self.get_regularization_constants())
    
    def calc_neg_log_prior_prob(self, phi, a):
        S1 = a / (2*self.n_p_faces) * self.DP.quad(phi)
        return(S1)
    
    def calc_neg_log_prior_prob_grad(self, phi, a):
        grad_S1 = a / self.n_p_faces * self.DP.dot(phi)
        return(grad_S1)
    
    def S(self, phi, a):
        S = self.calc_neg_log_prior_prob(phi, a)
        S += self.calc_neg_log_likelihood(phi)
        if hasattr(self, 'calc_regularization'):
            S += self.calc_regularization(phi)
        return(S)
    
    def S_grad(self, phi, a):
        S_grad = self.calc_neg_log_prior_prob_grad(phi, a)
        S_grad += self.calc_neg_log_likelihood_grad(phi)
        if hasattr(self, 'calc_regularization'):
            S_grad += self.calc_regularization_grad(phi) 
        return(S_grad)
    
    def S_inf(self, b):
        phi = self._b_to_phi(b)
        S = self.calc_neg_log_likelihood(phi)
        if hasattr(self, 'calc_regularization'):
            S += self.calc_regularization(phi)
        return(S)
    
    def S_grad_inf(self, b):
        phi = self._b_to_phi(b)
        S_grad = self.calc_neg_log_likelihood_grad(phi)
        if hasattr(self, 'calc_regularization'):
            S_grad += self.calc_regularization_grad(phi)
        S_grad = self._phi_to_b(S_grad)
        return(S_grad)
    
    def calc_cv_logL(self, cv_data, phi_initial=None):
        for a, fold, train, test in tqdm(cv_data, total=self.total_folds):
            (X_train, y_train), (X_test, y_test) = train, test

            self.set_data(X=X_train, y=y_train)
            phi = self._fit(a, phi_initial=phi_initial)
            
            self.set_data(X=X_test, y=y_test)
            test_logL = -self.calc_neg_log_likelihood(phi)
            yield({'a': a, 'fold': fold, 'logL': test_logL})

    def get_cv_logL_df(self, cv_logL):
        cv_log_L = pd.DataFrame(cv_logL)
        cv_log_L['log_a'] = np.log10(cv_log_L['a'])
        cv_log_L['sd'] = self._a_to_sd(cv_log_L['a'])
        cv_log_L['log_sd'] = np.log10(cv_log_L['sd'])
        return(cv_log_L)
    
    def get_ml_a(self, cv_logL_df):
        df = cv_logL_df.groupby('a')['logL'].mean()
        return(df.index[np.argmax(df)])
    
    def fit_a_cv(self, phi_inf=None):
        if phi_inf is None:
            phi_inf = self._fit(np.inf)
        a_values = self.get_a_values(phi_inf=phi_inf)
        cv_data = self.get_cv_iter(a_values)
        cv_logL = self.calc_cv_logL(cv_data, phi_initial=phi_inf) 
        
        self.logL_df = self.get_cv_logL_df(cv_logL)    
        self.a = self.get_ml_a(self.logL_df)
    
    def _phi_to_b(self, phi):
        return(self.DP.kernel_basis.T.dot(phi))
    
    def _b_to_phi(self, b):
        return(self.DP.kernel_basis.dot(b))

    def _a_to_sd(self, a):
        return(np.sqrt(self.n_p_faces / a))
    
    def _sd_to_a(self, sd):
        return(self.n_p_faces / sd ** 2)
    
    def _fit(self, a, phi_initial=None):
        check_error(a >= 0, msg='"a" must be larger or equal than 0')
        
        if phi_initial is None and a > 0:
            # Initialize with equal log density for every sequence
            phi_initial = np.log(self.n_genotypes) * np.ones(self.n_genotypes)
        
        if a == 0 and hasattr(self, '_fit_a_0'):
            phi = self._fit_a_0()
            
        elif a == np.inf:
            b_initial = self._phi_to_b(phi_initial)
            res = minimize(fun=self.S_inf, jac=self.S_grad_inf,
                           x0=b_initial, method=self.opt_method,
                           options=self.optimization_opts)
            if not res.success:
                self.report(res.message)
            b_a = res.x
            phi = self._b_to_phi(b_a)
            
        else:
            res = minimize(fun=self.S, jac=self.S_grad, args=(a,),
                           x0=phi_initial, method=self.opt_method,
                           options=self.optimization_opts)
            if not res.success:
                print(res.message)
            phi = res.x

        # a, N = a * scale_by, N *scale_by    
        return(phi)
    
    def fit(self, X, y, force_fit_a=True):
        """
        Infers the sequence-function relationship under the specified
        \Delta^{(P)} prior 
        
        Parameters
        ----------
        X : array-like of shape (n_obs,)
            Vector containing the genotypes for which have observations provided
            by `counts`. Missing sequences are assumed to have observed 0 times
            
        y : array-like of shape (n_obs,) or (n_obs, n_cols)
            Vector or matrix containing the expected type of data for each model
            
        force_fit_a : bool 
            Whether to re-fit ``a`` using cross-validation even if it is already
            defined a priori
            
        Returns
        -------
        
        landscape : pd.DataFrame (n_genotypes, 2)
            DataFrame containing the estimated function for each possible
            sequence in the space
        
        """
        self.init(genotypes=X)
        self.set_data(X, y)
        phi_inf = self._fit(np.inf)
        
        if not self.a_is_fixed and force_fit_a:
            self.fit_a_cv(phi_inf=phi_inf)
            self.set_data(X, y)
        
        # Fit model with a_star or provided a
        phi = self._fit(self.a, phi_initial=phi_inf)
        output = self.phi_to_output(phi)
        return(output)
    

class SeqDEFT(DeltaPEstimator):
    # Required methods
    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.y_var = None
        
        data = self.fill_zeros_counts(X, y)
        self.N = data.sum()
        self.R = (data / self.N)
    
    @property
    def count_data(self):
        return(True)
    
    def calc_regularization(self, phi):
        regularizer = 0
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.any() > 0:
                regularizer += np.sum((phi[flags] - PHI_UB)**2)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.any() > 0:
                regularizer += np.sum((phi[flags] - PHI_LB)**2)
        return(regularizer)
    
    def calc_regularization_grad(self, phi):
        regularizer = np.zeros(self.n_genotypes)
        if np.isfinite(PHI_UB):
            flags = (phi > PHI_UB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi[flags] - PHI_UB)
        if np.isfinite(PHI_LB):
            flags = (phi < PHI_LB)
            if flags.sum() > 0:
                regularizer[flags] += 2 * (phi[flags] - PHI_LB)
        return(regularizer)
    
    def calc_neg_log_likelihood(self, phi):
        S2 = self.N * np.sum(self.R * phi)
        S3 = self.N * np.sum(safe_exp(-phi))
        return(S2 + S3)
    
    def calc_neg_log_likelihood_grad(self, phi):
        grad_S2 = self.N * self.R
        grad_S3 = self.N * safe_exp(-phi)
        return(grad_S2 - grad_S3)
    
    def phi_to_output(self, phi):
        Q_star = self.phi_to_Q(phi)
        seq_densities = pd.DataFrame({'frequency': self.R, 'Q_star': Q_star},
                                     index=self.genotypes)
        return(seq_densities)
    
    # Optional methods
    def calc_a_max(self, phi_inf):
        a_max = self.n_p_faces * self.fac_max
        
        phi_max = self._fit(a_max, phi_initial=phi_inf)
        distance = D_geo(phi_max, phi_inf)
        
        while distance > self.a_resolution and a_max < self.max_a_max:
            a_max *= 10
            phi_max = self._fit(a_max, phi_initial=phi_inf)
            distance = D_geo(phi_max, phi_inf)
            
        return(a_max)
    
    def calc_a_min(self, phi_inf):
        a_min = self.n_p_faces * self.fac_min
        
        phi_0 = self._fit(0)
        phi_min = self._fit(a_min, phi_initial=phi_inf)
        
        distance = D_geo(phi_min, phi_0)
        
        while distance > self.a_resolution:
            a_min /= 10
            phi_min = self._fit(a_min, phi_initial=phi_inf)
            distance = D_geo(phi_min, phi_0)
        return(a_min)

    def get_a_values(self, phi_inf=None):
        if phi_inf is None:
            phi_inf = self._fit(np.inf)
        a_min = self.calc_a_min(phi_inf) 
        a_max = self.calc_a_max(phi_inf)
        a_values = np.geomspace(a_min, a_max, self.num_reg)
        return(a_values)

    def fill_zeros_counts(self, X, y):
        data = pd.Series(np.zeros(self.n_genotypes), index=self.genotypes)
        data.loc[X] = y
        data = data.astype(int)
        return(data)

    def phi_to_logQ(self, phi):
        return(-phi - logsumexp(-phi))
    
    def phi_to_Q(self, phi):
        return(np.exp(self.phi_to_logQ(phi)))
    
    def _fit_a_0(self):
        with np.errstate(divide='ignore'):
            phi = -np.log(self.R)
        return(phi)
    
    # TODO: fix and refactor simulation code
    def simulate(self, N, a_true, random_seed=None):
        # Set random seed
        np.random.seed(random_seed)
    
        # Simulate phi from prior distribution
        v = np.random.normal(size=self.n_genotypes)
        phi_true = np.zeros(self.n_genotypes)
        
        for k in range(self.P, self.seq_length+1):
            eta_k = np.sqrt(self.n_p_faces) / np.sqrt(a_true * self.D_eig_vals[k])
            self.solve_b_k(k)
            phi_true += eta_k * self.W_k_opt(v)
    
        # Construct Q_true from the simulated phi
        Q_true = np.exp(-phi_true) / np.sum(np.exp(-phi_true))
    
        # Simulate N data points from Q_true
        data = np.random.choice(self.n_genotypes, size=N, replace=True, p=Q_true)
    
        # Obtain count data
        values, counts = np.unique(data, return_counts=True)
        Ns = np.zeros(self.n_genotypes)
        Ns[values] = counts
    
        # Normalize count data
        R = Ns / N
    
        # Save N and R
        data_dict = {'N': int(N), 'R': R, 'Q_true': Q_true}
    
        # Return
        return data_dict
    
    def solve_b_k(self, k):
        # Tabulate w_k(d)
        w_k = np.zeros(self.seq_length+1)
        for d in range(self.seq_length+1):
            w_k[d] = self.w(k, d)
    
        # Solve for b_k
        b_k = solve(self.MAT, w_k)
        
        self.b_k = b_k
    
    def w(self, k, d):
        ss = 0
        for q in range(self.seq_length+1):
            ss += (-1)**q * (self.n_alleles-1)**(k-q) * comb(d,q) * comb(self.seq_length-d,k-q)
        return 1/self.n_alleles**self.seq_length * ss
    
    def W_k_opt(self, v):
        max_power = len(self.b_k) - 1
        Lsv = np.zeros([self.n_genotypes,len(self.b_k)])
        Lsv[:,0] = self.b_k[0] * v
        power = 1
        while power <= max_power:
            v = self.L_opt(v)
            Lsv[:,power] = self.b_k[power] * v
            power += 1
        Wkv = Lsv.sum(axis=1)
        return Wkv


def D_geo(phi1, phi2):
    logQ1 = -phi1 - logsumexp(-phi1)
    logQ2 = -phi2 - logsumexp(-phi2)
    s = np.exp(logsumexp(0.5 * (logQ1 + logQ2)))
    x = min(s, 1)
    return 2 * np.arccos(x)


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)
