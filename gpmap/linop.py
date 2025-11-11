#!/usr/bin/env python
from itertools import combinations

import numpy as np
from numpy.linalg.linalg import matrix_power
from scipy.linalg import lu_factor, lu_solve, orth, solve_triangular
from scipy.special import comb, factorial
from tqdm import tqdm

try:
    from scipy.sparse.linalg.interface import _CustomLinearOperator
except ImportError:
    from scipy.sparse.linalg._interface import _CustomLinearOperator

from gpmap.matrix import (
    inv_dot,
    is_lower_triangular,
    tensordot,
)
from gpmap.utils import check_error


class ExtendedLinearOperator(_CustomLinearOperator):
    def _init_dtype(self):
        v = np.random.normal(size=2)
        self.dtype = v.dtype

    def get_column(self, i):
        vec = np.zeros(self.shape[1])
        vec[i] = 1
        return self.dot(vec)

    def get_diag(self):
        return np.array([self.get_column(i)[i] for i in range(self.shape[0])])

    def submatrix(self, row_idx=None, col_idx=None):
        return SubMatrixOperator(self, row_idx, col_idx)

    def todense(self):
        return self.dot(np.eye(self.shape[1]))

    def rowsum(self):
        v = np.ones(self.shape[0])
        return self.dot(v)

    def _matmat(self, B):
        x = []
        for i in range(B.shape[1]):
            x.append(self._matvec(B[:, i]))
        return np.array(x).T


class TriangularInverseOperator(ExtendedLinearOperator):
    def __init__(self, tri):
        self.tri = tri
        self.shape = tri.shape
        self.dtype = tri.dtype
        self.lower = is_lower_triangular(tri)

    def _matvec(self, b):
        return solve_triangular(self.tri, b, lower=self.lower)

    def _matmat(self, B):
        return solve_triangular(self.tri, B, lower=self.lower)


class InverseOperator(ExtendedLinearOperator):
    def __init__(
        self, linop, method="minres", atol=1e-14, maxiter=1000, **kwargs
    ):
        # TODO: implement preconditioner option
        self.linop = linop
        self.shape = linop.shape
        self.dtype = linop.dtype
        self.method = method
        self.atol = atol
        self.maxiter = maxiter
        self.kwargs = kwargs

    def _matvec(self, v):
        return inv_dot(
            self.linop,
            v,
            method=self.method,
            maxiter=self.maxiter,
            atol=self.atol,
            **self.kwargs,
        )


class DiagonalOperator(ExtendedLinearOperator):
    def __init__(self, diag):
        self.diag = diag
        self.shape = (diag.shape[0], diag.shape[0])
        self._init_dtype()

    def _matvec(self, v):
        return self.diag * v

    def _matmat(self, B):
        return np.expand_dims(self.diag, 1) * B

    def transpose(self):
        return self

    def logdet(self):
        msg = "All diagonal entries must be larger than 0 to compute logdet"
        check_error(np.all(self.diag > 0), msg=msg)
        return np.sum(np.log(self.diag))

    def det(self):
        return np.product(self.diag)


class IdentityOperator(DiagonalOperator):
    def __init__(self, n):
        self.shape = (n, n)

    def _matvec(self, v):
        return v

    def _rmatvec(self, B):
        return B

    def _matmat(self, B):
        return B

    def _rmatmat(self, B):
        return B

    def transpose(self):
        return self


class MatMulOperator(ExtendedLinearOperator):
    def __init__(self, linops):
        self.linops = linops
        self._init_dtype()

        if len(linops) > 1:
            for A1, A2 in zip(linops, linops[1:]):
                msg = "Dimensions of the operators do not match"
                check_error(A1.shape[1] == A2.shape[0], msg=msg)

        self.shape = (linops[0].shape[0], linops[-1].shape[1])

    def _matvec(self, v):
        u = v.copy()
        for linop in self.linops[::-1]:
            u = linop @ u
        return u

    def _matmat(self, B):
        return self._matvec(B)

    def transpose(self):
        return MatMulOperator([x.transpose() for x in self.linops[::-1]])


class StackedOperator(ExtendedLinearOperator):
    def __init__(self, linops, axis):
        self.linops = linops
        self.axis = axis
        self._init_dtype()
        ncols = [linop.shape[1] for linop in linops]
        nrows = [linop.shape[0] for linop in linops]

        if axis == 0:
            ncol = np.unique(ncols)
            msg = "Missmatch in number of columns: {}".format(ncols)
            check_error(ncol.shape[0] == 1, msg=msg)
            self.shape = (np.sum(nrows), ncol[0])

        elif axis == 1:
            nrow = np.unique(nrows)
            msg = "Missmatch in number of rows: {}".format(nrows)
            check_error(nrow.shape[0] == 1, msg=msg)
            self.shape = (nrow[0], np.sum(ncols))
        else:
            raise ValueError("Axis can only take values [0, 1]")

    def hstack_dot(self, As, v):
        u = 0
        start = 0
        for A in As:
            end = start + A.shape[1]
            u += A @ v[start:end]
            start = end
        return u

    def vstack_dot(self, As, v):
        u = np.zeros(self.shape[0])
        start = 0
        for A in As:
            end = start + A.shape[0]
            u[start:end] += A @ v
            start = end
        return u

    def _matvec(self, v):
        if self.axis == 1:
            return self.hstack_dot(self.linops, v)
        elif self.axis == 0:
            return self.vstack_dot(self.linops, v)

    def _rmatvec(self, v):
        if self.axis == 1:
            return self.vstack_dot([A.transpose() for A in self.linops], v)
        elif self.axis == 0:
            return self.hstack_dot([A.transpose() for A in self.linops], v)

    def transpose(self):
        return StackedOperator(
            [A.transpose() for A in self.linops], axis=1 - self.axis
        )


class SubMatrixOperator(ExtendedLinearOperator):
    def __init__(self, linop, row_idx=None, col_idx=None):
        self.linop = linop
        self.dtype = linop.dtype
        shape = [i for i in linop.shape]
        self.row_idx = row_idx
        self.col_idx = col_idx

        if row_idx is not None:
            shape[0] = row_idx.shape[0]
        if col_idx is not None:
            shape[1] = col_idx.shape[0]
        self.shape = tuple(shape)

    def _matvec(self, v):
        u = v.copy()
        if self.col_idx is not None:
            u = np.zeros(self.linop.shape[0])
            u[self.col_idx] = v

        u = self.linop @ u

        if self.row_idx is not None:
            u = u[self.row_idx]
        return u

    def _matmat(self, v):
        u = v.copy()

        if self.col_idx is not None:
            u = np.zeros((self.linop.shape[0], v.shape[1]))
            u[self.col_idx, :] = v

        u = self.linop @ u

        if self.row_idx is not None:
            u = u[self.row_idx, :]
        return u

    def _rmatmat(self, v):
        u = v.copy()

        if self.row_idx is not None:
            u = np.zeros((self.linop.shape[1], v.shape[1]))
            u[self.row_idx, :] = v

        u = self.linop.transpose() @ u

        if self.col_idx is not None:
            u = u[self.col_idx, :]
        return u

    def transpose(self):
        return SubMatrixOperator(
            self.linop.transpose(), row_idx=self.col_idx, col_idx=self.row_idx
        )


class ExpandIdxOperator(ExtendedLinearOperator):
    def __init__(self, n, idx):
        self.n = n
        self.idx = idx
        self.shape = (n, self.idx.shape[0])

    def _matvec(self, v):
        u = np.zeros(self.n)
        u[self.idx] = v
        return u

    def transpose(self):
        return SelIdxOperator(self.n, self.idx)


class SelIdxOperator(ExtendedLinearOperator):
    def __init__(self, n, idx):
        self.n = n
        self.idx = idx
        self.shape = (self.idx.shape[0], n)
        self._init_dtype()

    def _matvec(self, v):
        return v[self.idx]

    def transpose(self):
        return ExpandIdxOperator(self.n, self.idx)


class KronOperator(ExtendedLinearOperator):
    def __init__(self, matrices):
        self.matrices = matrices
        self.n_matrices = len(matrices)
        self.v_shape = [m_i.shape[1] for m_i in self.matrices]
        self.shape = (
            np.prod([m_i.shape[0] for m_i in self.matrices]),
            np.prod(self.v_shape),
        )
        self.dtype = self.matrices[0].dtype

    def _matvec(self, v):
        check_error(
            v.shape[0] == self.shape[1],
            msg="Incorrect dimensions of matrices and `v`",
        )
        u_tensor = v.reshape(self.v_shape)
        for i, m in enumerate(self.matrices):
            u_tensor = tensordot(m, u_tensor, i)
        u = u_tensor.transpose().flatten()
        return u

    def _todense(self, matrices):
        if len(matrices) == 1:
            return matrices[0]
        matrices = matrices[:-2] + [np.kron(matrices[-2], matrices[-1])]
        return self._todense(matrices)

    def todense(self):
        return self._todense(self.matrices)

    def transpose(self):
        return KronOperator([m.T for m in self.matrices])

    def cholesky(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Cannot compute cholesky of non-square matrix")
        return KronSquareTriangularOperator(
            [np.linalg.cholesky(m) for m in self.matrices]
        )


class KronSquareTriangularOperator(KronOperator):
    def logdet(self):
        # TODO: needs to be updated to allow arbitrary linops instead of
        # matrices alone
        logdet = 0
        n = self.shape[0]
        for matrix in self.matrices:
            is_lower_triangular(matrix)
            k = n / matrix.shape[0]
            logdet += np.log(np.diag(matrix)).sum() * k
        return logdet


class KronTriangularInverseOperator(KronOperator):
    def __init__(self, kron_linop):
        self.kron_linop = kron_linop
        self.shape = kron_linop.shape
        self.dtype = kron_linop.dtype
        matrices = [TriangularInverseOperator(m) for m in kron_linop.matrices]
        KronOperator.__init__(self, matrices)


class PolynomialOperator(ExtendedLinearOperator):
    def __init__(self, linop, coeffs):
        self.linop = linop
        self.shape = linop.shape
        self.set_coeffs(coeffs)
        self._init_dtype()

    def set_coeffs(self, coeffs):
        self.coeffs = np.array(coeffs)
        self.degree = self.coeffs.shape[0]

    def _matvec(self, v):
        power = v
        u = self.coeffs[0] * v
        for c in self.coeffs[1:]:
            power = self.linop.dot(power)
            u += c * power
        return u

    def calc_trace_hutchinson(self, n_vectors):
        trace = []
        for _ in range(n_vectors):
            v = 0.5 - (np.random.uniform(size=self.shape[1]) > 0.5).astype(
                float
            )
            power = v
            trace_i = 0
            for c in self.coeffs[1:]:
                if c == 0:
                    continue
                power = self.linop.dot(power)
                trace_i += c * np.sum(v * power)
            trace.append(trace_i)
        return np.array(trace)


class TruncatedMatrixExp(PolynomialOperator):
    def __init__(self, linop, m):
        coeffs = np.array([1 / factorial(i) for i in range(m + 1)])
        super().__init__(linop, coeffs)


class TruncatedMatrixLog(PolynomialOperator):
    def __init__(self, linop, m, alpha=1):
        self.shape = linop.shape
        j = np.arange(1, m)
        coeffs = -(alpha**j) / j
        coeffs = np.append([0], coeffs)
        A = ExtendedLinearOperator(
            shape=self.shape,
            dtype=float,
            matvec=lambda v: v - alpha * linop.dot(v),
        )

        super().__init__(A, coeffs)
        self.alpha = alpha


class SeqOperator(ExtendedLinearOperator):
    def __init__(self, n_alleles, seq_length):
        self.alpha = n_alleles
        self.seq_length = seq_length
        self.lp1 = seq_length + 1
        self.n = self.alpha**seq_length
        self.shape = (self.n, self.n)
        self._init_dtype()
        self.shape_contracted = tuple([self.alpha] * self.seq_length)
        self.positions = np.arange(self.seq_length)

    def contract_v(self, v):
        return v.reshape(self.shape_contracted)

    def expand_v(self, v):
        return v.reshape(self.n)


class ConstantDiagSeqOperator(SeqOperator):
    def get_diag(self):
        return np.full(self.n, self.d)

    def _calc_trace(self):
        return self.n * self.d


class LaplacianOperator(ConstantDiagSeqOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.d = (self.alpha - 1) * self.seq_length
        self.lambdas = np.arange(self.seq_length + 1) * self.alpha
        self.lambdas_multiplicity = [
            comb(self.seq_length, k) * (self.alpha - 1) ** k
            for k in range(self.lp1)
        ]

    def _matvec(self, v):
        v = self.contract_v(v)
        u = self.seq_length * self.alpha * v
        for i in range(self.seq_length):
            u -= np.add.reduce(v, axis=i, keepdims=True)
        return self.expand_v(u)


class DeltaPOperator(ConstantDiagSeqOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.dtype = self.L.dtype
        self.m_k = self.L.lambdas_multiplicity
        self.set_P(P)
        self.calc_kernel_dimension()
        self.calc_n_p_faces()
        self.calc_n_p_faces_genotype()
        self.calc_lambdas()

    def calc_kernel_dimension(self):
        self.kernel_dimension = np.sum(self.m_k[: self.P])
        self.rank = self.n - self.kernel_dimension

    def calc_kernel_basis(self):
        return DeltaKernelBasisOperator(self.alpha, self.seq_length, self.P)

    def calc_n_p_faces_genotype(self):
        n_mut = self.seq_length * (self.alpha - 1)
        self.n_p_faces_genotype = float(comb(n_mut, self.P))

    def calc_n_p_faces(self):
        n_p_sites = comb(self.seq_length, self.P)
        n_p_faces_per_sites = comb(self.alpha, 2) ** self.P
        allelic_comb_remaining_sites = self.alpha ** (self.seq_length - self.P)
        self.n_p_faces = (
            n_p_sites * n_p_faces_per_sites * allelic_comb_remaining_sites
        )

    def set_P(self, P):
        self.P = P
        if self.P == (self.lp1):
            msg = '"P" = l+1, the optimal density is equal '
            msg += "to the empirical frequency."
            raise ValueError(msg)
        elif not 1 <= self.P <= self.seq_length:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
        self.Pfactorial = factorial(self.P)
        self.d = comb(self.seq_length, self.P) * (self.alpha - 1) ** self.P

    def _L_minus_p_a_dot(self, v, p=0):
        return self.L.dot(v) - p * self.alpha * v

    def _matvec(self, v):
        dotv = v.copy()
        for p in range(self.P):
            dotv = self._L_minus_p_a_dot(dotv, p)
        return dotv / self.Pfactorial

    def calc_lambdas(self):
        lambdas = []
        for L_lambda_k in self.L.lambdas:
            lambda_k = 1
            for p in range(self.P):
                lambda_k *= L_lambda_k - p * self.alpha
            lambdas.append(lambda_k / self.Pfactorial)
        self.lambdas = np.array(lambdas)

    def calc_log_det(self):
        return self.m_k[self.P :] * np.log(self.lambdas[self.P :])


class KrawtchoukOperator(SeqOperator, PolynomialOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, **params):
        SeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.L_lambdas = L.lambdas
        self.m_k = L.lambdas_multiplicity
        self.calc_eig_vandermonde_matrix_inverse(numeric=False)
        self.calc_polynomial_coefficients(**params)
        PolynomialOperator.__init__(self, L, self.coeffs)

    def calc_eig_vandermonde_matrix(self):
        self.V = np.vstack([self.L_lambdas**i for i in range(self.lp1)]).T
        self.V_LU = lu_factor(self.V)
        return self.V

    def calc_eig_vandermonde_matrix_inverse(self, numeric=False):
        """
        Calculates the coefficients of the polynomial in L that represent
        projection matrices into each of the kth eigenspaces.

        Returns
        -------
        B : array-like of shape (seq_length + 1, seq_length + 1)
            Matrix containing the b_i,k coefficients for power i on rows
            and order k on columns. One can obtain the coefficients for any
            combination of $\\lambda_k$ values by scaling the coefficients
            for each eigenspace by its eigenvalue and adding them up across
            different powers
        """
        if numeric:
            self.calc_eig_vandermonde_matrix()
            self.V_inv = np.linalg.inv(self.V)

        else:
            V_inv = np.zeros((self.lp1, self.lp1))
            idx = np.arange(self.lp1)

            for k in idx:
                k_idx = idx != k
                k_lambdas = self.L_lambdas[k_idx]
                norm_factor = 1 / np.prod(k_lambdas - self.L_lambdas[k])

                for power in idx:
                    p = np.sum(
                        [
                            np.prod(v)
                            for v in combinations(
                                k_lambdas, self.seq_length - power
                            )
                        ]
                    )
                    V_inv[power, k] = norm_factor * (-1) ** (power) * p

            self.V_inv = V_inv

        return self.V_inv

    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        sl, a = self.seq_length, self.alpha
        s = 0
        for q in range(k + 1):
            s += (
                (-1) ** q
                * (a - 1) ** (k - q)
                * comb(d, q)
                * comb(sl - d, k - q)
            )
        return 1 / a**sl * s

    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.lp1, self.lp1])
        for k in range(self.lp1):
            for d in range(self.lp1):
                self.W_kd[k, d] = self.calc_w(k, d)


class ProjectionOperator(ConstantDiagSeqOperator, KrawtchoukOperator):
    def __init__(self, n_alleles, seq_length, k=None, lambdas=None):
        KrawtchoukOperator.__init__(
            self,
            n_alleles=n_alleles,
            seq_length=seq_length,
            k=k,
            lambdas=lambdas,
        )
        self._init_dtype()

    def calc_polynomial_coefficients(self, k=None, lambdas=None):
        self.lambdas = self.get_lambdas(lambdas=lambdas, k=k)
        self.coeffs = self.lambdas_to_coeffs(self.lambdas)

    def get_lambdas(self, lambdas=None, k=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)

        if lambdas is None:
            lambdas = np.zeros(self.lp1)
            lambdas[k] = 1

        return lambdas

    def lambdas_to_coeffs(self, lambdas, use_lu=False):
        if use_lu:
            coeffs = lu_solve(self.V_LU, lambdas)
        else:
            coeffs = self.V_inv.dot(lambdas)
        return coeffs

    @property
    def d(self):
        if not hasattr(self, "_d"):
            self._d = self.calc_covariance_distance()[0]
        return self._d

    def calc_covariance_distance(self):
        self.calc_W_kd_matrix()
        return self.W_kd.T.dot(self.lambdas)

    def inv(self):
        return ProjectionOperator(
            self.alpha, self.seq_length, lambdas=1.0 / self.lambdas
        )

    def calc_log_det(self):
        if np.any(self.lambdas == 0.0):
            return -np.inf
        return np.sum(np.log(self.lambdas) * self.m_k)

    def power(self, b):
        return ProjectionOperator(
            self.alpha, self.seq_length, lambdas=self.lambdas**b
        )

    def matrix_sqrt(self):
        return ProjectionOperator(
            self.alpha, self.seq_length, lambdas=np.sqrt(self.lambdas)
        )

    def transpose(self):
        return self


class ExtendedDeltaPOperator(ProjectionOperator):
    def __init__(self, n_alleles, seq_length, P, lambdas0, **params):
        msg = "Ensure that lambdas0 has size P"
        check_error(lambdas0.shape[0] == P, msg=msg)
        DP = DeltaPOperator(n_alleles, seq_length, P)
        DP.calc_lambdas()
        lambdas = DP.lambdas
        lambdas[:P] = 1 / lambdas0
        super().__init__(
            n_alleles=n_alleles,
            seq_length=seq_length,
            lambdas=lambdas,
            **params,
        )


class CovarianceDistanceOperator(SeqOperator, PolynomialOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, distance):
        SeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        coeffs = self.calc_polynomial_coefficients(distance=distance)
        PolynomialOperator.__init__(self, L, coeffs)

    def calc_L_powers_distance_matrix_inverse(self):
        """Construct entries of powers of L.
        Column: powers of L.
        Row: Hamming distance"""

        sl, a, s = self.seq_length, self.alpha, self.lp1

        # Auxiliary matrices
        C = np.zeros([s, s])
        for i in range(s):
            for j in range(s):
                if i == j:
                    C[i, j] = i * (a - 2)
                if i == j + 1:
                    C[i, j] = i
                if i == j - 1:
                    C[i, j] = (sl - j + 1) * (a - 1)
        D = np.array(np.diag(sl * (a - 1) * np.ones(s), 0))
        B = D - C
        u = np.zeros(s)
        u[0], u[1] = sl * (a - 1), -1

        # Construct L_powers_d column by column
        L_powers_d = np.zeros([s, s])
        L_powers_d[0, 0] = 1
        for j in range(1, s):
            L_powers_d[:, j] = matrix_power(B, j - 1).dot(u)

        self.L_powers_d_inv = np.linalg.inv(L_powers_d)

    def calc_polynomial_coefficients(self, distance):
        self.calc_L_powers_distance_matrix_inverse()
        return self.L_powers_d_inv[:, distance]


class CovarianceVjOperator(ConstantDiagSeqOperator, KronOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, j):
        self.j = j

        ConstantDiagSeqOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length
        )
        KronOperator.__init__(self, self.get_matrices())

    def get_matrices(self):
        C0 = np.eye(self.alpha)
        C1 = np.ones((self.alpha, self.alpha)) - C0
        return [C1 if i in self.j else C0 for i in range(self.seq_length)]


class VjOperator(ConstantDiagSeqOperator, KronOperator):
    def __init__(self, n_alleles, seq_length, j):
        self.j = j
        self.k = len(j)

        ConstantDiagSeqOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length
        )
        self.repeats = self.alpha ** (self.seq_length - self.k)

        KronOperator.__init__(self, self.get_matrices(j))


class VjBasisOperator(VjOperator):
    def get_matrices(self, j):
        site_L = self.alpha * np.eye(self.alpha) - np.ones(
            (self.alpha, self.alpha)
        )
        b = [np.full((self.alpha, 1), 1 / np.sqrt(self.alpha)), orth(site_L)]
        return [b[int(i in j)] for i in range(self.seq_length)]


class VjProjectionOperator(VjOperator):
    symmetric = True

    def get_matrices(self, j):
        self.W0 = np.full((self.alpha, self.alpha), fill_value=1.0 / self.alpha)
        self.W1 = np.eye(self.alpha) - self.W0
        W = [self.W0, self.W1]
        return [W[int(i in j)] for i in range(self.seq_length)]

    def dot_square_norm(self, v):
        axis = tuple([p for p in range(self.seq_length) if p not in self.j])
        u = self.contract_v(v)
        if axis:
            u = u.mean(axis=axis)

        if self.k == 0:
            sqnorm = self.repeats * u**2
        else:
            A = KronOperator([self.W1] * self.k)
            sqnorm = self.repeats * np.sum((A @ u.flatten()) ** 2)
        return sqnorm


class RhoProjectionOperator(ConstantDiagSeqOperator, KronOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, rho):
        ConstantDiagSeqOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length
        )

        self.set_rho(rho)
        KronOperator.__init__(self, self.get_matrices())

    def get_matrices(self):
        W0 = np.full((self.alpha, self.alpha), fill_value=1.0 / self.alpha)
        W1 = np.eye(self.alpha) - W0
        return [W0 + r * W1 for r in self.rho]

    def check_rho(self, rho, ignore_bound=False):
        msg = "rho vector size must be equal to sequence length"
        check_error(rho.shape[0] == self.seq_length, msg=msg)

        checked = rho > 0
        msg = "rho larger than 0"
        if not ignore_bound:
            checked = checked & (rho < 1)
            msg = "rho must be between 0 and 1"
        check_error(np.all(checked), msg=msg)

    def set_rho(self, rho, ignore_bound=True):
        self.rho = (
            np.full(self.seq_length, rho)
            if isinstance(rho, float)
            else np.array(rho)
        )
        self.check_rho(self.rho, ignore_bound=ignore_bound)
        self.d = np.prod([1 + (self.alpha - 1) * r for r in self.rho]) / self.n

    def inv(self):
        return RhoProjectionOperator(
            self.alpha, self.seq_length, rho=1.0 / self.rho
        )

    def calc_log_det(self):
        log_rho = np.log(self.rho)
        k = np.sum(
            [comb(self.seq_length, i - 1) for i in range(self.seq_length)]
        )
        return k * np.sum(log_rho)

    def matrix_sqrt(self):
        return RhoProjectionOperator(
            self.alpha, self.seq_length, rho=np.sqrt(self.rho)
        )

    def matrix_power(self, b):
        return RhoProjectionOperator(
            self.alpha, self.seq_length, rho=self.rho**b
        )


class EigenBasisOperator(StackedOperator):
    def __init__(self, n_alleles, seq_length, k):
        positions = np.arange(seq_length)
        self.k = k
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        As = [
            VjBasisOperator(n_alleles, seq_length, j)
            for j in combinations(positions, k)
        ]
        super().__init__(linops=As, axis=1)


class DeltaKernelBasisOperator(StackedOperator):
    def __init__(self, n_alleles, seq_length, P):
        self.P = P
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        As = [EigenBasisOperator(n_alleles, seq_length, k) for k in range(P)]
        self.m_k = [A.shape[1] for A in As]
        self.rank = np.sum(self.m_k)
        super().__init__(linops=As, axis=1)


class DeltaKernelRegularizerOperator(ExtendedLinearOperator):
    def __init__(self, basis, lambdas_inv):
        s = basis.shape[0]
        msg = (
            "Basis size ({}) is different from number of provided lambdas ({})"
        )
        msg = msg.format(basis.P, lambdas_inv.shape[0])
        check_error(basis.P == lambdas_inv.shape[0], msg)
        self.shape = (s, s)
        self.B = basis
        self.D = self.set_regularizer(lambdas_inv)

    def set_regularizer(self, lambdas_inv):
        reg = []
        for k, lda in enumerate(lambdas_inv):
            reg += [lda] * int(self.B.m_k[k])
        return DiagonalOperator(np.array(reg))

    def _matvec(self, v):
        return self.B @ self.D @ self.B.transpose() @ v

    def calc_loss_grad_hess_b(self, b):
        Wb = self.D @ b
        loss = np.dot(b, Wb)
        grad = 2 * Wb
        hess = self.D
        return (loss, grad, hess)

    def calc_loss_grad_hess_phi(self, phi):
        D = self.B @ self.D @ self.B.transpose()
        Wb = D @ phi
        loss = np.dot(phi, Wb)
        grad = 2 * Wb
        hess = D
        return (loss, grad, hess)


class ProjectionOperator2(SeqOperator):
    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.calc_b()

    def set_lambdas(self, lambdas=None, k=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)

        if lambdas is None:
            lambdas = np.zeros(self.lp1)
            lambdas[k] = 1

        self.lambdas = lambdas
        self.coeffs = self.lambdas_to_coeffs(lambdas)

    def calc_b(self):
        b = np.zeros((self.lp1, self.lp1))
        for i in range(self.lp1):
            for k, j in enumerate(range(i, self.lp1)):
                b[i, j] = (-1) ** k * comb(self.seq_length - i, k)
        self.b = b

    def lambdas_to_coeffs(self, lambdas):
        return self.b.dot(lambdas)

    def _matvec(self, v):
        u = self.contract_v(v)
        r = np.zeros(self.shape_contracted)

        for k in range(self.lp1):
            c_k = self.coeffs[k]

            if c_k == 0:
                continue

            for j in combinations(self.positions, k):
                axis = tuple([p for p in self.positions if p not in j])
                r += c_k * np.expand_dims(u.mean(axis=axis), axis=axis)

        return self.expand_v(r)


class KernelOperator(SubMatrixOperator):
    symmetric = True

    def __init__(self, linop, x1=None, x2=None):
        super().__init__(linop, x1, x2)
        self._init_dtype()

    def _get_diag(self):
        msg = "rho need to be set to get diagonal"
        check_error(hasattr(self, "rho"), msg=msg)
        if hasattr(self, "n_obs"):
            return self.P.d + self.y_var
        else:
            return np.full(self.P.d, self.n)

    def _calc_trace(self):
        msg = "rho need to be set to calculate trace"
        check_error(hasattr(self.P, "rho"), msg=msg)
        if hasattr(self, "n_obs"):
            return self.n_obs * self.P.d + np.sum(self.y_var)
        else:
            return self.n * self.P.d


class Kernel(object):
    def compute(self, x1=None, x2=None, D=None):
        K = KernelOperator(self, x1, x2)
        if D is not None:
            K = K + D
        return K


class VarianceComponentKernel(ProjectionOperator, Kernel):
    def set_params(self, params):
        self.set_lambdas(lambdas=np.exp(params))

    def get_params(self):
        return np.log(self.get_lambdas())


class ConnectednessKernel(RhoProjectionOperator, Kernel):
    def set_params(self, params):
        self.set_rho(params)

    def get_params(self):
        return self.rho


class MultivariateGaussian(object):
    def __init__(self, mu, Sigma):
        msg = "The size of the mean should match the covariance matrix"
        check_error(mu.shape[0] == Sigma.shape[0], msg=msg)
        self.mu = mu
        self.Sigma = Sigma
        self.n = mu.shape[0]

    def get_cholesky(self):
        if not hasattr(self, "L"):
            self.L = self.Sigma.cholesky()
        return self.L

    def logp(self, x):
        # This implementation only works when Sigma is a KronOperator
        n = x.shape[0]
        logp = -0.5 * n * np.log(2 * np.pi)

        if hasattr(self.Sigma, "cholesky"):
            L = self.get_cholesky()
            L_inv = KronTriangularInverseOperator(L)
            z = L_inv @ (x - self.mu)
            logp -= 0.5 * np.sum(np.square(z)) + L.logdet()
        else:
            msg = "Only covariance matrices with cholesky method are available"
            raise NotImplementedError(msg)
        return logp

    def sample(self, n_samples):
        L = self.get_cholesky()
        x = L @ np.random.normal(size=(self.n, n_samples))
        return x


def get_diag(A, progress=False):
    s = min(A.shape)
    d = []

    idxs = range(s)
    if progress:
        idxs = tqdm(idxs)

    for i in idxs:
        v = np.zeros(s)
        v[i] = 1.0
        d.append(np.dot(v, A @ v))
    return np.array(d)


# ################### Skewed operators ##################################

# class SkewedLaplacianOperator(SeqOperator):
#     def __init__(self, n_alleles, seq_length, ps=None, max_size=None):
#         super().__init__(n_alleles=n_alleles, seq_length=seq_length)
#         self.set_ps(ps)
#         self.calc_Kns()
#         self.calc_lambdas(ps=ps)
#         self.calc_lambdas_multiplicity()

#         self.max_size = max_size
#         if max_size is None:
#             self.calc_L()
#             self._matvec = self.dot1
#         else:
#             self._matvec = self.dot2

#     def guess_n_products(self):
#         if self.max_size is None:
#             return(None)

#         size = 1
#         for k in range(self.seq_length):
#             size *= self.alpha
#             if size >= self.max_size:
#                 break
#         return(k)

#     def set_ps(self, ps):
#         self.variable_ps = ps is not None

#         if ps is None:
#             ps = [np.ones(self.alpha)] * self.seq_length
#         check_error(len(ps) == self.seq_length,
#                     msg='Number of ps should be equal to length')

#         # Normalize ps to have the eigenvalues in the right scale
#         self.ps = np.vstack([p / p.sum() * self.alpha for p in ps])
#         self.pi = calc_tensor_product([p.reshape((p.shape[0], 1))
#                                        for p in ps]).flatten()

#     def calc_lambdas(self, ps=None):
#         self.lambdas = np.arange(self.seq_length + 1) * self.alpha

#     def calc_lambdas_multiplicity(self):
#         self.lambdas_multiplicity = [comb(self.seq_length, k)
#                                      * (self.alpha-1) ** k
#                                      for k in range(self.lp1)]

#     def calc_Kn(self, p):
#         if self.variable_ps:
#             Kn = np.vstack([p] * p.shape[0])
#             np.fill_diagonal(Kn, np.zeros(Kn.shape[0]))
#         else:
#             Kn = np.full((self.alpha, self.alpha), True)
#             np.fill_diagonal(Kn, np.full(self.alpha, False))
#         return(Kn)

#     def calc_Kns(self):
#         Kns = [self.calc_Kn(p) for p in self.ps]
#         if self.max_size is not None:
#             size = self.guess_n_products()
#             i = self.seq_length - size
#             Kns = Kns[:i] + [calc_cartesian_product([csr_matrix(m)
#                              for m in Kns[i:]]).tocsr()]
#         self.Kns = Kns
#         self.Kns_shape = [x.shape for x in Kns]

#     def calc_L(self):
#         L = -calc_cartesian_product(self.Kns).astype(np.float64)
#         L.setdiag(-L.sum(1).A1)
#         self.L = L.tocsr()

#     def dot1(self, v):
#         return(self.L.dot(v))

#     def dot2(self, v):
#         return(self.d * v - calc_cartesian_product_dot(self.Kns, v))


# class LapDepOperator(SeqOperator):
#     def __init__(self, n_alleles=None, seq_length=None, L=None):
#         if L is None:
#             msg = 'either L or both seq_length and n_alleles must be given'
#             check_error(n_alleles is not None and seq_length is not None,
#                         msg=msg)
#             L = LaplacianOperator(n_alleles, seq_length)
#         else:
#             msg = 'either L or both seq_length and n_alleles must be given'
#             check_error(L is not None, msg=msg)
#             n_alleles, seq_length = L.alpha, L.l

#         super().__init__(n_alleles=n_alleles, seq_length=seq_length)
#         self.L = L


# class SkewedVjProjectionOperator(LapDepOperator):
#     def __init__(self, n_alleles=None, seq_length=None, L=None,
#                  max_size=None):
#         super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L)
#         self.calc_elementary_W()
#         self.max_size = max_size
#         self.size = self.guess_n_products()
#         self.cache = {}

#     def calc_elementary_W(self):
#         if self.L.ps is None:
#             self.b = [np.ones(self.alpha)] * self.seq_length
#         else:
#             self.b = self.L.ps
#         self.D_pi = [np.diag(b) for b in self.b]
#         self.pi = calc_tensor_product([b.reshape((b.shape[0], 1))
#                                        for b in self.b]).flatten()
#         self.W0 = [np.outer(b, b).dot(D) / np.sum(b * D.dot(b))
#                    for b, D in zip(self.b, self.D_pi)]
#         self.W1 = [np.eye(self.alpha) - w0 for w0 in self.W0]
#         self.Ws = [[w0, w1] for w0, w1 in zip(self.W0, self.W1)]

#     def get_Vj_matrix(self, js):
#         js_tuple = tuple(js)
#         if js_tuple not in self.cache:
#             m = self.seq_length - self.size
#             ms = [self.Ws[m+p][j] for p, j in enumerate(js)]
#             self.cache[js_tuple] = calc_tensor_product(ms)
#         return(self.cache[js_tuple])

#     def set_j(self, positions):
#         js = np.zeros(self.seq_length, dtype=int)
#         js[positions] = 1

#         if self.max_size is None:
#             self.matrices = [self.Ws[p][j] for p, j in enumerate(js)]
#         else:
#             m = self.seq_length - self.size
#             self.matrices = [self.Ws[p+m][j] for p, j in enumerate(js[:m])]
#             self.matrices.append(self.get_Vj_matrix(js[m:]))

#     def _matvec(self, v):
#         return(calc_tensor_product_dot(self.matrices, v))

#     def dot_square_norm(self, v):
#         '''
#         Note: we are calculating the squared D_pi-norm of the projection to
#         be able to do it directly through recursive product
#         '''
#         return(calc_tensor_product_quad(self.matrices, v1=self.pi * v, v2=v))


# class SkewedKernelOperator(SeqOperator):
#     def __init__(self, W):
#         self.W = W
#         self.sqrt_pi_inv = 1 / np.sqrt(W.L.pi)
#         self.pi_inv = 1 / W.L.pi
#         self.n = W.n
#         self.shape = (self.n, self.n)
#         self.known_var = False

#     def set_y_var(self, y_var=None, obs_idx=None):

#         if y_var is not None and obs_idx is not None:
#             msg = 'y_var and obs_idx should have same dimension: {} vs {}'
#             msg = msg.format(y_var.shape[0], obs_idx.shape[0])
#             check_error(y_var.shape[0] == obs_idx.shape[0], msg=msg)
#             self.known_var = True
#             self.homoscedastic = np.unique(y_var).shape[0] == 1
#             self.mean_var = y_var.mean()
#             self.y_var = y_var
#             self.n_obs = obs_idx.shape[0]
#             self.calc_gt_to_data_matrix(obs_idx)
#             self.shape = (self.n_obs, self.n_obs)

#         else:
#             msg = 'y_var and obs_idx must be provided with each other'
#             check_error(y_var is None and obs_idx is None, msg=msg)

#     @property
#     def lambdas_multiplicity(self):
#         return(self.W.L.lambdas_multiplicity)

#     def set_lambdas(self, lambdas):
#         self.W.set_lambdas(lambdas)

#     def get_lambdas(self):
#         return(self.W.lambdas)

#     def _dot(self, v):
#         if hasattr(self.W.L, 'variable_ps') and self.W.L.variable_ps:
#             return(self.W.dot(self.D_pi_inv.dot(v)))
#         else:
#             return(self.W.dot(v))

#     def calc_gt_to_data_matrix(self, obs_idx):
#         n_obs = obs_idx.shape[0]
#         self.gt2data = csr_matrix((np.ones(n_obs),
#                                   (obs_idx, np.arange(n_obs))),
#                                   shape=(self.n, n_obs))

#     def _matvec(self, v, all_rows=False, add_y_var_diag=True, full_v=False):
#         if full_v or not self.known_var:
#             u = self._dot(v)
#         else:
#             u = self._dot(self.gt2data.dot(v))
#             if add_y_var_diag:
#                 u += self.gt2data.dot(diag_pre_multiply(self.y_var, v))

#         if not all_rows and self.known_var:
#             u = self.gt2data.T.dot(u)
#         return(u)

#     @property
#     def Kop(self):
#         if not hasattr(self, '_Kop'):
#             self._Kop = LinearOperator((self.n_obs, self.n_obs),
#                                        matvec=self.dot)
#         return(self._Kop)

#     def inv_dot(self, v, show=False):
#         res = minres(self.Kop, v, tol=1e-6, show=show)
#         self.res = res[1]
#         return(res[0])

#     def inv_quad(self, v, show=False):
#         u = self.inv_dot(v, show=show)
#         return(np.sum(u * v))
