#!/usr/bin/env python
from itertools import combinations, product

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import (
    lu_factor,
    lu_solve,
    orth,
    solve_triangular,
)
from scipy.sparse.linalg import aslinearoperator, cg, eigsh, minres
from scipy.special import comb, factorial
from tqdm import tqdm

try:
    from scipy.sparse.linalg.interface import _CustomLinearOperator
except ImportError:
    from scipy.sparse.linalg._interface import _CustomLinearOperator

from gpmap.matrix import is_lower_triangular, kron, tensordot
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
        return self @ np.eye(self.shape[1])

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


class LowRankPerturbationOperator(ExtendedLinearOperator):
    def __init__(self, A, rank):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.rank = min(rank, A.shape[0] - 1)
        lda, Q = eigsh(A, k=rank, v0=np.ones(A.shape[1]))

        self.Lambda = DiagonalOperator(lda)
        self.Q = aslinearoperator(Q)
        if hasattr(A, "get_diag"):
            diag = A.get_diag()
        else:
            diag = np.ones(A.shape[1])
        self.D = DiagonalOperator(diag)
        
    def _matvec(self, v):
        return (self.D + self.Q @ self.Lambda @ self.Q.T) @ v

    def inv(self):
        D_inv = self.D.inv()
        lda_inv = np.diag(1.0 / self.Lambda.diag)
        H = lda_inv + self.Q.transpose() @ D_inv @ self.Q.A
        H_inv = InverseOperator(H, method="direct")
        M = D_inv - D_inv @ self.Q @ H_inv @ self.Q.T @ D_inv
        return M


class InverseOperator(ExtendedLinearOperator):
    def __init__(
        self,
        linop,
        method="cg",
        atol=1e-5,
        rtol=1e-5,
        maxiter=10000,
        preconditioner_size=0,
        **kwargs,
    ):
        self.linop = linop
        self.shape = linop.shape
        self.dtype = linop.dtype
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter
        self.kwargs = kwargs

        if method not in ["minres", "cg", "direct", "exact"]:
            msg = "Method {} not allowed".format(method)
            raise ValueError(msg)
        if method == "exact":
            if not hasattr(self.linop, "inv"):
                msg = "Linop must have 'inv' method for exact inversion"
                raise ValueError(msg)
            self.linop_inv = self.linop.inv()
        self.method = method

        if preconditioner_size > 0 and method in ["minres", "cg"]:
            preconditioner_size = min(linop.shape[0], preconditioner_size)
            A = LowRankPerturbationOperator(self.linop, preconditioner_size)
            self.preconditioner = A.inv()
        else:
            self.preconditioner = None

    def _matvec(self, v):
        if self.method == "exact":
            res = self.linop_inv @ v

        elif self.method == "minres":
            res = minres(self.linop, v, M=self.preconditioner, **self.kwargs)
            if res[1] != 0:
                msg = "Minres did not converge"
                raise ValueError(msg)
            res = res[0]

        elif self.method == "cg":

            class cb(object):
                def __init__(self):
                    self.niter = 0

                def __call__(self, xk):
                    self.niter += 1

            counter = cb()
            try:
                res = cg(
                    self.linop,
                    v,
                    M=self.preconditioner,
                    atol=self.atol,
                    tol=self.rtol,
                    maxiter=self.maxiter,
                    callback=counter,
                    **self.kwargs,
                )
            except TypeError: # Captures error from newer scipy versions
                res = cg(
                    self.linop,
                    v,
                    M=self.preconditioner,
                    atol=self.atol,
                    rtol=self.rtol,
                    maxiter=self.maxiter,
                    callback=counter,
                    **self.kwargs,
                )
                
            self.cg_n_iter = counter.niter
            if res[1] != 0:
                msg = "Conjugate gradient did not converge"
                raise ValueError(msg)
            res = res[0]

        elif self.method == "direct":
            res = np.linalg.solve(self.linop, v)

        return res

    def quad(self, v):
        u = self._matvec(v)
        return np.sum(v * u)


class SymmetricOperator(ExtendedLinearOperator):
    symmetric = True

    def _rmatvec(self, v):
        return self._matvec(v)

    def _rmatmat(self, B):
        return self._matmat(B)

    def transpose(self):
        return self


class DiagonalOperator(SymmetricOperator):
    def __init__(self, diag):
        self.diag = diag
        self.shape = (diag.shape[0], diag.shape[0])
        self._init_dtype()
        self.A = np.expand_dims(self.diag, 1)

    def _matvec(self, v):
        if len(v.shape) == 1:
            return self.diag * v
        else:
            return self._matmat(v)

    def _matmat(self, B):
        return self.A * B

    def logdet(self):
        msg = "All diagonal entries must be larger than 0 to compute logdet"
        check_error(np.all(self.diag > 0), msg=msg)
        return np.sum(np.log(self.diag))

    def det(self):
        return np.product(self.diag)

    def inv(self):
        return DiagonalOperator(1.0 / self.diag)


class IdentityOperator(DiagonalOperator):
    def __init__(self, n):
        self.shape = (n, n)
        self._init_dtype()

    def _matvec(self, v):
        return v

    def _matmat(self, B):
        return B

    def inv(self):
        return self


class PconOperator(SymmetricOperator):
    def __init__(self, n):
        self.shape = (n, n)
        self._init_dtype()

    def _matvec(self, v):
        return np.full_like(v, np.mean(v))

    def _matmat(self, B):
        return np.tile(np.mean(B, axis=0), (B.shape[0], 1))


class PonesOperator(SymmetricOperator):
    def __init__(self, n):
        self.shape = (n, n)
        self._init_dtype()

    def _matvec(self, v):
        return np.full_like(v, np.sum(v))

    def _matmat(self, B):
        return np.tile(np.sum(B, axis=0), (B.shape[0], 1))


class PaddOperator(SymmetricOperator):
    def __init__(self, n):
        self.shape = (n, n)
        self._init_dtype()

    def _matvec(self, v):
        return v - np.full_like(v, np.mean(v))

    def _matmat(self, B):
        return B - np.tile(np.mean(B, axis=0), (B.shape[0], 1))


class PconPaddWeightedSumOperator(SymmetricOperator):
    def __init__(self, n, lda0, lda1):
        self.shape = (n, n)
        self.lda0 = lda0
        self.lda1 = lda1
        self._init_dtype()

    def _matvec(self, v):
        return self.lda1 * v  + v.mean() * (self.lda0 - self.lda1)

    def _matmat(self, B):
        return self.lda1 * B  + B.mean(axis=0, keepdims=True) * (self.lda0 - self.lda1)


class SiteLaplacianOperator(SymmetricOperator):
    def __init__(self, n):
        self.n = n
        self.shape = (n, n)
        self._init_dtype()

    def _matvec(self, v):
        return self.n * v - np.full_like(v, np.sum(v))

    def _matmat(self, B):
        return self.n * B - np.tile(np.sum(B, axis=0), (B.shape[0], 1))


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

    def _rmatvec(self, v):
        return v[self.idx]

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

    def _rmatvec(self, v):
        u = np.zeros(self.n)
        u[self.idx] = v
        return u

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

    def _get_dense_matrix(self, m):
        if isinstance(m, np.ndarray):
            return m
        else:
            return m @ np.eye(m.shape[1])

    def todense(self):
        return kron([self._get_dense_matrix(m) for m in self.matrices])

    def transpose(self):
        return KronOperator([m.T for m in self.matrices])

    def cholesky(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Cannot compute cholesky of non-square matrix")
        return KronSquareTriangularOperator(
            [np.linalg.cholesky(m @ np.eye(m.shape[1])) for m in self.matrices]
        )

    def inv(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Cannot invert a non-square matrix")
        return KronOperator([np.linalg.inv(m) for m in self.matrices])


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


class DeltaOperator(ConstantDiagSeqOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.set_P(P)
        self.calc_kernel_dimension()
        self.calc_n_p_faces()
        self.calc_n_p_faces_genotype()

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

    def calc_kernel_dimension(self):
        self.kernel_dimension = np.sum(
            [
                comb(self.seq_length, k) * (self.alpha - 1) ** k
                for k in range(self.P)
            ]
        )
        self.rank = self.n - self.kernel_dimension

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

    def calc_kernel_basis(self):
        return DeltaKernelBasisOperator(self.alpha, self.seq_length, self.P)


class DeltaPOperator(DeltaOperator):
    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, P=P)
        self.L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.dtype = self.L.dtype
        self.m_k = self.L.lambdas_multiplicity
        self.calc_lambdas()

    def calc_lambdas(self):
        lambdas = []
        for L_lambda_k in self.L.lambdas:
            lambda_k = 1
            for p in range(self.P):
                lambda_k *= L_lambda_k - p * self.alpha
            lambdas.append(lambda_k / self.Pfactorial)
        self.lambdas = np.array(lambdas)

    def _L_minus_p_a_dot(self, v, p=0):
        return self.L.dot(v) - p * self.alpha * v

    def _matvec(self, v):
        dotv = v.copy()
        for p in range(self.P):
            dotv = self._L_minus_p_a_dot(dotv, p)
        return dotv / self.Pfactorial

    def calc_log_det(self):
        return self.m_k[self.P :] * np.log(self.lambdas[self.P :])


class DeltaUOperator(SeqOperator, KronOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, U):
        self.U = U
        SeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        KronOperator.__init__(self, self.get_matrices())
        self.L = SiteLaplacianOperator(self.alpha)

    def get_matrices(self):
        C0 = IdentityOperator(self.alpha)
        C1 = SiteLaplacianOperator(self.alpha)
        return [C1 if i in self.U else C0 for i in range(self.seq_length)]


class DeltaUWeighedSumOperator(DeltaOperator, SymmetricOperator):
    def __init__(self, n_alleles, seq_length, P, a):
        self.ncombs = comb(seq_length, P)
        check_error(a.shape[0] == self.ncombs, msg="Incorrect size of a")
        self.a = a
        DeltaOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length, P=P
        )
        self.Deltap = [
            DeltaUOperator(n_alleles, seq_length, [i])
            for i in range(seq_length)
        ]

    def take_product(self, U, v, temp_products):
        if not U:
            return v
        elif U in temp_products:
            return temp_products[U]
        else:
            prev_u = self.take_product(U[:-1], v, temp_products)
            temp_products[U[:-1]] = prev_u
            u = self.Deltap[U[-1]] @ prev_u
            return u

    def _matvec(self, v):
        u = np.zeros_like(v)
        temp_products = {}
        for a_U, U in zip(self.a, combinations(self.positions, self.P)):
            u += a_U * self.take_product(U, v, temp_products)
        return u


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


class CovarianceSitesOperator(SeqOperator, KronOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, sites):
        self.sites = sites
        SeqOperator.__init__(self, n_alleles, seq_length)
        KronOperator.__init__(self, self.get_matrices())

    def get_matrices(self):
        C0 = IdentityOperator(self.alpha)
        C1 = np.ones((self.alpha, self.alpha)) - np.eye(self.alpha)
        return [C1 if i in self.sites else C0 for i in range(self.seq_length)]


class VUOperator(ConstantDiagSeqOperator, KronOperator):
    def __init__(self, n_alleles, seq_length, j):
        self.j = j
        self.k = len(j)

        ConstantDiagSeqOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length
        )
        self.repeats = self.alpha ** (self.seq_length - self.k)

        KronOperator.__init__(self, self.get_matrices(j))


class VUBasisOperator(VUOperator):
    def get_matrices(self, j):
        site_L = self.alpha * np.eye(self.alpha) - np.ones(
            (self.alpha, self.alpha)
        )
        b = [np.full((self.alpha, 1), 1 / np.sqrt(self.alpha)), orth(site_L)]
        return [b[int(i in j)] for i in range(self.seq_length)]


class VUProjectionOperator(VUOperator):
    symmetric = True

    def get_matrices(self, j):
        self.W0 = PconOperator(self.alpha)
        self.W1 = PaddOperator(self.alpha)
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

class VUProjectionWeightedSumOperator(SeqOperator, SymmetricOperator):
    def __init__(self, n_alleles, seq_length, lambdas=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.n_V_U = 2**seq_length
        self.set_lambdas(lambdas)
        self.W0 = PconOperator(self.alpha)
        self.W1 = PaddOperator(self.alpha)
        self.v_shape = [n_alleles] * seq_length

    def set_lambdas(self, lambdas):
        if lambdas is not None:
            check_error(
                lambdas.shape[0] == self.n_V_U, msg="Incorrect size of lambdas"
            )
            check_error(
                np.all(lambdas >= 0),
                msg=f"lambdas must be non-negative: {lambdas}",
            )
            self.lambdas = lambdas

    def calc_V_U_product(self, v, sites, sites_included):
        if not sites:
            return v
        elif np.all([self.cached_U[s] == sites_included[s] for s in sites]):
            return self.cached_matvecs[sites[-1]]
        else:
            site = sites[-1]
            site_included = sites_included[-1]
            P = self.W1 if site_included else self.W0
            v_prev = self.calc_V_U_product(v, sites[:-1], sites_included[:-1])
            u = tensordot(P, v_prev, site)
            self.n_products += 1

            self.cached_matvecs[site] = u
            self.cached_U[site] = site_included
            for i in range(site + 1, self.seq_length):
                self.cached_U[i] = None
                self.cached_matvecs[i] = None

            return u

    def _matvec(self, v):
        if self.lambdas is None:
            msg = "lambdas must be defined for computing matrix-vector products"
            raise ValueError(msg)

        self.cached_matvecs = [None] * self.seq_length
        self.cached_U = [None] * self.seq_length
        self.n_products = 0
        check_error(
            v.shape[0] == self.shape[1],
            msg="Incorrect dimensions of matrices and `v`",
        )
        v = v.reshape(self.v_shape)
        sites = list(range(self.seq_length))
        u = np.zeros_like(v)
        Us = product([False, True], repeat=self.seq_length)
        for U, lambda_U in zip(Us, self.lambdas):
            v_U = self.calc_V_U_product(v, sites, U)
            u += lambda_U * v_U
        return u.transpose().flatten()

    def inv(self):
        return VUProjectionWeightedSumOperator(
            self.alpha, self.seq_length, lambdas=1.0 / self.lambdas
        )

    def matrix_sqrt(self):
        lambdas = np.sqrt(self.lambdas)
        return VUProjectionWeightedSumOperator(
            self.alpha, self.seq_length, lambdas=lambdas
        )


class ConnectednessProjectionOperator(ConstantDiagSeqOperator, KronOperator):
    symmetric = True

    def __init__(self, n_alleles, seq_length, mu):
        ConstantDiagSeqOperator.__init__(
            self, n_alleles=n_alleles, seq_length=seq_length
        )

        self.set_mu(mu)
        KronOperator.__init__(self, self.get_matrices())

    def get_matrices(self):
        return [PconPaddWeightedSumOperator(self.alpha, self.mu[0], mu_i)
                for mu_i in self.mu[1:]]

    def get_mu(self):
        return self.mu
    
    def get_decay_factors(self):
        decay_factors = []
        for m in self.matrices:
            m = m @ np.eye(self.alpha)
            decay_factors.append(1 - m[0, 1] / m[0, 0])
        return np.array(decay_factors)

    def check_mu(self, mu, ignore_bound=False):
        msg = "mu vector size must be equal to sequence length + 1"
        check_error(mu.shape[0] == self.seq_length + 1, msg=msg)

        checked = mu >= 0
        msg = "mu must be non-negative"
        if not ignore_bound:
            checked = checked & (mu < 1)
            msg = "mu must be between 0 and 1"
        check_error(np.all(checked), msg=msg)

    def set_mu(self, mu, ignore_bound=True):
        self.mu = (
            np.full(self.seq_length, mu)
            if isinstance(mu, float)
            else np.array(mu)
        )
        self.check_mu(self.mu, ignore_bound=ignore_bound)
        self.d = np.prod([1 + (self.alpha - 1) * r for r in self.mu]) / self.n

    def inv(self):
        mu = 1.0 / self.mu
        return ConnectednessProjectionOperator(
            self.alpha, self.seq_length, mu=mu
        )

    def matrix_sqrt(self):
        mu = np.sqrt(self.mu)
        return ConnectednessProjectionOperator(
            self.alpha, self.seq_length, mu=mu
        )


class EigenBasisOperator(StackedOperator):
    def __init__(self, n_alleles, seq_length, k):
        positions = np.arange(seq_length)
        self.k = k
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        As = [
            VUBasisOperator(n_alleles, seq_length, j)
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


class KernelOperator(SubMatrixOperator):
    symmetric = True

    def __init__(self, linop, x1=None, x2=None):
        super().__init__(linop, x1, x2)
        self._init_dtype()

    def _get_diag(self):
        msg = "mu need to be set to get diagonal"
        check_error(hasattr(self, "mu"), msg=msg)
        if hasattr(self, "n_obs"):
            return self.P.d + self.y_var
        else:
            return np.full(self.P.d, self.n)

    def _calc_trace(self):
        msg = "mu need to be set to calculate trace"
        check_error(hasattr(self.P, "mu"), msg=msg)
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


class VUKernel(VUProjectionWeightedSumOperator, Kernel):
    def set_params(self, params):
        self.set_lambdas(lambdas=np.exp(params))

    def get_params(self):
        return np.log(self.get_lambdas())


class ConnectednessKernel(ConnectednessProjectionOperator, Kernel):
    symmetric = True

    def set_params(self, params):
        self.set_mu(np.exp(params))

    def get_params(self):
        return np.log(self.get_mu())


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
