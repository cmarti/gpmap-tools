import sys
import time

from time import ctime

import numpy as np

import scipy.sparse as sp

from scipy.sparse.csr import csr_matrix
from scipy.sparse._matrix_io import save_npz
from scipy.sparse.extract import triu
from scipy.sparse.dia import dia_matrix


def check_error(condition, msg, error_type=ValueError):
    if not condition:
        raise error_type(msg)


def check_symmetric(sparse_matrix, tol=1e-6):
    if not (abs(sparse_matrix - sparse_matrix.T)> tol).nnz == 0:
        raise ValueError('Re-scaled rate matrix is not symmetric')


def get_sparse_diag_matrix(values):
    n_genotypes = values.shape[0]
    m = dia_matrix((values, np.array([0])), shape=(n_genotypes, n_genotypes))
    return(m)


class LogTrack(object):
    '''Logger class'''

    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = '[ {} ] {}\n'.format(ctime(), msg)
        else:
            msg += '\n'
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write('Finished succesfully. Time elapsed: {:.1f} s'.format(t))


def write_log(log, msg):
    if log is not None:
        log.write(msg)


def check_eigendecomposition(matrix, eigenvalues, right_eigenvectors, tol=1e-3):
    for l, u in zip(eigenvalues, right_eigenvectors.T):
        v1 = matrix.dot(u)
        v2 = l * u
        abs_err = np.mean(np.abs(v1 - v2))

        msg = 'Numeric error in eigendecomposition: abs error = {:.5f} > {:.5f}'
        check_error(abs_err <= tol, msg.format(abs_err, tol))


def calc_Kn_matrix(k=None, p=None):
    msg = 'One and only one of "k" or "p" must be provided'
    check_error((k is None) ^ (p is None), msg=msg)
    
    if p is not None:
        k = p.shape[0]
        m = np.vstack([p] * k)
    else:
        m = np.ones((k, k))

    np.fill_diagonal(m, np.zeros(k))
    return(csr_matrix(m))


def stack_prod_matrices(m_diag, m_offdiag, a):
    rows = []
    for j in range(a):
        row = [m_offdiag] * j + [m_diag] + [m_offdiag] * (a - j - 1)
        rows.append(sp.hstack(row)) 
    m = sp.vstack(rows)
    return(m)


def calc_cartesian_product(matrices):
    if len(matrices) == 1:
        return(matrices[0])
    
    m1, m2 = matrices[0], calc_cartesian_product(matrices[1:])
    i = sp.identity(m2.shape[0])
    
    rows = []
    for j in range(m1.shape[0]):
        row = [m2 if k == j else m1[j, k] * i for k in range(m1.shape[0])]
        rows.append(sp.hstack(row)) 
    m = sp.vstack(rows)
    return(m)


def reciprocal(x, y):
    """calculate reciprocal of variable, if variable=0, return 0"""
    if y == 0:
        return 0
    else:
        return x / y


def Frob(lambdas, M, a):
    """calculate the cost function given lambdas and a"""
    Frob1 = np.dot(lambdas, M).dot(lambdas)
    Frob2 = 2 * np.sum(lambdas * a)
    return Frob1 - Frob2


def grad_Frob(lambdas, M, a):
    """gradient of the function Frob(lambdas, M, a)"""
    grad_Frob1 = 2 * M.dot(lambdas)
    grad_Frob2 = 2 * a
    return grad_Frob1 - grad_Frob2


def calc_matrix_polynomial_dot(coefficients, matrix, v):
    power = v
    polynomial = coefficients[0] * v
    
    for c in coefficients[1:]:
        power = matrix.dot(power)
        polynomial += c * power
    
    return(polynomial)