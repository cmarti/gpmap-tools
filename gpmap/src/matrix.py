import numpy as np
import scipy.sparse as sp

from itertools import product
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix

from gpmap.src.utils import check_error, get_length


def inner_product(x1, x2, metric=None):
    if metric is None:
        return(x1.dot(x2.T))
    else:
        return(x1.dot(metric.dot(x2.T)))


def quad(matrix, v1, v2=None):
    if v2 is None:
        v2 = v1
    return(np.sum(matrix.dot(v1) * v2))


def get_sparse_diag_matrix(values):
    n_genotypes = values.shape[0]
    m = dia_matrix((values, np.array([0])), shape=(n_genotypes, n_genotypes))
    return(m)


def _diag_multiply(d, m, axis):
    if len(m.shape) == 1:
        return(d * m)
    else:
        return(np.expand_dims(d, axis=axis) * m)


def diag_pre_multiply(d, m):
    return(_diag_multiply(d, m, axis=1))
    

def diag_post_multiply(d, m):
    return(_diag_multiply(d, m, axis=0))


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
    if len(matrices) == 0:
        return(None)
    
    if len(matrices) == 1:
        return(matrices[0])
    
    m1, m2 = matrices[0], calc_cartesian_product(matrices[1:])
    i = sp.identity(m2.shape[0], dtype=m1.dtype)
    
    rows = []
    for j in range(m1.shape[0]): 
        row = [m2 if k == j else m1[j, k] * i for k in range(m1.shape[1])]
        rows.append(sp.hstack(row)) 
    m = sp.vstack(rows)
    return(m)


def calc_cartesian_product_dot(matrices, v):
    if len(matrices) == 1:
        return(matrices[0].dot(v))
    a = matrices[0].shape[0]
    s = np.prod([m.shape[0] for m in matrices]) // a
    vs = [v[s*j:s*(j+1)] for j in range(a)]
    
    u = np.zeros(v.shape[0])
    for col in range(a):
        u_i = np.hstack([calc_cartesian_product_dot(matrices[1:], vs[col])
                        if k == col else vs[col]
                        for k in range(a)]) 
        u += u_i
    return(u)


def calc_tensor_product(matrices):
    if len(matrices) == 1:
        return(matrices[0])
    
    m1, m2 = matrices[0], calc_tensor_product(matrices[1:])
    rows = []
    for j in range(m1.shape[0]):
        row = [m2 * m1[j, k] for k in range(m1.shape[1])]
        if len(row) > 1:
            rows.append(np.hstack(row))
        else: 
            rows.append(row[0])
    m = np.vstack(rows)
    return(m)


def calc_tensor_product_dot(matrices, v):
    d1 = np.prod([m.shape[0] for m in matrices])
    check_error(d1 == v.shape[0], 'Dimension missmatch between matrix and vector')
    
    if len(matrices) == 1:
        return(matrices[0].dot(v))
    
    a = matrices[0].shape[0]
    s = v.shape[0] // a
    m = matrices[0]
    vs = [v[s*j:s*(j+1)] for j in range(a)]
    us = [calc_tensor_product_dot(matrices[1:], v_i) for v_i in vs]
    
    u = np.zeros(v.shape[0])
    for col in range(a):
        u_i = np.hstack([m[k, col] * us[col] for k in range(a)]) 
        u += u_i
    return(u)


def calc_tensor_product_dot2(m1, m2, v):
    m = v.reshape((m1.shape[1], m2.shape[1])).T
    return(m1.dot(m2.dot(m).T).reshape(v.shape))


def kron_dot(matrices, v):
    shape = tuple([m_i.shape[1] for m_i in matrices])
    if np.prod(shape) != v.shape[0]:
        msg = 'Incorrect dimensions of matrices and `v`'
        raise ValueError(msg)
    
    m = v.reshape(shape)
    
    for i, m_i in enumerate(matrices):
        axes = np.arange(len(shape))
        axes[[i, 0]] = np.array([0, i])
        new_shape = (m_i.shape[0], int(v.shape[0] / m_i.shape[0]))
        m = np.transpose(m_i.dot(np.transpose(m, axes=axes).reshape(new_shape)).reshape(shape), axes=axes)
    
    return(m.reshape(v.shape))


def calc_tensor_product_quad(matrices, v1, v2=None):
    if v2 is None:
        v2 = v1.copy()
    
    if len(matrices) == 1:
        return(quad(matrices[0], v1, v2))
    
    a = matrices[0].shape[0]
    s = v1.shape[0] // a
    m = matrices[0]
    v1s = [v1[s*j:s*(j+1)] for j in range(a)]
    v2s = [v2[s*j:s*(j+1)] for j in range(a)]

    ss = np.sum([m[row, col] * calc_tensor_product_quad(matrices[1:], v1s[row], v2s[col])
                 for row, col in product(np.arange(a), repeat=2)])
    return(ss)


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
        
        if c == 0:
            continue
        
        power = matrix.dot(power)
        polynomial += c * power
    
    return(polynomial)


def calc_matrix_polynomial_quad(coefficients, matrix, v):
    Av = calc_matrix_polynomial_dot(coefficients, matrix, v)
    return(np.sum(v * Av))


def calc_cartesian_prod_freqs(site_freqs):
    if get_length(site_freqs) == 1:
            return(site_freqs[0])
    site1 = site_freqs[0]
    site2 = calc_cartesian_prod_freqs(site_freqs[1:])
    freqs = np.hstack([f * site2 for f in site1])
    return(freqs)
