import numpy as np
import scipy.sparse as sp

from itertools import product
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix

from gpmap.src.utils import check_error, get_length
from numpy.linalg.linalg import norm


def inner_product(x1, x2, metric=None):
    if metric is None:
        return(x1.dot(x2.T))
    else:
        return(x1.dot(metric.dot(x2.T)))


def quad(linop, v1, v2=None):
    if v2 is None:
        v2 = v1
    return(np.sum(linop.dot(v1) * v2))


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


def kron_dot(matrices, v):
    shape = [m_i.shape[1] for m_i in matrices]
    check_error(np.prod(shape) == v.shape[0],
                msg='Incorrect dimensions of matrices and `v`')
    
    m = v.reshape(shape)
    for i, m_i in enumerate(matrices):
        axes = np.arange(len(shape))
        axes[[i, 0]] = np.array([0, i])

        n = np.prod(m.shape)
        tmp_shape = (m_i.shape[1], int(n / m_i.shape[1]))
        mx = np.transpose(m, axes=axes).reshape(tmp_shape)
        p = m_i @ mx

        shape[i] = m_i.shape[0]
        tmp_shape = shape.copy()
        tmp_shape[0], tmp_shape[i] = tmp_shape[i], tmp_shape[0]

        m = np.transpose(p.reshape(tmp_shape), axes=axes)
    return(m.reshape(np.prod(shape)))


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


def filter_csr_matrix(matrix, idxs):
    return(matrix[idxs, :][:, idxs])


def rate_to_jump_matrix(rate_matrix):
    try:
        rate_matrix.setdiag(0)
        rate_matrix = rate_matrix.tolil()
    except AttributeError:
        np.fill_diagonal(rate_matrix, 0)
        
    leaving_rates = np.array(rate_matrix.sum(1)).flatten()
    zero_idxs = leaving_rates == 0
    leaving_rates[zero_idxs] = 1
    diag = 1 / leaving_rates
    jump_matrix = get_sparse_diag_matrix(diag) @ rate_matrix
    
    if hasattr(jump_matrix, 'tolil'):
        jump_matrix = jump_matrix.tolil()
        
    jump_matrix[zero_idxs, zero_idxs] = 1
    return(jump_matrix)


def lanczos_conjugate_gradient(A, b, tol=1e-6, max_iter=100):
    x = np.zeros(b.shape)
    p = b
    r = b
    w = 0
    gamma = 1
    prev_u = 0
    
#     u = np.zeros(b.shape)
#     r = A.dot(u) - b
#     r_norm = norm(r)
#     d = r
#     
#     for _ in range(max_iter):
#         print(r)
#         v = A.dot(r)
#         alpha = r_norm / np.dot(d, v)
#         u = u + alpha * r
#         r = r - alpha * v
#         r_norm_prev = r_norm
#         r_norm = norm(r)
#         
#         if r_norm < tol:
#             break
#         
#         beta = r_norm / r_norm_prev
#         d = r - beta * d
#     
#     return(u)
    