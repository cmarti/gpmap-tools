import numpy as np

from scipy.sparse import csr_matrix, dia_matrix, hstack, vstack, identity
from scipy.sparse.linalg import minres, cg

from gpmap.utils import check_error, get_length


def inner_product(x1, x2, metric=None):
    if metric is None:
        return x1.dot(x2.T)
    else:
        return x1.dot(metric.dot(x2.T))


def dot_log(logA, signA, logB, signB):
    if len(logB.shape) > 1:
        logA = np.expand_dims(logA, 2)
        signA = np.expand_dims(signA, 2)

    logB = np.expand_dims(logB, 0)
    signB = np.expand_dims(signB, 0)

    msg = "Ensure the first dimension of A and B match"
    check_error(logA.shape[1] == logB.shape[1], msg=msg)

    res = logA + logB
    x_star = np.max(res)
    res = np.sum(signA * signB * np.exp(res - x_star), axis=1)
    sign = np.sign(res)
    res = x_star + np.log(np.abs(res))
    return (res, sign)


def quad(linop, v1, v2=None):
    if v2 is None:
        v2 = v1
    return np.sum(linop.dot(v1) * v2)


def tensordot(linop, v, axis):
    u = np.moveaxis(v, axis, 0)  # Shape becomes (contract_dim,...rest_of_dims)
    u_reshaped = u.reshape(
        u.shape[0], -1
    )  # Shape: (contract_dim, rest_product)
    x = (
        linop @ u_reshaped
    )  # linop shape: (m, contract_dim), result shape: (m, rest_product)
    final_shape = (linop.shape[0],) + u.shape[1:]  # (m, ...rest_of_dims)
    x = x.reshape(final_shape)
    return x


def is_lower_triangular(m):
    if np.all(np.triu(m, k=1) == 0):
        return True
    elif np.all(np.tril(m, k=1) == 0):
        return False
    else:
        raise ValueError("Matrix is not triangular")


def rayleigh_quotient(linop, v, metric=None):
    return quad(linop, v) / inner_product(v, v, metric=metric)


def inv_dot(linop, v, method="minres", **kwargs):
    if method == "minres":
        res = minres(linop, v, **kwargs)
    elif method == "cg":
        # iters = 0
        # def nonlocal_iterate(arr):
        #     nonlocal iters
        #     iters+=1
        # res = cg(linop, v, callback=nonlocal_iterate,  **kwargs)
        # print('n iters', iters)
        res = cg(linop, v, **kwargs)
    elif method == "direct":
        res = np.linalg.solve(linop, v)
    else:
        msg = "Method {} not allowed".format(method)
        raise ValueError(msg)
    return res[0]


def inv_quad(linop, v, method="minres", **kwargs):
    u = inv_dot(linop, v, method, **kwargs)
    return np.sum(v * u)


def kron(matrices):
    n = len(matrices)
    msg = "Provide at least two matrices to take Kron product"
    check_error(n >= 2, msg=msg)
    if n == 2:
        return np.kron(matrices[0], matrices[1])
    else:
        return np.kron(matrices[0], kron(matrices[1:]))


def get_sparse_diag_matrix(values):
    n_genotypes = values.shape[0]
    m = dia_matrix((values, np.array([0])), shape=(n_genotypes, n_genotypes))
    return m


def _diag_multiply(d, m, axis):
    if len(m.shape) == 1:
        return d * m
    else:
        return np.expand_dims(d, axis=axis) * m


def diag_pre_multiply(d, m):
    return _diag_multiply(d, m, axis=1)


def diag_post_multiply(d, m):
    return _diag_multiply(d, m, axis=0)


def calc_Kn_matrix(k=None, p=None):
    msg = 'One and only one of "k" or "p" must be provided'
    check_error((k is None) ^ (p is None), msg=msg)

    if p is not None:
        k = p.shape[0]
        m = np.vstack([p] * k)
    else:
        m = np.ones((k, k))

    np.fill_diagonal(m, np.zeros(k))
    return csr_matrix(m)


def stack_prod_matrices(m_diag, m_offdiag, a):
    rows = []
    for j in range(a):
        row = [m_offdiag] * j + [m_diag] + [m_offdiag] * (a - j - 1)
        rows.append(hstack(row))
    m = vstack(rows)
    return m


def calc_cartesian_product(matrices):
    if len(matrices) == 0:
        return None

    if len(matrices) == 1:
        return matrices[0]

    m1, m2 = matrices[0], calc_cartesian_product(matrices[1:])
    i = identity(m2.shape[0], dtype=m1.dtype)

    rows = []
    for j in range(m1.shape[0]):
        row = [m2 if k == j else m1[j, k] * i for k in range(m1.shape[1])]
        rows.append(hstack(row))
    m = vstack(rows)
    return m


def calc_cartesian_product_dot(matrices, v):
    if len(matrices) == 1:
        return matrices[0].dot(v)
    a = matrices[0].shape[0]
    s = np.prod([m.shape[0] for m in matrices]) // a
    vs = [v[s * j : s * (j + 1)] for j in range(a)]

    u = np.zeros(v.shape[0])
    for col in range(a):
        u_i = np.hstack(
            [
                calc_cartesian_product_dot(matrices[1:], vs[col])
                if k == col
                else vs[col]
                for k in range(a)
            ]
        )
        u += u_i
    return u


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


def calc_cartesian_prod_freqs(site_freqs):
    if get_length(site_freqs) == 1:
        return site_freqs[0]
    site1 = site_freqs[0]
    site2 = calc_cartesian_prod_freqs(site_freqs[1:])
    freqs = np.hstack([f * site2 for f in site1])
    return freqs


def filter_csr_matrix(matrix, idxs):
    return matrix[idxs, :][:, idxs]


def rate_to_jump_matrix(rate_matrix):
    """
    Converts a given rate matrix into a jump matrix by normalizing transition
    rates and ensuring valid diagonal entries.

    The function first removes self-transition rates (diagonal elements set
    to zero), then normalizes each row so that the sum of transition
    probabilities equals 1. Any rows with zero transition rates are set
    to self-transition probability 1.

    Parameters
    ----------
    rate_matrix : scipy.sparse.csr_matrix or scipy.sparse.lil_matrix
        A sparse rate matrix where entry (i, j) represents the transition rate
        from state i to state j.

    Returns
    -------
    jump_matrix : scipy.sparse.lil_matrix
        A normalized jump matrix where each row sums to 1, ensuring proper
        probability transitions.

    Notes
    -----
    - The input `rate_matrix` should be a sparse matrix.
    - If `rate_matrix` is not already in LIL format, it is converted before
      modifications.
    - The diagonal is modified to ensure correct transition probability
      behavior.
    - The output is returned in `lil_matrix` format for efficient further
      modifications.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> rate_matrix = csr_matrix([[0, 2, 0], [1, 0, 1], [0, 2, 0]])
    >>> jump_matrix = rate_to_jump_matrix(rate_matrix)
    >>> jump_matrix.toarray()
    array([[0.  , 1.  , 0.  ],
           [0.5 , 0.  , 0.5 ],
           [0.  , 1.  , 0.  ]])
    """

    try:
        rate_matrix = rate_matrix.tolil()
        rate_matrix.setdiag(0)
    except AttributeError:
        np.fill_diagonal(rate_matrix, 0)

    leaving_rates = np.array(rate_matrix.sum(1)).flatten()
    zero_idxs = leaving_rates == 0
    leaving_rates[zero_idxs] = 1
    diag = 1 / leaving_rates
    jump_matrix = get_sparse_diag_matrix(diag) @ rate_matrix

    if hasattr(jump_matrix, "tolil"):
        jump_matrix = jump_matrix.tolil()

    jump_matrix[zero_idxs, zero_idxs] = 1
    if hasattr(jump_matrix, "tocsr"):
        jump_matrix = jump_matrix.tocsr()
    return jump_matrix


def pivoted_cholesky_with_diag(A, D, rank, tol=1e-10):
    """
    Compute the low-rank pivoted Cholesky decomposition of a linear operator A
    with an explicitly given diagonal.

    Parameters:
    A : LinearOperator
        Symmetric positive definite linear operator representing matrix A.
    D : ndarray
        The explicit diagonal of the matrix A.
    rank : int
        The desired rank for the low-rank approximation.
    tol : float
        The tolerance for stopping the algorithm.

    Returns:
    L : ndarray
        The low-rank Cholesky factor (rank x n matrix).
    piv : list
        The list of pivot indices.
    """
    n = len(D)
    piv = []
    L = np.zeros((rank, n))
    perm = np.arange(n)

    # Start with the precomputed diagonal
    diagonal = D.copy()

    for k in range(rank):
        # Pivot step: find the largest diagonal element
        i = np.argmax(diagonal[k:]) + k
        if diagonal[i] < tol:
            # Stop if diagonal element is smaller than tolerance
            break

        # Swap pivot to the current position
        perm[[k, i]] = perm[[i, k]]
        diagonal[[k, i]] = diagonal[[i, k]]
        if k > 0:
            L[:k, [k, i]] = L[:k, [i, k]]

        # Compute the matrix-vector product for the current pivot
        e_k = np.zeros(n)
        e_k[perm[k]] = 1
        v = A @ e_k

        # Compute the current row of L
        L[k, k] = np.sqrt(diagonal[k])
        if k < n - 1:
            L[k, k + 1 :] = (v[perm[k + 1 :]] - L[:k, k] @ L[:k, k + 1:]) / L[
                k, k
            ]

        # Update the diagonal entries for the next iteration
        diagonal[k + 1 :] -= L[k, k + 1 :] ** 2

        # Record the pivot index
        piv.append(perm[k])

    return L[: k + 1, :], piv
