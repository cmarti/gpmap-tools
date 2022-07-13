import sys
import time

from time import ctime

import numpy as np

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
