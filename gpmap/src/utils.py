import sys
import time
from time import ctime
import numpy as np
import pandas as pd
import scipy.sparse as sp

from os import listdir
from os.path import join, dirname, abspath
from itertools import product
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix
from scipy.stats.stats import pearsonr
from scipy.sparse.extract import triu
from scipy.sparse._matrix_io import load_npz, save_npz



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


def get_length(x):
    try:
        length = x.shape[0]
    except AttributeError:
        length = len(x)
    return(length)


def inner_product(x1, x2, metric=None):
    if metric is None:
        return(x1.dot(x2.T))
    else:
        return(x1.dot(metric.dot(x2.T)))


def quad(matrix, v1, v2=None):
    if v2 is None:
        v2 = v1
    return(np.sum(matrix.dot(v1) * v2))


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
    for i, (l, u) in enumerate(zip(eigenvalues, right_eigenvectors.T)):
        v1 = matrix.dot(u)
        v2 = l * u
        abs_err = np.mean(np.abs(v1 - v2))

        msg = 'Numeric error in eigendecomposition: abs error = {:.5f} > {:.5f}'
        msg += ' in the {}th component'.format(i)
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
    if len(matrices) == 0:
        return(None)
    
    if len(matrices) == 1:
        return(matrices[0])
    
    m1, m2 = matrices[0], calc_cartesian_product(matrices[1:])
    i = sp.identity(m2.shape[0])
    
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


def counts_to_seqs(X, y):
    seqs = []
    for seq, counts in zip(X, y):
        seqs.extend([seq] * counts)
    seqs = np.array(seqs)
    return(seqs)


def seqs_to_counts(seqs):
    return(np.unique(seqs, return_counts=True))


def get_data_subset(data, idx):
    subset = tuple([x[idx] if x is not None else None for x in data])
    return(subset)


def subsample_data(data, max_pred=None):
    n_test = data[0].shape[0]
    if max_pred is not None and n_test > max_pred:
        test_idx = np.random.choice(np.arange(n_test), size=max_pred,
                                    replace=False)
        data = get_data_subset(data, test_idx)
    return(data)


def shuffle(x, n=1):
    for _ in range(n):
        np.random.shuffle(x)


def get_CV_splits(X, y, y_var=None, nfolds=10, count_data=False, max_pred=None):
    msg = 'X and y must have the same size'
    check_error(X.shape[0] == y.shape[0], msg=msg)
    
    if count_data:
        check_error(y_var is None,
                    msg='variance in estimation not allowed for count data')
        seqs = counts_to_seqs(X, y)
        
        msg = 'Number of observations must be >= nfolds'
        check_error(seqs.shape[0] >= nfolds, msg=msg)
        
        shuffle(seqs, n=3)
        n_test = seqs.shape[0] // nfolds
        
        for j in range(nfolds):
            i = j * n_test
            test = seqs_to_counts(seqs[i:i+n_test])
            train = seqs_to_counts(np.append(seqs[:i], seqs[i+n_test:]))
            yield(j, train, test)
    else:
        msg = 'Number of observations must be >= nfolds'
        check_error(X.shape[0] >= nfolds, msg=msg)
        
        data = (X, y, y_var)
        n_obs = X.shape[0]
        order = np.arange(n_obs)
        shuffle(order, n=3)
        n_test = np.round(n_obs / nfolds).astype(int)
        
        for j in range(nfolds):
            i = j * n_test
            test_data = get_data_subset(data, order[i:i+n_test])
            train_data = get_data_subset(data, np.append(order[:i], order[i+n_test:]))
            test_data = subsample_data(test_data, max_pred=max_pred)
            yield(j, train_data, test_data)


def data_to_df(data):
    x, y = data[:2]

    df = {'y': y}
    if len(data) > 2 and data[2] is not None:
        df['y_var'] = data[2]
    
    df = pd.DataFrame(df, index=x)
    return(df)


def generate_p_training_config(n_ps=10, nreps=3):
    ps = np.hstack([np.linspace(0.05, 0.95, n_ps)] * nreps)
    i = np.arange(ps.shape[0])
    rep = np.hstack([j * np.ones(n_ps) for j in range(nreps)])
    data = pd.DataFrame({'id': i, 'p': ps, 'rep': rep})
    return(data)


def sample_training_p_data(X, y, p, y_var=None, count_data=False, max_pred=None):
    msg = 'X and y must have the same size'
    check_error(X.shape[0] == y.shape[0], msg=msg)
    
    if count_data:
        check_error(y_var is None,
                    msg='variance in estimation not allowed for count data')
        seqs = counts_to_seqs(X, y)
        u = np.random.uniform(size=seqs.shape[0]) < p
        test = np.unique(seqs[~u], return_counts=True)
        train = np.unique(seqs[u], return_counts=True)
        return(train, test)
    
    else:
        data = (X, y, y_var)
        u = np.random.uniform(size=X.shape[0]) < p
        train_data = get_data_subset(data, u)
        test_data = get_data_subset(data, ~u)
        test_data = subsample_data(test_data, max_pred=max_pred)
        return(train_data, test_data)


def get_training_p_splits(config, X, y, y_var=None, count_data=False,
                          max_pred=None):
    for i, p in zip(config['id'], config['p']):
        train, test = sample_training_p_data(X, y, p, y_var=y_var,
                                             count_data=count_data,
                                             max_pred=max_pred)
        yield(i, train, test)


def write_seqs(seqs, fpath):
    with open(fpath, 'w') as fhand:
        for seq in seqs:
            fhand.write('{}\n'.format(seq))
        

def write_dataframe(df, fpath):
    suffix = fpath.split('.')[-1]
    if suffix == 'csv':
        df.to_csv(fpath)
    elif suffix == 'pq' or suffix == 'parquet':
        df.to_parquet(fpath)
    else:
        msg = 'output format {} not recognized'.format(suffix)
        raise ValueError(msg)


def read_dataframe(fpath):
    suffix = fpath.split('.')[-1]
    if suffix == 'csv':
        df = pd.read_csv(fpath, index_col=0)
    elif suffix == 'pq' or suffix == 'parquet':
        df = pd.read_parquet(fpath)
    else:
        msg = 'input format {} not recognized'.format(suffix)
        raise ValueError(msg)
    return(df)


def read_edges(fpath, log=None, return_df=True):
    '''
    Reads the incidence matrix containing the adjacency information among 
    genotypes from a sequence space
    
    Parameters
    ----------
    fpath : str
        File path containing the edges of a sequence space. The extension will 
        be used to differentiate between csv and the more efficient npz 
        format
        
    return_df : bool (True)
        Whether to return a pd.DataFrame with the edges. Alternatively
        it will return a csr_matrix
        
    Returns
    -------
    edges_df : pd.DataFrame of shape (n_edges, 2) or csr_matrix
        DataFrame with column names ``i`` and ``j`` containing the indexes
        of the genotypes that are separated by a single mutation in a 
        sequence space
    
    '''
    if fpath is not None:
        write_log(log, 'Reading edges data from {}'.format(fpath))
        edges_format = fpath.split('.')[-1]
        if edges_format == 'npz':
            A = load_npz(fpath).tocoo()
            if return_df:
                write_log(log, 'Transforming sparse matrix into dataframe')
                edges_df = pd.DataFrame({'i': A.row, 'j': A.col})
            else:
                edges_df = A
        else:
            edges_df = read_dataframe(fpath)
            if not return_df:
                edges_df = edges_df_to_csr_matrix(edges_df)
    else:
        write_log(log, 'No edges provided')
        edges_df = None
    
    return(edges_df)


def edges_df_to_csr_matrix(edges_df):
    size = max(edges_df['i'].max(), edges_df['j'].max()) + 1
    idxs = np.arange(edges_df.shape[0])
    
    # idxs are store for filtering edges later on rather than just ones
    m = csr_matrix((idxs, (edges_df['i'], edges_df['j'])),
                   shape=(size, size))
    return(m)


def csr_matrix_to_edges_df(A):
    try:
        edges_df = pd.DataFrame({'i': A.row, 'j': A.col})
    except AttributeError:
        A = A.tocoo()
        edges_df = pd.DataFrame({'i': A.row, 'j': A.col})
    return(edges_df)


def write_edges(edges, fpath, triangular=True):
    '''
    Writes the incidence matrix containing the adjacency information among 
    genotypes from a sequence space.
    
    Parameters
    ----------
    edges : csr_matrix or pd.DataFrame
        edges object to write into a file.
        
    fpath : str
        File path containing the edges of a sequence space. The extension will 
        be used to differentiate between csv or pq and the more efficient npz 
        format
        
    triangular : bool (True)
        Whether to write only the upper triangular for more efficient storing
        of the adjacency relationships when plotting. 
    '''
    # Transform into the right object given a format
    fmt = fpath.split('.')[-1]
    if isinstance(edges, pd.DataFrame):
        if triangular:
            edges = edges.loc[edges['j'] > edges['i'], :]
        if fmt == 'npz':
            edges = edges_df_to_csr_matrix(edges)
            
    elif isinstance(edges, csr_matrix):
        if triangular:
            edges = triu(edges)
        if fmt != 'npz':
            edges = csr_matrix_to_edges_df(edges)
    
    else:
        msg = 'Invalid edges object. Use csr_matrix or pd.DataFrame'
        raise ValueError(msg)

    # Write into disk
    if fmt == 'npz':
        save_npz(fpath, edges)
    else:
        write_dataframe(edges, fpath)


def write_split_data(out_prefix, splits, out_format='csv'):
    for i, train, test in splits:
        fpath = '{}.{}.train.{}'.format(out_prefix, i, out_format=out_format)
        train_df = data_to_df(train)
        write_dataframe(train_df, fpath)

        test_x = test[0]        
        write_seqs(test_x, fpath='{}.{}.test.txt'.format(out_prefix, i))
        

def read_split_data(prefix, suffix=None, in_format='csv'):
    fdir = abspath(dirname(prefix))
    prefix = prefix.split('/')[-1]
    
    suffix2 = 'test_pred.{}'.format(in_format)
    if suffix is None:
        suffix = suffix2
    else:
        suffix = '{}.{}'.format(suffix, suffix2)
         
    for fname in listdir(fdir):
        fpath = join(fdir, fname)
        if fname.startswith(prefix) and fname.endswith(suffix):
            label = '.'.join(fname.split('.')[1:-2])
            test_pred = pd.read_csv(fpath, index_col=0)
            yield(label, test_pred)
        

def calc_r2_values(test_pred_sets, data):
    r2 = []
    for label, test_pred in test_pred_sets:
        try:
            seqs = np.intersect1d(test_pred.index.values, data.index.values)
            ypred = test_pred.loc[seqs, :].iloc[:, 0].values
            yobs = data.loc[seqs, :].iloc[:, 0].values
            r2.append({'id': label, 'r2': pearsonr(ypred, yobs)[0] ** 2})
        except:
            print('problem found in {}. Skipping it'.format(label))
            continue
    return(pd.DataFrame(r2))
