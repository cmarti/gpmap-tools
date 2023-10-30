import sys
import time
from time import ctime
import numpy as np
import pandas as pd

from os import listdir
from os.path import join, dirname, abspath
from tqdm import tqdm

from scipy.sparse.csr import csr_matrix
from scipy.sparse.extract import triu
from scipy.sparse._matrix_io import load_npz, save_npz
from scipy.sparse.coo import coo_matrix
from scipy.stats import norm
from scipy.stats.stats import pearsonr


def check_error(condition, msg, error_type=ValueError):
    if not condition:
        raise error_type(msg)


def get_length(x):
    try:
        length = x.shape[0]
    except AttributeError:
        length = len(x)
    return(length)


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


def subsample_data(data, n=None):
    n_test = data[0].shape[0]
    if n is not None and n_test > n:
        test_idx = np.random.choice(np.arange(n_test), size=n,
                                    replace=False)
        data = get_data_subset(data, test_idx)
    return(data)


def shuffle(x, n=1):
    for _ in range(n):
        np.random.shuffle(x)


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
            
    elif isinstance(edges, csr_matrix) or isinstance(edges, coo_matrix):
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


def get_CV_splits(X, y, y_var=None, nfolds=10, max_pred=None):
    msg = 'X and y must have the same size'
    check_error(X.shape[0] == y.shape[0], msg=msg)
    
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
        test_data = subsample_data(test_data, n=max_pred)
        yield(j, train_data, test_data)


def data_to_df(data):
    x, y = data[:2]

    df = {'y': y}
    if len(data) > 2 and data[2] is not None:
        df['y_var'] = data[2]
    
    df = pd.DataFrame(df, index=x)
    return(df)


def generate_p_training_config(n_ps=10, nreps=3):
    ps = np.vstack([np.linspace(0.05, 0.95, n_ps)] * nreps).T.flatten()
    i = np.arange(ps.shape[0])
    rep = np.hstack([j * np.ones(n_ps) for j in range(nreps)])
    data = pd.DataFrame({'id': i, 'p': ps, 'rep': rep})
    return(data)


def get_training_p_splits(config, X, y, y_var=None, max_pred=None,
                          fixed_test=False):
    msg = 'X and y must have the same size'
    check_error(X.shape[0] == y.shape[0], msg=msg)
    data = (X, y, y_var)
    total = X.shape[0]
    
    for _, c in config.groupby('rep'):
        
        if fixed_test:
            msg = 'max_pred must be provided for fixed test'
            check_error(max_pred is not None, msg=msg)
            
            test_data = subsample_data(data, n=max_pred)
            training = get_data_subset(data, ~np.isin(X, test_data[0]))
            training_n = training[0].shape[0]
        
            for i, p in zip(c['id'], c['p']):
                p = p * total / training_n  
                u = np.random.uniform(size=X.shape[0]) < p
                train_data = get_data_subset(data, u)
                yield(i, train_data, test_data)
        else:
            
            for i, p in zip(c['id'], c['p']):
                u = np.random.uniform(size=total) < p
                train_data = get_data_subset(data, u)
                test_data = get_data_subset(data, ~u)
                test_data = subsample_data(test_data, n=max_pred)
                yield(i, train_data, test_data)


def write_split_data(out_prefix, splits, out_format='csv'):
    for i, train, test in splits:
        fpath = '{}.{}.train.{}'.format(out_prefix, i, out_format)
        train_df = data_to_df(train)
        write_dataframe(train_df, fpath)

        test_x = test[0]        
        write_seqs(test_x, fpath='{}.{}.test.txt'.format(out_prefix, i))
        

def read_split_data(prefix, suffix=None, in_format='csv', log=None):
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
            if log is not None:
                msg = '\tReading {} file: {}'.format(label, fpath)
                log.write(msg)
            test_pred = pd.read_csv(fpath, index_col=0)
            yield(label, test_pred)


def evaluate_predictions(test_pred_sets, data,
                         ypred_col='ypred', y_col='y', y_var_col='y_var'):
    results = []
    for label, test_pred in tqdm(test_pred_sets):
        df = data.join(test_pred).dropna()

        y = df[y_col].values
        ypred = df[ypred_col].values
        record = {'label': label, 'n': ypred.shape[0],
                  'mse': np.mean((y - ypred)**2)}
        
        if df.shape[0] > 1 and np.unique(ypred).shape[0] > 1 and np.unique(y).shape[0] > 1:
            record['r2'] = pearsonr(ypred, y)[0] ** 2
        
        if y_var_col in df.columns:
            y_var = df[y_var_col]
            record['loglikelihood'] = norm.logpdf(ypred, loc=y, scale=np.sqrt(y_var)).mean()
            
        results.append(record)
    return(pd.DataFrame(results))
