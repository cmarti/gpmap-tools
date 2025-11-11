import sys
import time
from os import listdir
from os.path import abspath, dirname, join
from time import ctime

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz, triu
from scipy.stats import norm, pearsonr
from tqdm import tqdm

from gpmap.settings import U_MAX


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)


def check_error(condition, msg, error_type=ValueError):
    if not condition:
        raise error_type(msg)


def get_length(x):
    try:
        length = x.shape[0]
    except AttributeError:
        length = len(x)
    return length


class LogTrack(object):
    """Logger class"""

    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = "[ {} ] {}\n".format(ctime(), msg)
        else:
            msg += "\n"
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write("Finished succesfully. Time elapsed: {:.1f} s".format(t))


def write_log(log, msg):
    if log is not None:
        log.write(msg)


def counts_to_seqs(X, y):
    seqs = []
    for seq, counts in zip(X, y):
        seqs.extend([seq] * counts)
    seqs = np.array(seqs)
    return seqs


def seqs_to_counts(seqs):
    return np.unique(seqs, return_counts=True)


def get_data_subset(data, idx):
    subset = tuple([x[idx] if x is not None else None for x in data])
    return subset


def subsample_data(data, n=None):
    n_test = data[0].shape[0]
    if n is not None and n_test > n:
        test_idx = np.random.choice(np.arange(n_test), size=n, replace=False)
        data = get_data_subset(data, test_idx)
    return data


def shuffle(x, n=1):
    for _ in range(n):
        np.random.shuffle(x)


def write_seqs(seqs, fpath):
    with open(fpath, "w") as fhand:
        for seq in seqs:
            fhand.write("{}\n".format(seq))


def write_dataframe(df, fpath):
    suffix = fpath.split(".")[-1]
    if suffix == "csv":
        df.to_csv(fpath)
    elif suffix == "pq" or suffix == "parquet":
        df.to_parquet(fpath)
    else:
        msg = "output format {} not recognized".format(suffix)
        raise ValueError(msg)


def read_dataframe(fpath):
    """
    Read a DataFrame from a file path in different formats.

    Parameters
    ----------
    fpath : str
        Path to the file containing the DataFrame. The file extension
        determines the format: 'csv' or 'parquet'.

    Returns
    -------
    pd.DataFrame
        The DataFrame read from the file.

    Raises
    ------
    ValueError
        If the file format is not recognized.
    """
    suffix = fpath.split(".")[-1]
    if suffix == "csv":
        df = pd.read_csv(fpath, index_col=0)
    elif suffix == "pq" or suffix == "parquet":
        df = pd.read_parquet(fpath)
    else:
        msg = "input format {} not recognized".format(suffix)
        raise ValueError(msg)
    return df


def read_edges(fpath, log=None, return_df=True):
    """
    Reads the incidence matrix containing the adjacency information among
    genotypes from a sequence space.

    Parameters
    ----------
    fpath : str
        File path containing the edges of a sequence space. The extension will
        be used to differentiate between csv, parquet, and the more efficient
        npz format.

    log : LogTrack, optional
        Logger instance to log messages. Default is None.

    return_df : bool, default=True
        Whether to return a pandas DataFrame with the edges. If False,
        it will return a csr_matrix.

    Returns
    -------
    edges_df : pd.DataFrame or csr_matrix
        If `return_df` is True, returns a DataFrame with columns ``i`` and ``j``
        containing the indices of the genotypes that are separated by a single
        mutation in a sequence space. If `return_df` is False, returns a
        csr_matrix representation of the edges.
    """
    if fpath is not None:
        write_log(log, "Reading edges data from {}".format(fpath))
        edges_format = fpath.split(".")[-1]
        if edges_format == "npz":
            A = load_npz(fpath).tocoo()
            if return_df:
                write_log(log, "Transforming sparse matrix into dataframe")
                edges_df = pd.DataFrame({"i": A.row, "j": A.col})
            else:
                edges_df = A
        else:
            edges_df = read_dataframe(fpath)
            if not return_df:
                edges_df = edges_df_to_csr_matrix(edges_df)
    else:
        write_log(log, "No edges provided")
        edges_df = None

    return edges_df


def edges_df_to_csr_matrix(edges_df):
    size = max(edges_df["i"].max(), edges_df["j"].max()) + 1
    idxs = np.arange(edges_df.shape[0])

    # idxs are store for filtering edges later on rather than just ones
    m = csr_matrix((idxs, (edges_df["i"], edges_df["j"])), shape=(size, size))
    return m


def csr_matrix_to_edges_df(A):
    try:
        edges_df = pd.DataFrame({"i": A.row, "j": A.col})
    except AttributeError:
        A = A.tocoo()
        edges_df = pd.DataFrame({"i": A.row, "j": A.col})
    return edges_df


def write_edges(edges, fpath, triangular=True):
    """
    Save the incidence matrix containing adjacency information among genotypes
    in a sequence space to a file.

    Parameters
    ----------
    edges : csr_matrix or pd.DataFrame
        The edges object to save. Can be a sparse matrix (csr_matrix) or a
        DataFrame containing edge information.

    fpath : str
        The file path where the edges will be saved. The file extension
        determines the format: 'csv', 'pq' (Parquet), or 'npz' (sparse matrix).

    triangular : bool, default=True
        If True, only the upper triangular part of the adjacency matrix is
        saved. This is useful for efficient storage when visualizing adjacency
        relationships.
    """
    # Transform into the right object given a format
    fmt = fpath.split(".")[-1]
    if isinstance(edges, pd.DataFrame):
        if triangular:
            edges = edges.loc[edges["j"] > edges["i"], :]
        if fmt == "npz":
            edges = edges_df_to_csr_matrix(edges)

    elif isinstance(edges, csr_matrix) or isinstance(edges, coo_matrix):
        if triangular:
            edges = triu(edges)
        if fmt != "npz":
            edges = csr_matrix_to_edges_df(edges)

    else:
        msg = "Invalid edges object. Use csr_matrix or pd.DataFrame"
        raise ValueError(msg)

    # Write into disk
    if fmt == "npz":
        save_npz(fpath, edges)
    else:
        write_dataframe(edges, fpath)


def get_CV_splits(X, y=None, y_var=None, nfolds=10, max_pred=None):
    msg = "X and y must have the same size"
    check_error(y is None or X.shape[0] == y.shape[0], msg=msg)

    msg = "Number of observations must be >= nfolds"
    check_error(X.shape[0] >= nfolds, msg=msg)

    data = (X, y, y_var)
    n_obs = X.shape[0]
    order = np.arange(n_obs)
    shuffle(order, n=3)
    n_test = np.round(n_obs / nfolds).astype(int)

    for j in range(nfolds):
        i = j * n_test
        test_data = get_data_subset(data, order[i : i + n_test])
        train_data = get_data_subset(
            data, np.append(order[:i], order[i + n_test :])
        )
        test_data = subsample_data(test_data, n=max_pred)
        yield (j, train_data, test_data)


def get_cv_iter(cv_splits, hyperparam_values, process_data=None):
    for fold, train, test in cv_splits:
        if process_data is not None:
            train = process_data(train)
            test = process_data(test)
        for param in hyperparam_values:
            yield (param, fold, train, test)


def calc_cv_loss(
    cv_iter,
    train_function,
    evaluate_function,
    total_folds=None,
    param_label="beta",
    loss_label="loss",
):
    for beta, fold, train, test in tqdm(cv_iter, total=total_folds):
        params = train_function(train, beta)
        loss = evaluate_function(test, params)
        yield ({param_label: beta, "fold": fold, loss_label: loss})


def data_to_df(data):
    x, y = data[:2]

    df = {"y": y}
    if len(data) > 2 and data[2] is not None:
        df["y_var"] = data[2]

    df = pd.DataFrame(df, index=x)
    return df


def generate_p_training_config(
    ps=None, n_ps=10, n_reps=3, pmin=0.05, pmax=0.95
):
    """
    Generate a configuration DataFrame for creating data splits with varying
    proportions of training data.

    Parameters
    ----------
    ps : array-like, optional
        Array specifying the training proportions to use. If None, a uniform
        set of proportions between `pmin` and `pmax` will be generated.
        Default is None.
    n_ps : int, optional
        Number of distinct training proportions to generate. Default is 10.
    n_reps : int, optional
        Number of replicates for each training proportion. Default is 3.
    pmin : float, optional
        Minimum training proportion. Default is 0.05.
    pmax : float, optional
        Maximum training proportion. Default is 0.95.

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing the configuration for the different subsets,
        including training proportions and replicate information. This can
        be used as input for `get_training_p_splits`.
    """

    if ps is None:
        ps = np.linspace(0.05, 0.95, n_ps)
    else:
        n_ps = ps.shape[0]
    ps = np.vstack([ps] * n_reps).T.flatten()

    i = np.arange(1, ps.shape[0] + 1)
    rep = np.hstack([np.arange(n_reps) for j in range(n_ps)])
    data = pd.DataFrame({"id": i, "p": ps, "rep": rep})
    return data


def get_training_p_splits(
    config, X, y, y_var=None, max_pred=None, fixed_test=False
):
    """
    Generate data splits into training and test subsets based on a training
    proportion configuration.

    Parameters
    ----------
    config : pd.DataFrame
        DataFrame containing the training proportions configuration, typically
        generated by `generate_p_training_config`.
    X : array-like of str
        Array containing the set of sequences from which to sample.
    y : array-like of float
        Array containing the corresponding quantitative information
        associated with each sequence.
    y_var : array-like of float, optional
        Variance associated with each measurement in `y`. Default is None.
    max_pred : int, optional
        Maximum number of test points to return for each split. Default is None.
    fixed_test : bool, optional
        Whether to use the same test data across each of the replicates.
        Default is False.

    Yields
    ------
    tuple
        A tuple containing:
        - int: The split ID.
        - tuple: The training data subset.
        - tuple: The test data subset.
    """
    msg = "X and y must have the same size"
    check_error(X.shape[0] == y.shape[0], msg=msg)
    data = (X, y, y_var)
    total = X.shape[0]

    for _, c in config.groupby("rep"):
        if fixed_test:
            msg = "max_pred must be provided for fixed test"
            check_error(max_pred is not None, msg=msg)

            test_data = subsample_data(data, n=max_pred)
            training = get_data_subset(data, ~np.isin(X, test_data[0]))
            training_n = training[0].shape[0]

            for i, p in zip(c["id"], c["p"]):
                p = p * total / training_n
                u = np.random.uniform(size=X.shape[0]) < p
                train_data = get_data_subset(data, u)
                yield (i, train_data, test_data)
        else:
            for i, p in zip(c["id"], c["p"]):
                u = np.random.uniform(size=total) < p
                train_data = get_data_subset(data, u)
                test_data = get_data_subset(data, ~u)
                test_data = subsample_data(test_data, n=max_pred)
                yield (i, train_data, test_data)


def write_split_data(out_prefix, splits, out_format="csv"):
    for i, train, test in splits:
        fpath = "{}.{}.train.{}".format(out_prefix, i, out_format)
        train_df = data_to_df(train)
        write_dataframe(train_df, fpath)

        test_x = test[0]
        write_seqs(test_x, fpath="{}.{}.test.txt".format(out_prefix, i))


def read_split_data(prefix, suffix=None, in_format="csv", log=None):
    fdir = abspath(dirname(prefix))
    prefix = prefix.split("/")[-1]

    suffix2 = "test_pred.{}".format(in_format)
    if suffix is None:
        suffix = suffix2
    else:
        suffix = "{}.{}".format(suffix, suffix2)

    for fname in listdir(fdir):
        fpath = join(fdir, fname)
        if fname.startswith(prefix) and fname.endswith(suffix):
            label = ".".join(fname.split(".")[1:-2])
            if log is not None:
                msg = "\tReading {} file: {}".format(label, fpath)
                log.write(msg)
            test_pred = pd.read_csv(fpath, index_col=0)
            yield (label, test_pred)


def evaluate_predictions(
    test_pred_sets, data, ypred_col="ypred", y_col="y", y_var_col="y_var"
):
    results = []
    for label, test_pred in tqdm(test_pred_sets):
        df = data.join(test_pred).dropna()

        y = df[y_col].values
        ypred = df[ypred_col].values
        record = {
            "label": label,
            "n": ypred.shape[0],
            "mse": np.mean((y - ypred) ** 2),
        }

        if (
            df.shape[0] > 1
            and np.unique(ypred).shape[0] > 1
            and np.unique(y).shape[0] > 1
        ):
            record["r2"] = pearsonr(ypred, y)[0] ** 2

        if y_var_col in df.columns:
            y_var = df[y_var_col]
            record["loglikelihood"] = norm.logpdf(
                ypred, loc=y, scale=np.sqrt(y_var)
            ).mean()

        results.append(record)
    return pd.DataFrame(results)
