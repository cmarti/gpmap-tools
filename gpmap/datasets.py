#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpmap.plot.mpl as plot

from os import listdir
from os.path import join, exists

from gpmap.utils import (
    check_error,
    read_dataframe,
    read_edges,
    write_edges,
    write_dataframe,
)
from gpmap.settings import RAW_DATA_DIR, LANDSCAPES_DIR, PROCESSED_DIR, VIZ_DIR
from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
from gpmap.inference import VCregression, SeqDEFT


class DataSet(object):
    """
    DataSet object that allows convenient manipulation of the different
    objets related with a given dataset. This includes the original data,
    the reconstructed landscape, visualization coordinates

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load from the built-in list. If `data`
        or `landscape` are provided, it will be the name given to the
        new dataset

    data: pd.DataFrame of shape (n_obs, n_features)
        Dataframe containing the experimental data using
        genotypes as index

    landscape: pd.DataFrame of shape (n_genotypes, 1)
        Dataframe containing the complete combinatorial landscape
        from which to build the remaining objects of the dataset

    """

    def __init__(self, dataset_name, data=None, landscape=None):
        self.name = dataset_name

        if data is None and landscape is None:
            datasets = list_available_datasets()
            check_error(
                dataset_name in datasets,
                msg="Dataset not available: check {}".format(datasets),
            )
        else:
            check_error(
                landscape is not None or data is not None,
                msg="landscape or data must be provided for new dataset",
            )
            if landscape is not None:
                self._landscape = landscape
            if data is not None:
                self._data = data

        self.Ns = None

    def rename(self, dataset_name):
        self.name = dataset_name

    def _load(self, fdir, label, suffix=""):
        fpath = join(fdir, "{}.pq".format(self.name + suffix))

        if not exists(fpath):
            fpath = join(fdir, "{}.npz".format(self.name + suffix))

            if not exists(fpath):
                msg = "{} for dataset {} not found".format(
                    label, self.name + suffix
                )
                raise ValueError(msg)
            else:
                df = read_edges(fpath, return_df=True)
        else:
            df = read_dataframe(fpath)
        return df

    def _write(self, df, fdir, suffix="", fmt="pq"):
        fpath = join(fdir, "{}.{}".format(self.name + suffix, fmt))
        if fmt == "npz":
            write_edges(df, fpath)
        else:
            write_dataframe(df, fpath)

    @property
    def landscape(self):
        if not hasattr(self, "_landscape"):
            self._landscape = self._load(
                fdir=LANDSCAPES_DIR, label="estimated landscape"
            )
        return self._landscape

    @property
    def raw_data(self):
        if not hasattr(self, "_raw_data"):
            self._raw_data = self._load(fdir=RAW_DATA_DIR, label="raw data")
        return self._raw_data

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = self._load(fdir=PROCESSED_DIR, label="processed data")
        return self._data

    @property
    def nodes(self):
        if not hasattr(self, "_nodes"):
            self._nodes = self._load(
                fdir=VIZ_DIR, label="nodes coordinates", suffix=".nodes"
            )
        return self._nodes

    @property
    def edges(self):
        if not hasattr(self, "_edges"):
            self._edges = self._load(
                fdir=VIZ_DIR, label="nodes coordinates", suffix=".edges"
            )
        return self._edges

    @property
    def relaxation_times(self):
        if not hasattr(self, "_relaxation_times"):
            self._relaxation_times = self._load(
                fdir=VIZ_DIR,
                label="relaxation times",
                suffix=".relaxation_times",
            )
        return self._relaxation_times

    def to_sequence_space(self):
        space = SequenceSpace(
            X=self.landscape.index.values, y=self.landscape.iloc[:, 0].values
        )
        return space

    def calc_visualization(
        self,
        n_components=20,
        Ns=None,
        mean_function=None,
        mean_function_perc=None,
    ):
        space = self.to_sequence_space()
        rw = WMWalk(space)
        rw.calc_visualization(
            n_components=n_components,
            Ns=Ns,
            mean_function=mean_function,
            mean_function_perc=mean_function_perc,
        )
        self._nodes = rw.nodes_df
        self._edges = space.get_edges_df()
        self._relaxation_times = rw.decay_rates_df

    def infer_landscape(
        self, P=2, vc_cross_validation=True, vc_cv_loss_function="logL"
    ):
        if "X" in self.data.columns:
            X = self.data.X.values
            model = SeqDEFT(P=P, genotypes=X)
            pred = model.fit(X=X)
            X, y = pred.index.values, np.log(pred.Q_star.values)
            self.Ns = 1

        elif "y" in self.data.columns:
            X = self.data.index.values
            y = self.data["y"].values
            y_var = (
                self.data["y_var"].values
                if "y_var" in self.data.columns
                else None
            )
            model = VCregression(
                cross_validation=vc_cross_validation,
                cv_loss_function=vc_cv_loss_function,
            )
            model.fit(X=X, y=y, y_var=y_var)
            pred = model.predict()
            X, y = pred.index.values, pred.y.values

        else:
            msg = "Model could not be selected for the provided data table. "
            msg += "Make sure to include a at least column `X` or `y`"
            raise ValueError(msg)

        self._landscape = pd.DataFrame({"y": y}, index=X)

    def build(self):
        if not hasattr(self, "_landscape"):
            self.infer_landscape()
        meanv, maxv = self.landscape["y"].mean(), self.landscape["y"].max()
        mean_function = meanv + 0.8 * (maxv - meanv)
        self.calc_visualization(Ns=self.Ns, mean_function=mean_function)
        self.save()

    def save(self, fdir=None):
        fdirs = [
            RAW_DATA_DIR,
            PROCESSED_DIR,
            LANDSCAPES_DIR,
            VIZ_DIR,
            VIZ_DIR,
            VIZ_DIR,
        ]
        suffixes = ["", "", "", ".nodes", ".edges", ".relaxation_times"]

        if fdir is not None:
            suffixes = [
                ".raw_data",
                ".data",
                ".landscape",
                ".nodes",
                ".edges",
                ".relaxation_times",
            ]
            fdirs = [fdir] * len(suffixes)

        attrs = [
            "_raw_data",
            "_data",
            "_landscape",
            "_nodes",
            "_edges",
            "_relaxation_times",
        ]
        fmts = ["pq", "pq", "pq", "pq", "npz", "pq"]

        for attr, fdir, suffix, fmt in zip(attrs, fdirs, suffixes, fmts):
            if hasattr(self, attr):
                df = getattr(self, attr)
                self._write(df, fdir, suffix, fmt=fmt)

    def plot(self):
        fig, subplots = plt.subplots(1, 2, figsize=(8, 3.5))
        axes = subplots[0]
        plot.plot_relaxation_times(self.relaxation_times, axes)
        axes.set_ylim(0, None)

        axes = subplots[1]
        plot.plot_visualization(axes, self.nodes, edges_df=self.edges)
        fig.tight_layout()


def list_available_datasets():
    """
    Returns a list with the names of all available built-in datasets
    """
    dataset_names = [
        ".".join(fname.split(".")[:-1]) for fname in listdir(LANDSCAPES_DIR)
    ]
    return dataset_names
