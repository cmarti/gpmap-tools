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
    DataSet object for managing and manipulating various components 
    related to a specific dataset. This includes the original data, 
    reconstructed landscape, and visualization coordinates.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load from the built-in list. If `data`
        or `landscape` are provided, this will be the name assigned to 
        the new dataset.

    data : pd.DataFrame, shape (n_obs, n_features), optional
        A DataFrame containing the experimental data with genotypes 
        as the index.

    landscape : pd.DataFrame, shape (n_genotypes, 1), optional
        A DataFrame containing the complete combinatorial landscape 
        used to build the remaining components of the dataset.
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
        """
        Generate a SequenceSpace object from the dataset's landscape.

        This method constructs a SequenceSpace object using the genotypes 
        and their corresponding values from the dataset's landscape.

        Returns
        -------
        SequenceSpace
            A SequenceSpace object representing the dataset's landscape.
        """
        space = SequenceSpace(
            X=self.landscape.index.values, y=self.landscape.iloc[:, 0].values
        )
        return space

    def calc_visualization(
        self,
        Ns=None,
        mean_function=None,
        mean_function_perc=None,
        n_components=20,
    ):
        """
        Calculates the state coordinates to use for visualization
        of the provided discrete space under a given time-reversible
        random walk. The coordinates consist of the right eigenvectors
        of the associated rate matrix `Q`, re-scaled by the corresponding
        quantity so that the embedding is in units of square root of
        time.

        Parameters
        ----------
        Ns : float, optional
            Scaled effective population size to use in the underlying
            evolutionary model. If not provided, it will be derived
            from `mean_function` or `mean_function_perc`.

        mean_function : float, optional
            Mean function at stationarity to derive the associated Ns.
            Either this or `mean_function_perc` must be provided if
            `Ns` is not specified.

        mean_function_perc : float, optional
            Percentile that the mean function at stationarity takes within
            the distribution of function values along sequence space. For
            example, if `mean_function_perc=98`, then the mean function at
            stationarity is set to be at the 98th percentile across all
            the function values. Either this or `mean_function` must be
            provided if `Ns` is not specified.

        n_components : int, default=10
            Number of eigenvectors or Diffusion axes to calculate.
        """
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
        self, P=2, vc_cross_validation=False, vc_cv_loss_function="logL"
    ):
        if "X" in self.data.columns:
            X = self.data.X.values
            model = SeqDEFT(P=P, genotypes=X)
            model.fit(X=X)
            pred = model.predict()
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
                genotypes=X,
                cross_validation=vc_cross_validation,
                cv_loss_function=vc_cv_loss_function,
            )
            model.fit(X=X, y=y, y_var=y_var)
            pred = model.predict()
            X, y = pred.index.values, pred.f.values

        else:
            msg = "Model could not be selected for the provided data table. "
            msg += "Make sure to include a at least column `X` or `y`"
            raise ValueError(msg)

        self._landscape = pd.DataFrame({"y": y}, index=X)

    def build(self):
        """
        Build the dataset by inferring the landscape, calculating visualization, 
        and saving the results.

        This method performs the following steps:
        1. Checks if the `landscape` attribute exists. If not, it infers the landscape.
        2. Computes a mean function based on the mean and maximum values of the 
           `y` attribute in the landscape.
        3. Calculates visualization data using the specified number of samples (`Ns`) 
           and the computed mean function.
        4. Saves the processed data.

        Notes
        -----
        The method assumes that the `landscape` attribute is a dictionary-like object 
        with a key `"y"` containing numerical data.

        Raises
        ------
        AttributeError
            If the `landscape` attribute is not properly initialized.
        """
        if not hasattr(self, "_landscape"):
            self.infer_landscape()
        meanv, maxv = self.landscape["y"].mean(), self.landscape["y"].max()
        mean_function = meanv + 0.8 * (maxv - meanv)
        self.calc_visualization(Ns=self.Ns, mean_function=mean_function)
        self.save()

    def save(self, fdir=None):
        """
        Saves the dataset to disk for direct access within the library.

        This method stores raw data, inferred genotype-phenotype map, and
        the computed visualization coordinates and relaxation times when
        available. 

        Parameters
        ----------
        fdir : str, optional
            Directory where the dataset should be saved. If not provided, 
            the default directories defined in the settings will be used.

        Notes
        -----
        - The dataset is stored by default in the installation folder and will
          be deleted upon re-installation.
        - If a custom directory is provided, the dataset will be saved with 
          specific suffixes for each component.
        """
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
        '''
        Makes a two panel figure with the relaxation times associated to
        the computed Diffusion axes and a low dimensional representation
        of the complete genotype-phenotype map from this ``DataSet``.
        '''
        fig, subplots = plt.subplots(1, 2, figsize=(8, 3.5))
        axes = subplots[0]
        plot.plot_relaxation_times(self.relaxation_times, axes)
        axes.set_ylim(0, None)

        axes = subplots[1]
        plot.plot_visualization(axes, self.nodes, edges_df=self.edges)
        fig.tight_layout()


def list_available_datasets():
    """
    Retrieve the names of all available built-in datasets.

    This function scans the directory specified by `LANDSCAPES_DIR` and 
    extracts the names of all files present, excluding their extensions. 
    It returns these names as a list.

    Returns:
        list: A list of strings, where each string is the name of a built-in dataset.
    """
    dataset_names = [
        ".".join(fname.split(".")[:-1]) for fname in listdir(LANDSCAPES_DIR)
    ]
    return dataset_names
