#!/usr/bin/env python
import argparse

import numpy as np

from gpmap.utils import (
    LogTrack,
    get_CV_splits,
    read_dataframe,
    generate_p_training_config,
    get_training_p_splits,
    write_split_data,
)


def main():
    description = "Splits data into smaller datasets for model evaluation. "
    description += (
        "Two modes are available: generating regular Cross-validation"
    )
    description += "split datasets or training and test data for a range "
    description += " of training proportions. It supports both quantitative "
    description += "measurements or count based data."

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "data", help="CSV or parquet table with genotype-phenotype data"
    )
    input_group.add_argument(
        "--seed", default=None, type=int, help="Random seed"
    )

    cv_group = parser.add_argument_group("Cross-validation options")
    cv_group.add_argument(
        "--cv",
        default=False,
        action="store_true",
        help="Generate cross-validation datasets",
    )
    cv_group.add_argument(
        "-k",
        "--k_folds",
        default=10,
        type=int,
        help="Number of folds for cross-validation (10)",
    )

    trainp_group = parser.add_argument_group("Training proportions options")
    trainp_group.add_argument(
        "-r",
        "--nreps",
        default=3,
        type=int,
        help="Number of replicates for each proportion (3)",
    )
    trainp_group.add_argument(
        "-n",
        "--n_ps",
        default=10,
        type=int,
        help="Number of different training proportions (10)",
    )
    trainp_group.add_argument(
        "-m",
        "--max_pred",
        default=None,
        type=int,
        help="Max number of test sequences to generate",
    )
    trainp_group.add_argument(
        "--ps",
        default=None,
        help="File containing specific training proportions to use",
    )
    trainp_group.add_argument(
        "--fixed_test",
        default=False,
        action="store_true",
        help="Keep a constant test set across splits",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )
    output_group.add_argument(
        "-f", "--format", default="parquet", help="Output files format"
    )
    output_group.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="Prefix for sub-datasets output files",
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    seed = parsed_args.seed

    run_cv = parsed_args.cv
    nfolds = parsed_args.k_folds

    n_reps = parsed_args.nreps
    n_ps = parsed_args.n_ps
    ps_fpath = parsed_args.ps
    max_pred = parsed_args.max_pred
    fixed_test = parsed_args.fixed_test

    out_fpath = parsed_args.output
    out_prefix = parsed_args.prefix
    out_format = parsed_args.format

    # Load counts data
    log = LogTrack()
    log.write("Loading data")
    data = read_dataframe(data_fpath)

    if seed is not None:
        np.random.seed(seed)

    # Get processed data
    X = data.index.values
    y = data.values[:, 0]
    y_var = data.values[:, 1] if data.shape[1] > 1 else None

    if run_cv:
        log.write("Generating {}-fold Cross-valiation sets".format(nfolds))
        splits = get_CV_splits(
            X, y, y_var=y_var, nfolds=nfolds, max_pred=max_pred
        )
    else:
        log.write("Generating training and test sets with variable proportions")
        ps = (
            None
            if ps_fpath is None
            else np.array([float(x.strip()) for x in open(ps_fpath)])
        )
        config = generate_p_training_config(ps=ps, n_ps=n_ps, n_reps=n_reps)
        config.to_csv(out_fpath)

        splits = get_training_p_splits(
            config, X, y, y_var=y_var, max_pred=max_pred, fixed_test=fixed_test
        )

    write_split_data(out_prefix, splits, out_format=out_format)

    log.finish()


if __name__ == "__main__":
    main()
