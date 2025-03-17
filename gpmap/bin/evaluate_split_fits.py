#!/usr/bin/env python
import argparse

from gpmap.utils import (
    read_split_data,
    evaluate_predictions,
    read_dataframe,
    write_dataframe,
    LogTrack,
)


def main():
    description = "Calculates evaluation metrics values of a set of predicted "
    description += "genotype-phenotype data subsets"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")

    input_group.add_argument(
        "data", help="CSV or parquet table with genotype-phenotype data"
    )
    input_group.add_argument(
        "-p", "--prefix", required=True, help="Files prefix"
    )
    input_group.add_argument(
        "-s", "--suffix", default=None, help="Files suffix"
    )
    help_msg = (
        "CSV or parquet table containing annotations to add to metrics table"
    )
    input_group.add_argument("-c", "--config", default=None, help=help_msg)
    help_msg = "Column name with y predictions (ypred_col)"
    input_group.add_argument(
        "-f", "--ypred_col", default="y_pred", help=help_msg
    )
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    prefix = parsed_args.prefix
    suffix = parsed_args.suffix
    config_fpath = parsed_args.config
    ypred_col = parsed_args.ypred_col
    out_fpath = parsed_args.output

    # Load data
    log = LogTrack()

    log.write("Reading landscape data from {}".format(data_fpath))
    data = read_dataframe(data_fpath)

    log.write("Reading split data from {}*{}".format(prefix, suffix))
    splits = read_split_data(prefix, suffix=suffix, log=log)

    log.write("Evaluating predictions")
    results = evaluate_predictions(splits, data, ypred_col=ypred_col)

    if config_fpath is not None:
        log.write("Merging with datasets information")
        config = read_dataframe(config_fpath)
        results.join(config, on="label")

    log.write("Writing output to {}".format(out_fpath))
    write_dataframe(results, out_fpath)


if __name__ == "__main__":
    main()
