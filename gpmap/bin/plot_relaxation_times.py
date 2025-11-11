#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.plot.mpl import plot_relaxation_times


def main():
    description = "Screeplot of the decay rates associated to the eigenvalues"
    description += " of the rate matrix"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    help_msg = "CSV file containing decay rates for the different eigenvalues"
    input_group.add_argument("input", help=help_msg)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    fpath = parsed_args.input
    out_fpath = parsed_args.output

    # Load data
    log = LogTrack()
    log.write("Start analysis")

    log.write("Reading decay rates and relaxation times from {}".format(fpath))
    decay_rates = pd.read_csv(fpath)

    log.write("Making screeplot at {}".format(out_fpath))
    plot_relaxation_times(decay_rates, fpath=out_fpath)
    log.finish()


if __name__ == "__main__":
    main()
