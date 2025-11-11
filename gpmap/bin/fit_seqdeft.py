#!/usr/bin/env python
import argparse
import numpy as np

from gpmap.utils import LogTrack, write_dataframe
from gpmap.inference import SeqDEFT
from gpmap.plot.mpl import figure_SeqDEFT_summary, savefig


def main():
    description = (
        "Runs Sequence Density Estimation using Field Theory (SeqDEFT)"
    )
    description += " from a table with the number of time each sequence was"
    description += " observed e.g. the number of times each splice donor or"
    description += " acceptor is found in the human genome"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "sequences", help="File containing a list of equally long sequences"
    )
    help_msg = "File containing the weights associated to each input sequence"
    input_group.add_argument("-y", "--weights", default=None, help=help_msg)

    help_msg = (
        "Comma separated list of positions to select from the input sequences"
    )
    input_group.add_argument("-p", "--positions", default=None, help=help_msg)

    options_group = parser.add_argument_group("SeqDEFT options")
    help_msg = "Value of the hyperparameter a for SeqDEFT. If not provided, "
    help_msg += "grid search with cross-validated likelihood will be used"
    options_group.add_argument(
        "-a", "--a_value", default=None, type=float, help=help_msg
    )
    options_group.add_argument(
        "-k",
        "--num_a",
        default=20,
        type=int,
        help="Number of a values to test (20)",
    )
    help_msg = "P for the Delta(P) operator to build prior (2)"
    options_group.add_argument(
        "-P", "--P_delta", default=2, help=help_msg, type=int
    )
    options_group.add_argument(
        "--get_a_values",
        default=False,
        action="store_true",
        help="Only calculate the series of a values for CV",
    )
    help_msg = "Apply phylogenetic correction to input sequences (--positions required)"
    options_group.add_argument(
        "--phylo_correction", default=False, action="store_true", help=help_msg
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    x_fpath = parsed_args.sequences
    y_fpath = parsed_args.weights
    positions = parsed_args.positions

    P = parsed_args.P_delta
    a_value = parsed_args.a_value
    num_a = parsed_args.num_a
    get_a_values = parsed_args.get_a_values
    phylo_correction = parsed_args.phylo_correction

    out_fpath = parsed_args.output

    # Load counts data
    log = LogTrack()
    log.write("Loading input sequences")
    X = np.array([x.strip() for x in open(x_fpath)])

    if positions is not None:
        positions = [int(x) for x in positions.split(",")]

    if y_fpath is None:
        y = None
    else:
        log.write("Loading weights for input sequences")
        y = np.array([x.strip() for x in open(y_fpath)])

    # Load annotation data
    seqdeft = SeqDEFT(P, a=a_value, num_reg=num_a, genotypes=X)
    if get_a_values:
        log.write("Initializint SeqDEFT model")
        seqdeft.set_data(
            X=X, y=y, phylo_correction=phylo_correction, positions=positions
        )

        log.write("Calculating only a values")
        with open(out_fpath, "w") as fhand:
            for a_value in seqdeft.get_a_values():
                fhand.write("{}\n".format(a_value))
    else:
        log.write("Fitting SeqDEFT model to observed data")
        seqdeft.fit(
            X=X, y=y, phylo_correction=phylo_correction, positions=positions
        )
        result = seqdeft.predict()

        log.write("Writing output to {}".format(out_fpath))
        write_dataframe(result, out_fpath)

        log.write(
            "Creating summary plot and saving to {}.png".format(out_fpath)
        )
        fig = figure_SeqDEFT_summary(seqdeft.logL_df, result)
        savefig(fig, out_fpath)

    log.finish()


if __name__ == "__main__":
    main()
