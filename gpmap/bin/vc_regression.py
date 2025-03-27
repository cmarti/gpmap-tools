#!/usr/bin/env python
import argparse

import numpy as np

from gpmap.utils import LogTrack, read_dataframe, write_dataframe
from gpmap.inference import VCregression


def main():
    description = "Runs Variance Component regression using data from"
    description += " quantitative phenotypes associated to their corresponding"
    description += " sequences. If provided, the variance of the estimated "
    description += " quantitative measure can be incorporated into the model"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "data", help="CSV table with genotype-phenotype data"
    )

    options_group = parser.add_argument_group("VC options")
    options_group.add_argument(
        "--lambdas",
        default=None,
        help="File containing known lambdas to use for prediction directly",
    )
    help_msg = "Regularize variance components to exponential decay through CV"
    options_group.add_argument(
        "-r", "--regularize", action="store_true", default=False, help=help_msg
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )
    help_msg = (
        "File containing sequences for predicting the associated phenotype"
    )
    output_group.add_argument("-p", "--pred", help=help_msg)
    help_msg = "Report also posterior variance for the predicted values"
    output_group.add_argument("--var", action="store_true", help=help_msg)

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data

    regularize = parsed_args.regularize
    lambdas_fpath = parsed_args.lambdas

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    calc_variance = parsed_args.var

    # Load counts data
    log = LogTrack()
    log.write("Start analysis")
    data = read_dataframe(data_fpath)

    # Get processed data
    X = data.index.values
    y = data.values[:, 0]
    y_var = data.values[:, 1] if data.shape[1] > 1 else None
    X_pred = (
        [line.strip().strip('"') for line in open(pred_fpath)]
        if pred_fpath is not None
        else None
    )

    # Create VC model, fit and predict
    vc = VCregression(genotypes=X, cross_validation=regularize)

    if lambdas_fpath is None:
        log.write("Estimate variance components through kernel alignment")
        vc.fit(X=X, y=y, y_var=y_var)
        lambdas = vc.lambdas
    else:
        log.write("Load variance components from {}".format(lambdas_fpath))
        lambdas = np.array([float(x.strip()) for x in open(lambdas_fpath)])

    msg = "Obtain phenotypic predictions"
    if calc_variance:
        msg += ". Calculating posterior variance"
    log.write(msg)
    vc.set_data(X=X, y=y, y_var=y_var)
    vc.set_lambdas(lambdas)
    result = vc.predict(X_pred=X_pred, calc_variance=calc_variance)

    # Save output
    write_dataframe(result, out_fpath)

    # Save lambdas
    prefix = ".".join(out_fpath.split(".")[:-1])
    with open("{}.lambdas.txt".format(prefix), "w") as fhand:
        for lda in lambdas:
            fhand.write("{}\n".format(lda))

    # Save running time
    with open("{}.time.txt".format(prefix), "w") as fhand:
        if hasattr(vc, "fit_time"):
            fhand.write("fit,{}\n".format(vc.fit_time))
        if hasattr(vc, "pred_time"):
            fhand.write("pred,{}\n".format(vc.pred_time))

    log.finish()


if __name__ == "__main__":
    main()
