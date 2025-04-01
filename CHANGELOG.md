# CHANGELOG

All notable changes to this project will be documented in this file.


## New release [0.3.1] - 2025-03-31

### Added

- **Computation of posterior covariance** under the Minimum epistasis interpolation model. If `a` is not provided by the user, it can be inferred from the MEI solution before computing the posterior covariance. Uniqueness of the solution is now checked before calculations are done. 
- **Gaussian likelihood implemented** for consistency and sampling from the likelihood when simulating data as in SeqDEFT

### Changed

- **Generalization of Gaussian process** class so that all the models have the same set of basic methods under the more general class, including `sample_prior`, `simulate`, `fit`, `predict` and `make_contrast`. 

- **Homogeneization of inference classes** to all work under the same logic with the same methods for defining and fitting hyperparameters, making predictions, computing the posterior of linear combinations and sampling from the prior. 

## First stable release [0.3.0] - 2025-03-14

### Added

- **Independent plotting modules** for the three backends: `matplotlib`, `plotly`, and `datashader`, ensuring a **consistent interface** across all backends.
- **Computation of posterior variances** for any **linear combination** of sequence-phentoypes, including mutational effects and epistatic coefficients across different backgrounds.
- **Minimum epistasis interpolation** of complete genotype-phenotype maps given incomplete data implemenented. 
- **Common interface to inference methods** with main methods `predict`, `make_contrast`, `calc_posterior`

### Changed

- **Refactored project structure** by removing the `"src"` package, simplifying module organization.
- **Enhanced computation efficiency** through optimized **linear operators**, improving performance in key calculations.



