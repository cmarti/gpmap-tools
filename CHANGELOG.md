# CHANGELOG

All notable changes to this project will be documented in this file.


## New release [0.4.2] - 2026-06-05

### Changed

- **Bug fixed**  when running the inference of GB1 landscape from the documentation. Default CG tolerance was reduced in a previous version to a numerical precision that cannot be reached in this dataset. Tests have been added to ensure that GB1 and the 5ss examples from the documentation run as expected.

## New release [0.4.0] - 2026-03-27

### Added

- **New method for inference of genotype-phenotype maps**. `LocalEpistasisRegression` learns which sites tend to interact with each other from the empirical correlations in the data and uses this information to build a prior for Gaussian process inference of complete combinatorial landscapes. 

- **Implementation of the Connectedness model regression**. `ConnectednessModelRegression` learns how much mutations at each site decrease the predictabiligy of other mutations from the empirical correlations in the data and uses this information to build a prior for Gaussian process inference of complete combinatorial landscapes. 

- **New module for computing summary statistics in empirical data**. It consists of a new class `GPDataSummarizer` that provides a high level interface to computation of the empirical covariance for pairs of sequences located at different Hamming distance from each other or differing at every possible combination of sites. These statistics are very useful for understanding the structure and complexity of epistatic interactions and are used for estimating the parameters of the prior via kernel alignment. 

- **New functions for plotting summary statistics**. We implemented new functions to easily plot the diverse summary statistics that can be computed either from the data or from the inferred combinatorial genotype-phenotype maps to characterize the structure of genetic interactions in the posterior distribution.

### Changed

- **Documentation updated** with the new functionality and updated references for published papers describing the software or its applications. 
- 
- **Covariance-distance** statistics can be now computed before or after centering explicitly and take into account the provided experimental error variances. 

## New release [0.3.3] - 2025-11-11

### Added

- **New module for computing summary statistics** of a complete combinatiorial genotype-phenotype map added to the library. It consists of a new class `GPmapSummarizer` that provides a high level interface to computation of the root mean squared epistatic coefficients and variance explained by epistatic interactions of any order and involving any subset of sites. 

### Changed

- **Documentation updated** with new tutorials for inference, statistical analysis, visualization of comparison of genotype-phenotype maps using different types of data for the Shine-Dalgarno fitness landscape, showing some of the analysis presented in the associated manuscript. 

## New release [0.3.2] - 2025-07-21

### Changed

- **Documentation updated** following Arlin Stoltzfus feedback and comments. Motivation and history of the package
added to the introductory page together with links to other studies using gpmap-tools and online talks. Getting started
section shows the basic tasks one can do with gpmap-tools. Installation instructions simplified to use pip alone for users and download github repository and test for developers. Docstrings updated for completeness for API in rtd. 

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



