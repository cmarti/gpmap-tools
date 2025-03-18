# CHANGELOG

All notable changes to this project will be documented in this file.

## First stable release [0.3.0] - 2025-03-14

### Added

- **Independent plotting modules** for the three backends: `matplotlib`, `plotly`, and `datashader`, ensuring a **consistent interface** across all backends.
- **Computation of posterior variances** for any **linear combination** of sequence-phentoypes, including mutational effects and epistatic coefficients across different backgrounds.
- **Minimum epistasis interpolation** of complete genotype-phenotype maps given incomplete data implemenented. 
- **Common interface to inference methods** with main methods `predict`, `make_contrast`, `calc_posterior`

### Changed

- **Refactored project structure** by removing the `"src"` package, simplifying module organization.
- **Enhanced computation efficiency** through optimized **linear operators**, improving performance in key calculations.



