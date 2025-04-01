# gpmap-tools: tools for inference and visualization of complex genotype-phenotype maps

gpmap-tools is a python library containing a suite of tools for inference and visualization of large and complex genotype-phenotype maps for genotypic spaces containing up to several million genotypes. 

gpmap-tools uses Gaussian process models to infer complete genotype-phenotype maps in the presence of genetic interactions of every possible order from high throughput experimental data and observations of natural sequences. gpmap-tools enables the statistical analysis of features of these complex objects by handling the high dimensional posterior distribution over the set of genotype-phenotype maps. 

gpmap-tools provides an open-source and easy to use interface of a slightly
modified version of the original method. Briefly, genotypes are embedded into a 
low dimensional coordinate system where the square distances are proportional to the required time to evolve from one genotype to another under a Weak Mutation evolutionary model. This technique highlights regions of sequence space that are highly inaccessible to each other and display the main qualitative features of genotype-phenotype maps. 

gpmap-tools is written for Python 3 and is provided under an MIT open source license.
The documentation provided [here](https://gpmap-tools.readthedocs.io) is meant guide users through the basic principles underlying the method as well as explain how to use it for calculating the embedding coordinates and use the functionalities provided for advanced plotting of their own fitness landscapes. 

Please do not hesitate to contact us with any questions or suggestions for improvements.

- For technical assistance or to report bugs, please contact Carlos Martí-Gómez (<martigo@cshl.edu>)
- For more general correspondence, please contact David M. McCandlish (<mccandlish@cshl.edu>)


## Installation

### Create a new environment and activate it

The library has only been tested with specific versions of the dependent libraries, so we recommend the same to be used to avoid potential incompatibilities.

```
conda create -n gpmap python=3.8
conda activate gpmap
```

### Users: install from PyPI

```
pip install gpmap-tools
```

### Developers: from from repository

Download the repository using git and cd into it

```bash
git clone https://github.com/cmarti/gpmap-tools.git
cd gpmap_tools
pip install .
```

Run tests using pytest

```bash
pytest test
```

# References
- Martí-Gómez, C.; Zhou, J.; Chen W.; Kinney J. B.; Mccandlish, D. M. Inference and visualization of complex genotype-phenotype maps using gpmap-tools. bioRxiv (2025). [doi](https://www.biorxiv.org/content/10.1101/2025.03.09.642267v2)
- Zhou, J.; McCandlish D. M.; Minimum epistasis interpolation for sequence-function relationships. Nat. Comm. (2020) [doi](https://www.nature.com/articles/s41467-020-15512-5)
- Zhou, J.; Wong, M. S.; Chen, W.; Krainer, A. R.; Kinney J. B.; Mccandlish, D. M. Higher-Order Epistasis and Phenotypic Prediction. PNAS (2022). [doi](https://doi.org/10.1073/pnas.2204233119).
- Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. Field theoretic density estimation for biological sequence space with applications to 5' splice site diversity and aneuploidy in cancer. PNAS (2021). [doi](https://www.pnas.org/doi/10.1073/pnas.2025782118)
- Mccandlish, D. M. Visualizing fitness landscapes. Evolution (2011). [doi](https://doi.org/10.1111/j.1558-5646.2011.01236.x)
