# Visualizing fitness landscapes

GPMAP-tools is a python library to create low dimensional visualizations for large fitness landscapes
or the structure of any sequence-function relationship for large genotypic spaces, up to several millions. 
It provides an open-source and easy to use interface of a slightly modified version of the original method. 
Briefly, genotypes are embedded into a low dimensional representation such that the square distances are 
proportional to the required time to evolve from one genotype to another under a Weak Mutation Weak Selection 
evolutionary model.


The main limitation for the applicability of these method for visualizing a landscape is that we need to have 
a full characterization of the funcion associated to each possible genotype in the sequence space. As the 
total number of genotypes scales exponentially with sequence length, this method is inherently limited to 
the study of relatively short sequences. Another limitation that derives from the difficulty to experimentally 
assess every possible sequence in a high throughput experiment reliably. To tackle this issue, we provide also 
a simple interface to previously published methods for inference of quantitative sequence-function
relationships in presence of complex epistatic interactions, either from high throughput experiments 
or from the number of times a sequences has been observed in nature .

GPMAP-tools is written for Python 3 and is provided under an MIT open source license. The documentation 
provided [here](https://gpmap-tools.readthedocs.io) is meant guide users through the basic principles 
underlying the method as well as explain how to use it for calculating the embedding coordinates and use 
the functionalities provided for advanced plotting of their own fitness landscapes. Please do not hesitate 
to contact us with any questions or suggestions for improvements.

- For technical assistance or to report bugs, please contact Carlos Martí-Gómez(<martigo@cshl.edu>
- For more general correspondence, please contact David M. McCandlish<mccandlish@cshl.edu>


## Installation

### Create a new environment and activate it

The library has only been tested with specific versions of the dependent libraries, so we recommend the same to be used to avoid potential incompatibilities.

```
conda create -n gpmap python=3.8.13
conda activate gpmap
```

### From PyPI (old version only)

```
pip install gpmap-tools
```

### From latest version in repository

Download the repository using git and cd into it

```bash
git clone https://github.com/cmarti/gpmap-tools.git
```

Install using setuptools
```bash
cd gpmap
python setup.py install
```

### Test installation

```bash
python -m unittest gpmap/test/*py
```

# Cite
- Zhou, J.; Wong, M. S.; Chen, W.; Krainer, A. R.; Justin, B.; Mccandlish, D. M. Higher-Order Epistasis and Phenotypic Prediction. PNAS (2022). [doi](https://doi.org/10.1073/pnas.2204233119).
- Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. Field theoretic density estimation for biological sequence space with applications to 5' splice site diversity and aneuploidy in cancer. PNAS (2021). [doi](https://www.pnas.org/doi/10.1073/pnas.2025782118)
- Mccandlish, D. M. Visualizing fitness landscapes. Evolution (2011). [doi](https://doi.org/10.1111/j.1558-5646.2011.01236.x)
