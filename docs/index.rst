=====================================================================================
gpmap-tools: tools for inference and visualization of complex genotype-phenotype maps
=====================================================================================

gpmap-tools is a python library containing a suite of tools for inference and visualization of
large and complex genotype-phenotype maps for genotypic spaces containing 
up to several million genotypes [#gpmap]_. 

gpmap-tools uses Gaussian process models to infer complete genotype-phenotype maps in the presence
of genetic interactions of every possible order from high throughput experimental data [#mem]_ [#vc]_ and observations of
natural sequences [#seqdeft]_. gpmap-tools enables the statistical analysis of features
of these complex objects by handling the high dimensional posterior distribution over
the set of genotype-phenotype maps. 

gpmap-tools provides an open-source and easy to use interface of a slightly
modified version of the original method [#viz]_. Briefly, genotypes are embedded into a 
low dimensional coordinate system where the square distances are proportional to the required time
to evolve from one genotype to another under a Weak Mutation evolutionary model. This technique highlights
regions of sequence space that are highly inaccessible to each other and display the main qualitative
features of genotype-phenotype maps. 

gpmap-tools is written for Python 3 and is provided under an MIT open source license.
The documentation provided here is meant guide users through the basic principles underlying 
the method as well as explain how to use it for calculating the embedding coordinates and use
the functionalities provided for advanced plotting of their own fitness landscapes. 

Please do not hesitate to contact us with any questions or suggestions for improvements.

* For technical assistance or to report bugs, please contact Carlos Marti (`Email: martigo@cshl.edu <martigo@cshl.edu>`_).

* For more general correspondence, please contact David McCandlish (`Email: mccandlish@cshl.edu <mccandlish@cshl.edu>`_).


.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    installation
    quickstart
    inference
    visualization
    datasets
    api 

References
==========

.. [#gpmap] `Martí-Gómez C, Zhou J, Chen WC, Kinney JB, McCandlish DM.
   Inference and visualization of complex genotype-phenotype maps with
   gpmap-tools (2025) <https://www.biorxiv.org/content/10.1101/2025.03.09.642267v1>`_

.. [#viz] `McCandlish DM.
    Visualizing fitness landscapes (2011)
    <https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.2011.01236.x>`_

.. [#mem] `Zhou J and McCandlsih DM.
    Minimum epistasis interpolation for sequence-function relationships (2020)
    <https://www.nature.com/articles/s41467-020-15512-5>`_

.. [#vc] `Zhou J, Wong MS, Chen WC, Krainer AR, Kinney JB, McCandlsih DM.
    Higher order epistasis and phenotypic prediction (2022) 
    <https://www.pnas.org/doi/full/10.1073/pnas.2204233119>`_

.. [#seqdeft] `Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. 
    Field theoretic density estimation for biological sequence space with
    applications to 5' splice site diversity and aneuploidy in cancer (2021) 
    <https://www.pnas.org/doi/10.1073/pnas.2025782118>`_


Links
=====

- `McCandlish Lab <https://www.cshl.edu/research/faculty-staff/david-mccandlish/#research>`_
