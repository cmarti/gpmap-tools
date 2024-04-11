=======================================================================================
GPMAP-tools: tools for visualization and inference of large complete fitness landscapes
=======================================================================================

GPMAP-tools is a python library to create low dimensional visualizations for large fitness
landscapes or the structure of any sequence-function relationship for genotypic spaces containing 
up to several million genotypes. It provides an open-source and easy to use interface of a slightly
modified version of the original method [#McCandlish2011]_. Briefly, genotypes are embedded into a 
low dimensional coordinate system where the square distances are proportional to the required time
to evolve from one genotype to another under a Weak Mutation evolutionary model.

The main limitation of this method for visualizing a landscape is that we need to know the
phenotype associated to every possible sequence in the space. As the total number of genotypes
scales exponentially with sequence length, this method is inherently limited to the study of
relatively short sequences.

When studying empirical landscapes, it is often challenging to experimentally measure
the phenotype of every possible sequence of a certain length in a high throughput experiment.
These experiments indeed often consist in rather incomplete and noisy data. To tackle this issue,
we provide a simple interface to our previously published Gaussian Process regression methods for
inference of complete quantitative sequence-function relationships in presence of complex epistatic
interactions. This library contains methods to perform such inference using both high throughput
experiments [#mem]_,[#vc]_ or observations of natural sequences e.g. ocurrences of 
the 5´splice site sequence in the human genome [#Chen2021]_.

GPMAP-tools is written for Python 3 and is provided under an MIT open source license.
The documentation provided here is meant guide users through the basic principles underlying 
the method as well as explain how to use it for calculating the embedding coordinates and use
the functionalities provided for advanced plotting of their own fitness landscapes. 


Please do not hesitate to contact us with any questions or suggestions for improvements.

* For technical assistance or to report bugs,
please contact Carlos Marti (`Email: martigo@cshl.edu <martigo@cshl.edu>`_, `Twitter: @cmarti_ga <https://twitter.com/cmarti_ga>`_).

* For more general correspondence,
please contact David McCandlish (`Email: mccandlish@cshl.edu <mccandlish@cshl.edu>`_, `Twitter: @TheDMMcC <https://twitter.com/TheDMMcC>`_).


.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    installation
    quickstart
    inference
    visualization
    evolution
    datasets
    api 

References
""""""""""

.. [#McCandlish2011] `McCandlish DM.
    Visualizing fitness landscapes (2011)
    <https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.2011.01236.x>`_

.. [#mem] `Zhou J and McCandlsih DM.
    Minimum epistasis interpolation for sequence-function relationships (2020)
    <https://www.nature.com/articles/s41467-020-15512-5>`_

.. [#vc] `Zhou J, Wong MS, Chen WC, Krainer AR, Kinney JB, McCandlsih DM.
    Higher order epistasis and phenotypic prediction (2022) 
    <https://www.pnas.org/doi/full/10.1073/pnas.2204233119>`_

.. [#Chen2021] `Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. 
    Field theoretic density estimation for biological sequence space with
    applications to 5' splice site diversity and aneuploidy in cancer (2021) 
    <https://www.pnas.org/doi/10.1073/pnas.2025782118>`_


Links
==============

- `McCandlish Lab <https://www.cshl.edu/research/faculty-staff/david-mccandlish/#research>`_
