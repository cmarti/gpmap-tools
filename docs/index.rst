=======================================================================================
GPMAP-tools: tools for visualization and inference of large complete fitness landscapes
=======================================================================================

GPMAP-tools [#Marti-Gomez2022]_ is a python library to create low dimensional visualizations for large fitness landscapes or the structure of any sequence-function relationship for large genotypic spaces, up to several millions. It provides an open-source and easy to use interface of a slightly modified version of the original method [#McCandlish2011]_. Briefly, genotypes are embedded into a low dimensional representation such that the square distances are proportional to the required time to evolve from one genotype to another under a Weak Mutation Weak Selection evolutionary model.

.. [#Marti-Gomez2022] Marti-Gomez C, McCandlish DM.
    GPMAP-tools: a python library for visualizing large fitness landscapes
    Biorxiv (2022). `<https://doi.org/10.1101/2020.07.14.201475>`_
.. [#McCandlish2011] McCandlish DM.
    Visualizing fitness landscapes
    Evolution (2011). `<https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.2011.01236.x>`

GPMAP-tools is written for Python 3 and is provided under an MIT open source license. The documentation provided here is meant guide users through the basic principles underlying the method as well as explain how to use it for calculating the embedding coordinates and use the functionalities provided for advanced plotting of their own fitness landscapes. Please do not hesitate to contact us with any questions or suggestions for improvements. For technical assistance or to report bugs, please contact Carlos Marti (`Email: martigo@cshl.edu <martigo@cshl.edu>`_, `Twitter: @cmarti_ga <https://twitter.com/cmarti_ga>`_) . For more general correspondence, please contact David McCandlish (`Email: mccandlish@cshl.edu <mccandlish@cshl.edu>`_, `Twitter: @TheDMMcC <https://twitter.com/TheDMMcC>`_).

.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    installation
    usage 

Links
==============

- `McCandlish Lab <https://www.cshl.edu/research/faculty-staff/david-mccandlish/#research>`_
