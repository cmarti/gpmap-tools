.. _inference:

Inference of complex landscapes
===============================

In the previous section, we illustrate of how to visualize a complete
combinatorial landscape when it is perfectly known. However, we usually only have access 
to noisy or indirect measurements of such genotype-genotype maps. Moreover, not all possible sequences 
are evaluated, especially when dealing with high-throughput experiments. In this section, the user 
will learn how to use our methods to estimate complete combinatorial landsapces from both
experimental data [#Zhou2022]_ and observations of natural sequences [#Chen2021]_ using our exact
Gaussian Process regression techniques.

.. toctree::
    :maxdepth: 1

    usage/3_VCregression.ipynb
    usage/4_SeqDEFT.ipynb

Read more
"""""""""

.. [#Zhou2022] `Zhou J, Wong MS, Chen WC, Krainer AR, Kinney JB, McCandlsih DM.
    Higher order epistasis and phenotypic prediction (2022) <https://www.pnas.org/doi/full/10.1073/pnas.2204233119>`_

.. [#Chen2021] `Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. 
    Field theoretic density estimation for biological sequence space with
    applications to 5' splice site diversity and aneuploidy in cancer (2021) <https://www.pnas.org/doi/10.1073/pnas.2025782118>`_
