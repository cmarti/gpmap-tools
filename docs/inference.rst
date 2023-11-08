.. _inference:

Inference of complex landscapes
===============================

In the previous section we illustrate a couple of examples of how to visualize a complete combinatorial landscape when it is perfectly known. However, we usually only have access to noisy or indirect measurements of such genotype-genotype maps and not all possible sequences are evaluate, especially when dealing with high-throughput experiments. In this section the user will learn how to use our methods to perform inference of complete combinatorial landsapces from experimental data and observations of natural equences using our exact Gaussian Process regression techniques.

.. toctree::
    :maxdepth: 1

    usage/3_VCregression.ipynb
    usage/4_SeqDEFT.ipynb

Read more
---------
.. [#Zhou2022] Zhou J, Wong MS, Chen WC, Krainer AR, Kinney JB, McCandlsih DM.
    Higher order epistasis and phenotypic prediction.
    Biorxiv (2022) <https://www.biorxiv.org/content/10.1101/2020.10.14.339804v3>_

.. [#Chen2021] Chen WC, Zhou J, Sheltzer JM, Kinney JB, McCandlish DM. 
    Field theoretic density estimation for biological sequence space with
    applications to 5' splice site diversity and aneuploidy in cancer. 
    PNAS (2021) <https://www.pnas.org/doi/10.1073/pnas.2025782118>_
