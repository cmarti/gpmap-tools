.. _installation:

Installation
============

We recommend using an new independent environment with python3.8, as used during 
development and testing of gpmap-tools to minimize problems with dependencies. For instance,
one can create and activate a new conda environment as follows: ::

    $ conda create -n gpmap python=3.8
    $ conda activate gpmap

gpmap-tools is available in PyPI and installable through ``pip`` package manager: ::

    $ pip install gpmap-tools

You can also install the latest or specific versions from `GitHub <https://github.com/cmarti/gpmap-tools>`_ as follows: ::

    $ git clone https://github.com/cmarti/gpmap-tools.git

and install it in the current python environment: ::
    
    $ cd gpmap-tools
    $ pip install .

For developers, tests can be run with using ``pytest``: ::

    $ pytest test

