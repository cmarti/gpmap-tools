.. _installation:

Installation Instructions
=========================

We recommend using an new independent environment with python3.8, as used during 
development and testing of gpmap-tools to minimize problems with dependencies. For instance,
one can create and activate a new conda environment as follows: ::

    $ conda create -n gpmap python=3.8
    $ conda activate gpmap

Now you can clone GPMAP-tools from `GitHub <https://github.com/cmarti/gpmap-tools>`_ as follows: ::

    $ git clone https://github.com/cmarti/gpmap-tools.git

and install it in the current python environment: ::
    
    $ cd gpmap-tools
    $ pip install .

An older version of gpmap-tools is availabel in PyPI and installable through ``pip`` package
manager, but we recommend using the most recent version from GitHub so far: ::

    $ pip install gpmap-tools

While this should install all required dependencies automatically, there are sometimes
problems with the installation of `datashader <https://datashader.org/>` and their own
dependencies. We are still trying to figure out incompatibilities
in the dependencies but generally we find that installing it first seems to work: ::
    
    $ pip install datashader==0.13

For using the last version of the library install from 
`BitBucket <https://bitbucket.org/cmartiga/gpmap_tools>`_ or
 `GitHub <https://github.com/cmarti/gpmap-tools>`_ and switch to
the `dev` branch. To test installation is working properly you can run all tests or a
subset of them. Running all of them may take some time. ::

    $ python -m unittest gpmap/test/*py

