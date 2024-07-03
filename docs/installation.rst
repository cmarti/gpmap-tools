.. _installation:

Installation Instructions
=========================

Soon but not yet, using the ``pip`` package manager by executing the following at the
command line: ::

    $ pip install gpmap-tools

Alternatively, you can clone GPMAP-tools from
`GitHub <https://github.com/cmarti/gpmap-tools>`_ as follows: ::

    $ git clone https://github.com/cmarti/gpmap-tools.git

and install it in the current python environment: ::
    
    $ python setup.py install

or: ::

    $ pip install .

While this should install all required dependencies automatically, there are sometimes
problems with the installation of `datashader <https://datashader.org/>` and their own
dependencies. We are still trying to figure out incompatibilities
in the dependencies but generally we find that installing it first seems to work: ::
    
    $ pip install datashader==0.13

For using the last version of the library install from bitbucket or github and switch to
the `dev` branch. To test installation is working properly you can run all tests or a
subset of them. Running all of them may take some time. ::

    $ python -m unittest gpmap/test/*py

