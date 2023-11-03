.. _installation:

Installation Instructions
=========================

Soon but not yet, using the ``pip`` package manager by executing the following at the
command line: ::

    $ pip install gpmap-tools

Alternatively, you can clone GPMAP-tools from
`BitBucket <https://bitbucket.org/cmartiga/gpmap_tools/src/master/>`_ by doing
this at the command line: ::

    $ git clone https://cmartiga@bitbucket.org/cmartiga/gpmap_tools.git

and install it in the current python environment: ::
    
    $ python setup.py install

There are sometimes problems with the installation of `datashader <https://datashader.org/>` 
and their own dependencies. We are still trying to figure out incompatibilities
in the dependencies but generally we find that installing it first seems to work: ::
    
    $ pip install datashader==0.13

For using the last version of the library install from bitbucket or git and switch to the `dev` branch

