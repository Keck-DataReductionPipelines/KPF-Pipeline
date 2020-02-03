Repository
==========
This section contains topic on the ``KPF-Pipeline`` repository.

The current repository is organized into the following Structure::

    KPF-Pipe
    ├── kpfpipe
    │   ├── models 
    |   |   └── ... (module implementation)
    │   ├── primitives 
    |   |   └── ... 
    │   ├── tools 
    |   |   └── ... 
    │   └── pipelines
    |       └── ...
    ├── modules
    │   ├── module 1
    |   |   └── ...
    │   └── module2
    |       └── ...
    ├── makefile
    ├── requirement.txt
    └── examples
        └── ...

The ``kpfpipe`` directory contains source code to the pipeline, and
the ``modules`` directory contains implementation of KPF modules. 
See :doc:`pipeline` and :doc:`modules` for more details. 

The ``makefile`` is used for general project management. The currently
supported command are::

    make init       # build package
    make update     # upgrade package dependencies
    make clean      # purge current build
    make clear      # purge all .log files
    make test       # run all test with pytest

..note:: running ``make`` will have the same effect as running ``make init``

As this package uses ``pip`` for setup, a ``requirement.txt`` is required to
contain all dependencies of the package. This include dependencies of the
pipeline and all modules.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    modules.rst
    pipeline.rst