Getting Started
===============

This section contains topics about getting started with using 
the ``KPF-Pipeline`` module.

To install, clone the repository and navigate into it::

    git clone https://github.com/California-Planet-Search/KPF-Pipeline.git

One of the requirements of ``KPF-Pipe`` is the ``KeckDRPFramework``
package. This package is not on PyPl, so we will install in from source.
The current pipelineis setup such that the ``KeckDRPFramework``
repository is expected to be at thesame directory level as ``KPF-Pipeline``.
In other words, if you cloned '`KPF-Pipeline`' into a directory
named ``MyProject``, your directory structure
is expected to look like::

    MyProject
    ├── KPF-Pipeline
    └── KeckDRPFramework

Refer to `KeckDRPFramework
<https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_
for Framework related issues.

If you plan on building from branch ``master`` then simply
use the provided ``makefile`` for package installation and setup::

    cd KPF-Pipeline
    make

.. warning:: Refer to :doc:`install_develop` for setting up other branches

To run the pipeline, use the following command::

    kpf

This will prompt the use case message, which should be::

    usage: kpf [-h] recipe config_file
    kpf: error: the following arguments are required: recipe, config_file

The two mandatory input arguments to ``kpf`` is: a ``recipe`` file and a
``config`` configuration file. The recipe file is expected to be a ``.py`` script,
while the ``config`` file is expected to be a ``.cfg`` file. See :doc:`example_trivial`
for a basic example.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    install_develop.rst
    example_trivial.rst
