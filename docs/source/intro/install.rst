Install
=======

To begin, clone the repository and navigate into it::

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

    cd KPF-Pipe
    make

.. warning:: Refer to :doc:`install_develop` for setting up other branches



