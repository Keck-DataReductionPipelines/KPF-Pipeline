Getting Started
===============

This section contains topics about getting started with using 
the ``KPF-Pipeline`` module.

First, download and install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_ if you don't already have it.

Then clone the repository and navigate into it::

    git clone https://github.com/California-Planet-Search/KPF-Pipeline.git
    cd KPF-Pipeline

.. warning:: Refer to :doc:`install_develop` for setting up other branches

Build the package into a docker container::
    
    docker build --cache-from kpf-drp:latest --tag kpf-drp:latest .

Launch an interactive bash shell in the docker container::

    docker run -it -v $OWNCLOUD_BASE/KPF-Pipeline-TestData:/data kpf-drp:latest bash
    
where ``$OWNCLOUD_BASE`` is the base path of your local ownCloud directory.

Install the package once the container launches::

    make init

To run the pipeline, use the following command::

    kpf

This will prompt the use case message, which should be::

    usage: kpf [-h] recipe config_file
    kpf: error: the following arguments are required: recipe, config_file

The two mandatory input arguments to ``kpf`` is: a ``recipe`` file and a
``config`` configuration file. The recipe file is expected to be a ``.recipe`` script,
while the ``config`` file is expected to be a ``.cfg`` file. See :doc:`example_trivial`
for a basic example.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    install_develop.rst
    example_trivial.rst
    logging.rst
    config.rst
