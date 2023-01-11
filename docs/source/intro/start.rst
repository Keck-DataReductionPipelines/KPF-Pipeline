Getting Started
===============

This section contains topics about getting started with using 
the ``KPF-Pipeline`` module.

First, download and install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_ if you don't already have it.

Obtain a copy of the test datasets stored in ownCloud. Download and install the `ownCloud desktop client  <https://owncloud.com/desktop-app/>`_ and direct it to `<http://shrek.caltech.edu:5555>`_.

Then clone the repository and navigate into it::

    git clone https://github.com/California-Planet-Search/KPF-Pipeline.git
    cd KPF-Pipeline

.. warning:: Refer to :doc:`install_develop` for setting up other branches

Define the ``KPFPIPE_TEST_DATA`` environment variable and point it to the ``KPF-Pipeline-TestData`` directory inside your copy of the ownCloud directory.
Define the ``KPFPIPE_DATA`` environment variable and point it to the incoming data directory.
If you would like to work in Jupyter notebooks then also define the ``KPFPIPE_PORT`` environment variable to assign a port to use for the notebook server.

Build the package into a docker container and launch an interactive bash shell::
    
    make docker

Install the package once the container launches::

    make init

To run the pipeline, use the following command::

    kpf

This will prompt the use case message, which should be::

    usage: kpf [-h] -r recipe -c config_file
    kpf: error: the following arguments are required: recipe, config_file

The two mandatory input arguments to ``kpf`` is: a ``recipe`` file and a
``config`` configuration file. The recipe file is expected to be a ``.recipe`` script,
while the ``config`` file is expected to be a ``.cfg`` file. See :doc:`example_trivial`
for a basic example.

To start a notebook server run the following and follow the on-screen instructions::

    make notebook


.. toctree::
    :maxdepth: 1
    :titlesonly:

    install_develop.rst
    example_trivial.rst
    logging.rst
    config.rst
