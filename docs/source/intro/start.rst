Getting Started with the KPF DRP
================================

This section contains topics about getting started with using 
the ``KPF-Pipeline`` module.

First, download and install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_ if you don't already have it.

Then clone the repository and navigate into it::

    git clone https://github.com/Keck-DataReductionPipelines/KPF-Pipeline.git
    cd KPF-Pipeline

.. warning:: Refer to :doc:`install_develop` for setting up other branches

Define the ``KPFPIPE_DATA`` environment variable and point it to a location where you want to store the input and ouput files.
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

The two mandatory input arguments to ``kpf`` are: a ``recipe`` file and a
``config`` configuration file. The recipe file is expected to be a ``.recipe`` script,
while the ``config`` file is expected to be a ``.cfg`` file. See :doc:`example_trivial`
for a basic example and :doc:`process_night` to process a night of data.

To start a notebook server run the following and follow the on-screen instructions::

    make notebook

Finally, some of the recipes interact with a PostgresSQL database,
which is the so-called *pipeline operations database*.  Refer to :doc:`database_setup`
for instructions on how to initially set up the pipeline operations database.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    install_develop.rst
    example_trivial.rst
    process_night.rst
    processscripts.rst
    logging.rst
    config.rst
    database_setup.rst
