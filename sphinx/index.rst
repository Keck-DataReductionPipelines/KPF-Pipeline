.. KPFpipe documentation master file, created by
   sphinx-quickstart on Tue Oct 29 09:44:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the KPF Data Reduction Pipeline
=================================================


Installation
++++++++++++

Install ``rvsearch`` directly from the
`GitHub repository <https://github.com/California-Planet-Search/rvsearch>`_
using pip:

.. code-block:: bash

    $ pip install git+https://github.com/California-Planet-Search/KPF-Pipeline

Please report any bugs or feature requests to the
`GitHub issue tracker <https://github.com/California-Planet-Search/KPF-Pipeline>`_.


Quickstart
++++++++++

Check out the features available in the command-line-interface:

.. code-block:: bash

   $ kpf --help
   usage: kpf [-h] [-r RECIPE_FILE] [-c CONFIG_FILE] [-o DIRNAME]
              [--pipeline PIPELINE_NAME]

   KPF Pipeline CLI

   optional arguments:
     -h, --help            show this help message and exit
     -r RECIPE_FILE        Recipe file with list of actions to take.
     -c CONFIG_FILE        Run Configuration file
     -o DIRNAME, --output_dir DIRNAME
                           Output directory
     --pipeline PIPELINE_NAME
                           Name of the pipeline class

Example calling syntax:

.. code-block:: bash

   $ kpf -c run.cfg -r recipe.txt

The recipe file is required and lists actions to take.
Each line must correspond to a primitive class availble in the Pipeline namespace.

A run configuration file is also required which specifies things like the list of files to process
and their processing levels.

See the example recipe and configuration file in the `sample_run` directory.


API
+++

.. toctree::
   :maxdepth: 3

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
