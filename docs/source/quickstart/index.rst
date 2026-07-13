Quickstart
==========

Get the KPF-DRP installed and run a first reduction.

Installation
------------

The pipeline runs in a dedicated conda environment (``kpfpipe``, Python 3.14).
Clone the repository, create the environment, and install the package in
editable mode:

.. code-block:: bash

   git clone https://github.com/Keck-DataReductionPipelines/KPF-Pipeline.git
   cd KPF-Pipeline
   conda env create -f environment.yml
   conda activate kpfpipe
   pip install -e .

Running the DRP
---------------

The ``kpfpipe`` command has two main entry points: ``masters`` builds nightly
calibration products (bias, dark, flat, wavelength solution), and ``science``
reduces science exposures into RVs. Both take an input data directory, an output
directory, and the units to process.

Build the master calibrations for a night (identified by its ``YYYYMMDD``
datecode):

.. code-block:: bash

   kpfpipe masters \
       --input_dir /path/to/data \
       --output_dir /path/to/output \
       --dates 20240405

Reduce one or more science frames (identified by their obs_ids):

.. code-block:: bash

   kpfpipe science \
       --input_dir /path/to/data \
       --output_dir /path/to/output \
       --obs_ids KP.20240405.40113.57
