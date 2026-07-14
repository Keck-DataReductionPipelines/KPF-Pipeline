==========
Quickstart
==========

To install and run the vNext pipeline, follow the quickstart guide below.


Installation
============

The vNext pipeline runs in a dedicated conda environment (``kpfpipe``, Python 3.14).
Clone the repository, create the environment, and install the package in
editable mode:

.. code-block:: bash

   git clone -b kpf-next https://github.com/Keck-DataReductionPipelines/KPF-Pipeline.git
   cd KPF-Pipeline
   conda env create -f environment.yml
   conda activate kpfpipe
   pip install -e .


Running the DRP
===============

The easiest way to run the pipeline is via the command line interface (CLI).

The ``kpfpipe`` command has two main entry points: ``masters`` builds nightly
calibration products (bias, dark, flat, wavelength solution), and ``science``
reduces science exposures into RVs. Each takes input/output data directories
and a datecode (masters) or obs_id (science) to act on.

Input/output directories specify the root directory for the data files:

* Input L0 data files should be placed in {kpf_data_input}/L0/
* Reduced L1/L2/L4 science files will output to {kpf_science_output}/L{n}
* Masters files will output to {kpf_masters_output}/masters

Masters
-------

Build master calibrations for one or more nights (identified by a ``YYYYMMDD``
datecode) directly in the CLI:

.. code-block:: bash

   kpfpipe masters \
       --kpf_data_input /path/to/data \
       --kpf_masters_output /path/to/masters/output \
       --dates 20240405 20250912 20250111

Or by passing a plain text file which lists a single datecode per line:

.. code-block:: bash

   kpfpipe masters \
       --kpf_data_input /path/to/data \
       --kpf_masters_output /path/to/masters/output \
       --dates /path/to/datecodes.txt

You can also generate masters over a range of nights:

.. code-block:: bash

   kpfpipe masters \
       --kpf_data_input /path/to/data \
       --kpf_masters_output /path/to/masters/output \
       --date_range 20240405 20240418


Science
-------

Reduce one or more science frames (identified by their obs_ids):

.. code-block:: bash

   kpfpipe science \
       --kpf_data_input /path/to/data \
       --kpf_science_output /path/to/science/output \
       --kpf_masters_output /path/to/masters/output \
       --obs_ids KP.20240405.40113.57 KP.20240912.84491.73

Or by passing a plain text file which lists a single obs_id per line:

.. code-block:: bash

   kpfpipe science \
       --kpf_data_input /path/to/data \
       --kpf_science_output /path/to/science/output \
       --kpf_masters_output /path/to/masters/output \
       --obs_ids /path/to/obs_ids.txt

Make sure that ``--kpf_masters_output`` specifies the same directory you used
to run the masters pipeline so the science pipeline knows where to look for
masters calibrations.


Timeseries
----------

To conveniently process all masters and science for a single star over a
given date range, use the timeseries command:

.. code-block:: bash

   kpfpipe timeseries \
       --target 10700 \
       --kpf_data_input /path/to/data \
       --kpf_science_output /path/to/science/output \
       --kpf_masters_output /path/to/masters/output \
       --date_range 20240405 20240418
