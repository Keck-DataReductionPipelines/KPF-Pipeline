KPF DRP Overview
================

The KPF DRP processes raw KPF spectra and associated data products stored in the Level 0 (L0) files into reduced spectra (L1) and radial velocities and other final data products associated with individual observations (L2).  The KPF DRP operates within the `WMKO DRP Framework <https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_ using recipes to specify the algorithms applied and config files to specify parameters associated with the processing.  A summary of the main recipe used to process KPF data products is below.   

Main Recipe
-----------

Level 0 to 2D
^^^^^^^^^^^^^

* Subtract overscan
* Subtract maser bias
* Subtract scaled master dark
* Apply master flat
* Apply bad-pixel mask


2D to Level 1
^^^^^^^^^^^^^

* Step 1
* Step 2
* Step 3
* ...


Level 1 to Level 2
^^^^^^^^^^^^^^^^^^

* Step 1
* Step 2
* Step 3
* ...


The DRP also creates 'master' files that are stacks of particular observations of a particular type (e.g., darks, bias, flats).  The DRP also has a set of 'quick-look' recipes to produce diagnostic plots and measurements.

.. |date| date::

Last Updated on |date|
