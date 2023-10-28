KPF DRP Overview
================

The KPF DRP processes raw KPF spectra and associated data products stored in the Level 0 (L0) files into reduced spectra (L1) and radial velocities and other final data products associated with individual observations (L2).  The KPF DRP operates within the `WMKO DRP Framework <https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_ using recipes to specify the algorithms applied and config files to specify parameters associated with the processing.  A summary of the main recipe used to process KPF data products is below.   

Main Recipe
-----------

Level 0 to 2D
^^^^^^^^^^^^^

#. Remove the overscan region from each amplifier region
#. Stitch amplifier regions into a single image (per CCD)
#. Subtract master bias (per CCD)
#. Subtract scaled master dark (per CCD)
#. Apply master flat (per CCD)
#. Apply bad-pixel mask (per CCD)


2D to Level 1
^^^^^^^^^^^^^

#. Spectral Extraction

   #. <part 1>

   #. <part 2>

   #. <part 3>

#. ... <more>

#. Ca H&K Spectrometer Spectra

   #. Subtract master bias

   #. Subtract scaled master dark

   #. Spectral Extraction

   #. Apply default wavelength solution to extracted spectra


Level 1 to Level 2
^^^^^^^^^^^^^^^^^^

#. Compute Cross-correlation functions (CCFs) with binary masks
#. Compute RVs per order and per orderlet from fitting CCF peaks
#. Compute reweighted RVs based on information content per order
#. <more ?>


Wavelength Solution Recipes
---------------------------

ThAr
^^^^

#. Step 1
#. Step 2
#. Step 3
#. ...

LFC
^^^^

#. Step 1
#. Step 2
#. Step 3
#. ...

Etalon
^^^^

#. Step 1
#. Step 2
#. Step 3
#. ...


Master Construction Recipes
---------------------------

The DRP also creates 'master' files that are stacks of particular observations of a particular type (e.g., darks, bias, flats).  The DRP also has a set of 'quick-look' recipes to produce diagnostic plots and measurements.

.. |date| date::

Last Updated on |date|
