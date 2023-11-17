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

   #. Map the orderlet locations using a rectified master flat.

   #. Extract each order and orderlet using a weighted sum. The weights are determined from the rectified flat.

   #. Reject outliers by comparing the cross-dispersion profile in each column to that expected in a scaled master flat.

   #. Subtract background by measuring the inter-order light.

   #. Copy wavelength solution from the day's master ThAr or LFC frames.

   #. Calculate photon-weighted midpoint for each order from the exposure meter spectral timeseries.

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
#. Compute and correct for the barycentric RV using the photon-weighted midpoints.


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

