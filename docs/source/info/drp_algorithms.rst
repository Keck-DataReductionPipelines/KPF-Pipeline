KPF DRP Algorithms
==================

Please refer to the `KPF-Pipeline GitHub Repository <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_
for any source code referred to below.

Overview
--------
The target audience here is astronomers who will use the data.
The description of each algorithm should include:

* a brief description of how the algorithm works
* performance estimates, if appropriate
* any caveats, if needed
* development status (if partially implemented or not yet implemented)


Current Limitations and Development Plans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

<Add a list of current limitations and plans to address them>

Include:

* Doppler stability on timescales longer than 1 day
* consistency of wavelength solutions
* intranight drift corrections for wavelength solutions
* scattered light correction
* outlier rejection in spectral extraction
* sky subtraction
* telluric corrections
* Stellar activity indicators (see below)


CCD Image Processing
--------------------

A CCD (charged-coupled device) is a sensor of many pixels that allows
photons to be detected and a digital image to be produced.
The CCD is exposed to light for a certain amount of requested time,
called exposure time (EXPTIME in the FITS PRIMARY header, in seconds).
In the case of KPF, spectroscopic-image data are taken by CCDs with GREEN and RED filters
(and a separate CCD with a Ca H&K line filter),
and these are exposed simultaneously via a beamsplitter.
The CCD image processing consists of several steps, as discussed below.
It starts with an L0 FITS file for a given exposure and ends with a 2D FITS file.
The L0 FITS file contains several HDUs (header-data units) with CCD subimage data from
separate readout amplifiers.
The size in pixels of the subimages depends on the mode in which the instrument operating,
which may be utilizing 2 or 4 amplifiers for a given GREEN or RED filter.
For example, with 2 amplifiers the subimage is 2094x4110 pixels.
The description in this section mainly refers to the GREEN and RED CCDs.

Here are the basic image-processing steps for a given L0 FITS file:

1. Overscan subtraction
2. Mosaicing amplifier subimage data into a full image.
3. Master bias subtraction
4. Master dark subtraction
5. Master flat correction
6. Production of a 2D FITS file for the exposure

Overscan subtraction is a fundamental data-reduction step in CCD image processing.
It is the removal of the bias in the data that is associated
with signal from pixels that are not exposed to light.  By design, the edges of a CCD are masked
off so that there is no light exposure, and so sensor data from these overscan regions
(or so-called bias strips) can be used to compute the overscan bias.
Overscan subtraction is also known as "floating-bias subtraction", which is not to be confused with the
master bias subtraction to be discussed below.
The method of determining the overscan bias is, for a given readout amplifier, to compute the clipped mean of data
well into the overscan strip, avoiding the 4-pixel pre-scan region, with an addition 5-pixel buffer away from
the edges of the overscan strip.  The specific clipped-mean algorithm involves rejecting data that are
+/- some number of sigmas from the median of the data, where sigma is robustly computed by::

    sigma = 0.5 * (p84 - p16)
    p84 = 84th percentile of the data
    p16 = 16th percentile of the data

The mean or average is computed from the remaining data.
The level of data clipping is 2.1 sigma.
The overscan bias, which is just a number for each readout amplifier (for a given filter), is then subtracted from
the image data at each pixel in the unmasked or light-exposed portion of the CCD subimage data for that
readout amplifier.
The CCD subimage is trimmed to completely remove the bias strips that are now superfluous.
With the overscan bias removed, the CCD subimage data are a step closer to a regime that is
linearly proportional to the amount of light exposure.  The python module ``overscan_subtract.py``
under git repository ``KPF-Pipeline/modules/Utils`` handles both overscan subtraction and
mosaicing amplifier subimage data into a full image.  This module is called from subrecipe
``watchfor_kpf_l0.recipe`` of the KPF data reduction pipeline ``kpf_drp.recipe``
under git repository ``KPF-Pipeline/recipes``.  The ``WATCHFOR_L0`` block in configuration file
``kpf_drp.cfg`` under git repository ``KPF-Pipeline/configs`` gives the parameter settings used.

The mosaicing of subimages from different readout amplifiers into a full CCD image for a given filter (GREEN or RED)
is straightforward.  The relative positions of the subimages are described in the following parameter files under
git repository ``KPF-Pipeline/static``: ``kpfsim_ccd_orient_green.txt`` and ``kpfsim_ccd_orient_red.txt``.
At this point the product is in 2D FITS format, but the actual 2D FITS file is not actually written to disk
until after all the remaining CCD image processing has been done, which is described in the remainder of this section.

Next is master bias subtraction.  This is done in machine memory, continued after the aforementioned steps.
Whereas overscan subtraction involves subtracting off the floating bias specific to an exposure, master bias
subtraction involves the pixel-by-pixel subtraction of a master bias image that is common to a collection of
exposures taken within some short period of time (such as 24 hours).  The master bias is a data-clipped,
pixel-by-pixel stack average of some number of bias frames that were taken in the near vicinity of time.
A database query is performed to obtain the nearest-in-time available master bias 2D FITS file (past or future).
The section below called **Master Files Creation** gives more details about the generation of master bias files.

<Describe master dark subtraction>

<Describe master flat correction>

In the end, the 2D FITS file contains HDUs for GREEN and RED full spectroscopic-data images,
each 4080x4080 pixels, with FITS extension names GREEN_CCD and RED_CCD, respectively.
The overscan biases that were subtracted are recorded in the FITS headers of
these HDUs; for example::

    OSCANV1 =    3086.385215099043 / Overscan clipped mean (e-), GREEN_AMP1
    OSCANV2 =    2783.307279684444 / Overscan clipped mean (e-), GREEN_AMP2

The physical units of the image data is electrons.
There are also associated variance images with FITS extension names
GREEN_VAR and RED_VAR, respectively, with physical units of electrons squared.


Master Files Creation
---------------------

<TBD to add content here>

Include a description of how master stacks are made for bias, dark, flats, LFC, etalon, and ThAr.

Scattered light correction
--------------------------

<TBD to add content here>

Spectral Extraction
-------------------

<TBD to add content here>

Sky Correction (not yet implemented)
------------------------------------

<TBD to add content here>

Wavelength Calibration
----------------------

<TBD to add content here>

Barycentric Correction
----------------------

<TBD to add content here>

Telluric Model (not yet implemented)
------------------------------------

<TBD to add content here>

Cross-Correlation based RVs
---------------------------

<TBD to add content here>

Include a note about RV header information

Stellar Activity Information
----------------------------
KPF does not yet have stellar activity indicators produced as a standard data product from the DRP.  The Ca H & K spectrometer covers the Ca H & K lines and we expect the DRP to produce S-values on the Mt. Wilson scale.  Future DRP developments are also expected to include code to generate other activity indicators (Ca IR triplet, Hα, etc.)


Ca H&K Spectrometer Data Processing
-----------------------------------

<TBD to add content here>

Exposure Meter Data Processing
------------------------------

<TBD to add content here>

Quality Control
---------------

<TBD to add content here>

Explain how the QC framework operates and describe the current status.

Guider Data Processing
----------------------
The DRP does not further process the data from the KPF Guider that are stored in FITS extensions in the L0 files.  These data include a guider image summed over the spectrometer integration and a table of guiding corrections, flux measurements, and other diagnostics taken from real-time Source Extractor analysis of the guider frames (typically at 100 Hz speed).

