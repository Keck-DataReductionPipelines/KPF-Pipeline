KPF DRP Algorithms
==================

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

<TBD to add content here>


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

