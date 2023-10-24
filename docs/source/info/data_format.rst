KPF Data Format
===============

Overview
--------

KPF data products are defined for these data levels:

* **Level 0 (L0)**: Raw data products produced by KPF at the W. M. Keck Observatory
* **2D**: Assembled CCD images with minimal processing.  This data product is produced by the DRP during processing from L0 to L1, but is not fundamental and is frequently not archived.
* **Level 1 (L1)**: Extracted, wavelength-calibrated spectra
* **Level 2 (L2)**: Derived data products including cross-correlation functions, radial velocities, and activity indicators

Each of these data levels is a standardized, multi-extension FITS format, and can be read using standard fits tools (e.g., `astropy.fits.io <https://docs.astropy.org/en/stable/io/fits/>`_) and the `KPF-Pipeline <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_.

KPF L0 files follow the naming convention: KP.YYYYMMDD.SSSSS.ss.fits, where YYYYMMDD is a date and SSSSS.ss is the number of decimal seconds after UT midnight corresponding to XXXX (when the file was read out?).  2D/L1/L2 files have similar file names, but with '_2D', '_L1', or '_L2' before '.fits'.  For example, KP.YYYYMMDD.SSSSS.ss_2D.fits is a 2D file name.

Level 0 Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 0 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

'2D' Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 1 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

Level 1 Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 1 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

Level 2 Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 2 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

