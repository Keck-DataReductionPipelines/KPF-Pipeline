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

KPF L0 files follow the naming convention: KP.YYYYMMDD.SSSSS.ss.fits, where YYYYMMDD is a date and SSSSS.ss is the number of decimal seconds after UT midnight corresponding to the start of the exposure.  2D/L1/L2 files have similar file names, but with '_2D', '_L1', or '_L2' before '.fits'.  For example, KP.YYYYMMDD.SSSSS.ss_2D.fits is a 2D file name.

Level 0 Data Format
-------------------

*Add a list of important Level 0 primary keywords.  Add a link to the tutorial showing how to open and examine L0 files.*

Level 0 FITS Extensions
^^^^^^^^^^^^^^^^^^^^^^^

===================  =========  ==============  =======
Extension Name       Data Type  Data Dimension  Description    
===================  =========  ==============  =======
GREEN_AMP1           image      4110 x 2094     CCD image from Green amplifier 1   
GREEN_AMP2           image      4110 x 2094     CCD image from Green amplifier 2 [a]       
RED_AMP1             image      4110 x 2094     CCD image from Red amplifier 1   
RED_AMP2             image      4110 x 2094     CCD image from Red amplifier 2    
CA_HK                image      255 x 1024      CCD image from Ca H&K Spectrometer    
EXPMETER_SCI         table      variable        Table of Exposure Meter measurements for SCI channel
EXPMETER_SKY         table      variable        Table of Exposure Meter measurements for SKY channel
TELEMETRY            table      variable        Table of telemetry measurements
SOLAR_IRRADIANCE     table      variable        Table of pyrheliometer measurements (for SoCal spectra)
GUIDER_AVG           image      512 x 640       Average image from the guide camera
GUIDER_CUBE_ORIGINS  table      variable        Table of time-series guide camera measurements            
===================  =========  ==============  =======

[a] - the example shown above is for two-amplifier mode.  When KPF is operated in fast' read mode, four amplifiers per CCD will be used and there will be a corresponding number of AMP extensions.

'2D' Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 1 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

Level 1 Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 1 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

Level 2 Data Format
-------------------

Add a table of the HDUs and their contents.  Add a list of important Level 2 primary keywords.  Add a link to the Tutorial showing how to open and examine L0 files.

.. |date| date::

Last Updated on |date|
