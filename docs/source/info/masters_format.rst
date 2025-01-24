KPF Calibration-Masters Data Format
===============

Overview
--------

KPF master calibration-file products, made by combining similar subsets of input L0 image data,
are defined for these data levels:

* **2D**: Assembled CCD images with minimal processing.
* **Level 1 (L1)**: Extracted, wavelength-calibrated spectra
* **Level 2 (L2)**: Derived data products

Derived data products for L2 include cross-correlation functions, radial velocities, and activity indicators.
Each of these data levels has associated product files in a standardized, multi-extension FITS format
(with a few exceptions), and can be read using standard fits tools
(e.g., `astropy.fits.io <https://docs.astropy.org/en/stable/io/fits/>`_)
and the `KPF-Pipeline <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_.

Master calibration files are generated daily, for each observation date.
The canonical location for persisted master files on the shrek machine is::

    /data/kpf/masters/yyyymmdd

Less frequently, the masters for some or all observation dates are reprocessed to
fix bugs, correct processing abnormalities, or incorporate new features.

The file-naming convention for 2D master calibration files generally adheres to the following prototype::

    kpf_<yyymmdd>_master_<type>_<object>.fits

where ``<yyyymmdd>`` is the observation date, and there is no explicit '_2D' suffix.
Master ``<type>`` can be either bias, dark, flat, arclamp, smooth lamp, and WLS.
Order-trace products unfortunately do not include anything like 'order_trace' in the filename
(please see below for more information).
Some master filenames include '_GREEN' or '_RED' as suffixes before
the '.fits' filename extension to indicate that they pertain to that specific filter only.
Master ``<object>`` in the file-naming scheme is a descriptive hyphenated subtype string
(derived from the OBJECT FITS keyword of a representation input data L0 FITS file), and
can be like any of the examples listed in the following table (the list is not exhaustive, but
gives the reader an idea of what to expect)::

    autocal-bias
    autocal-dark
    (none for master flat)
    (none for master smooth lamp)
    autocal-une-all-morn
    autocal-une-cal-eve
    autocal-une-sci-eve
    autocal-une-sky-eve
    autocal-une-all-eve
    slewcal
    autocal-etalon-all-eve
    autocal-etalon-all-midday
    autocal-etalon-all-morn
    autocal-etalon-all-night
    autocal-lfc-all-eve
    autocal-lfc-all-midnight
    autocal-lfc-all-morn
    autocal-thar-all-eve
    autocal-thar-all-morn
    autocal-thar-cal-eve
    autocal-thar-sci-eve
    autocal-thar-sky-eve

The master flat and smooth-lamp filenames do not include the ``<object>`` placeholder by quirk of the software.

The L1 and L2 master files have similar file names, but with '_L1', or '_L2' suffixes before the '.fits' filename extension.

There are four exceptions to this general file-naming scheme, namely:

*  The 2D (no explicit suffix), L1, and L2 master-flat products have filenames like the following (no ``<object>`` placeholder)::

    kpf_<yyyymmdd>_master_flat.fits
    kpf_<yyyymmdd>_master_flat_L1.fits
    kpf_<yyyymmdd>_master_flat_L2.fits

*  The 2D (no explicit suffix), L1, and L2 smooth-lamp products have filenames like the following (no explicit '_master' and no ``<object>`` placeholder)::

    kpf_<yyyymmdd>_smooth_lamp.fits
    kpf_<yyyymmdd>_smooth_lamp_L1.fits
    kpf_<yyyymmdd>_smooth_lamp_L2.fits

*  The order-trace products have filenames like the following (CSV files instead of FITS, and explicit filter suffix is included)::

    kpf_<yyyymmdd>_master_flat_GREEN_CCD.csv
    kpf_<yyyymmdd>_master_flat_RED_CCD.csv

* The etalon wavelength masks have filenames like the following (CSV files instead of FITS, and in the masks subdirectory, with derived ``<object>`` placeholder broken down by fiber)::

    masks/<yyyymmdd>_eve_CAL_etalon_wavelengths.csv
    masks/<yyyymmdd>_eve_SCI1_etalon_wavelengths.csv
    masks/<yyyymmdd>_eve_SCI2_etalon_wavelengths.csv
    masks/<yyyymmdd>_eve_SCI3_etalon_wavelengths.csv
    masks/<yyyymmdd>_morn_CAL_etalon_wavelengths.csv
    masks/<yyyymmdd>_morn_SCI1_etalon_wavelengths.csv
    masks/<yyyymmdd>_morn_SCI2_etalon_wavelengths.csv
    masks/<yyyymmdd>_morn_SCI3_etalon_wavelengths.csv
    masks/<yyyymmdd>_night_CAL_etalon_wavelengths.csv
    masks/<yyyymmdd>_night_SCI1_etalon_wavelengths.csv
    masks/<yyyymmdd>_night_SCI2_etalon_wavelengths.csv
    masks/<yyyymmdd>_night_SCI3_etalon_wavelengths.csv


Data Format of KPF Master Files
-------------------------------

Master Bias
^^^^^^^^^^^

A 2D master-bias file is a pixel-by-pixel clipped mean of a stack of L0 FITS image-data with
``IMTYPE='Bias'`` and ``OBJECT='autocal-bias'`` observed on the same date.

Here are the FITS extensions of interest in a 2D master-bias file:

===================  =========  ==============  ==========  ========================================================
Extension Name       Data Type  Data Dimension  Data Units  Description
===================  =========  ==============  ==========  ========================================================
GREEN_CCD            image      4080 x 4080     electrons   Master bias image for GREEN
RED_CCD              image      4080 x 4080     electrons   Master bias image for RED
CA_HK                image      1024 x 255      electrons   Master bias image for CA_HK
GREEN_CCD_UNC        image      4080 x 4080     electrons   Master bias-image uncertainty for GREEN
GREEN_CCD_CNT        image      4080 x 4080     count       Master bias-image number of stack samples for GREEN
RED_CCD_UNC          image      4080 x 4080     electrons   Master bias-image uncertainty for RED
RED_CCD_CNT          image      4080 x 4080     count       Master bias-image number of stack samples for RED
CA_HK_UNC            image      1024 x 255      electrons   Master bias-image uncertainty for CA_HK
CA_HK_CNT            image      1024 x 255      count       Master bias-image number of stack samples for CA_HK
===================  =========  ==============  ==========  ========================================================

Here is an example of the keywords in the GREEN_CCD extension of master bias file
``kpf_20250122_master_bias_autocal-bias.fits``::

    ==================================================================================
    HDU number and type = 4 and 0
    Number of header cards in HDU = 26
    ==================================================================================
    XTENSION= 'IMAGE   '           / Image extension
    BITPIX  =                  -64 / array data type
    NAXIS   =                    2 / number of array dimensions
    NAXIS1  =                 4080
    NAXIS2  =                 4080
    PCOUNT  =                    0 / number of parameters
    GCOUNT  =                    1 / number of groups
    BUNIT   = 'electrons'          / Units of master bias
    EXTNAME = 'GREEN_CCD'          / extension name
    NFRAMES =                    6 / Number of frames in input stack
    NSIGMA  =                  2.1 / Number of sigmas for data-clipping
    MINMJD  =         60697.043256 / Minimum MJD of bias observations
    MAXMJD  =         60697.712638 / Maximum MJD of bias observations
    MIDMJD  =         60697.377947 / Middle MJD of bias observations
    DATE-MID= '2025-01-22T09:04:14.621Z' / Middle timestamp of bias observations
    CREATED = '2025-01-23T03:07:21Z' / UTC of master-bias creation
    INFOBITS=                    7 / Bit-wise flags defined below
    BIT00   = '2**0 = 1'           / GREEN_CCD has gt 1% pixels with lt 10 samples
    BIT01   = '2**1 = 2'           / RED_CCD has gt 1% pixels with lt 10 samples
    BIT02   = '2**2 = 4'           / CA_HK" has gt 1% pixels with lt 10 samples
    INFL0   = 'KP.20250122.03737.36_2D.fits'
    INFL1   = 'KP.20250122.03937.09_2D.fits'
    INFL2   = 'KP.20250122.03987.04_2D.fits'
    INFL3   = 'KP.20250122.04037.05_2D.fits'
    INFL4   = 'KP.20250122.61521.77_2D.fits'
    INFL5   = 'KP.20250122.61571.78_2D.fits'

It includes useful metadata about the image stacking, including the specific input bias L0 FITS files.
The input bias L0 FITS files are preprocessed to subtract the overscan biases, and assemble the CCD images.


Master Dark
^^^^^^^^^^^

A 2D master-dark file is a pixel-by-pixel clipped mean of a stack of L0 FITS image-data with
``IMTYPE='Dark'`` and ``OBJECT='autocal-dark'`` observed on the same date.

Here are the FITS extensions of interest in a 2D master-dark file:

===================  =========  ==============  ==============  ========================================================
Extension Name       Data Type  Data Dimension  Data Units      Description
===================  =========  ==============  ==============  ========================================================
GREEN_CCD            image      4080 x 4080     electrons/sec   Master dark image for GREEN
RED_CCD              image      4080 x 4080     electrons/sec   Master dark image for RED
CA_HK                image      1024 x 255      electrons/sec   Master dark image for CA_HK
GREEN_CCD_UNC        image      4080 x 4080     electrons/sec   Master dark-image uncertainty for GREEN
GREEN_CCD_CNT        image      4080 x 4080     count           Master dark-image number of stack samples for GREEN
RED_CCD_UNC          image      4080 x 4080     electrons/sec   Master dark-image uncertainty for RED
RED_CCD_CNT          image      4080 x 4080     count           Master dark-image number of stack samples for RED
CA_HK_UNC            image      1024 x 255      electrons/sec   Master dark-image uncertainty for CA_HK
CA_HK_CNT            image      1024 x 255      count           Master dark-image number of stack samples for CA_HK
===================  =========  ==============  ==============  ========================================================

Here is an example of the keywords in the GREEN_CCD extension of master dark file
``kpf_20250122_master_dark_autocal-dark.fits``::

    ==================================================================================
    HDU number and type = 4 and 0
    Number of header cards in HDU = 27
    ==================================================================================
    XTENSION= 'IMAGE   '           / Image extension
    BITPIX  =                  -64 / array data type
    NAXIS   =                    2 / number of array dimensions
    NAXIS1  =                 4080
    NAXIS2  =                 4080
    PCOUNT  =                    0 / number of parameters
    GCOUNT  =                    1 / number of groups
    BUNIT   = 'electrons/sec'      / Units of master dark
    EXTNAME = 'GREEN_CCD'          / extension name
    NFRAMES =                    5 / Number of frames in input stack
    MINEXPTM=                300.0 / Minimum exposure time of input darks (seconds)
    NSIGMA  =                  2.2 / Number of sigmas for data-clipping
    MINMJD  =         60697.048439 / Minimum MJD of dark observations
    MAXMJD  =         60697.916909 / Maximum MJD of dark observations
    MIDMJD  =         60697.482674 / Middle MJD of dark observations
    DATE-MID= '2025-01-22T11:35:03.034Z' / Middle timestamp of dark observations
    INPBIAS = 'kpf_20250122_master_bias_autocal-bias.fits'
    CREATED = '2025-01-23T03:08:44Z' / UTC of master-dark creation
    INFOBITS=                    7 / Bit-wise flags defined below
    BIT00   = '2**0 = 1'           / GREEN_CCD has gt 1% pixels with lt 10 samples
    BIT01   = '2**1 = 2'           / RED_CCD has gt 1% pixels with lt 10 samples
    BIT02   = '2**2 = 4'           / CA_HK" has gt 1% pixels with lt 10 samples
    INFL0   = 'KP.20250122.04185.19_2D.fits'
    INFL1   = 'KP.20250122.08949.47_2D.fits'
    INFL2   = 'KP.20250122.61770.27_2D.fits'
    INFL3   = 'KP.20250122.65367.88_2D.fits'
    INFL4   = 'KP.20250122.79221.17_2D.fits'

It includes useful metadata about the image stacking, including the specific input dark L0 FITS files.
The input dark L0 FITS files are preprocessed to subtract the overscan biases, assemble the CCD images, and subtract
the master bias.  The header keyword ``INPBIAS`` gives the master bias employed.

Master Flat
^^^^^^^^^^^

Add content here.

Master Smooth Lamp
^^^^^^^^^^^^^^^^^^

Add content here.

Master Arclamp
^^^^^^^^^^^^^^

Add content here.

Master Order Trace
^^^^^^^^^^^^^^^^^^

Add content here.

Master WLS
^^^^^^^^^^

Add content here.
