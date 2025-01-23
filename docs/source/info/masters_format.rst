KPF Calibration-Masters Data Format
===============

Overview
--------

KPF master calibration-file products, made by combining subsets of input L0 image data,
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
correct processing abnormalities or incorporate new features.

The file-naming convention for 2D master calibration files generally adheres to the following prototype::

    kpf_<yyymmdd>_master_<type>_<object>.fits

where ``<yyyymmdd>`` is the observation date, and there is no explicit '_2D' suffix.
Master ``<type>`` can be either bias, dark, flat, arclamp, smooth lamp, order trace, and WLS.
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
There are thus four exceptions to this general file-naming scheme, namely:

*  The 2D (no explicit suffix), L1, and L2 master-flat products have filenames like the following (no ``<object>`` placeholder)::

    kpf_<yyyymmdd>_master_flat.fits
    kpf_<yyyymmdd>_master_flat_L1.fits
    kpf_<yyyymmdd>_master_flat_L2.fits

*  The 2D (no explicit suffix), L1, and L2 smooth-lamp products have filenames like the following (no explicit '_master' and no ``<object>`` placeholder)::

    kpf_<yyyymmdd>_smooth_lamp.fits
    kpf_<yyyymmdd>_smooth_lamp_L1.fits
    kpf_<yyyymmdd>_smooth_lamp_L2.fits

*  The order-trace products have filenames like the following::

    kpf_<yyyymmdd>_master_flat_GREEN_CCD.csv
    kpf_<yyyymmdd>_master_flat_RED_CCD.csv

* The etalon masks have filenames like the following (in the masks subdirectory, with derived ``<object>`` placeholder)::

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

Add content here.

Master Dark
^^^^^^^^^^^

Add content here.

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
