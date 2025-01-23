KPF Calibration-Masters Data Format
===============

Overview
--------

KPF master calibration-file products, made by combining subsets of input L0 FITS files,
are defined for these data levels:

* **2D**: Assembled CCD images with minimal processing.
* **Level 1 (L1)**: Extracted, wavelength-calibrated spectra
* **Level 2 (L2)**: Derived data products including cross-correlation functions,
                    radial velocities, and activity indicators.

Each of these data levels is a standardized, multi-extension FITS format,
and can be read using standard fits tools
(e.g., `astropy.fits.io <https://docs.astropy.org/en/stable/io/fits/>`_)
and the `KPF-Pipeline <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_.

Master calibration files are generated daily, for each observation date.
Less frequently, the masters for some or all observation dates are reprocessed to
correct processing abnormalities or incorporate new features.

The file-naming convention for master calibration files generally follows the following prototype::

    kpf_<yyymmdd>_master_<type>_<object>.fits

where <yyyymmdd> is the observation date.
Master <type> can be either bias, dark, flat, arclamp, smooth lamp, order trace, and WLS.
Some master filenames include GREEN or RED as suffixes to indicate that they pertain to that filter only.
Master <object> in the file-naming scheme is a descriptive hyphenated subtype string
(derived from the OBJECT FITS keyword of a representation input data L0 FITS file), and
can be like any of those listed in the following table (the list is not exhaustive, but
gives the reader an idea of what to expect).  The master flat and smooth-lamp filenames
do not include <object> by quirk of software.

======================================
Examples of KPF Master-File Objects
======================================
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
======================================


L1 and L2 master files have similar file names, but with '_L1', or '_L2' before
the '.fits' filename extension.
There are three exceptions to this file-naming scheme, namely:

*  The 2D, L1, and L2 smooth-lamp products have filenames like the following::

    kpf_<yyyymmdd>_smooth_lamp.fits
    kpf_<yyyymmdd>_smooth_lamp_L2.fits
    kpf_<yyyymmdd>_smooth_lamp_L1.fits

*  The order-trace products have filenames like the following::

    kpf_<yyyymmdd>_master_flat_GREEN_CCD.csv
    kpf_<yyyymmdd>_master_flat_RED_CCD.csv

* The etalon masks

<yyyymmdd>_eve_CAL_etalon_wavelengths.csv


Data Format of KPF Master Files
-------------------------------

Master Bias
^^^^^^^^^^^

Master Dark
^^^^^^^^^^^

Master Flat
^^^^^^^^^^^

Master Smooth Lamp
^^^^^^^^^^^^^^^^^^

Master Arclamp
^^^^^^^^^^^^^^

Master Order Trace
^^^^^^^^^^^^^^^^^^

Master WLS
^^^^^^^^^^
