KPF Calibrations and Master Files
=================================

Calibrations
------------

With rare exceptions, calibration spectra are taken on a daily basis with KPF to characterize the instrument.  While the calibration set has evolved with time, but is now mostly fixed (the table below is current as of October 2023).  This set of calibrations is designed to be sufficient for regular DRP processing without 'manual' calibrations taken by KPF observers.

======  ===========================  ===============  =======  ==================
Type    Object name                  Exp. time (sec)  Num/day  Comment
======  ===========================  ===============  =======  ==================
Dark    autocal-dark                 1200             5          
Bias    autocal-bias                 0                22
Flat    autocal-flat-all             30               100
LFC     autocal-lfc-all-morn         60               5        morning sequence

        autocal-lfc-all-eve          60               5        evening sequence
ThAr    autocal-thar-all-morn        10               9        morning sequence

        autocal-thar-all-eve         10               10       evening sequence
Etalon  autocal-etalon-all-morn      60               10       morning sequence

        autocal-etalon-all-eve       60               10       evening sequence

        autocal-etalon-all-night     60               varies   30x per qtr. night when off-sky

        slewcal                      60               varies   ~once per hour when on-sky
UNe     autocal-une-all-morn         5                5        not used

        autocal-une-all-eve          5                5        not used 
======  ===========================  ===============  =======  ==================

The Ca H&K spectrometer shares several calibration exposures with the main spectrometer.  The full set is listed below.  

======  ===========================  ===============  =======  ==================
Type    Object name                  Exp. time (sec)  Num/day  Comment
======  ===========================  ===============  =======  ==================
Dark    autocal-dark                 1200             5        the same exposures as for the main spectrometer (above)
Bias    autocal-bias                 0                22       the same exposures as for the main spectrometer (above)
ThAr    autocal-thar-hk              60               3        light detected in two bluest orders only
======  ===========================  ===============  =======  ==================


Processing data from KPF's Exposure Meter (EM) is handled in real-time with a separate pipeline.  For documentation proposes, the table below lists the EM calibrations.  Note that because of coatings on optics in the Fiber Injection Unit that are specific to the light path for calibrations, most calibrations of that type do not deliver measurable flux to the Ca H&K Spectrometer.

======  ===========================  ===============  =======  ==================
Type    Object name                  Exp. time (sec)  Num/day  Comment
======  ===========================  ===============  =======  ==================
TBD
======  ===========================  ===============  =======  ==================


Master Files
------------

Master files for bias, dark, flat, and the wavelength calibration sources listed above are created each day from the calibrations during the UT date.  The co-addition process involves iterative outlier rejection per pixel.  

.. |date| date::

Last Updated on |date|
