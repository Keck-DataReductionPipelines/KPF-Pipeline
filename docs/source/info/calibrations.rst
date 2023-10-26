KPF Calibrations and Master Files
=================================

Calibrations
------------

With rare exceptions, calibration spectra are taken on a daily basis with KPF to characterize the instrument.  While the set of calibrations has evolved with time, it is mostly fixed, as shown in the table below (current as of October 2023).  This set of calibrations is designed to be sufficient for regular DRP processing without 'manual' calibrations taken by KPF observers.

======  ===========================  ==============  =======  ==================
Type    Object name                  Ex. time (sec)  Num/day  Comment
======  ===========================  ==============  =======  ==================
Dark    autocal-dark                 1200            5          
Bias    autocal-bias                 0               22
Flat    autocal-flat-all             30              100
LFC     autocal-lfc-all-morn         60              5        morning sequence

        autocal-lfc-all-eve          60              5        evening sequence
ThAr    autocal-thar-all-morn        10              9        morning sequence

        autocal-thar-all-eve         10              10       evening sequence
Etalon  autocal-etalon-all-morn      60              10       morning sequence

        autocal-etalon-all-eve       60              10       evening sequence

        autocal-etalon-all-night     60              varies   30x per quarter night while off-sky

        slewcal                      60              varies   approx. once per hour while on-sky
UNe     autocal-une-all-morn         5               5        not used

        autocal-une-all-eve          5               5        not used 
======  ===========================  ==============  =======  ==================


<Add note about Ca H&K spectrometer calibrations>

<Add note about exposure meter calibrations>

Master Files
------------

<Add a list of the master files that are created from standard calibrations.>

.. |date| date::
*Last Updated on |date|*
