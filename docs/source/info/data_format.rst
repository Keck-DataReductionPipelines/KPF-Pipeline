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

See the section titled :ref:`label-tutorials` for a set of tutorials on the various KPF data files.

In addition, the DRP is able produce WLS Dictionaries that contain detailed diagnostic information about the fits of individual lines, orders, and orderlets for the wavelength solutions.  These are described at the bottom of this page.

Data Format of KPF Files
------------------------

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


2D File FITS Extensions
^^^^^^^^^^^^^^^^^^^^^^^

===================  =========  ==============  =======
Extension Name       Data Type  Data Dimension  Description    
===================  =========  ==============  =======
RECEIPT              table      variable        Receipt of DRP processing
CONFIG               table      variable        Configuration parameters
GREEN_CCD            image      4080 x 4080     Assembled Green CCD image with bias/dark correction   
RED_CCD              image      4080 x 4080     Assembled Red CCD image with bias/dark correction   
CA_HK                image      255 x 1024      Same as in L0 file    
EXPMETER_SCI         table      variable        Same as in L0 file 
EXPMETER_SKY         table      variable        Same as in L0 file 
TELEMETRY            table      variable        Same as in L0 file 
SOLAR_IRRADIANCE     table      variable        Same as in L0 file 
GUIDER_AVG           image      512 x 640       Same as in L0 file 
GUIDER_CUBE_ORIGINS  table      variable        Same as in L0 file          
===================  =========  ==============  =======


Level 1 FITS Extensions
^^^^^^^^^^^^^^^^^^^^^^^

===================  =========  ==============  =======
Extension Name       Data Type  Data Dimension  Description    
===================  =========  ==============  =======
RECEIPT              table      variable        Receipt of DRP processing
CONFIG               table      variable        Configuration parameters
TELEMETRY            table      variable        Table of telemetry measurements
GREEN_SCI_FLUX1      image      35 x 4080       1D spectra for 35 GREEN CCD orders of SCI1 orderlet
GREEN_SCI_FLUX2      image      35 x 4080       1D spectra for 35 GREEN CCD orders of SCI2 orderlet
GREEN_SCI_FLUX3      image      35 x 4080       1D spectra for 35 GREEN CCD orders of SCI3 orderlet
GREEN_SKY_FLUX       image      35 x 4080       1D spectra for 35 GREEN CCD orders of SKY orderlet
GREEN_CAL_FLUX       image      35 x 4080       1D spectra for 35 GREEN CCD orders of CAL orderlet
GREEN_SCI_VAR1       image      35 x 4080       Variance vs. pixel for GREEN_SCI_FLUX1
GREEN_SCI_VAR2       image      35 x 4080       Variance vs. pixel for GREEN_SCI_FLUX2
GREEN_SCI_VAR3       image      35 x 4080       Variance vs. pixel for GREEN_SCI_FLUX3
GREEN_SKY_VAR        image      35 x 4080       Variance vs. pixel for GREEN_SKY_FLUX
GREEN_CAL_VAR        image      35 x 4080       Variance vs. pixel for GREEN_CAL_FLUX
GREEN_SCI_WAV1       image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX1
GREEN_SCI_WAV2       image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX2
GREEN_SCI_WAV3       image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX3
GREEN_SKY_WAV        image      35 x 4080       Wavelength vs. pixel for GREEN_SKY_FLUX
GREEN_CAL_WAV        image      35 x 4080       Wavelength vs. pixel for GREEN_CAL_FLUX
GREEN_TELLURIC       table      n/a             Not used yet (will include telluric spectrum)
GREEN_SKY            table      n/a             Not used yet (will include modeled sky spectrum)
RED_SCI_FLUX1        image      32 x 4080       1D spectra for 32 RED CCD orders of SCI1 orderlet
RED_SCI_FLUX2        image      32 x 4080       1D spectra for 32 RED CCD orders of SCI2 orderlet
RED_SCI_FLUX3        image      32 x 4080       1D spectra for 32 RED CCD orders of SCI3 orderlet
RED_SKY_FLUX         image      32 x 4080       1D spectra for 32 RED CCD orders of SKY orderlet
RED_CAL_FLUX         image      32 x 4080       1D spectra for 32 RED CCD orders of CAL orderlet
RED_SCI_VAR1         image      32 x 4080       Variance vs. pixel for RED_SCI_FLUX1
RED_SCI_VAR2         image      32 x 4080       Variance vs. pixel for RED_SCI_FLUX2
RED_SCI_VAR3         image      32 x 4080       Variance vs. pixel for RED_SCI_FLUX3
RED_SKY_VAR          image      32 x 4080       Variance vs. pixel for RED_SCI_FLUX
RED_CAL_VAR          image      32 x 4080       Variance vs. pixel for RED_SCI_FLUX
RED_SCI_WAV1         image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX1
RED_SCI_WAV2         image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX2
RED_SCI_WAV3         image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX3
RED_SKY_WAV          image      32 x 4080       Wavelength vs. pixel for RED_SKY_FLUX
RED_CAL_WAV          image      32 x 4080       Wavelength vs. pixel for RED_CAL_FLUX
RED_TELLURIC         table      n/a             Not used yet (will include telluric spectrum)
RED_SKY              table      n/a             Not used yet (will include modeled sky spectrum)
CA_HK_SCI            image      6 x 1024        1D spectra (6 orders) of SCI in Ca H&K spectrometer
CA_HK_SKY            image      6 x 1024        1D spectra (6 orders) of SKY in Ca H&K spectrometer
CA_HK_SCI_WAVE       image      6 x 1024        Wavelength vs. pixel for CA_HK_SCI
CA_HK_SKY_WAVE       image      6 x 1024        Wavelength vs. pixel for CA_HK_SKY
BARY_CORR            table      67              Table of barycentric corrections by spectral order
===================  =========  ==============  =======


Level 2 FITS Extensions
^^^^^^^^^^^^^^^^^^^^^^^

===================  =========  ==============  =======
Extension Name       Data Type  Data Dimension  Description    
===================  =========  ==============  =======
RECEIPT              table      variable        Receipt of DRP processing
CONFIG               table      variable        Configuration parameters
TELEMETRY            table      variable        Table of telemetry measurements
GREEN_CCF            image      5 x 52 x 804    CCFs (orderlet x order x RV step) for GREEN
RED_CCF              image      5 x 52 x 804    CCFs (orderlet x order x RV step) for RED
GREEN_CCF            image      5 x 52 x 804    Reweighted CCFs (orderlet x order x RV step) for GREEN
RED_CCF              image      5 x 52 x 804    Reweighted CCFs (orderlet x order x RV step) for RED
RV                   table      67              Table of RVs by spectral order
ACTIVITY             table      n/a             Not used yet (will include activity measurements)
===================  =========  ==============  =======

Important FITS Header Keywords
------------------------------

Level 0 Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the important keywords are stored in the primary extension of the Level 0 file, which is written immediately after each KPF exposure.

========  ==============================  =========
Keyword   Value (example)                 Comment
========  ==============================  =========
DATE-BEG  2023-10-22T15:30:01.056733      Start of exposure from kpfexpose
DATE-MID  2023-10-22T15:32:31.065         Halfway point of the exposure (unweighted)
DATE-END  2023-10-22T15:35:01.072797      End of exposure
EXPTIME   300.0                           Requested exposure time
ELAPSED   300.0                           Actual exposure time
PROGNAME  N226                            Program name from kpfexpose
OBJECT    42813                           Object name
TARGRA    06:12:13.80                     Right ascension [hr] from DCS
TARGDEC   -14:38:56.0                     Declination [deg] from DCS
TARGEPOC  2000.0                          Target epoch from DCS
TARGEQUI  2000.0                          Target equinox from DCS
TARGPLAX  14.7                            Target parallax [arcsec] from DCS
TARGPMDC  0.0                             Target proper motion [arcsec/yr] in declination from DCS
TARGPMRA  0.0                             Target proper motion [s/yr] in right ascension from DCS
TARGRADV  81.87                           Target radial velocity [km/s]
AIRMASS   1.26                            Airmass from DCS
PARANTEL  23.58                           Parallactic angle of the telescope from DCS
HA        +01:01:37.22                    Hour angle
EL        52.46                           Elevation [deg]
AZ        204.46                          Azimuth [deg]
LST       07:13:51.02                     Local sidereal time
GAIAID    DR3 2993561629444856960         GAIA Target name
2MASSID   J06121397-1439002               2MASS Target name
GAIAMAG   9.28                            GAIA G band magnitude
2MASSMAG  8.06                            2MASS J band magnitude
TARGTEFF  5398.0                          Target effective temperature (K)
OCTAGON   EtalonFiber                     Selected octagon calibration source (not necessarily powered on)
TRIGTARG  Green,Red,Ca_HK,ExpMeter,Guide  Cameras that were sent triggers
IMTYPE    Object                          Image Type
CAL-OBJ   None                            Calibration fiber source
SKY-OBJ   Sky                             Sky fiber source
SCI-OBJ   Target                          Science fiber source
AGITSTA   Running                         Agitator status
========  ==============================  =========

2D Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All keywords from Level 0 are inherited by the 2D file.  Below are additional keywords.

========  ==========================================  =========
Keyword   Value (example)                             Comment
========  ==========================================  =========
DRPTAG    v2.5.2                                      Git version number of KPF-Pipeline used for processing
DRPHASH   'ccf5f6ebe0c9ae7d43706cc57fed2ecdeb540a17'  Git commit hash version of KPF-Pipeline used for processing
RNGREEN1  4.85283                                     Read noise for GREEN_AMP1 [e-] (first amplifier region on Green CCD)
RNGREEN2  4.14966                                     Read noise for GREEN_AMP2 [e-] (second amplifier region on Green CCD)
RNGREEN3  4.85283                                     Read noise for GREEN_AMP3 [e-] (third amplifier region on Green CCD)
RNGREEN4  4.14966                                     Read noise for GREEN_AMP4 [e-] (fourth amplifier region on Green CCD)
RNRED1    4.0376                                      Read noise for RED_AMP1 [e-] (first amplifier region on Red CCD)
RNRED2    4.12717                                     Read noise for RED_AMP2 [e-] (second amplifier region on Red CCD)
RNRED3    4.0376                                      Read noise for RED_AMP3 [e-] (third amplifier region on Red CCD)
RNRED4    4.12717                                     Read noise for RED_AMP4 [e-] (fourth amplifier region on Red CCD)
GREENTRT  46.804                                      Green CCD read time [sec]
REDTRT    46.839                                      Red CCD read time [sec]
READSPED  'regular '                                  Categorization of CCD read speed ('regular' or 'fast')
FLXREG1G  1.00                                        Dark current [e-/hr] - Green CCD region 1 - coords = [1690:1990,1690:1990]
FLXREG2G  1.00                                        Dark current [e-/hr] - Green CCD region 2 - coords = [1690:1990,2090:2390]
FLXREG3G  1.00                                        Dark current [e-/hr] - Green CCD region 3 - coords = [2090:2390,1690:1990]
FLXREG4G  1.00                                        Dark current [e-/hr] - Green CCD region 4 - coords = [2090:2390,2090:2390]
FLXREG5G  1.00                                        Dark current [e-/hr] - Green CCD region 5 - coords = [80:380,3080:3380]
FLXREG6G  1.00                                        Dark current [e-/hr] - Green CCD region 6 - coords = [1690:1990,1690:1990]
FLXAMP1G  1.00                                        Dark current [e-/hr] - Green CCD amplifier region 1 - coords = [3700:4000,700:1000]
FLXAMP2G  1.00                                        Dark current [e-/hr] - Green CCD amplifier region 2 - coords = [3700:4000,3080:3380]
FLXCOLLG  1.00                                        Dark current [e-/hr] - Green CCD collimator-side region = [3700:4000,700:1000]
FLXECHG   1.00                                        Dark current [e-/hr] - Green CCD echelle-side region = [3700:4000,700:1000]
FLXREG1R  1.00                                        Dark current [e-/hr] - Red CCD region 1 - coords = [1690:1990,1690:1990]
FLXREG2R  1.00                                        Dark current [e-/hr] - Red CCD region 2 - coords = [1690:1990,2090:2390]
FLXREG3R  1.00                                        Dark current [e-/hr] - Red CCD region 3 - coords = [2090:2390,1690:1990]
FLXREG4R  1.00                                        Dark current [e-/hr] - Red CCD region 4 - coords = [2090:2390,2090:2390]
FLXREG5R  1.00                                        Dark current [e-/hr] - Red CCD region 5 - coords = [80:380,3080:3380]
FLXREG6R  1.00                                        Dark current [e-/hr] - Red CCD region 6 - coords = [1690:1990,1690:1990]
FLXAMP1R  1.00                                        Dark current [e-/hr] - Red CCD amplifier region 1 = [3700:4000,700:1000]
FLXAMP2R  1.00                                        Dark current [e-/hr] - Red CCD amplifier region 2 = [3700:4000,3080:3380]
FLXCOLLR  1.00                                        Dark current [e-/hr] - Red CCD collimator-side region = [3700:4000,700:1000]
FLXECHR   1.00                                        Dark current [e-/hr] - Red CCD echelle-side region = [3700:4000,700:1000]
GDRXRMS   10.123                                      x-coordinate RMS guiding error in milliarcsec (mas)
GDRYRMS   10.123                                      y-coordinate RMS guiding error in milliarcsec (mas)
GDRRRMS   10.123                                      r-coordinate RMS guiding error in milliarcsec (mas)
GDRXBIAS  0.0010                                      x-coordinate bias guiding error in milliarcsec (mas)
GDRYBIAS  0.0010                                      y-coordinate bias guiding error in milliarcsec (mas)
GDRSEEJZ  0.450                                       Seeing (arcsec) in J+Z-band from Moffat func fit
GDRSEEV   0.450                                       Scaled seeing (arcsec) in V-band from J+Z-band
MOONSEP   55.0                                        Separation between Moon and target star (deg)
SUNALT    -45.0                                       Altitude of Sun (deg); negative = below horizon
SKYSCIMS  0.0000123                                   SKY/SCI flux ratio in main spectrometer scaled from EM data. 
EMSCCT48  100000000.1234                              cumulative EM counts [ADU] in SCI in 445-870 nm
EMSCCT45  100000000.1234                              cumulative EM counts [ADU] in SCI in 445-551 nm
EMSCCT56  100000000.1234                              cumulative EM counts [ADU] in SCI in 551-658 nm
EMSCCT67  100000000.1234                              cumulative EM counts [ADU] in SCI in 658-764 nm
EMSCCT78  100000000.1234                              cumulative EM counts [ADU] in SCI in 764-870 nm
EMSKCT48  100000000.1234                              cumulative EM counts [ADU] in SKY in 445-870 nm
EMSKCT45  100000000.1234                              cumulative EM counts [ADU] in SKY in 445-551 nm
EMSKCT56  100000000.1234                              cumulative EM counts [ADU] in SKY in 551-658 nm
EMSKCT67  100000000.1234                              cumulative EM counts [ADU] in SKY in 658-764 nm
EMSKCT78  100000000.1234                              cumulative EM counts [ADU] in SKY in 764-870 nm
========  ==========================================  =========

Keywords related to read noise are only computed for the amplifiers used.  In regular read mode, two amplifiers are used (AMP1 and AMP2), while in fast read mode, four amplifiers are used (AMP1, AMP2, AMP3, and AMP4).

Keywords related to dark current (starting with FLX) are only added for 2D files of Dark observations (no illumination and exposure time > 0). The regions for those keywords refer to the CCD coordinates where the dark current measurements were made (using modules/quicklook/arc/analyze_2d.py).  The image below (click to enlarge) shows the regions and dark current estimates for a 2D spectrum taken when the dark current was high.

Keywords related to the Guider are only added for 2D files that have Guider data products.  Similar for Exposure Meter data products.

.. image:: dark_current_example.png
   :alt: Image of KPF Green CCD showing regions where dark current is measured
   :align: center
   :height: 400px
   :width: 500px

L1 Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All keywords from Level 0 and 2D are inherited by the L1 file.  Below are additional keywords.

========  ===============  =========
Keyword   Value (example)  Comment
========  ===============  =========
SNRSC452  250.0            SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 452 nm (second bluest order); on Green CCD
SNRSK452  250.0            SNR of L1 SKY spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
SNRCL452  250.0            SNR of L1 CAL spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
SNRSC548  250.0            SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 548 nm; on Green CCD
SNRSK548  250.0            SNR of L1 SKY spectrum (95th %ile) near 548 nm; on Green CCD
SNRCL548  250.0            SNR of L1 CAL spectrum (95th %ile) near 548 nm; on Green CCD
SNRSC652  250.0            SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 652 nm; on Red CCD
SNRSK652  250.0            SNR of L1 SKY spectrum (95th %ile) near 652 nm; on Red CCD
SNRCL652  250.0            SNR of L1 CAL spectrum (95th %ile) near 652 nm; on Red CCD
SNRSC747  250.0            SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 747 nm; on Red CCD
SNRSK747  250.0            SNR of L1 SKY spectrum (95th %ile) near 747 nm; on Red CCD
SNRCL747  250.0            SNR of L1 CAL spectrum (95th %ile) near 747 nm; on Red CCD
SNRSC852  250.0            SNR of L1 SCI (SCI1+SCI2+SCI3; 95th %ile) near 852 nm (second reddest order); on Red CCD
SNRSK852  250.0            SNR of L1 SKY spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
SNRCL852  250.0            SNR of L1 CAL spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
FR452652  1.2345           Peak flux ratio between orders (452nm/652nm) using SCI2
FR548652  1.2345           Peak flux ratio between orders (548nm/652nm) using SCI2
FR747652  1.2345           Peak flux ratio between orders (747nm/652nm) using SCI2
FR852652  1.2345           Peak flux ratio between orders (852nm/652nm) using SCI2
FR12M452  0.9000           median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FR12U452  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FR32M452  0.9000           median(SCI3/SCI2) flux ratio near 452 nm; on Green CCD
FR32U452  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FRS2M452  0.9000           median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
FRS2U452  0.0010           uncertainty on the median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
FRC2M452  0.9000           median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
FRC2U452  0.0010           uncertainty on the median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
FR12M548  0.9000           median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FR12U548  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FR32M548  0.9000           median(SCI3/SCI2) flux ratio near 548 nm; on Green CCD
FR32U548  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FRS2M548  0.9000           median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
FRS2U548  0.0010           uncertainty on the median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
FRC2M548  0.9000           median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
FRC2U548  0.0010           uncertainty on the median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
FR12M652  0.9000           median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FR12U652  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FR32M652  0.9000           median(SCI3/SCI2) flux ratio near 652 nm; on Red CCD
FR32U652  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FRS2M652  0.9000           median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
FRS2U652  0.0010           uncertainty on the median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
FRC2M652  0.9000           median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
FRC2U652  0.0010           uncertainty on the median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
FR12M747  0.9000           median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FR12U747  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FR32M747  0.9000           median(SCI3/SCI2) flux ratio near 747 nm; on Red CCD
FR32U747  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FRS2M747  0.9000           median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
FRS2U747  0.0010           uncertainty on the median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
FRC2M747  0.9000           median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
FRC2U747  0.0010           uncertainty on the median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
FR12M852  0.9000           median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FR12U852  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FR32M852  0.9000           median(SCI3/SCI2) flux ratio near 852 nm; on Red CCD
FR32U852  0.0010           uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FRS2M852  0.9000           median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
FRS2U852  0.0010           uncertainty on the median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
FRC2M852  0.9000           median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
FRC2U852  0.0010           uncertainty on the median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
========  ===============  =========

The keywords above related to the signal-to-noise ratio in L1 spectra all start with 'SNR'.  These measurements were made using modules/quicklook/src/analyze_l1.py.  The image below (click to enlarge) shows the spectral orders and wavelengths at which SNR is measured.

Keywords related to flux ratios between orders (FR452652, FR548652, FR747652, FR852652) are the ratios between the 95th percentile in flux for the spectral orders containing 452 nm, 548 nm, 747 nm, and 852 nm, all normalized by the spectral order containing 652 nm.  These are the same spectral orders used for the SNR calculations and use the SCI2 orderlet.

Keywords related to orderlet flux ratios (e.g., FR12M452 and its uncertainty FR12U452) are computed in 500-pixel regions in the centers in the same spectral orders as are used for the SNR calculations.

.. image:: KPF_L1_SNR.png
   :alt: L1 Spectrum show wavelengths where SNR is measured
   :align: center
   :height: 400px
   :width: 600px


*To-do: add a list of additional 2D, Level 1, and Level 2 primary keywords.*


WLS Dictionaries
----------------

See :doc:`../analysis/dictonary_format` for details.
