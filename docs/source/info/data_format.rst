KPF Data Format
===============

Overview
--------

KPF data products are defined for these data levels:

* **Level 0 (L0)**: Raw data products produced by KPF at the W. M. Keck Observatory
* **2D**: Assembled CCD images with minimal processing.  This data product is produced by the DRP during processing from L0 to L1 but is not fundamental and is frequently not archived.
* **Level 1 (L1)**: Extracted, wavelength-calibrated spectra
* **Level 2 (L2)**: Derived data products including cross-correlation functions, radial velocities, and activity indicators

Each of these data levels is a standardized, multi-extension FITS format, and can be read using standard fits tools (e.g., `astropy.fits.io <https://docs.astropy.org/en/stable/io/fits/>`_) and the `KPF-Pipeline <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_.

KPF L0 files follow the naming convention: KP.YYYYMMDD.SSSSS.ss.fits, where YYYYMMDD is a date and SSSSS.ss is the number of decimal seconds after UT midnight corresponding to the start of the exposure.  2D/L1/L2 files have similar file names, but with '_2D', '_L1', or '_L2' before '.fits'.  For example, KP.YYYYMMDD.SSSSS.ss_2D.fits is a 2D file name.

See the section titled :ref:`label-tutorials` for a set of tutorials on the various KPF data files.

In addition, the DRP is able to produce WLS Dictionaries that contain detailed diagnostic information about the fits of individual lines, orders, and orderlets for the wavelength solutions.  These are described at the bottom of this page.

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

========  ==========================================  =========
Keyword   Value (example)                             Comment
========  ==========================================  =========
DATE-BEG  2023-10-22T15:30:01.056733                  Start of exposure from kpfexpose
DATE-MID  2023-10-22T15:32:31.065                     Halfway point of the exposure (unweighted)
DATE-END  2023-10-22T15:35:01.072797                  End of exposure
EXPTIME   300.0                                       Requested exposure time
ELAPSED   300.0                                       Actual exposure time
PROGNAME  N226                                        Program name from kpfexpose
OBJECT    42813                                       Object name
TARGRA    06:12:13.80                                 Right ascension [hr] from DCS
TARGDEC   -14:38:56.0                                 Declination [deg] from DCS
TARGEPOC  2000.0                                      Target epoch from DCS
TARGEQUI  2000.0                                      Target equinox from DCS
TARGPLAX  14.7                                        Target parallax [arcsec] from DCS
TARGPMDC  0.0                                         Target proper motion [arcsec/yr] in declination from DCS
TARGPMRA  0.0                                         Target proper motion [s/yr] in right ascension from DCS
TARGRADV  81.87                                       Target radial velocity [km/s]
AIRMASS   1.26                                        Airmass from DCS
PARANTEL  23.58                                       Parallactic angle of the telescope from DCS
HA        +01:01:37.22                                Hour angle
EL        52.46                                       Elevation [deg]
AZ        204.46                                      Azimuth [deg]
LST       07:13:51.02                                 Local sidereal time
GAIAID    DR3 2993561629444856960                     GAIA Target name
2MASSID   J06121397-1439002                           2MASS Target name
GAIAMAG   9.28                                        GAIA G band magnitude
2MASSMAG  8.06                                        2MASS J band magnitude
TARGTEFF  5398.0                                      Target effective temperature (K)
OCTAGON   EtalonFiber                                 Selected octagon calibration source (not necessarily powered on)
TRIGTARG  Green,Red,Ca_HK,ExpMeter,Guide              Cameras that were sent triggers
IMTYPE    Object                                      Image Type
CAL-OBJ   None                                        Calibration fiber source
SKY-OBJ   Sky                                         Sky fiber source
SCI-OBJ   Target                                      Science fiber source
AGITSTA   Running                                     Agitator status
FIUMODE   Observing                                   FIU operating mode
TOTCNTS   1.1299e+08 1.959e+08 1.8185e+08 1.1561e+08  Total Exp. Meter counts (DN) - four channels (445.0-551.25, 551.25-657.5, 657.5-763.75, 763.75-870.0 nm) 
TOTCORR   2.3994e+08 4.1319e+08 3.8088e+08 2.403e+08  Total Exp. Meter counts (DN), corrected for dead time - four channels (445.0-551.25, 551.25-657.5, 657.5-763.75, 763.75-870.0 nm) 
ETAV1C1T  23.990154                                   Etalon Vescent 1 Channel 1 temperature
ETAV1C2T  23.79949                                    Etalon Vescent 1 Channel 2 temperature
ETAV1C3T  23.599987                                   Etalon Vescent 1 Channel 3 temperature
ETAV1C4T  23.900118                                   Etalon Vescent 1 Channel 4 temperature
ETAV2C3T  24.000668                                   Etalon Vescent 2 Channel 3 temperature
========  ==========================================  =========

2D Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All keywords from Level 0 are inherited by the 2D file.  Below are additional keywords.

========  ==========================================  =========
Keyword   Value (example)                             Comment
========  ==========================================  =========
DRPTAG    v2.5.2                                      Git version number of KPF-Pipeline used for processing
DRPHASH   'ccf5f6ebe0c9ae7d43706cc57fed2ecdeb540a17'  Git commit hash version of KPF-Pipeline used for processing
NOTJUNK   1                                           Quality Control: 1 = not in the list of junk files check; this QC is rerun on L1 and L2
DATAPRL0  1                                           Quality Control: 1 = L0 data products present with non-zero array sizes
KWRDPRL0  1                                           Quality Control: 1 = L0 expected keywords present 
EMSAT     1                                           Quality Control: 1 = Exp Meter not saturated; 0 = 2+ reduced EM pixels within 90% of saturation in EM-SCI or EM-SKY 
EMNEG     1                                           Quality Control: 1 = Exp Meter not negative flux; 0 = 20+ consecutive pixels in summed spectra with negative flux 
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
MEDGRN1   3.9642348e+07                               Median for GREEN_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
P16GRN1   3.9340188e+07                               16th-percentile for GREEN_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
P84GRN1   3.9340188e+07                               84th-percentile for GREEN_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
MEDGRN2   3.9642348e+07                               Median for GREEN_AMP2 [DN] (includes overscan region, excludes NaNs explicitly)
P16GRN2   3.9340188e+07                               16th-percentile for GREEN_AMP2 [DN] (includes overscan region, excludes NaNs explicitly)
P84GRN2   3.9340188e+07                               84th-percentile for GREEN_AMP2 [DN] (includes overscan region, excludes NaNs explicitly)
MEDGRN3   3.9642348e+07                               Median for GREEN_AMP3 [DN] (includes overscan region, excludes NaNs explicitly)
P16GRN3   3.9340188e+07                               16th-percentile for GREEN_AMP3 [DN] (includes overscan region, excludes NaNs explicitly)
P84GRN3   3.9340188e+07                               84th-percentile for GREEN_AMP3 [DN] (includes overscan region, excludes NaNs explicitly)
MEDGRN4   3.9642348e+07                               Median for GREEN_AMP4 [DN] (includes overscan region, excludes NaNs explicitly)
P16GRN4   3.9340188e+07                               16th-percentile for GREEN_AMP4 [DN] (includes overscan region, excludes NaNs explicitly)
P84GRN4   3.9340188e+07                               84th-percentile for GREEN_AMP4 [DN] (includes overscan region, excludes NaNs explicitly)
MEDRED1   3.9642348e+07                               Median for RED_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
P16RED1   3.9340188e+07                               16th-percentile for RED_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
P84RED1   3.9340188e+07                               84th-percentile for RED_AMP1 [DN] (includes overscan region, excludes NaNs explicitly)
MEDRED2   3.9642348e+07                               Median for RED_AMP2 [e-] (includes overscan region, excludes NaNs explicitly)
P16RED2   3.9340188e+07                               16th-percentile for RED_AMP2 [DN] (includes overscan region, excludes NaNs explicitly)
P84RED2   3.9340188e+07                               84th-percentile for RED_AMP2 [DN] (includes overscan region, excludes NaNs explicitly)
MEDCAHK   3.9642348e+07                               Median for CA_HK_AMP [DN] (includes overscan region, excludes NaNs explicitly)
P16CAHK   3.9340188e+07                               16th-percentile for CA_HK_AMP [DN] (includes overscan region, excludes NaNs explicitly)
P84CAHK   3.9340188e+07                               84th-percentile for CA_HK_AMP [DN] (includes overscan region, excludes NaNs explicitly)
========  ==========================================  =========

Keywords related to read noise are only computed for the amplifiers used.  In regular read mode, two amplifiers are used (AMP1 and AMP2), while in fast read mode, four amplifiers are used (AMP1, AMP2, AMP3, and AMP4).

Keywords related to dark current (starting with FLX) are only added for 2D files of Dark observations (no illumination and exposure time > 0). The regions for those keywords refer to the CCD coordinates where the dark current measurements were made (using modules/quicklook/arc/analyze_2d.py).  The image below (click to enlarge) shows the regions and dark current estimates for a 2D spectrum taken when the dark current was high.

Keywords related to the Guider are only added for 2D files that have Guider data products.  Similar for Exposure Meter data products.

Keywords related to L0 amplifier-image statistics (e.g., MEDGRN1) are only added to 2D files.  A robust estimator of data dispersion width is
sigma = 0.5 * (P84 - P16), equivalent to one standard deviation for normally distributed data.

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
MONOTWLS  1                Quality Control: 1 = L1 wavelength solution is monotonic
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

L2 Primary Extension Header and RV Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
========  ===============  =========
Keyword   Value (example)  Comment
========  ===============  =========

DATE-OBS= '2023-11-05'         / Date dcs1.DATE                                 
UT      = '17:47:51.73'        / DCS Universal Time                             
DATE-BEG= '2023-11-05T17:47:51.691957' / Start of exposure from kpfexpose       
EXPTIME =                 10.0 / Requested exposure time                        
PROGNAME= 'ENG     '           / Program name from kpfexpose                    
OBJECT  = 'autocal-thar-all-morn' / Object                                      
INSTRUME= 'KPF     '           / Instrument Name                                
DATALVL = '2       '           / Data Product Base Level                        
OBSERVAT= 'KECK    '           / Observatory name                               
OBSMODE = 'SPEC    '           / Basic mode of observation                      
HEADVER = '127561  '           / Header config version                          
CURRINST= 'HIRES   '           / Selected instrument                            
TELESCOP= 'Keck I  '           / Telescope                                      
TARGNAME= '' / KPF Target name                                                  
DCSNAME = 'HORIZON STOW'       / DCS Target name - may not make any sense       
TARGRA  = '00:00:00.00'        / [h] DCS Target RA                              
TARGDEC = '+00:00:00.0'        / [deg] DCS Target Dec                           
TARGEPOC=                  0.0 / DCS Target epoch                               
TARGEQUI=                  0.0 / DCS Target equinox                             
TARGPLAX=                  0.0 / [arcsec] target parallax                       
TARGPMDC=                  0.0 / [arcsec/yr] target proper motion dec           
TARGPMRA=                  0.0 / [s/yr] target proper motion ra                 
TARGRADV=                  0.0 / [km/s] target radial velocity                  
TARGWAVE=                 0.65 / target-wavelength                              
TARGFRAM= 'mount az/el'        / target-frame                                   
FULLTARG= '' / Full Target name from kpfconfig                                  
GAIAID  = '' / GAIA Target name from kpfconfig                                  
2MASSID = '' / 2MASS Target name from kpfconfig                                 
GAIAMAG = '' / GAIA G band magnitude                                            
2MASSMAG= '' / 2MASS J band magnitude                                           
TARGTEFF=              45000.0 / Target effective temperature (K)               
RA      = '22:40:00.00'        / [h] Right ascension                            
DEC     = '+00:02:19.3'        / [deg] Declination                              
EQUINOX =               2000.0 / DCS Equinox                                    
MJD-OBS =         60253.741571 / Modified Julian day                            
PMFM    =                  0.0 / PMFM value                                     
LFCMODE = 'StandbyHigh'        / LFC Operation Mode                             
AMPON   =             2499.022 / LFC: Amount of time amplifier on               
LFCFO   =          250000000.0 / LFC filtered Offset Freq RR Comb counted       
LFCFREF =          250000000.0 / LFC filtered Offset Freq RR Comb setpoint      
LFCFR   =     19999999999.9804 / LFC filtered Offset Freq RR Filter counted     
LFCFRREF=        20000000000.0 / LFC filtered Offset Freq RR Filter setpoint    
LFCCEOFR=         5220000000.0 / CEO Filtered Setpoint Freq                     
LFCCWFRF=    288005220000000.0 / CW Freq Reference                              
LFCCWFRQ=    288005219851191.0 / CW Freq                                        
LFCCWFER=            -148809.0 / CW Freq Error: Ref-Actual                      
LFCCWMDN=              1152021 / CW mode number                                 
LFCBIACT=                0.033 / Blue cut amp diode current (A)                 
LFCBISET=                  0.0 / Blue cut amp diode setting (A)                 
OCTAGON = 'Th_daily'           / selected octagon value                         
CALMON  = 'Out     '           / Calibration intensity monitor position         
HATCH   = 'Closed  '           / Status of the hatch on the exterior of the FIU 
FIUMODE = 'Calibration'        / FIU operating mode                             
FOLDNAM = 'In      '           / Named Position of FIU Fold mirror              
FOLDVAL =                -13.5 / [mm] Position of FIU Fold mirror               
ADCTRACK= 'Off     '           / Is the ADC Tracking                            
HCLSN   = 'L82932  '           / S/N of lamp in use                             
CAREQ   = 'Yes     '           / Ca HK fibers (6,7) requested                   
EXPSCREQ= 'Yes     '           / ExpM Science fiber (8ab) requested             
FFREQ   = 'No      '           / Flatfield fiber (10ab) requested               
SICALREQ= 'Yes     '           / SimCal fiber (3ab) requested                   
VACSCREQ= 'Yes     '           / Vac Sci fiber (1b) requested                   
VACSKREQ= 'Yes     '           / Vac Sci fiber (2b) requested                   
THDAYON = '20231105T07:00:10 HST' / ThAr Daily was last turned on at this time  
THDAYTON=               2861.4 / ThAr Daily has been on for this long           
THAUON  = '20230503T12:39:28 HST' / ThAr Gold was last turned on at this time   
THAUTON =               1465.7 / ThAr Gold has been on for this long            
UDAYON  = '20231104T16:11:07 HST' / UNe Daily was last turned on at this time   
UDAYTON =               3038.9 / UNe Daily has been on for this long            
UAUON   = '19691231T14:00:00 HST' / UNe Gold was last turned on at this time    
UAUTON  =                  0.0 / UNe Gold has been on for this long             
PTHDAY  =            7.818E-06 / Last ThAr Daily calibration power value (Watts)
PTHDAYT = '20231105T07:38:37 HST' / Time of ThAr Daily last power value         
PTHAU   =                  0.0 / Last ThAr Gold calibration power value (Watts) 
PTHAUT  = '19691231T14:00:00 HST' / Time of ThAr Gold last power value          
PUDAY   =            1.512E-05 / Last UNe Daily calibration power value (Watts) 
PUDAYT  = '20231104T16:57:37 HST' / Time of UNe Daily last power value          
PUAU    =                  0.0 / Last UNe Gold  calibration power value (Watts) 
PUAUT   = '19691231T14:00:00 HST' / Time of UNe Gold last power value           
PLFC    =                  0.0 / Last LFC calibration power value (Watts)       
PLFCT   = '19691231T14:00:00 HST' / Time of LFC last power value                
PETAL   =            2.038E-09 / Last Etalon calibration power value (Watts)    
PETALT  = '20231105T07:30:32 HST' / Time of Etalon last power value             
PBRB    =            9.696E-06 / Last Broadband calibration power value (Watts) 
PBRBT   = '20231104T12:31:09 HST' / Time of Broadband last power value          
PSOL    =                  0.0 / Last SoCal-CalFib cal. power value (Watts)     
PSOLT   = '19691231T14:00:00 HST' / Time of SoCal-CalFib last power value       
COMMENT                                                                         

SRCSHTTR= 'SciSelect,SkySelect,Cal_SciSky' / Source shutters commanded          
TIMSHTTR= 'Scrambler,SimulCal,Ca_HK' / Timed shutters commanded                 
OTIMSHTR= 'Scrambler,SimulCal,Ca_HK' / Timed shutters open exp. midpoint        
SCISEL  = 'open    '           / Science Select shutter at exp. midpoint        
SKYSEL  = 'open    '           / Sky Select Shutter at exp. midpoint            
FFSHTR  = 'closed  '           / Flat field fiber shutter at exp. midpoint      
SCRAMSHT= 'open    '           / Scrambler shutter at exp. midpoint             
SIMCALSH= 'open    '           / Simult Cal shutter at exp. midpoint            
TRIGTARG= 'Green,Red,ExpMeter' / Cameras that were sent triggers                
IMTYPE  = 'Arclamp '           / Image Type                                     
CAL-OBJ = 'Th_daily'           / Calibration fiber source                       
SKY-OBJ = 'Th_daily'           / Sky fiber source                               
SCI-OBJ = 'Th_daily'           / Science fiber source                           
CAFBS   = 'Yes     '           / Ca HK fibers (6,7) on                          
EXPSCIFB= 'Yes     '           / ExpM Science fiber (8ab) on                    
EXPSKYFB= 'No      '           / ExpM Science fiber (9) on                      
FFFB    = 'No      '           / Flatfield fiber (10ab) on                      
FICALFBS= 'Yes     '           / FIU Cal fibers (4,5) on                        
SICALFB = 'Yes     '           / SimCal fiber (3ab) on                          
VACSCFB = 'Yes     '           / Vac Sci fiber (1b) on                          
VACSKFB = 'Yes     '           / Vac Sci fiber (2b) on                          
SCIFB   = 'Yes     '           / Science fiber (1a) on                          
SKYFB   = 'Yes     '           / Sky fiber (2a) on                              
CURRBASE= '(335.5, 258.0)'     / [pix] Selected pointing origin                 
PIXTARG = '(13155.357, 266.95)' / [pix] Selected object tip/tilt target         
GRACFMD5= '1EF8CF3BA1C2905FE9980C295DCB9953' / Green MD5 sum for the acf file   
GRACFFLN= 'regular-read-green.acf' / Green acf file loaded                      
RDACFMD5= '46B91FD6D29EEBBDED920738326414BA' / Red MD5 sum for the acf file     
RDACFFLN= 'regular-read-red.acf' / Red acf file loaded                          
AGITSTA = 'Running '           / Agitator status kpfmot.AGITATOR                
THARGD  = 'Off     '           / Gold ThAr power status                         
UNEGD   = 'Off     '           / Gold UNe power status                          
THARDAY = 'On      '           / Daily ThAr power status                        
UNEDAY  = 'Off     '           / Daily UNe power status                         
OCTBB   = 'Off     '           / Octagon broad band power status                
FFSOURCE= 'Off     '           / Flat field broad band power status             
SCIFBILL= 'Off     '           / Science fiber LED back illuminator power       
SKYFBILL= 'Off     '           / Sky fiber LED back illuminator power           
HKFBILL = 'Off     '           / Ca H and K fiber LED back illuminator power    
EXPFBILL= 'Off     '           / Exposure meter fiber LED back illuminator power
SSCALFW = 'OD 0.1  '           / Sci/Cal FW Position                            
SIMCALFW= 'OD 0.1  '           / Simual Cal FW Position                         
FFFW    = 'Blank   '           / Flatfield FW Position                          
OCTAGON = 'Th_daily'           / selected octagon value                         
PRES    =              624.479 / [hPa] Pressure at Vaisala kpfmet.PRES          
RELH    =               13.465 / Relative humidity at Vaisala kpfmet.RELH       
PONAME  = 'REF     '           / DCS Point origin name                          
PONAME1 = 'REF     '           / DCS Point origin name1                         
DRA     =                  0.0 / [s/s] DCS Diff RA rate                         
DDEC    =                  0.0 / [asec/s] DCS Diff Dec rate                     
RABASE  = '22:40:00.00'        / DCS RA base                                    
RAOFF   =                  0.0 / [asec] DCS RA offset                           
DECBASE = '+00:02:19.3'        / DCS Dec base                                   
DECOFF  =                  0.0 / [asec] DCS Dec offset                          
HA      = '+00:00:00.00'       / DCS Hour angle                                 
AIRMASS =                13.37 / DCS Airmass                                    
PARANG  =                  0.0 / [deg] DCS Parallactic angle astrometric        
PARANTEL=                  0.0 / [deg] DCS Parallactic angle telescope          
EL      =                 0.04 / [deg] DCS Elevation                            
AZ      =                -20.0 / [deg] DCS Azimuth                              
LST     = '10:24:50.37'        / DCS Local sidereal time                        
AXESTAT = 'not controlling'    / DCS axes control status                        
TRACKING= 'no      '           / DCS Servos tracking status                     
DTRACK  = 'disabled'           / DCS differential tracking status               
GUIDING = 'false   '           / DCS Guiding status                             
AUTACTIV= 'no      '           / DCS Guider active                              
AUTFWHM =             1.334309 / [pix] DCS Guider fwhm                          
AUTXCENT=                 -0.6 / [asec] DCS Guider x centroid                   
AUTYCENT=                 -0.1 / [asec] DCS Guider y centroid                   
SECFOCUS=               -2.482 / [mm] Secondary focus                           
TELFOCUS=               -1.631 / [mm] Telescope focus                           
TUBEFLEX=                -13.3 / [arcsec] Telescope tube flexure                
TUBETEMP=                 1.57 / [degC] Telescope tube temperature              
PRIMTEMP=             2.154095 / [degC] Telescope pri temperature               
SECMTEMP=                1.354 / [degC] Telescope sec temperature               
DIFFPTDW=            25.554095 / [decC] Diff between pri mirro temp dewpt       
DIFFSTDW=               24.754 / [decC] Diff between sec mirro temp dewpt       
VIGNETTE= 'false   '           / dome vignette (t/f)                            
STVIGNE = 'false   '           / top shutter vignette (t/f)                     
SBVIGNE = 'false   '           / bottom shutter vignette (t/f)                  
SBELEV  =                23.99 / [deg] bottom shutter elevation                 
STELEV  =                23.99 / [deg] top shutter elevation                    
SECSTST = 'UNINIT  '           / DCS Secondary status string                    
SECTHETX=               -441.0 / [asec] DCS Secondary theta x                   
SECTHETY=                139.0 / [asec] DCS Secondary theta y                   
TERTSTST= 'STANDBY '           / DCS Tertiary status string                     
TERTDEST= 'stowed  '           / DCS Tertiary user destination                  
TERTPOSN= 'stowed  '           / DCS Tertiary user position                     
DOMEPOSN=               140.77 / DCS Dome user position                         
DOMESTST= 'UNINIT  '           / DCS Dome status string                         
CALOCAL =                  2.7 / collimation-azimuth-local                      
CELOCAL =                  1.7 / collimation-elevation-local                    
FOCALSTN= 'rnas (right nasmyth)' / focal-station                                
INSTANGL=                180.0 / porg-to-instrument angle                       
POXPOS  =                  0.0 / pointing-origin-x-position                     
POYPOS  =                  0.0 / pointing-origin-y-position                     
ROTCALAN=                  0.0 / rotator-calibration-angle                      
ROTZERO =                  0.0 / rotator-zero-angle                             
GUIDWAVE=                 0.65 / guidestar-wavelength                           
TIMEERR = 'ok 2 2 {NTP time correct to within 2 ms}' / resp time serv           
ETAV1C1T=                  0.0 / Etalon v1ch1temp                               
ETAV1C2T=                  0.0 / Etalon v1ch2temp                               
ETAV1C3T=                  0.0 / Etalon v1ch3temp                               
ETAV1C4T=                  0.0 / Etalon v1ch4temp                               
ETAV2C3T=                  0.0 / Etalon v2ch3temp                               
TSHKEXP = '2023-11-05 07:47:03.302' / Time of signal sent to start HK exposure  
WSHKEXP =                0.004 / Window of time start HK exposure active        
TSHKSHT = '2023-11-05 07:47:51.695' / Time of signal open to close HK shutter   
WSHKSHT = 0.008999999999999999 / Window of time open HK shutter active          
TSTMSHT = '2023-11-05 07:47:51.691' / Time of signal sent to open timed shutter 
WSTMSHT =                0.005 / Window of time start timed shutter active      

DATE    = '2023-11-05T17:48:01.708332' / End of exposure from kpfexpose.ENDTIME 
DATE-END= '2023-11-05T17:48:01.708332' / End of exposure from kpfexpose.ENDTIME 
ELAPSED =                 10.0 / Actual exposure time from kpfexpose.ELAPSED    
TOTCNTS = '6.3263e+05 1.307e+06 1.6356e+07 3.1946e+07' / Total counts kpf_expmet
TOTCORR = '4.2026e+06 8.6818e+06 1.0864e+08 2.1221e+08' / Total counts corrected
TEHKEXP = '2023-11-05 07:47:03.302' / Time of signal sent to end HK exposure    
WEHKEXP =                0.004 / Window of time end HK exposure active          
TEHKSHT = '2023-11-05 07:48:01.706' / Time of signal sent to close HK shutter   
WEHKSHT =                0.059 / Window of time close HK shutter active         
TETMSHT = '2023-11-05 07:48:01.703' / Time of signal sent to close timed shutter
WETMSHT =                0.004 / Window of time close timed shutter active      
OUTDIR  = '/s/sdata1701/kpfeng/2023nov04/L0' / Output directory                 
OFNAME  = 'KP.20231105.64071.69.fits' / Filename of output file                 
GREENFN = '/s/sdata1701/kpfeng/2023nov04/Green/kpf_137054.fits'                 
REDFN   = '/s/sdata1701/kpfeng/2023nov04/Red/kpf_137054.fits'                   
EXPMETFN= '/s/sdata1701/kpfeng/2023nov04/ExpMeter/kpf_em_137054.fits'           
FRAMENO =               137054                                                  
GRCAMD_V= 'Nov  9 2022 09:50:15' / camerad build date Kwd green CAMD_VER        
GREXPTI =                    0 / exposure time in msec Kwd green EXPTIME        
GRFILENA= 'kpf_137054.fits'    / this filename Kwd green FILENAME               
GRFIRMWA= '/kroot/rel/default/data/kpfgreen/ACF/regular-read-green.acf' / contro
GRGAIN02=                    1 / gain for AD chan 2 Kwd green GAIN02            
GRGAIN03=                    1 / gain for AD chan 3 Kwd green GAIN03            
GRHDRSHI=                    0 / number of HDR right-shift bits Kwd green HDRSHI
GROFFSET=                  100 / offset for AD chan 3 Kwd green OFFSET03        
GRSHUTT =                    T / shutter was enabled Kwd green SHUTTEN          
GRTM_ZO = 'GMT     '           / time zone Kwd green TM_ZONE                    
GRCDS0  = '#eval RGsettleT'    /  Kwd green CDS0                                
GRCDS1  = '#eval CDump - 1'    /  Kwd green CDS1                                
GRCDS2  = '#eval PixelT'       / 2 + SWsettleT Kwd green CDS2                   
GRCDS3  =                  950 /  Kwd green CDS3                                
GRDATAS1= '[4:2044,0:4080]'    / left Kwd green DATASEC1                        
GRDATAS2= '[50:2090,0:4080]'   / right Kwd green DATASEC2                       
GRDATE  = '2023-11-05T17:48:48.562842' / FITS file write time Kwd green DATE    
GROBSERV= '' / Observer name Kwd green OBSERVER                                 
GRPROGNA= 'ENG     '           / Program name Kwd green PROGNAME                
GRACF   = 'regular-read-green' / Last user-chosen ACF key Kwd green ACF         
GRACFFI = 'regular-read-green.acf' / ACF file from ACF key Kwd green ACFFILE    
GRACFMD5= '1EF8CF3BA1C2905FE9980C295DCB9953' / MD5 sum for ACFFILE; unknown if p
GRFRAME =               137054 /  Kwd green FRAMENO                             
GRDATE-B= '2023-11-05T17:47:51.691957' / Shutter-open time Kwd green DATE-BEG   
GRDATE-E= '2023-11-05T17:48:01.708332' / Shutter-close time Kwd green DATE-END  
GRELAPS =               10.016 / Shutter-elapsed time Kwd green ELAPSED         
RDCAMD_V= 'Nov  9 2022 09:50:15' / camerad build date Kwd red CAMD_VER          
RDEXPTI =                    0 / exposure time in msec Kwd red EXPTIME          
RDFILENA= 'kpf_137054.fits'    / this filename Kwd red FILENAME                 
RDFIRMWA= '/kroot/rel/default/data/kpfred/ACF/regular-read-red.acf' / controller
RDHDRSHI=                    0 / number of HDR right-shift bits Kwd red HDRSHIFT
RDSHUTT =                    T / shutter was enabled Kwd red SHUTTEN            
RDTM_ZO = 'GMT     '           / time zone Kwd red TM_ZONE                      
RDCDS0  = '#eval RGsettleT'    /  Kwd red CDS0                                  
RDCDS1  = '#eval CDump - 1'    /  Kwd red CDS1                                  
RDCDS2  = '#eval PixelT'       / 2 + SWsettleT Kwd red CDS2                     
RDCDS3  =                  950 /  Kwd red CDS3                                  
RDTEST  =                  123 / test fitskey from modes file Kwd red TEST      
RDDATE  = '2023-11-05T17:48:48.570956' / FITS file write time Kwd red DATE      
RDOBSERV= '' / Observer name Kwd red OBSERVER                                   
RDPROGNA= 'ENG     '           / Program name Kwd red PROGNAME                  
RDACF   = 'regular-read-red'   / Last user-chosen ACF key Kwd red ACF           
RDACFFI = 'regular-read-red.acf' / ACF file from ACF key Kwd red ACFFILE        
RDACFMD5= '46B91FD6D29EEBBDED920738326414BA' / MD5 sum for ACFFILE; unknown if p
RDFRAME =               137054 /  Kwd red FRAMENO                               
RDDATE-B= '2023-11-05T17:47:51.691957' / Shutter-open time Kwd red DATE-BEG     
RDDATE-E= '2023-11-05T17:48:01.708332' / Shutter-close time Kwd red DATE-END    
RDELAPS =               10.016 / Shutter-elapsed time Kwd red ELAPSED           
EMFRAME =               137054 / Frame number from kpf_expmeter Kwd expmeter FRA
EMFILEN = '/s/sdata1701/kpfeng/2023nov04/ExpMeter/kpf_em_137054.fits' / Output f
EMSEQBEG=                    1 / Sequence number of first observation Kwd expmet
EMSEQEND=                   11 / Sequence number of last observation Kwd expmete
EMDATE-B= '2023-11-05T17:48:01.919' / Date-Beg of first observation Kwd expmeter
EMDATE-E= '2023-11-05T17:48:02.178' / Date-End of last observation Kwd expmeter 
DRPTAG  = 'v2.5.3  '                                                            
DRPHASH = '82cf70d1be2f67a26a39758b50a91c5fa5ebf82d'                            
NOTJUNK =                    1 / QC: Not in list of junk files check            
DATAPRL0=                    1 / QC: L0 data present check                      
KWRDPRL0=                    1 / QC: L0 keywords present check                  
REDAMPS =                    2                                                  
GRNAMPS =                    2                                                  
BIASFILE= 'kpf_20231105_master_bias_autocal-bias.fits'                          
DARKFILE= 'kpf_20231105_master_dark_autocal-dark.fits'                          
FLATFILE= 'kpf_20231105_master_flat.fits'                                       
EXTNAME = 'PRIMARY '           / extension name                                 
ORIGIN  = 'astropy.fits'       / File Originator                                
MONOTWLS=                    1 / QC: Monotonic wavelength-solution check        
WLSFILE = '/masters/20231105/kpf_20231105_master_WLS_autocal-thar-all-eve_L1.f&'
CONTINUE  'its'                                                                 


*To-do: add a list of additional 2D, Level 1, and Level 2 primary keywords.*


WLS Dictionaries
----------------

See :doc:`../analysis/dictonary_format` for details.
