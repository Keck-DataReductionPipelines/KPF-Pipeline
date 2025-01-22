KPF Data Format
===============

Overview
--------

KPF data products are defined for these data levels:

* **Level 0 (L0)**: Raw data products produced by KPF at the W. M. Keck Observatory
* **2D**: Assembled CCD images with minimal processing.  This data product is produced by the DRP during processing from L0 to L1 but is not fundamental and is frequently not archived.
* **Level 1 (L1)**: Extracted, wavelength-calibrated spectra
* **Level 2 (L2)**: Derived data products including cross-correlation functions, radial velocities, and activity indicators.

Each of these data levels is a standardized, multi-extension FITS format, and can be read using standard fits tools (e.g., `astropy.fits.io <https://docs.astropy.org/en/stable/io/fits/>`_) and the `KPF-Pipeline <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline>`_.

KPF L0 files follow the naming convention: KP.YYYYMMDD.SSSSS.ss.fits, where YYYYMMDD is a date and SSSSS.ss is the number of decimal seconds after UT midnight corresponding to the start of the exposure.  2D/L1/L2 files have similar file names, but with '_2D', '_L1', or '_L2' before '.fits'.  For example, KP.YYYYMMDD.SSSSS.ss_2D.fits is a 2D file name.

See the section titled :ref:`label-tutorials` for a set of tutorials on the various KPF data files.

In addition, the DRP is able to produce WLS Dictionaries that contain detailed diagnostic information about the fits of individual lines, orders, and orderlets for the wavelength solutions.  These are described at the bottom of this page.

Data Format of KPF Files
------------------------

L0 FITS Extensions
^^^^^^^^^^^^^^^^^^

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


L1 FITS Extensions
^^^^^^^^^^^^^^^^^^

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
GREEN_SCI_WAVE1      image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX1
GREEN_SCI_WAVE2      image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX2
GREEN_SCI_WAVE3      image      35 x 4080       Wavelength vs. pixel for GREEN_SCI_FLUX3
GREEN_SKY_WAVE       image      35 x 4080       Wavelength vs. pixel for GREEN_SKY_FLUX
GREEN_CAL_WAVE       image      35 x 4080       Wavelength vs. pixel for GREEN_CAL_FLUX
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
RED_SCI_WAVE1        image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX1
RED_SCI_WAVE2        image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX2
RED_SCI_WAVE3        image      32 x 4080       Wavelength vs. pixel for RED_SCI_FLUX3
RED_SKY_WAVE         image      32 x 4080       Wavelength vs. pixel for RED_SKY_FLUX
RED_CAL_WAVE         image      32 x 4080       Wavelength vs. pixel for RED_CAL_FLUX
RED_TELLURIC         table      n/a             Not used yet (will include telluric spectrum)
RED_SKY              table      n/a             Not used yet (will include modeled sky spectrum)
CA_HK_SCI            image      6 x 1024        1D spectra (6 orders) of SCI in Ca H&K spectrometer
CA_HK_SKY            image      6 x 1024        1D spectra (6 orders) of SKY in Ca H&K spectrometer
CA_HK_SCI_WAVE       image      6 x 1024        Wavelength vs. pixel for CA_HK_SCI
CA_HK_SKY_WAVE       image      6 x 1024        Wavelength vs. pixel for CA_HK_SKY
BARY_CORR            table      67              Table of barycentric corrections by spectral order
===================  =========  ==============  =======


L2 FITS Extensions
^^^^^^^^^^^^^^^^^^

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
RV                   table      67              Table of RVs by spectral order (described below)
ACTIVITY             table      n/a             Not used yet (will include activity measurements)
===================  =========  ==============  =======

Primary Extension Header Keywords
---------------------------------

L0 Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
TARGFRAM  FK5                                         Target frame
AIRMASS   1.26                                        Airmass from DCS
PARANTEL  23.58                                       Parallactic angle of the telescope from DCS
HA        +01:01:37.22                                Hour angle
EL        52.46                                       Elevation [deg]
AZ        204.46                                      Azimuth [deg]
LST       07:13:51.02                                 Local sidereal time
RA        06:12:13.80                                 [h] Right ascension
DEC       -14:38:56.0                                 [deg] Declination
EQUINOX   2000.0                                      DCS Equinox
MJD-OBS   60310.21291                                 Modified Julian days
GAIAID    DR3 2993561629444856960                     GAIA Target name
2MASSID   J06121397-1439002                           2MASS Target name
GAIAMAG   9.28                                        GAIA G band magnitude
2MASSMAG  8.06                                        2MASS J band magnitude
TARGTEFF  5398.0                                      Target effective temperature (K)
OCTAGON   EtalonFiber                                 Selected octagon calibration source (not necessarily powered on)
TRIGTARG  Green,Red,Ca_HK,ExpMeter,Guide              Cameras that were sent triggers
IMTYPE    Object                                      Image Type
TARGNAME  42813                                       KPF Target Name
DCSNAME   42813                                       DCS Target Name
FULLTARG  42813                                       Full Target name from kpfconfig
CAL-OBJ   None                                        Calibration fiber source
SKY-OBJ   Sky                                         Sky fiber source
SCI-OBJ   Target                                      Science fiber source
AGITSTA   Running                                     Agitator status
FIUMODE   Observing                                   FIU operating mode
FFFB      Yes                                         Flatfield fiber on
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

The 2D file inherits all L0 keywords.  Below are additional keywords.

DATAPRL0|bool|QC: L0 data products present with non-zero array sizes|None
DATAPR2D|bool|QC: 2D red and green data present|None
CAHKPR2D|bool|QC: 2D CaHK data present|None
KWRDPRL0|bool|QC: L0 expected keywords present|None
TIMCHKL0|bool|QC: Consistent times in L0 file|None
EMSAT|bool|QC: Exp Meter not saturated|None
EMNEG|bool|QC: Exp Meter not negative flux|None
GOODREAD|bool|QC: Texp not consistent with CCD readout error|None
POS2DSNR|bool|QC: 2D Red/Green data/var^0.5 not significantly negative|None
LOWBIAS|bool|QC: 2D bias flux not low|None
LOWDARK|bool|QC: 2D dark flux not low|None
LFC2DFOK|bool|QC: LFC flux meets threshold of 4000 counts|None


========  ==========================================  =========
Keyword   Value (example)                             Comment
========  ==========================================  =========
DRPTAG    v2.5.2                                      Git version number of KPF-Pipeline used for processing
DRPHASH   'ccf5f6ebe0c9ae7d43706cc57fed2ecdeb540a17'  Git commit hash version of KPF-Pipeline used for processing
NOTJUNK   1                                           QC: 1 = not in the list of junk files check; this QC is rerun on L1 and L2
DATAPRL0  1                                           QC: 1 = L0 data products present with non-zero array sizes
KWRDPRL0  1                                           QC: 1 = L0 expected keywords present 
TIMCHKL0  1                                           QC: 1 = consistent times in L0 file
EMSAT     1                                           QC: 1 = Exp Meter not saturated; 0 = 2+ reduced EM pixels within 90% of saturation in EM-SCI or EM-SKY 
EMNEG     1                                           QC: 1 = Exp Meter not negative flux; 0 = 20+ consecutive pixels in summed spectra with negative flux 
DATAPR2D  1                                           QC: 1 = 2D data products present with non-zero array sizes
CAHKPR2D  1                                           QC: 1 = 2D CaHK data present with non-zero array sizes
GOODREAD  1                                           QC: 1 = Exposure time not consistent with CCD readout error (~6 sec)
POS2DSNR  1                                           QC: 1 = 2D Red and Green SNR (data/var^0.5) not significantly negative
LOWBIAS   1                                           QC: 1 = 2D bias flux not low
LOWDARK   1                                           QC: 1 = 2D dark flux not low
LFC2DFOK  1                                           QC: 1 = LFC flux meets threshold of 4000 counts
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
GR2DF99P  30552.46                                    99th percentile flux in 2D Green image (e-)
GR2DF90P  14860.21                                    90th percentile flux in 2D Green image (e-)
GR2DF50P  234.62                                      50th percentile flux in 2D Green image (e-)
GR2DF10P  42.05                                       10th percentile flux in 2D Green image (e-)
RD2DF99P  62520.97                                    99th percentile flux in 2D Red image (e-)
RD2DF90P  40589.16                                    90th percentile flux in 2D Red image (e-)
RD2DF50P  613.23                                      50th percentile flux in 2D Red image (e-)
RD2DF10P  128.83                                      10th percentile flux in 2D Red image (e-)
HK2DF99P  62520.97                                    99th percentile flux in the 2D header (e-)
HK2DF90P  40589.16                                    90th percentile flux in the 2D header (e-)
HK2DF50P  613.23                                      50th percentile flux in the 2D header (e-)
HK2DF10P  128.83                                      10th percentile flux in the 2D header (e-)
AGEBIAS   0                                           Age of master bias file compared to this file (whole days)
AGEDARK   0                                           Age of master dark file compared to this file (whole days)
AGEFLAT   0                                           Age of master flat file compared to this file (whole days)
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

The L1 file inherits all L0 and 2D keywords.  Below are additional important keywords.

========  =======================================================================  =========
Keyword   Value (example)                                                          Comment
========  =======================================================================  =========
WLSFILE   /masters/20231230/kpf_20231230_master_WLS_autocal-lfc-all-eve_L1.fits    First wavelength interpolation reference for this L1 file
WLSFILE2  /masters/20231231/kpf_20231231_master_WLS_autocal-lfc-all-morn_L1.fits   Second wavelength interpolation reference for this L1 file
MONOTWLS  1                                                                        QC: 1 = L1 wavelength solution is monotonic
DATAPRL1  1                                                                        QC: 1 = L1 red and green data present
CAHKPRL1  1                                                                        QC: 1 = CaHK data present in L1 with expected shape
WLSL1     1                                                                        QC: 1 = L1 WLS file check passed
LFCSAT    1                                                                        QC: 1 = L1 LFC spectrum not saturated
SNRSC452  250.0                                                                    SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 452 nm (second bluest order); on Green CCD
SNRSK452  250.0                                                                    SNR of L1 SKY spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
SNRCL452  250.0                                                                    SNR of L1 CAL spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
SNRSC548  250.0                                                                    SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 548 nm; on Green CCD
SNRSK548  250.0                                                                    SNR of L1 SKY spectrum (95th %ile) near 548 nm; on Green CCD
SNRCL548  250.0                                                                    SNR of L1 CAL spectrum (95th %ile) near 548 nm; on Green CCD
SNRSC652  250.0                                                                    SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 652 nm; on Red CCD
SNRSK652  250.0                                                                    SNR of L1 SKY spectrum (95th %ile) near 652 nm; on Red CCD
SNRCL652  250.0                                                                    SNR of L1 CAL spectrum (95th %ile) near 652 nm; on Red CCD
SNRSC747  250.0                                                                    SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 747 nm; on Red CCD
SNRSK747  250.0                                                                    SNR of L1 SKY spectrum (95th %ile) near 747 nm; on Red CCD
SNRCL747  250.0                                                                    SNR of L1 CAL spectrum (95th %ile) near 747 nm; on Red CCD
SNRSC852  250.0                                                                    SNR of L1 SCI (SCI1+SCI2+SCI3; 95th %ile) near 852 nm (second reddest order); on Red CCD
SNRSK852  250.0                                                                    SNR of L1 SKY spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
SNRCL852  250.0                                                                    SNR of L1 CAL spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
FR452652  1.2345                                                                   Peak flux ratio between orders (452nm/652nm) using SCI2
FR548652  1.2345                                                                   Peak flux ratio between orders (548nm/652nm) using SCI2
FR747652  1.2345                                                                   Peak flux ratio between orders (747nm/652nm) using SCI2
FR852652  1.2345                                                                   Peak flux ratio between orders (852nm/652nm) using SCI2
FR12M452  0.9000                                                                   median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FR12U452  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FR32M452  0.9000                                                                   median(SCI3/SCI2) flux ratio near 452 nm; on Green CCD
FR32U452  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
FRS2M452  0.9000                                                                   median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
FRS2U452  0.0010                                                                   uncertainty on the median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
FRC2M452  0.9000                                                                   median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
FRC2U452  0.0010                                                                   uncertainty on the median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
FR12M548  0.9000                                                                   median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FR12U548  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FR32M548  0.9000                                                                   median(SCI3/SCI2) flux ratio near 548 nm; on Green CCD
FR32U548  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
FRS2M548  0.9000                                                                   median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
FRS2U548  0.0010                                                                   uncertainty on the median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
FRC2M548  0.9000                                                                   median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
FRC2U548  0.0010                                                                   uncertainty on the median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
FR12M652  0.9000                                                                   median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FR12U652  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FR32M652  0.9000                                                                   median(SCI3/SCI2) flux ratio near 652 nm; on Red CCD
FR32U652  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
FRS2M652  0.9000                                                                   median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
FRS2U652  0.0010                                                                   uncertainty on the median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
FRC2M652  0.9000                                                                   median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
FRC2U652  0.0010                                                                   uncertainty on the median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
FR12M747  0.9000                                                                   median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FR12U747  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FR32M747  0.9000                                                                   median(SCI3/SCI2) flux ratio near 747 nm; on Red CCD
FR32U747  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
FRS2M747  0.9000                                                                   median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
FRS2U747  0.0010                                                                   uncertainty on the median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
FRC2M747  0.9000                                                                   median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
FRC2U747  0.0010                                                                   uncertainty on the median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
FR12M852  0.9000                                                                   median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FR12U852  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FR32M852  0.9000                                                                   median(SCI3/SCI2) flux ratio near 852 nm; on Red CCD
FR32U852  0.0010                                                                   uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
FRS2M852  0.9000                                                                   median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
FRS2U852  0.0010                                                                   uncertainty on the median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
FRC2M852  0.9000                                                                   median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
FRC2U852  0.0010                                                                   uncertainty on the median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
AGEWLS    -0.2205656666666667                                                      Approx age of WLSFILE compared to this file (days)
AGEWLS2   0.1419343333333333                                                       Approx age of WLSFILE2 compared to this file (days)
========  =======================================================================  =========

The keywords above related to the signal-to-noise ratio in L1 spectra all start with 'SNR'.  These measurements were made using modules/quicklook/src/analyze_l1.py.  The image below (click to enlarge) shows the spectral orders and wavelengths at which SNR is measured.

Keywords related to flux ratios between orders (FR452652, FR548652, FR747652, FR852652) are the ratios between the 95th percentile in flux for the spectral orders containing 452 nm, 548 nm, 747 nm, and 852 nm, all normalized by the spectral order containing 652 nm.  These are the same spectral orders used for the SNR calculations and use the SCI2 orderlet.

Keywords related to orderlet flux ratios (e.g., FR12M452 and its uncertainty FR12U452) are computed in 500-pixel regions in the centers in the same spectral orders as are used for the SNR calculations.

.. image:: KPF_L1_SNR.png
   :alt: L1 Spectrum show wavelengths where SNR is measured
   :align: center
   :height: 400px
   :width: 600px

L2 Primary Extension Header
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The L2 file inherits all L0, 2D, and L1 keywords.  Below are additional important keywords.

========  ==========================================  =========
Keyword   Value (example)                             Comment
========  ==========================================  =========
CCFRV     19.4247572623                               Average of CCD1RV and CCD2RV using weights from RV table
CCFERV    0.001175044                                 Error on CCFRV
CCFRVC    19.4247572623                               Average of CCD1RVC and CCD2RVC using weights from RV table
CCFERVC   0.001175044                                 Error on CCFRVC
CCFBJD    2460662.094073044                           Weighted average of BJD times for spectral orders
CCFBCV    21.751977696646478                          Weighted average of barycentric RV (km/s) for spectral orders
TIMCHKL2  1                                           QC: 1 = consistent times in L2 file
DATAPRL2  1                                           QC: 1 = L2 data is present
========  ==========================================  =========

Radial Velocities
-----------------

L2 RV Extension Header
^^^^^^^^^^^^^^^^^^^^^^

The header to the RV extension (not the primary extension) contains this information about RVs computed using the CCF technique. CCD1 refers to the Green CCD (445-600 nm) and CCD2 refers to the Red CCD (600-870 nm).

To-do, add notes on: 

- recommendations for which RVs to use in papers
- how the orders are averaged using weights.  
- precisely how the RVs are computed (refer to a paper on the CCF algorithm that we're using)
- how the errors are computed
- is BJD = BJD :sub:`TBD`? 
- Test equation for rst syntax: :math:`y = x^2`

=============  =================  =========
Keyword        Value (example)    Comment
=============  =================  =========
CCD1ROW        0                  Row number in the RV table (below) of the bluest order on the Green CCD
CCD1RV1        19.4247572623      RV (km/s) of SCI1 (all orders, Green CCD); corrected for barycentric RV
CCD1ERV1       0.0013815112       Error on CCD1RV1
CCD1RV2        19.3879442221      RV (km/s) of SCI2 (all orders, Green CCD); corrected for barycentric RV
CCD1ERV2       0.001175044        Error on CCD1RV2
CCD1RV3        19.3740241724      RV (km/s) of SCI3 (all orders, Green CCD); corrected for barycentric RV
CCD1ERV3       0.0012185926       Error on CCD1RV3
CCD1RVC        0.0                RV (km/s) of CAL (all orders, Green CCD); corrected for barycentric RV
CCD1ERV        0.0                Error on CCD1RVC
CCD1RVS        18.2490292404      RV (km/s) of SKY (all orders, Green CCD); corrected for barycentric RV
CCD1ERVS       0.0                Error on CCD1RVS
CCD1RV         19.395608349       RV (km/s) of average of SCI1/SCI2/SCI3 (all orders, Green CCD); corrected for barycentric RV
CCD1ERV        0.0007214256       Error on CCD1RV  
CCD1BJD        2460237.787166463  Photon-weighted mid-time (BJD) for CCD1RV
CCD2ROW        35                 Row number in the RV table (below) of the bluest order on the Red CCD
CCD2RV1        19.4423673077      RV (km/s) of SCI1 (all orders, Red CCD); corrected for barycentric RV
CCD2ERV1       0.004087698        Error on CCD2RV1
CCD2RV2        19.3979186805      RV (km/s) of SCI2 (all orders, Red CCD); corrected for barycentric RV
CCD2ERV2       0.0034324475       Error on CCD2RV2
CCD2RV3        19.3808011301      RV (km/s) of SCI3 (all orders, Red CCD); corrected for barycentric RV
CCD2ERV3       0.0035412025       Error on CCD2RV3
CCD2RVC        0.0                RV (km/s) of CAL (all orders, Red CCD); corrected for barycentric RV
CCD2ERVC       0.0                Error on CCD2RVC
CCD2RVS        51.9730319697      RV (km/s) of SKY (all orders, Red CCD); corrected for barycentric RV
CCD2ERVS       0.0                Error on CCD2RVS
CCD2RV         19.4069470745      RV (km/s) of average of SCI1/SCI2/SCI3 (all orders, Red CCD); corrected for barycentric RV
CCD2ERV        0.0021111409       Error on CCD2RV  
CCD2BJD        2460237.787150946  Photon-weighted mid-time (BJD) for CCD2RV
=============  =================  =========

L2 RV Extension
^^^^^^^^^^^^^^^

The RV extension in an L2 file contains the order-by-order RV information for each orderlet (SCI1, SCI2, SCI3, CAL, SKY) determined by the CCF technique.  This extension is a FITS table that is converted into a Pandas dataframe if the L2 file is read by `kpfpipe.models.level2.KPF2.from_fits()`.  The table rows correspond to the spectral orders, with the values of the keywords `CCD1ROW` and `CCD2ROW` in the RV extension header giving the rows where the Green and Red orders start, respectively.  The table columns are listed below.

=============  =================  =========
Column         Value (example)    Comment
=============  =================  =========
orderlet1      19.250267          RV (km/s) of SCI1 (Green CCD); corrected for barycentric RV
orderlet2      19.264743          RV (km/s) of SCI2 (Green CCD); corrected for barycentric RV
orderlet3      19.388630          RV (km/s) of SCI3 (Green CCD); corrected for barycentric RV
s_wavelength   4505.907677        starting wavelength for order
e_wavelength   4462.664498        ending wavelength for order
segment no.    0                  Segment number (for full-order CCF RVs, segment no. = order no.)
order no.      0                  Order number
RV             19.306370          RV (km/s) of average of SCI1/SCI2/SCI3 (Green CCD); corrected for barycentric RV
RV error       0.019248           error on 'RV'
CAL RV         0.0                RV (km/s) of CAL (Green CCD); corrected for barycentric RV
CAL error      0.0                error on 'CAL RV'
SKY RV         0.0                RV (km/s) of sKY (Green CCD); corrected for barycentric RV
SKY error      0.0                error on 'SKY RV'
CCFBJD         2.460238e+06       Photon-weighted mid-time (BJD) for CCD1RV
Bary_RVC       -8.729925          Barycentric RV (km/s)
source1        GREEN_SCI_FLUX1    name of array for orderlet1 (SCI1)
source2        GREEN_SCI_FLUX2    name of array for orderlet2 (SCI2)
source3        GREEN_SCI_FLUX3    name of array for orderlet3 (SCI3)
source CAL     GREEN_CAL_FLUX     name of array for CAL
source SKY     GREEN_SKY_FLUX     name of array for SKY
CCF Weights    0.2590             weight for this order
=============  =================  =========


WLS Dictionaries
----------------

See :doc:`../analysis/dictonary_format` for details.


Notes on Dates and Times in KPF Files
-------------------------------------
* To do: add notes here about how DATE-BEG, DATE-MID, and DATE-END are computed.  There are other datetimes in the header that should be clarified.  Also, explain how exposure midpoints are computed (using the exposure meter and DATE-BEG??), which leads to BJD and ultimately the barycentric corrections.
