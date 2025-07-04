# This file contains methods to write diagnostic information to the KPF headers.

# Standard dependencies
import os
import traceback
import numpy as np
from datetime import datetime, timedelta
from astropy.time import Time

# Local dependencies
from kpfpipe.models.level1 import KPF1
from modules.Utils.utils import DummyLogger
from modules.quicklook.src.analyze_2d import Analyze2D
from modules.quicklook.src.analyze_guider import AnalyzeGuider
from modules.quicklook.src.analyze_hk import AnalyzeHK
from modules.quicklook.src.analyze_em import AnalyzeEM
from modules.quicklook.src.analyze_l1 import AnalyzeL1
from modules.quicklook.src.analyze_l1 import uncertainty_median
from modules.quicklook.src.analyze_l2 import AnalyzeL2
from modules.Utils.kpf_parse import get_data_products_2D
from modules.Utils.kpf_parse import get_data_products_L1
from modules.Utils.kpf_parse import get_data_products_L2
from modules.Utils.kpf_parse import get_datecode_from_filename
from modules.Utils.kpf_parse import HeaderParse, get_datetime_obsid, get_kpf_level, get_data_products_expected
from modules.Utils.utils import get_moon_sep, get_sun_alt
from modules.calibration_lookup.src.alg import GetCalibrations

DEFAULT_CALIBRATION_CFG_PATH = os.path.join(os.path.dirname(__file__), '../../calibration_lookup/configs/default.cfg')
DEFAULT_CALIBRATION_CFG_PATH = os.path.normpath(DEFAULT_CALIBRATION_CFG_PATH)


def add_headers_L0_nonGaussian_read_noise(L0, logger=None):
    """
    Adds keywords to the L0 object header for a read noise metric equal to 
        (0.7979*stdev(region)/mad(region)), where stdev 
        is the standard deviation of a given overscan region, mad is the mean 
        absolute deviation of a given overscan region.  
        This should be = 1.00 for Gaussian noise.  
        
        For a simple noise model with two Gaussians centered on zero with 
        a ratio of sigma values s and with the high-noise Gaussian having 
        a relative total flux (area under the Gaussian) f, 
        this metric: 0.7979*stdev(region)/mad(region) = f * (s^2 - 1)
    
    Keywords:
        RNNGGR1 - Non-Gaussian read noise GREEN1, 0.8*stddev/mad of overscan
        RNNGGR2 - Non-Gaussian read noise GREEN2, 0.8*stddev/mad of overscan
        RNNGGR3 - Non-Gaussian read noise GREEN3, 0.8*stddev/mad of overscan
        RNNGGR4 - Non-Gaussian read noise GREEN4, 0.8*stddev/mad of overscan
        RNNGRD1 - Non-Gaussian read noise RED1, 0.8*stddev/mad of overscan
        RNNGRD2 - Non-Gaussian read noise RED2, 0.8*stddev/mad of overscan
        RNNGRD3 - Non-Gaussian read noise RED3, 0.8*stddev/mad of overscan
        RNNGRD4 - Non-Gaussian read noise RED4, 0.8*stddev/mad of overscan

    Args:
        L0 - a KPF L0 object 

    Returns:
        L0 - a L0 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L0(L0)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L0)) != "<class 'kpfpipe.models.level0.KPF0'>" or chips == []:
        logger.info('Not a valid L0 or no Gree/Red CCD data.')
        return L0
        
    # Use the AnalyzeL0 class measure non-Gaussian read noise
    try:
        myL0 = AnalyzeL0(L0, logger=logger)
        for chip in chips:
            if chip == 'green':
                try:
                    if 'GREEN_AMP1' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGGR1'] = (round(myL0.std_mad_norm_ratio_overscan['GREEN_AMP1'],5), 'Non-Gaussian read noise GREEN1, 0.8*stddev/mad of overscan')
                    if 'GREEN_AMP2' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGGR2'] = (round(myL0.std_mad_norm_ratio_overscan['GREEN_AMP2'],5), 'Non-Gaussian read noise GREEN2, 0.8*stddev/mad of overscan')
                    if 'GREEN_AMP3' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGGR3'] = (round(myL0.std_mad_norm_ratio_overscan['GREEN_AMP3'],5), 'Non-Gaussian read noise GREEN3, 0.8*stddev/mad of overscan')
                    if 'GREEN_AMP4' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGGR4'] = (round(myL0.std_mad_norm_ratio_overscan['GREEN_AMP4'],5), 'Non-Gaussian read noise GREEN4, 0.8*stddev/mad of overscan')
                except Exception as e:
                    logger.error(f"Problem with L0 non-Gaussian read noise measurements Green: {e}\n{traceback.format_exc()}")
            if chip == 'red':
                try:
                    if 'RED_AMP1' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGRD1'] = (round(myL0.std_mad_norm_ratio_overscan['RED_AMP1'],5), 'Non-Gaussian read noise RED1, 0.8*stddev/mad of overscan')
                    if 'RED_AMP2' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGRD2'] = (round(myL0.std_mad_norm_ratio_overscan['RED_AMP2'],5), 'Non-Gaussian read noise RED2, 0.8*stddev/mad of overscan')
                    if 'RED_AMP3' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGRD3'] = (round(myL0.std_mad_norm_ratio_overscan['RED_AMP3'],5), 'Non-Gaussian read noise RED3, 0.8*stddev/mad of overscan')
                    if 'RED_AMP4' in myL0.std_mad_norm_ratio_overscan:
                        L0.header['PRIMARY']['RNNGRD4'] = (round(myL0.std_mad_norm_ratio_overscan['RED_AMP4'],5), 'Non-Gaussian read noise RED4, 0.8*stddev/mad of overscan')
                except Exception as e:
                    logger.error(f"Problem with L0 non-Gaussian read noise measurements Red: {e}\n{traceback.format_exc()}")
    except:
        logger.error(f"Problem with L0 non-Gaussian read noise measurements: {e}\n{traceback.format_exc()}")

    return L0


def add_headers_2D_flux(D2, logger=None):
    """
    Adds keywords to the 2D object header for flux measurements
    
    Keywords:
        GR2DF99P - 99th percentile flux in the 2D Green image (e-)
        GR2DF90P - 90th percentile flux in the 2D Green image (e-)
        GR2DF50P - 50th percentile flux in the 2D Green image (e-)
        GR2DF10P - 10th percentile flux in the 2D Green image (e-)
        RD2DF99P - 99th percentile flux in the 2D Red image (e-)
        RD2DF90P - 90th percentile flux in the 2D Red image (e-)
        RD2DF50P - 50th percentile flux in the 2D Red image (e-)
        RD2DF10P - 10th percentile flux in the 2D Red image (e-)

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_2D(D2)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>" or chips == []:
        logger.info('Not a valid 2D or no Gree/Red CCD data.')
        return D2
        
    # Use the Analyze2D class to compute flux
    my2D = Analyze2D(D2, logger=logger)
    for chip in chips:
        if chip == 'green':
            try:
                D2.header['PRIMARY']['GR2DF99P'] = (round(my2D.green_percentile_99, 3), '99th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF90P'] = (round(my2D.green_percentile_90, 3), '90th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF50P'] = (round(my2D.green_percentile_50, 3), '50th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF10P'] = (round(my2D.green_percentile_10, 3), '10th percentile flux in 2D Green image (e-)')
            except Exception as e:
                logger.error(f"Problem with Green 2D flux measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                D2.header['PRIMARY']['RD2DF99P'] = (round(my2D.red_percentile_99, 3), '99th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF90P'] = (round(my2D.red_percentile_90, 3), '90th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF50P'] = (round(my2D.red_percentile_50, 3), '50th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF10P'] = (round(my2D.red_percentile_10, 3), '10th percentile flux in 2D Red image (e-)')
            except Exception as e:
                logger.error(f"Problem with Red 2D flux measurements: {e}\n{traceback.format_exc()}")
    return D2


def add_headers_dark_current_2D(D2, logger=None):
    """
    Compute the dark current for dark files and adds keywords to the 2D object header

    Keywords:
        FLXREG1G - Dark current [e-/hr] - Green CCD region 1 - coords = [1690:1990,1690:1990]
        FLXREG2G - Dark current [e-/hr] - Green CCD region 2 - coords = [1690:1990,2090:2390]
        FLXREG3G - Dark current [e-/hr] - Green CCD region 3 - coords = [2090:2390,1690:1990]
        FLXREG4G - Dark current [e-/hr] - Green CCD region 4 - coords = [2090:2390,2090:2390]
        FLXREG5G - Dark current [e-/hr] - Green CCD region 5 - coords = [80:380,3080:3380]
        FLXREG6G - Dark current [e-/hr] - Green CCD region 6 - coords = [1690:1990,1690:1990]
        FLXAMP1G - Dark current [e-/hr] - Green CCD amplifier region 1 - coords = [3700:4000,700:1000]
        FLXAMP2G - Dark current [e-/hr] - Green CCD amplifier region 2 - coords = [3700:4000,3080:3380]
        FLXCOLLG - Dark current [e-/hr] - Green CCD collimator-side region = [3700:4000,700:1000]
        FLXECHG  - Dark current [e-/hr] - Green CCD echelle-side region = [3700:4000,700:1000]
        FLXREG1R - Dark current [e-/hr] - Red CCD region 1 - coords = [1690:1990,1690:1990]
        FLXREG2R - Dark current [e-/hr] - Red CCD region 2 - coords = [1690:1990,2090:2390]
        FLXREG3R - Dark current [e-/hr] - Red CCD region 3 - coords = [2090:2390,1690:1990]
        FLXREG4R - Dark current [e-/hr] - Red CCD region 4 - coords = [2090:2390,2090:2390]
        FLXREG5R - Dark current [e-/hr] - Red CCD region 5 - coords = [80:380,3080:3380]
        FLXREG6R - Dark current [e-/hr] - Red CCD region 6 - coords = [1690:1990,1690:1990]
        FLXAMP1R - Dark current [e-/hr] - Red CCD amplifier region 1 = [3700:4000,700:1000]
        FLXAMP2R - Dark current [e-/hr] - Red CCD amplifier region 2 = [3700:4000,3080:3380]
        FLXCOLLR - Dark current [e-/hr] - Red CCD collimator-side region = [3700:4000,700:1000]
        FLXECHR  - Dark current [e-/hr] - Red CCD echelle-side region = [3700:4000,700:1000]

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    data_products = get_data_products_2D(D2)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    if str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>" or chips == []:
        print('Not a valid 2D.')
        return D2
    
    # list of dark current measurements generated by Analyze2D.measure_2D_dark_current
    keywords = {
        'g_ref1': {'key': 'ref1', 'keyword': 'FLXREG1G', 'comment': 'dark e-/hr Green reg1=[1690:1990,1690:1990]'},
        'g_ref2': {'key': 'ref2', 'keyword': 'FLXREG2G', 'comment': 'dark e-/hr Green reg2=[1690:1990,2090:2390]'},
        'g_ref3': {'key': 'ref3', 'keyword': 'FLXREG3G', 'comment': 'dark e-/hr Green reg3=[2090:2390,1690:1990]'},
        'g_ref4': {'key': 'ref4', 'keyword': 'FLXREG4G', 'comment': 'dark e-/hr Green reg4=[2090:2390,2090:2390]'},
        'g_ref5': {'key': 'ref5', 'keyword': 'FLXREG5G', 'comment': 'dark e-/hr Green reg5=[80:380,3080:3380]'},
        'g_ref6': {'key': 'ref6', 'keyword': 'FLXREG6G', 'comment': 'dark e-/hr Green reg6=[1690:1990,1690:1990]'},
        'g_amp1': {'key': 'amp1', 'keyword': 'FLXAMP1G', 'comment': 'dark e-/hr Green amp reg1=[3700:4000,700:1000]'},
        'g_amp2': {'key': 'amp2', 'keyword': 'FLXAMP2G', 'comment': 'dark e-/hr Green amp reg2=[3700:4000,3080:3380]'},
        'g_coll': {'key': 'coll', 'keyword': 'FLXCOLLG', 'comment': 'dark e-/hr Green coll reg=[3700:4000,700:1000]'},
        'g_ech':  {'key': 'ech',  'keyword': 'FLXECHG' , 'comment': 'dark e-/hr Green ech reg=[3700:4000,700:1000]'},
        'r_ref1': {'key': 'ref1', 'keyword': 'FLXREG1R', 'comment': 'dark e-/hr Red reg1=[1690:1990,1690:1990]'},
        'r_ref2': {'key': 'ref2', 'keyword': 'FLXREG2R', 'comment': 'dark e-/hr Red reg2=[1690:1990,2090:2390]'},
        'r_ref3': {'key': 'ref3', 'keyword': 'FLXREG3R', 'comment': 'dark e-/hr Red reg3=[2090:2390,1690:1990]'},
        'r_ref4': {'key': 'ref4', 'keyword': 'FLXREG4R', 'comment': 'dark e-/hr Red reg4=[2090:2390,2090:2390]'},
        'r_ref5': {'key': 'ref5', 'keyword': 'FLXREG5R', 'comment': 'dark e-/hr Red reg5=[80:380,3080:3380]'},
        'r_ref6': {'key': 'ref6', 'keyword': 'FLXREG6R', 'comment': 'dark e-/hr Red reg6=[1690:1990,1690:1990]'},
        'r_amp1': {'key': 'amp1', 'keyword': 'FLXAMP1R', 'comment': 'dark e-/hr Red amp reg1=[3700:4000,700:1000]'},
        'r_amp2': {'key': 'amp2', 'keyword': 'FLXAMP2R', 'comment': 'dark e-/hr Red amp reg2=[3700:4000,3080:3380]'},
        'r_coll': {'key': 'coll', 'keyword': 'FLXCOLLR', 'comment': 'dark e-/hr Red coll reg=[3700:4000,700:1000]'},
        'r_ech':  {'key': 'ech',  'keyword': 'FLXECHR' , 'comment': 'dark e-/hr Red ech reg=[3700:4000,700:1000]'}
               }

    # Use the Analyze2D class to compute dark current
    my2D = Analyze2D(D2, logger=logger)
    for chip in chips:
         my2D.measure_2D_dark_current(chip=chip)
         for k in keywords:
             if k[0] == chip[0]: # match the 'g' in 'green' (first character); similar for 'r'
                keyword = keywords[k]['keyword']
                comment = keywords[k]['comment']
                value = None
                if chip == 'green':
                    try:
                        if hasattr(my2D, 'green_dark_current_regions'):
                            if 'med_elec' in my2D.green_dark_current_regions[keywords[k]['key']]:
                                value = "{:.3f}".format(my2D.green_dark_current_regions[keywords[k]['key']]['med_elec'])
                    except Exception as e:
                        logger.error(f"Problem with green dark current : {e}\n{traceback.format_exc()}")
                if chip == 'red':
                    try:
                        if hasattr(my2D, 'red_dark_current_regions'):
                            if 'med_elec' in my2D.red_dark_current_regions[keywords[k]['key']]:
                                value = "{:.3f}".format(my2D.red_dark_current_regions[keywords[k]['key']]['med_elec'])
                    except Exception as e:
                        logger.error(f"Problem with red dark current: {e}\n{traceback.format_exc()}")                
                if value != None:
                    D2.header['PRIMARY'][keyword] = (value, comment)
    
    return D2


def add_headers_guider(D2, logger=None):
    """
    Adds guider-related information to the header of a 2D object
    
    Keywords:
        GDRXRMS  - x-coordinate RMS guiding error in milliarcsec (mas)
        GDRYRMS  - y-coordinate RMS guiding error in milliarcsec (mas)
        GDRRRMS  - r-coordinate RMS guiding error in milliarcsec (mas)
        GDRXBIAS - x-coordinate bias guiding error in milliarcsec (mas)
        GDRYBIAS - y-coordinate bias guiding error in milliarcsec (mas)
        GDRSEEJZ - Seeing (arcsec) in J+Z-band from Moffat func fit
        GDRSEEV  - Scaled seeing (arcsec) in V-band from J+Z-band
        GDRFWMD - Guider images: median(FWHM [mas])
        GDRFWSTD - Guider images: std(FWHM [mas])
        GDRFXMD - Guider images: median(flux [ADU])
        GDRFXSTD - Guider images: std(flux [ADU])
        GDRPKMD - Guider images: median(peak_flux [ADU])
        GDRPKSTD - Guider images: std(peak_flux [ADU])
        GDRFRSAT - Guider: frac of frames w/in 90% saturated
        GDRNSAT - Guider: num 90% saturated pix in co-added image          
        MOONSEP  - Separation between Moon and target star (deg)
        SUNALT   - Altitude of Sun (deg)

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    data_products = get_data_products_2D(D2)
    if (str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>") or not ('Guider' in data_products):
        logger.info('Guider not in the 2D file or not a valid.  Guider data products not added to header.')
        return D2
        
    # Use the AnalyzeGuider class to compute data products
    myGuider = AnalyzeGuider(D2, logger=logger)
    myGuider.measure_seeing()
    try: 
        D2.header['PRIMARY']['GDRXRMS']  = (round(myGuider.x_rms, 2),
                                           'x-coordinate RMS guiding error [milliarcsec]')
        D2.header['PRIMARY']['GDRYRMS']  = (round(myGuider.y_rms, 2),
                                           'y-coordinate RMS guiding error [milliarcsec]')
        D2.header['PRIMARY']['GDRRRMS']  = (round(myGuider.r_rms, 2),
                                           'r-coordinate RMS guiding error [milliarcsec]')
        D2.header['PRIMARY']['GDRXBIAS'] = (round(myGuider.x_bias, 2),
                                           'x-coordinate bias guiding error [milliarcsec]')
        D2.header['PRIMARY']['GDRYBIAS'] = (round(myGuider.y_bias, 2),
                                           'y-coordinate bias guiding error [milliarcsec]')
        D2.header['PRIMARY']['GDRFWMD']  = (round(myGuider.fwhm_mas_median, 1),
                                           'Guider images: median(FWHM [mas])')
        D2.header['PRIMARY']['GDRFWSTD'] = (round(myGuider.fwhm_mas_std, 1),
                                           'Guider images: std(FWHM [mas])')
        D2.header['PRIMARY']['GDRFXMD']  = (round(myGuider.flux_median, 1),
                                           'Guider images: median(flux [ADU])')
        D2.header['PRIMARY']['GDRFXSTD'] = (round(myGuider.flux_std, 1),
                                           'Guider images: std(flux [ADU])')
        D2.header['PRIMARY']['GDRPKMD']  = (round(myGuider.peak_flux_median, 1),
                                           'Guider images: median(peak_flux [ADU])')
        D2.header['PRIMARY']['GDRPKSTD'] = (round(myGuider.peak_flux_std, 1),
                                           'Guider images: std(peak_flux [ADU])')
        D2.header['PRIMARY']['GDRFRSAT'] = (round(myGuider.frac_frames_saturated, 5),
                                           'Guider: frac of frames w/in 90% saturated')
        D2.header['PRIMARY']['GDRNSAT']  = (int(myGuider.n_saturated_pixels),
                                           'Guider: num 90% saturated pix in co-added image')            
    except Exception as e:
        logger.error(f"Problem with guider measurements: {e}\n{traceback.format_exc()}")
    try: 
        D2.header['PRIMARY']['MOONSEP']  = (round(get_moon_sep(myGuider.date_mid, myGuider.ra, myGuider.dec), 2),
                                           'Separation between Moon and target star [deg]')
    except Exception as e:
        logger.error(f"Problem with moon separation: {e}\n{traceback.format_exc()}")
    try: 
        D2.header['PRIMARY']['SUNALT']  = (round(get_sun_alt(myGuider.date_mid), 1),
                                           'Altitude of Sun [deg]; negative = below horizon')
    except Exception as e:
        logger.error(f"Problem with Sun altitude: {e}\n{traceback.format_exc()}")
    try: 
        if myGuider.good_fit:
            D2.header['PRIMARY']['GDRSEEJZ'] = (round(myGuider.seeing*myGuider.pixel_scale, 3),
                                               'Seeing [arcsec] in J+Z-band from Moffat fit')
            D2.header['PRIMARY']['GDRSEEV']  = (round(myGuider.seeing_550nm*myGuider.pixel_scale, 3),
                                               'Scaled seeing [arcsec] in V-band from J+Z-band')
    except Exception as e:
        logger.error(f"Problem with guider fit: {e}\n{traceback.format_exc()}")
                                           
    return D2


def add_headers_hk(D2, logger=None):
    """
    Adds HK-related information to the header of a 2D object
    
    Keywords:
        HK2DF99P - 99th percentile flux in the 2D HK image (e-)
        HK2DF90P - 90th percentile flux in the 2D HK image (e-)
        HK2DF50P - 50th percentile flux in the 2D HK image (e-)
        HK2DF10P - 10th percentile flux in the 2D HK image (e-)

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    data_products = get_data_products_2D(D2)
    if (str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>") or not ('HK' in data_products):
        logger.info('CaHK not in the 2D file or not a valid 2D.  CaHK data products not added to header.')
        logger.info(data_products)
        return D2
        
    # Use the AnalyzeGuider class to compute data products
    myHK = AnalyzeHK(D2, logger=logger)
    try: 
        D2.header['PRIMARY']['HK2DF99P'] = (round(myHK.percentile_99, 2), '99th percentile flux in 2D CaHK image')
        D2.header['PRIMARY']['HK2DF90P'] = (round(myHK.percentile_90, 2), '90th percentile flux in 2D CaHK image')
        D2.header['PRIMARY']['HK2DF50P'] = (round(myHK.percentile_50, 2), '50th percentile flux in 2D CaHK image')
        D2.header['PRIMARY']['HK2DF10P'] = (round(myHK.percentile_10, 2), '10th percentile flux in 2D CaHK image')
    except Exception as e:
        logger.error(f"Problem with HK measurements: {e}\n{traceback.format_exc()}")
                                           
    return D2


def add_headers_exposure_meter(D2, logger=None):
    """
    Computes the SCI/SKY flux ratio in the main spectrometer based on exposure meter data products
    
    Keywords:
        SKYSCIMS - SKY/SCI flux ratio in main spectrometer scaled from EM data. 
        EMSCCT48 - cumulative EM counts [ADU] in SCI in 445-870 nm
        EMSCCT45 - cumulative EM counts [ADU] in SCI in 445-551 nm
        EMSCCT56 - cumulative EM counts [ADU] in SCI in 551-658 nm
        EMSCCT67 - cumulative EM counts [ADU] in SCI in 658-764 nm
        EMSCCT78 - cumulative EM counts [ADU] in SCI in 764-870 nm
        EMSKCT48 - cumulative EM counts [ADU] in SKY in 445-870 nm
        EMSKCT45 - cumulative EM counts [ADU] in SKY in 445-551 nm
        EMSKCT56 - cumulative EM counts [ADU] in SKY in 551-658 nm
        EMSKCT67 - cumulative EM counts [ADU] in SKY in 658-764 nm
        EMSKCT78 - cumulative EM counts [ADU] in SKY in 764-870 nm

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    data_products = get_data_products_2D(D2)
    if (str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>") or not ('ExpMeter' in data_products):
        logger.info('ExpMeter not in the 2D file or not a valid 2D.  EM data products not added to header.')
        return D2
        
    # Use the Analyze EM class to data products
    myEM = AnalyzeEM(D2, logger=logger)
    try: 
        D2.header['PRIMARY']['SKYSCIMS'] = (round(myEM.SKY_SCI_main_spectrometer,10),
                                           'SKY/SCI flux ratio in main spectro. based on EM')
        D2.header['PRIMARY']['EMSCCT48'] = (int(myEM.counts_SCI.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SCI in 445-870 nm')
        D2.header['PRIMARY']['EMSCCT45'] = (int(myEM.counts_SCI_551m.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SCI in 445-551 nm')
        D2.header['PRIMARY']['EMSCCT56'] = (int(myEM.counts_SCI_551_658.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SCI in 551-658 nm')
        D2.header['PRIMARY']['EMSCCT67'] = (int(myEM.counts_SCI_658_764.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SCI in 658-764 nm')
        D2.header['PRIMARY']['EMSCCT78'] = (int(myEM.counts_SCI_764p.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SCI in 764-870 nm')
        D2.header['PRIMARY']['EMSKCT48'] = (int(myEM.counts_SKY.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SKY in 445-870 nm')
        D2.header['PRIMARY']['EMSKCT45'] = (int(myEM.counts_SKY_551m.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SKY in 445-551 nm')
        D2.header['PRIMARY']['EMSKCT56'] = (int(myEM.counts_SKY_551_658.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SKY in 551-658 nm')
        D2.header['PRIMARY']['EMSKCT67'] = (int(myEM.counts_SKY_658_764.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SKY in 658-764 nm')
        D2.header['PRIMARY']['EMSKCT78'] = (int(myEM.counts_SKY_764p.sum(axis=0)),
                                           'cumulative EM counts [ADU] in SKY in 764-870 nm')

    except Exception as e:
        logger.error(f"Problem with exposure meter measurements: {e}\n{traceback.format_exc()}")

    return D2


def add_headers_masters_age_2D(D2, logger=None, verbose=False):
    """
    Computes the the number of days between the observation and the
    date of observations for the master bias, master dark, and master flat.
    
    Keywords:
        AGEBIAS - Age of master bias file compared to this file (whole days)
        AGEDARK - Age of master dark file compared to this file (whole days)
        AGEFLAT - Age of master flat file compared to this file (whole days)

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with PRIMARY header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    if (str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>"):
        logger.info('Not a valid 2D.  Master age keywords not added to header.')
        return D2
     
    date_obs_str = D2.header['PRIMARY']['DATE-OBS']
    date_obs_datetime = datetime.strptime(date_obs_str, "%Y-%m-%d").date()
   
    my2D = Analyze2D(D2, logger=logger)
    master_files = ['BIASFILE', 'DARKFILE', 'FLATFILE']
    new_keywords = ['AGEBIAS', 'AGEDARK', 'AGEFLAT']
    for master_file, new_keyword in zip(master_files, new_keywords):
        age_master_file = my2D.measure_master_age(kwd=master_file, verbose=verbose)
        file_error = False
        try:
            if type(age_master_file) == type(0):
                D2.header['PRIMARY'][new_keyword] = (age_master_file, f'{master_file} age compared to this file (whole days)')
            else:
                file_error = True
        except Exception as e:
            file_error = True
            logger.error(f"Problem with {new_keyword} age determination: {e}\n{traceback.format_exc()}")
    
        if file_error:
            logger.error(f"Problem with {new_keyword} age determination: Age of {master_file} compared to this file (whole days) = {new_keyword}")
            D2.header['PRIMARY'][new_keyword] = (-99, 'ERROR: Age of {master_file} compared to this file (whole days)')

    return D2


def add_headers_2D_xdisp_offset(D2, logger=None):
    """
    Adds keywords to the 2D object header for measurements of offsets in 
    cross-dispersion
    
    Keywords:
        XDSPDYG1 - Green cross-dispersion offset [pix] compared to master reference
        XDSPDYG2 - Green cross-dispersion offset [pix] compared to reference in era
        XDSPDYR1 - Red cross-dispersion offset [pix] compared to master reference
        XDSPDYR2 - Red cross-dispersion offset [pix] compared to reference in era
        XDSPSYG1 - Uncertainty [pix] in XDSPDYG1 
        XDSPSYG2 - Uncertainty [pix] in XDSPDYG2
        XDSPSYR1 - Uncertainty [pix] in XDSPDYR1
        XDSPSYR2 - Uncertainty [pix] in XDSPDYR2

    Args:
        D2 - a KPF 2D object 

    Returns:
        D2 - a 2D file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_2D(D2)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>" or chips == []:
        print('Not a valid 2D.')
        return D2
        
    # Compute cross-dispersion offsets with two references: global and in era
    for ref in ['global', 'era']:
        if ref == 'era':
            dt = get_datetime_obsid(my2D.ObsID).strftime('%Y-%m-%dT%H:%M:%S.%f')
            keyword_suffix = '2'
            comment_txt = 'in-era reference'
        elif ref == 'global':
            dt = '2024-02-11T00:00:00.000000' # reference time for all KPF observations
            keyword_suffix = '1'
            comment_txt = 'global reference'
        default_config_path = '/code/KPF-Pipeline/modules/calibration_lookup/configs/default.cfg'
        GC = GetCalibrations(dt, default_config_path, use_db=False)
        wls_dict = GC.lookup(subset=['xdisp_ref'])
        reference_file = wls_dict['xdisp_ref']
        my2D = Analyze2D(D2, logger=logger)
        if 'master' in reference_file:
            ref_extension = 'CCD_STACK'
        else:
            ref_extension = None
        
        for chip in chips:
            if chip == 'green':
                try:
                    my2D.measure_xdisp_offset(chip='green', ref_image=reference_file, ref_extension=ref_extension)
                    keyword_value = f'{my2D.green_offset:.5f}'
                    keyword_sigma = f'{my2D.green_offset_sigma:.5f}'
                    D2.header['PRIMARY']['XDSPDYG'+keyword_suffix] = (keyword_value, '[pix] Green x-disp offset; '+comment_txt)
                    D2.header['PRIMARY']['XDSPSYG'+keyword_suffix] = (keyword_sigma, '[pix] uncertainty in XDSPDYG'+keyword_suffix)
                except Exception as e:
                    logger.error(f"Problem with Green 2D cross-dispersion offset measurements: {e}\n{traceback.format_exc()}")
            if chip == 'red':
                try:
                    my2D.measure_xdisp_offset(chip='red', ref_image=reference_file, ref_extension=ref_extension)
                    keyword_value = f'{my2D.red_offset:.5f}'
                    keyword_sigma = f'{my2D.red_offset_sigma:.5f}'
                    D2.header['PRIMARY']['XDSPDYR'+keyword_suffix] = (keyword_value, '[pix] Red x-disp offset; '+comment_txt)
                    D2.header['PRIMARY']['XDSPSYR'+keyword_suffix] = (keyword_sigma, '[pix] uncertainty in XDSPDYR'+keyword_suffix)
                except Exception as e:
                    logger.error(f"Problem with Red 2D cross-dispersion offset measurements: {e}\n{traceback.format_exc()}")
    return D2


def add_headers_masters_age_L1(L1, logger=None, verbose=False):
    """
    Computes the the number of days between the observation and the
    date of observations for the WLS files.  

    Keywords:
        AGEWLS  - Approx age of WLSFILE compared to this file (days)
        AGEWLS2 - Approx age of WLSFILE2 compared to this file (days)

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with PRIMARY header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    if (str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>"):
        logger.info('Not a valid L1.  Master age keywords not added to header.')
        return L1
    
    # Make datetime object of the observation time
    date_mjd_str = L1.header['PRIMARY']['MJD-OBS']
    date_obs_datetime = Time(date_mjd_str, format='mjd').datetime

    # Loops over WLSFILE keywords
    myL1 = AnalyzeL1(L1, logger=logger)
    for wlsfile, new_keyword in zip(['WLSFILE', 'WLSFILE2'], ['AGEWLS', 'AGEWLS2']):
        try:
            age_wls_file = myL1.measure_WLS_age(kwd=wlsfile, verbose=verbose)
            if verbose:
                logger.info(f'{wlsfile} age compared to this file (days): {age_wls_file}')
            
            if age_wls_file == None:
                age_wls_file = -99

            # Write WLS age to primary header
            L1.header['PRIMARY'][new_keyword] = (age_wls_file, f'{wlsfile} age compared to this file (days)')

        except Exception as e:
            logger.error(f"Problem with determining age of {wlsfile}: {e}\n{traceback.format_exc()}")
            L1.header['PRIMARY'][new_keyword] = (-99, 'ERROR: {wlsfile} age compared to this file (days).')

    return L1


def add_headers_trace_lamp_age_L1(L1, logger=None, verbose=False):
    """
    Computes the the number of days between the observation and the
    date of observations for the smooth lamp and order traces files.  

    Keywords:
        AGETRAC - Approx age of TRACFILE compared to this file (days)
        AGEFLAT - Approx age of LAMPFILE compared to this file (days)

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with PRIMARY header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    # Check that the input object is of the right type
    if (str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>"):
        logger.info('Not a valid L1.  Master age keywords not added to header.')
        return L1
    
    # Make datetime object of the observation time
    date_mjd_str = L1.header['PRIMARY']['MJD-OBS']
    date_obs_datetime = Time(date_mjd_str, format='mjd').datetime

    # Loops over WLSFILE keywords
    myL1 = AnalyzeL1(L1, logger=logger)
    for file, new_keyword in zip(['TRACFILE', 'LAMPFILE'], ['AGETRAC', 'AGELAMP']):
        try:
            age_file = myL1.measure_master_age(kwd=file, verbose=verbose)
            if verbose:
                logger.info(f'{file} age compared to this file (days): {age_file}')
            
            if age_file == None:
                age_file = -99

            # Write age to primary header
            L1.header['PRIMARY'][new_keyword] = (age_file, f'{file} age compared to this file (days)')

        except Exception as e:
            logger.error(f"Problem with determining age of {file}: {e}\n{traceback.format_exc()}")
            L1.header['PRIMARY'][new_keyword] = (-99, 'ERROR: {file} age compared to this file (days).')

    return L1


def add_headers_L1_SNR(L1, logger=None):
    """
    Computes the SNR of L1 spectra and adds keywords to the L1 object headers
    
    Keywords:
        SNRSC452 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 452 nm (second bluest order); on Green CCD
        SNRSK452 - SNR of L1 SKY spectrum near 452 nm (second bluest order); on Green CCD
        SNRCL452 - SNR of L1 CAL spectrum near 452 nm (second bluest order); on Green CCD
        SNRSC548 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 548 nm; on Green CCD
        SNRSK548 - SNR of L1 SKY spectrum near 548 nm; on Green CCD
        SNRCL548 - SNR of L1 CAL spectrum near 548 nm; on Green CCD
        SNRSC652 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 652 nm; on Red CCD
        SNRSK652 - SNR of L1 SKY spectrum near 652 nm; on Red CCD
        SNRCL652 - SNR of L1 CAL spectrum near 652 nm; on Red CCD
        SNRSC747 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 747 nm; on Red CCD
        SNRSK747 - SNR of L1 SKY spectrum near 747 nm; on Red CCD
        SNRCL747 - SNR of L1 CAL spectrum near 747 nm; on Red CCD
        SNRSC852 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 852 nm (second reddest order); on Red CCD
        SNRSK852 - SNR of L1 SKY spectrum near 852 nm (second reddest order); on Red CCD
        SNRCL852 - SNR of L1 CAL spectrum near 852 nm (second reddest order); on Red CCD

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        print('Not a valid L1.')
        return L1
        
    # Use the AnalyzeL1 class to compute SNR
    myL1 = AnalyzeL1(L1, logger=logger)
    myL1.measure_L1_snr(snr_percentile=95)
    for chip in chips:
        if chip == 'green':
            try:
                L1.header['PRIMARY']['SNRSC452'] = (round(myL1.GREEN_SNR[1,-1],3), 
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 452 nm')
                L1.header['PRIMARY']['SNRSK452'] = (round(myL1.GREEN_SNR[1,-2],3),
                                                    'SNR of L1 SKY near 452 nm')
                L1.header['PRIMARY']['SNRCL452'] = (round(myL1.GREEN_SNR[1,0],3),
                                                    'SNR of L1 CAL near 452 nm')
                L1.header['PRIMARY']['SNRSC548'] = (round(myL1.GREEN_SNR[25,-1],3),
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 548 nm')
                L1.header['PRIMARY']['SNRSK548'] = (round(myL1.GREEN_SNR[25,-2],3),
                                                    'SNR of L1 SKY near 548 nm')
                L1.header['PRIMARY']['SNRCL548'] = (round(myL1.GREEN_SNR[25,0],3),
                                                    'SNR of L1 CAL near 548 nm')
            except Exception as e:
                logger.error(f"Problem with green L1 SNR measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                L1.header['PRIMARY']['SNRSC652'] = (round(myL1.RED_SNR[8,-1],3),
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 652 nm')
                L1.header['PRIMARY']['SNRSK652'] = (round(myL1.RED_SNR[8,-2],3),
                                                    'SNR of L1 SKY near 652 nm')
                L1.header['PRIMARY']['SNRCL652'] = (round(myL1.RED_SNR[8,0],3),
                                                    'SNR of L1 CAL near 652 nm')
                L1.header['PRIMARY']['SNRSC747'] = (round(myL1.RED_SNR[20,-1],3),
                                                    'SNR of L1 SCI near 747 nm')
                L1.header['PRIMARY']['SNRSK747'] = (round(myL1.RED_SNR[20,-2],3),
                                                    'SNR of L1 SKY (SCI1+SCI2+SCI3) near 747 nm')
                L1.header['PRIMARY']['SNRCL747'] = (round(myL1.RED_SNR[20,0],3),
                                                    'SNR of L1 CAL near 747 nm')
                L1.header['PRIMARY']['SNRSC852'] = (round(myL1.RED_SNR[30,-1],3),
                                                    'SNR of L1 SCI near 852 nm')
                L1.header['PRIMARY']['SNRSK852'] = (round(myL1.RED_SNR[30,-2],3),
                                                    'SNR of L1 SKY (SCI1+SCI2+SCI3) near 852 nm')
                L1.header['PRIMARY']['SNRCL852'] = (round(myL1.RED_SNR[30,0],3),
                                                    'SNR of L1 CAL near 852 nm')
            except Exception as e:
                logger.error(f"Problem with red L1 SNR measurements: {e}\n{traceback.format_exc()}")
    return L1


def add_headers_L1_order_flux_ratios(L1, logger=None):
    """
    Computes the SNR of L1 spectra and adds keywords to the L1 object headers
    
    Keywords:
        FR452652 - Peak flux ratio (452nm/652nm) - SCI2
        FR548652 - Peak flux ratio (548nm/652nm) - SCI2
        FR747652 - Peak flux ratio (747nm/652nm) - SCI2
        FR852652 - Peak flux ratio (852nm/652nm) - SCI2

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        print('Not a valid L1.')
        return L1
        
    # Use the AnalyzeL1 class to compute ratios between spectral orders
    myL1 = AnalyzeL1(L1, logger=logger)
    myL1.measure_L1_snr(counts_percentile=95, snr_percentile=95)
    for chip in chips:
        if chips == ['green', 'red']:
            try: 
                L1.header['PRIMARY']['FR452652'] = (round(myL1.GREEN_PEAK_FLUX[1,2]/myL1.RED_PEAK_FLUX[8,2],6), 
                                                    'Peak flux ratio (452nm/652nm) - SCI2')
                L1.header['PRIMARY']['FR548652'] = (round(myL1.GREEN_PEAK_FLUX[25,2]/myL1.RED_PEAK_FLUX[8,2],6), 
                                                    'Peak flux ratio (548nm/652nm) - SCI2')
            except Exception as e:
                logger.error(f"Problem with green L1 SNR measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                L1.header['PRIMARY']['FR747652'] = (round(myL1.RED_PEAK_FLUX[20,2]/myL1.RED_PEAK_FLUX[8,2],6), 
                                                    'Peak flux ratio (747nm/652nm) - SCI2')
                L1.header['PRIMARY']['FR852652'] = (round(myL1.RED_PEAK_FLUX[30,2]/myL1.RED_PEAK_FLUX[8,2],6), 
                                                    'Peak flux ratio (852nm/652nm) - SCI2')
            except Exception as e:
                logger.error(f"Problem with red L1 SNR measurements: {e}\n{traceback.format_exc()}")
    return L1


def add_headers_L1_orderlet_flux_ratios(L1, logger=None):
    """
    Computes the orderlet flux ratios of L1 spectra and 
    adds keywords to the L1 object headers
    
    Keywords:
        FR12M452 - median(SCI1/SCI2) flux ratio near 452 nm
        FR12U452 - uncertainty on median(SCI1/SCI2) flux ratio near 452 nm
        FR32M452 - median(SCI3/SCI2) flux ratio near 452 nm
        FR32U452 - uncertainty on median(SCI1/SCI2) flux ratio near 452 nm
        FRS2M452 - median(SKY/SCI2) flux ratio near 452 nm
        FRS2U452 - uncertainty on median(SKY/SCI2) flux ratio near 452 nm
        FRC2M452 - median(CAL/SCI2) flux ratio near 452 nm
        FRC2U452 - uncertainty on median(CAL/SCI2) flux ratio near 452 nm
        FR12M548 - median(SCI1/SCI2) flux ratio near 548 nm
        FR12U548 - uncertainty on median(SCI1/SCI2) flux ratio near 548 nm
        FR32M548 - median(SCI3/SCI2) flux ratio near 548 nm
        FR32U548 - uncertainty on median(SCI1/SCI2) flux ratio near 548 nm
        FRS2M548 - median(SKY/SCI2) flux ratio near 548 nm
        FRS2U548 - uncertainty on median(SKY/SCI2) flux ratio near 548 nm
        FRC2M548 - median(CAL/SCI2) flux ratio near 548 nm
        FRC2U548 - uncertainty on median(CAL/SCI2) flux ratio near 548 nm
        FR12M652 - median(SCI1/SCI2) flux ratio near 652 nm
        FR12U652 - uncertainty on median(SCI1/SCI2) flux ratio near 652 nm
        FR32M652 - median(SCI3/SCI2) flux ratio near 652 nm
        FR32U652 - uncertainty on median(SCI1/SCI2) flux ratio near 652 nm
        FRS2M652 - median(SKY/SCI2) flux ratio near 652 nm
        FRS2U652 - uncertainty on median(SKY/SCI2) flux ratio near 652 nm
        FRC2M652 - median(CAL/SCI2) flux ratio near 652 nm
        FRC2U652 - uncertainty on median(CAL/SCI2) flux ratio near 652 nm
        FR12M747 - median(SCI1/SCI2) flux ratio near 747 nm
        FR12U747 - uncertainty on median(SCI1/SCI2) flux ratio near 747 nm
        FR32M747 - median(SCI3/SCI2) flux ratio near 747 nm
        FR32U747 - uncertainty on median(SCI1/SCI2) flux ratio near 747 nm
        FRS2M747 - median(SKY/SCI2) flux ratio near 747 nm
        FRS2U747 - uncertainty on median(SKY/SCI2) flux ratio near 747 nm
        FRC2M747 - median(CAL/SCI2) flux ratio near 747 nm
        FRC2U747 - uncertainty on median(CAL/SCI2) flux ratio near 747 nm
        FR12M852 - median(SCI1/SCI2) flux ratio near 852 nm
        FR12U852 - uncertainty on median(SCI1/SCI2) flux ratio near 852 nm
        FR32M852 - median(SCI3/SCI2) flux ratio near 852 nm
        FR32U852 - uncertainty on median(SCI1/SCI2) flux ratio near 852 nm
        FRS2M852 - median(SKY/SCI2) flux ratio near 852 nm
        FRS2U852 - uncertainty on median(SKY/SCI2) flux ratio near 852 nm
        FRC2M852 - median(CAL/SCI2) flux ratio near 852 nm
        FRC2U852 - uncertainty on median(CAL/SCI2) flux ratio near 852 nm

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        print('Not a valid L1.')
        return L1
        
    # Use the AnalyzeL1 class to compute flux ratios between orderlets
    myL1 = AnalyzeL1(L1, logger=logger)
    myL1.measure_orderlet_flux_ratios()
    for chip in chips:
        if chip == 'green':
            try:
                # Order 1 (Green) - 452 nm
                o=1
                imin = 2040-250
                imax = 2040+250
                L1.header['PRIMARY']['FR12M452'] = (np.median(myL1.f_g_sci1_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SCI1/SCI2) near 452 nm')
                L1.header['PRIMARY']['FR12U452'] = (uncertainty_median(myL1.f_g_sci1_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI1/SCI2) near 452 nm')
                L1.header['PRIMARY']['FR32M452'] = (np.median(myL1.f_g_sci3_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SCI3/SCI2) near 452 nm')
                L1.header['PRIMARY']['FR32U452'] = (uncertainty_median(myL1.f_g_sci3_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI3/SCI2) near 452 nm')
                L1.header['PRIMARY']['FRS2M452'] = (np.median(myL1.f_g_sky_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SKY/SCI2) near 452 nm')
                L1.header['PRIMARY']['FRS2U452'] = (uncertainty_median(myL1.f_g_sky_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SKY/SCI2) near 452 nm')
                L1.header['PRIMARY']['FRC2M452'] = (np.median(myL1.f_g_cal_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(CAL/SCI2) near 452 nm')
                L1.header['PRIMARY']['FRC2U452'] = (uncertainty_median(myL1.f_g_cal_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(CAL/SCI2) near 452 nm')
                # Order 25 (Green) - 548 nm
                o=25
                imin = 2040-250
                imax = 2040+250
                L1.header['PRIMARY']['FR12M548'] = (np.median(myL1.f_g_sci1_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SCI1/SCI2) near 548 nm')
                L1.header['PRIMARY']['FR12U548'] = (uncertainty_median(myL1.f_g_sci1_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI1/SCI2) near 548 nm')
                L1.header['PRIMARY']['FR32M548'] = (np.median(myL1.f_g_sci3_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SCI3/SCI2) near 548 nm')
                L1.header['PRIMARY']['FR32U548'] = (uncertainty_median(myL1.f_g_sci3_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI3/SCI2) near 548 nm')
                L1.header['PRIMARY']['FRS2M548'] = (np.median(myL1.f_g_sky_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(SKY/SCI2) near 548 nm')
                L1.header['PRIMARY']['FRS2U548'] = (uncertainty_median(myL1.f_g_sky_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(SKY/SCI2) near 548 nm')
                L1.header['PRIMARY']['FRC2M548'] = (np.median(myL1.f_g_cal_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'median(CAL/SCI2) near 548 nm')
                L1.header['PRIMARY']['FRC2U548'] = (uncertainty_median(myL1.f_g_cal_int[o,imin:imax] / myL1.f_g_sci2[o,imin:imax]), 
                                                    'unc. of median(CAL/SCI2) near 548 nm')
            except Exception as e:
                logger.error(f"Problem with green L1 SNR measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                # Order 8 (Red) - 652 nm
                o=8
                imin = 2040-250
                imax = 2040+250
                L1.header['PRIMARY']['FR12M652'] = (np.median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI1/SCI2) near 652 nm')
                L1.header['PRIMARY']['FR12U652'] = (uncertainty_median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI1/SCI2) near 652 nm')
                L1.header['PRIMARY']['FR32M652'] = (np.median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI3/SCI2) near 652 nm')
                L1.header['PRIMARY']['FR32U652'] = (uncertainty_median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI3/SCI2) near 652 nm')
                L1.header['PRIMARY']['FRS2M652'] = (np.median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SKY/SCI2) near 652 nm')
                L1.header['PRIMARY']['FRS2U652'] = (uncertainty_median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SKY/SCI2) near 652 nm')
                L1.header['PRIMARY']['FRC2M652'] = (np.median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(CAL/SCI2) near 652 nm')
                L1.header['PRIMARY']['FRC2U652'] = (uncertainty_median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(CAL/SCI2) near 652 nm')
                # Order 20 (Red) - 747 nm
                o=20
                imin = 2040-250
                imax = 2040+250
                L1.header['PRIMARY']['FR12M747'] = (np.median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI1/SCI2) near 747 nm')
                L1.header['PRIMARY']['FR12U747'] = (uncertainty_median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI1/SCI2) near 747 nm')
                L1.header['PRIMARY']['FR32M747'] = (np.median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI3/SCI2) near 747 nm')
                L1.header['PRIMARY']['FR32U747'] = (uncertainty_median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI3/SCI2) near 747 nm')
                L1.header['PRIMARY']['FRS2M747'] = (np.median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SKY/SCI2) near 747 nm')
                L1.header['PRIMARY']['FRS2U747'] = (uncertainty_median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SKY/SCI2) near 747 nm')
                L1.header['PRIMARY']['FRC2M747'] = (np.median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(CAL/SCI2) near 747 nm')
                L1.header['PRIMARY']['FRC2U747'] = (uncertainty_median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(CAL/SCI2) near 747 nm')
                # Order 30 (Red) - 852 nm
                o=20
                imin = 2040-250
                imax = 2040+250
                L1.header['PRIMARY']['FR12M852'] = (np.median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI1/SCI2) near 852 nm')
                L1.header['PRIMARY']['FR12U852'] = (uncertainty_median(myL1.f_r_sci1_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI1/SCI2) near 852 nm')
                L1.header['PRIMARY']['FR32M852'] = (np.median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SCI3/SCI2) near 852 nm')
                L1.header['PRIMARY']['FR32U852'] = (uncertainty_median(myL1.f_r_sci3_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SCI3/SCI2) near 852 nm')
                L1.header['PRIMARY']['FRS2M852'] = (np.median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(SKY/SCI2) near 852 nm')
                L1.header['PRIMARY']['FRS2U852'] = (uncertainty_median(myL1.f_r_sky_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(SKY/SCI2) near 852 nm')
                L1.header['PRIMARY']['FRC2M852'] = (np.median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'median(CAL/SCI2) near 852 nm')
                L1.header['PRIMARY']['FRC2U852'] = (uncertainty_median(myL1.f_r_cal_int[o,imin:imax] / myL1.f_r_sci2[o,imin:imax]), 
                                                    'unc. of median(CAL/SCI2) near 852 nm')
            except Exception as e:
                logger.error(f"Problem with red L1 SNR measurements: {e}\n{traceback.format_exc()}")
    return L1


def add_headers_L1_cal_line_quality(L1, intensity_thresh=40**2, min_lines=100, 
                                    divisions_per_order=8, cal=None,logger=None):
    """
    Computes the min/max order per chip and orderlet with good LFC lines.
    An order is good if at least min_lines with amplitude intensity_thresh.  
    It also checks that there is at least one line of that amplitude in 8 
    (set by divisions_per_order) equal-spaced regions per order.
    
    Keywords:
        LFCLGS0 - Min order with good LFC lines on SCI orders of Green CCD
        LFCLGS1 - Max order with good LFC lines on SCI orders of Green CCD
        LFCLGC0 - Min order with good LFC lines on CAL orders of Green CCD
        LFCLGC1 - Max order with good LFC lines on CAL orders of Green CCD
        LFCLGK0 - Min order with good LFC lines on SKY orders of Green CCD
        LFCLGK1 - Max order with good LFC lines on SKY orders of Green CCD
        LFCLRS0 - Min order with good LFC lines on SCI orders of Red CCD
        LFCLRS1 - Max order with good LFC lines on SCI orders of Red CCD
        LFCLRC0 - Min order with good LFC lines on CAL orders of Red CCD
        LFCLRC1 - Max order with good LFC lines on CAL orders of Red CCD
        LFCLRK0 - Min order with good LFC lines on SKY orders of Red CCD
        LFCLRK1 - Max order with good LFC lines on SKY orders of Red CCD
        ETALGS0 - Min order with good Etalon lines on SCI orders of Green CCD
        ETALGS1 - Max order with good Etalon lines on SCI orders of Green CCD
        ETALGC0 - Min order with good Etalon lines on CAL orders of Green CCD
        ETALGC1 - Max order with good Etalon lines on CAL orders of Green CCD
        ETALGK0 - Min order with good Etalon lines on SKY orders of Green CCD
        ETALGK1 - Max order with good Etalon lines on SKY orders of Green CCD
        ETALRS0 - Min order with good Etalon lines on SCI orders of Red CCD
        ETALRS1 - Max order with good Etalon lines on SCI orders of Red CCD
        ETALRC0 - Min order with good Etalon lines on CAL orders of Red CCD
        ETALRC1 - Max order with good Etalon lines on CAL orders of Red CCD
        ETALRK0 - Min order with good Etalon lines on SKY orders of Red CCD
        ETALRK1 - Max order with good Etalon lines on SKY orders of Red CCD

    Args:
        L1 - a KPF L1 object 
        intensity_thresh (float): minimum line amplitude to be considered good
        min_lines (int):          minimum number of lines in a spectral 
                                  order for it to be considered good
        divisions_per_order (int): number of contiguous subregions each order 
                                   must have at least one peak in
        cal: one of ['Etalon', 'LFC'] - sets type of calibration

    Returns:
        L1 - a L1 file with header keywords added
    """
    if logger == None:
        logger = DummyLogger()

    # Use the AnalyzeL1 class 
    myL1 = AnalyzeL1(L1, logger=logger)
    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')

    # Determine which fibers are illuminated by LFC or Etalon
    use_CAL, use_SCI, use_SKY = False, False, False
    if cal == 'LFC':
        cal_fiber = 'LFCFiber'
        name = 'LFC'
        prefix = 'LFC'
    elif cal == 'Etalon':
        cal_fiber = 'EtalonFiber'
        name = 'Etalon'
        prefix = 'ETA'
    else:
        self.logger.error('Calibration type not specified.')
        return L1
    if 'CAL-OBJ' in myL1.L1.header['PRIMARY']:
        if myL1.L1.header['PRIMARY']['CAL-OBJ'] == cal_fiber:
            use_CAL = True
    if 'SCI-OBJ' in myL1.L1.header['PRIMARY']:
        if myL1.L1.header['PRIMARY']['SCI-OBJ'] == cal_fiber:
            use_SCI = True
    if 'SKY-OBJ' in myL1.L1.header['PRIMARY']:
        if myL1.L1.header['PRIMARY']['SKY-OBJ'] == cal_fiber:
            use_SKY = True


    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        print('Not a valid L1.')
        return L1
        
    for chip in chips:
        if chip == 'green':
            try:
                # Compute first and last good orders
                SCI_g_fl, CAL_g_fl, SKY_g_fl = myL1.measure_good_comb_orders(chip='green', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)

                # Replace None values with -1 to indicate no good orders
                SCI_g_fl = [-1 if x is None else x for x in SCI_g_fl]
                CAL_g_fl = [-1 if x is None else x for x in CAL_g_fl]
                SKY_g_fl = [-1 if x is None else x for x in SKY_g_fl]

                if use_SCI:
                    L1.header['PRIMARY'][f'{prefix}LGS0'] = (SCI_g_fl[0], f'Min Green SCI order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LGS1'] = (SCI_g_fl[1], f'Max Green SCI order with good {name} lines')
                if use_CAL:
                    L1.header['PRIMARY'][f'{prefix}LGC0'] = (CAL_g_fl[0], f'Min Green CAL order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LGC1'] = (CAL_g_fl[1], f'Max Green CAL order with good {name} lines')
                if use_SKY:
                    L1.header['PRIMARY'][f'{prefix}LGK0'] = (SKY_g_fl[0], f'Min Green SKY order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LGK1'] = (SKY_g_fl[1], f'Max Green SKY order with good {name} lines')

            except Exception as e:
                logger.error(f"Problem with green L1 {name} line measurements: {e}\n{traceback.format_exc()}")

        if chip == 'red':
            try:
                # Compute first and last good orders
                SCI_r_fl, CAL_r_fl, SKY_r_fl = myL1.measure_good_comb_orders(chip='red', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)

                # Replace None values with -1 to indicate no good orders
                SCI_r_fl = [-1 if x is None else x for x in SCI_r_fl]
                CAL_r_fl = [-1 if x is None else x for x in CAL_r_fl]
                SKY_r_fl = [-1 if x is None else x for x in SKY_r_fl]

                if use_SCI:
                    L1.header['PRIMARY'][f'{prefix}LRS0'] = (SCI_r_fl[0], f'Min RED SCI order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LRS1'] = (SCI_r_fl[1], f'Max RED SCI order with good {name} lines')
                if use_CAL:
                    L1.header['PRIMARY'][f'{prefix}LRC0'] = (CAL_r_fl[0], f'Min RED CAL order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LRC1'] = (CAL_r_fl[1], f'Max RED CAL order with good {name} lines')
                if use_SKY:
                    L1.header['PRIMARY'][f'{prefix}LRK0'] = (SKY_r_fl[0], f'Min RED SKY order with good {name} lines')
                    L1.header['PRIMARY'][f'{prefix}LRK1'] = (SKY_r_fl[1], f'Max RED SKY order with good {name} lines')

            except Exception as e:
                logger.error(f"Problem with red L1 {name} line measurements: {e}\n{traceback.format_exc()}")

    return L1


def add_headers_L1_saturated_lines(L1, logger=None):
    """
    Counts the number of saturated lines and adds keywords to the L1 object headers
    
    Keywords:
        NSATGS2 - Number of saturated lines in Green SCI2
        NSATGC  - Number of saturated lines in Green CAL
        NSATGK  - Number of saturated lines in Green SKY
        NSATRS2 - Number of saturated lines in Red SCI2
        NSATRC  - Number of saturated lines in Red CAL
        NSATRK  - Number of saturated lines in Red SKY

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        print('Not a valid L1.')
        return L1
        
    # Use the AnalyzeL1 class to compute ratios between spectral orders
    myL1 = AnalyzeL1(L1, logger=logger)
    for chip in chips:
        if chip == 'green':
            try: 
                (SCI1_lines, SCI2_lines, SCI3_lines, CAL_lines, SKY_lines) =  myL1.count_saturated_lines(chip='green')
                L1.header['PRIMARY']['NSATGS2'] = (SCI2_lines, 'Number of saturated lines in Green SCI2')
                L1.header['PRIMARY']['NSATGC']  = (CAL_lines,  'Number of saturated lines in Green CAL')
                L1.header['PRIMARY']['NSATGK']  = (SKY_lines,  'Number of saturated lines in Green SKY')
            except Exception as e:
                logger.error(f"Problem counting satured lines for green chip: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                (SCI1_lines, SCI2_lines, SCI3_lines, CAL_lines, SKY_lines) =  myL1.count_saturated_lines(chip='red')
                L1.header['PRIMARY']['NSATRS2'] = (SCI2_lines, 'Number of saturated lines in Red SCI2')
                L1.header['PRIMARY']['NSATRC']  = (CAL_lines,  'Number of saturated lines in Red CAL')
                L1.header['PRIMARY']['NSATRK']  = (SKY_lines,  'Number of saturated lines in Red SKY')
            except Exception as e:
                logger.error(f"Problem counting satured lines for red chip: {e}\n{traceback.format_exc()}")
    return L1


def add_headers_L1_std_wls(L1, logger=None, debug=False):
    """
    Computes the standard deviation of the L1 wavelength solution compared to a 
    reference wavelength solution. The output is in units of pixels.  Keywords 
    are generated for combinations of [Green, Red] and [SCI, SKY, CAL].
    
    Keywords:
        STDWREF - filename of reference wavelength solution
        STDWGSNN (35 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Green SCI1, SCI2, SCI3 order NN
        STDWGKNN (35 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Green SKY order NN
        STDWGCNN (35 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Green CAL order NN
        STDWRSNN (32 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Red SCI1, SCI2, SCI3 order NN
        STDWRKNN (32 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Red SKY order NN
        STDWRCNN (32 keywords for orders NN) - stdev of the WLS (in pixels) compared to reference for Red CAL order NN

    Args:
        L1 - a KPF L1 object 

    Returns:
        L1 - a L1 file with header keywords added
    """
    if logger == None:
        logger = DummyLogger()

    # Use the AnalyzeL1 class 
    myL1 = AnalyzeL1(L1, logger=logger)
    data_products = get_data_products_L1(L1)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')

    # Check that the input object is of the right type
    if str(type(L1)) != "<class 'kpfpipe.models.level1.KPF1'>" or chips == []:
        self.logger.error('Not a valid L1.')
        return L1

    # Get reference wavelength solution
    dt = get_datetime_obsid(myL1.ObsID).strftime('%Y-%m-%dT%H:%M:%S.%f')
    if debug:
        print(f'DEFAULT_CALIBRATION_CFG_PATH = ' + DEFAULT_CALIBRATION_CFG_PATH)
    GC = GetCalibrations(dt, DEFAULT_CALIBRATION_CFG_PATH, use_db=False)
    wls_filename = GC.lookup(subset=['rough_wls']) 
    if debug:
        print(f'wls_filename = ' + wls_filename['rough_wls'])
    L1_ref = KPF1.from_fits(wls_filename['rough_wls'])
    myL1_ref = AnalyzeL1(L1_ref)  
    myL1_ref.add_dispersion_arrays()

    # Method to compute Stdev of WLS
    def compute_stats_wls(L1, L1_ref, EXT=['SCI'], CHIP=['GREEN'], ORDER=[0], debug=False):
        """
        Compute the median and the standard deviation of the difference between 
        the wavelength solution (L1) and a reference (L1_ref).  
        The output is in units of pixels.

        Args:
            EXT (array): possible values in the array are 'SCI', 'SKY', 'CAL'
            CHIP (array): possible values in the array are 'GREEN', 'RED'
            ORDER (array): possible values are integers from 0 to 34
            debug: if True, print debugging statements

        Returns:
            med_wls, std_wls (floats): Median and standard deviation 
                                       of the differen between the WLS and 
                                       reference WLS for EXT, CHIP, ORDER
        """

        # Determine which extensions to check
        WAVE_extensions   = []
        if 'CAL' in EXT:
            if 'GREEN' in CHIP:
                WAVE_extensions.append("GREEN_CAL_WAVE")
            if 'RED' in CHIP:
                WAVE_extensions.append("RED_CAL_WAVE")
        if 'SCI' in EXT:
            if 'GREEN' in CHIP:
                WAVE_extensions.append("GREEN_SCI_WAVE1")
                WAVE_extensions.append("GREEN_SCI_WAVE2")
                WAVE_extensions.append("GREEN_SCI_WAVE3")
            if 'RED' in CHIP:
                WAVE_extensions.append("RED_SCI_WAVE1")
                WAVE_extensions.append("RED_SCI_WAVE2")
                WAVE_extensions.append("RED_SCI_WAVE3")
        if 'SKY' in EXT:
            if 'GREEN' in CHIP:
                WAVE_extensions.append("GREEN_SKY_WAVE")
            if 'RED' in CHIP:
                WAVE_extensions.append("RED_SKY_WAVE")

        for EXT_WAVE in WAVE_extensions:
            if debug:
                print(f'EXT_WAVE = ' + EXT_WAVE)
            EXT_DISP = EXT_WAVE.replace('WAVE', 'DISP')
            pix_diff_med = 0
            pix_diff_std = 0
            for o in ORDER:
                if not (L1_ref[EXT_DISP][o,:] == 0).all():
                    numerator = L1[EXT_WAVE][o,:] - L1_ref[EXT_WAVE][o,:]
                    denominator = L1_ref[EXT_DISP][o, :]
                    zero_diff_mask = numerator == 0
                    pix_diff_array = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
                    pix_diff_array[zero_diff_mask] = 0
                    this_pix_diff_med = np.nanmedian(pix_diff_array)
                    this_pix_diff_std = np.nanstd(pix_diff_array)

                    if abs(this_pix_diff_med) > abs(pix_diff_med):
                        pix_diff_med = this_pix_diff_med
                    if abs(this_pix_diff_std) > abs(pix_diff_std):
                        pix_diff_std = this_pix_diff_std
                    if debug:
                        print(o, this_pix_diff_median, this_pix_diff_std)
                    
        return pix_diff_med, pix_diff_std

    for chip in chips:
        L1.header['PRIMARY']['STATWREF'] = (wls_filename['rough_wls'], 'ref fn for WLS-ref')
        for EXT in ['SCI', 'SKY', 'CAL']:
            norder = L1[chip+'_CAL_WAVE'].shape[0]
            for o in range(norder):
                try:
                    med_wls, std_wls = compute_stats_wls(myL1.L1, myL1_ref.L1, EXT=[EXT], CHIP=[chip.upper()], ORDER=[o])
                    if chip == 'green':
                        if EXT == 'SCI':
                            L1.header['PRIMARY'][f'MEDWGS{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Green SCI order {o:02d}')
                            L1.header['PRIMARY'][f'STDWGS{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Green SCI order {o:02d}')
                        elif EXT == 'SKY':
                            L1.header['PRIMARY'][f'MEDWGK{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Green SKY order {o:02d}')
                            L1.header['PRIMARY'][f'STDWGK{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Green SKY order {o:02d}')
                        elif EXT == 'CAL':
                            L1.header['PRIMARY'][f'MEDWGC{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Green CAL order {o:02d}')
                            L1.header['PRIMARY'][f'STDWGC{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Green CAL order {o:02d}')
                    if chip == 'red':
                        if EXT == 'SCI':
                            L1.header['PRIMARY'][f'MEDWRS{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Red SCI order {o:02d}')
                            L1.header['PRIMARY'][f'STDWRS{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Red SCI order {o:02d}')
                        elif EXT == 'SKY':
                            L1.header['PRIMARY'][f'MEDWRK{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Red SKY order {o:02d}')
                            L1.header['PRIMARY'][f'STDWRK{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Red SKY order {o:02d}')
                        elif EXT == 'CAL':
                            L1.header['PRIMARY'][f'MEDWRC{o:02d}'] = (med_wls, f'median(WLS-ref) [pix], Red CAL order {o:02d}')
                            L1.header['PRIMARY'][f'STDWRC{o:02d}'] = (std_wls, f'stddev(WLS-ref) [pix], Red CAL order {o:02d}')
    
                except Exception as e:
                    logger.error(f"Problem with green L1 {name} line measurements: {e}\n{traceback.format_exc()}")

    return L1


def add_headers_L2_barycentric(L2, logger=None):
    """
    Adds Barycentric RV correction and BJD to the L2 primary header
    
    Keywords:
        CCFBCV - Barycentric radial velocity correction (km/s), averaged
                 over the BCV values for each spectral order and weighted 
                 by the CCF Weights
        CCFBJD - Weighted avg of BJD values (days)
        BCVRNG - Range of values of barycentric radial velocity correction 
                 (m/s) for the spectral orders, with zero-weight orders 
                 excluded
        BCVSTD - Standard deviation of values of barycentric radial velocity 
                 correction (m/s) for the spectral orders and weighted by the 
                 CCF Weights
        BJDRNG - Range of BJD values for the spectral orders, with zero-weight
                 orders excluded (sec)
        BJDSTD - Standard deviation of BJD values for the spectral orders, 
                 weighted by the CCF Weights (sec)
        MAXPCBCV - The maximum of the spectral orders' percent change 
                 differences from an observation's CCFBCV, with zero-weight 
                 orders excluded (%)
        MINPCBCV - The minimum of the spectral orders' percent change
                 differences from an observation's CCFBCV, with zero-weight
                 orders excluded (%)

    Args:
        L2 - a KPF L2 object 

    Returns:
        L2 - a L2 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    try:
        data_products = get_data_products_L2(L2)
        chips = []
        if 'Green' in data_products: chips.append('green')
        if 'Red'   in data_products: chips.append('red')
        
        # Check that the input object is of the right type
        if str(type(L2)) != "<class 'kpfpipe.models.level2.KPF2'>" or chips == []:
            print('Not a valid L2.')
            return L2
            
        # Use the AnalyzeL2 class to compute BCV
        myL2 = AnalyzeL2(L2, logger=logger)
    
        # Add values to header
        if hasattr(myL2, 'CCFBCV'):
            L2.header['PRIMARY']['CCFBCV'] = (myL2.CCFBCV, 'Weighted avg of barycentricRV correction (km/s)')
        # remove the two lines below when CCFBJD is computed where the RV table is assembled
        if hasattr(myL2, 'CCFBJD'):
            L2.header['PRIMARY']['CCFBJD']  = (myL2.CCFBJD, 'Weighted avg of BJD values (days)')
    
        # Add range, standard deviation, and percent difference stats
        if hasattr(myL2, 'Delta_CCFBJD_weighted_std'):
            L2.header['PRIMARY']['BJDSTD'] = (myL2.Delta_CCFBJD_weighted_std, 'Weighted stddev of BJD for orders (sec)')
        if hasattr(myL2, 'Delta_CCFBJD_weighted_range'):
            L2.header['PRIMARY']['BJDRNG'] = (myL2.Delta_CCFBJD_weighted_range, 'Range(BJD) for non-zero-weight orders (sec)')
        if hasattr(myL2, 'Delta_Bary_RVC_weighted_std'):
            L2.header['PRIMARY']['BCVSTD'] = (myL2.Delta_Bary_RVC_weighted_std, 'Weighted stddev of BCV for orders (m/s)')
        if hasattr(myL2, 'Delta_Bary_RVC_weighted_range'):
            L2.header['PRIMARY']['BCVRNG'] = (myL2.Delta_Bary_RVC_weighted_range, 'Range(BCV) for non-zero-weight orders (m/s)')
        if hasattr(myL2, 'Max_Perc_Delta_Bary_RV'):
            L2.header['PRIMARY']['MAXPCBCV'] = (myL2.Max_Perc_Delta_Bary_RV, 'Maximum percent change from CCFBCV for non-zero-weight orders (%)')
        if hasattr(myL2, 'Min_Perc_Delta_Bary_RV'):
            L2.header['PRIMARY']['MINPCBCV'] = (myL2.Min_Perc_Delta_Bary_RV, 'Minimum percent change from CCFBCV for non-zero-weight orders (%)')

    except Exception as e:
        logger.error(f"Problem with L2 BJD/BCV measurements: {e}\n{traceback.format_exc()}")

    return L2

