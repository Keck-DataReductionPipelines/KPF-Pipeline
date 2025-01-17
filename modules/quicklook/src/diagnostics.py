# This file contains methods to write diagnostic information to the KPF headers.

# Standard dependencies
import traceback
import numpy as np

# Local dependencies
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
from modules.Utils.utils import get_moon_sep, get_sun_alt

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
        D2 - a KPF L1 object 

    Returns:
        D2 - a L1 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_2D(D2)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(D2)) != "<class 'kpfpipe.models.level0.KPF0'>" or chips == []:
        print('Not a valid 2D KPF file.')
        return D2
        
    # Use the Analyze2D class to compute flux
    my2D = Analyze2D(D2, logger=logger)
    for chip in chips:
        if chip == 'green':
            try:
                D2.header['PRIMARY']['GR2DF99P'] = (round(my2D.green_percentile_99, 2), '99th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF90P'] = (round(my2D.green_percentile_90, 2), '90th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF50P'] = (round(my2D.green_percentile_50, 2), '50th percentile flux in 2D Green image (e-)')
                D2.header['PRIMARY']['GR2DF10P'] = (round(my2D.green_percentile_10, 2), '10th percentile flux in 2D Green image (e-)')
            except Exception as e:
                logger.error(f"Problem with Green 2D flux measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                D2.header['PRIMARY']['RD2DF99P'] = (round(my2D.red_percentile_99, 2), '99th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF90P'] = (round(my2D.red_percentile_90, 2), '90th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF50P'] = (round(my2D.red_percentile_50, 2), '50th percentile flux in 2D Red image (e-)')
                D2.header['PRIMARY']['RD2DF10P'] = (round(my2D.red_percentile_10, 2), '10th percentile flux in 2D Red image (e-)')
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
        print('Not a valid 2D KPF file.')
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
                                value = "{:.2f}".format(my2D.green_dark_current_regions[keywords[k]['key']]['med_elec'])
                    except Exception as e:
                        logger.error(f"Problem with green dark current : {e}\n{traceback.format_exc()}")
                if chip == 'red':
                    try:
                        if hasattr(my2D, 'red_dark_current_regions'):
                            if 'med_elec' in my2D.red_dark_current_regions[keywords[k]['key']]:
                                value = "{:.2f}".format(my2D.red_dark_current_regions[keywords[k]['key']]['med_elec'])
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
        logger.info('Guider not in the 2D file or not a valid 2D KPF file.  Guider data products not added to header.')
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
                                           'Guider images: median(flux [ADU])')
        D2.header['PRIMARY']['GDRPKSTD'] = (round(myGuider.peak_flux_std, 1),
                                           'Guider images: std(flux [ADU])')
        D2.header['PRIMARY']['GDRFRSAT'] = (round(myGuider.frac_saturated, 5),
                                           'Guider images: frac of frames w/in 90% saturated')
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
        logger.info('CaHK not in the 2D file or not a valid 2D KPF file.  CaHK data products not added to header.')
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
        logger.info('ExpMeter not in the 2D file or not a valid 2D KPF file.  EM data products not added to header.')
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
        print('Not a valid L1 KPF file.')
        return L1
        
    # Use the AnalyzeL1 class to compute SNR
    myL1 = AnalyzeL1(L1, logger=logger)
    myL1.measure_L1_snr(snr_percentile=95)
    for chip in chips:
        if chip == 'green':
            try:
                L1.header['PRIMARY']['SNRSC452'] = (round(myL1.GREEN_SNR[1,-1],1), 
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 452 nm')
                L1.header['PRIMARY']['SNRSK452'] = (round(myL1.GREEN_SNR[1,-2],1),
                                                    'SNR of L1 SKY near 452 nm')
                L1.header['PRIMARY']['SNRCL452'] = (round(myL1.GREEN_SNR[1,0],1),
                                                    'SNR of L1 CAL near 452 nm')
                L1.header['PRIMARY']['SNRSC548'] = (round(myL1.GREEN_SNR[25,-1],1),
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 548 nm')
                L1.header['PRIMARY']['SNRSK548'] = (round(myL1.GREEN_SNR[25,-2],1),
                                                    'SNR of L1 SKY near 548 nm')
                L1.header['PRIMARY']['SNRCL548'] = (round(myL1.GREEN_SNR[25,0],1),
                                                    'SNR of L1 CAL near 548 nm')
            except Exception as e:
                logger.error(f"Problem with green L1 SNR measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                L1.header['PRIMARY']['SNRSC652'] = (round(myL1.RED_SNR[8,-1],1),
                                                    'SNR of L1 SCI (SCI1+SCI2+SCI3) near 652 nm')
                L1.header['PRIMARY']['SNRSK652'] = (round(myL1.RED_SNR[8,-2],1),
                                                    'SNR of L1 SKY near 652 nm')
                L1.header['PRIMARY']['SNRCL652'] = (round(myL1.RED_SNR[8,0],1),
                                                    'SNR of L1 CAL near 652 nm')
                L1.header['PRIMARY']['SNRSC747'] = (round(myL1.RED_SNR[20,-1],1),
                                                    'SNR of L1 SCI near 747 nm')
                L1.header['PRIMARY']['SNRSK747'] = (round(myL1.RED_SNR[20,-2],1),
                                                    'SNR of L1 SKY (SCI1+SCI2+SCI3) near 747 nm')
                L1.header['PRIMARY']['SNRCL747'] = (round(myL1.RED_SNR[20,0],1),
                                                    'SNR of L1 CAL near 747 nm')
                L1.header['PRIMARY']['SNRSC852'] = (round(myL1.RED_SNR[30,-1],1),
                                                    'SNR of L1 SCI near 852 nm')
                L1.header['PRIMARY']['SNRSK852'] = (round(myL1.RED_SNR[30,-2],1),
                                                    'SNR of L1 SKY (SCI1+SCI2+SCI3) near 852 nm')
                L1.header['PRIMARY']['SNRCL852'] = (round(myL1.RED_SNR[30,0],1),
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
        print('Not a valid L1 KPF file.')
        return L1
        
    # Use the AnalyzeL1 class to compute ratios between spectral orders
    myL1 = AnalyzeL1(L1, logger=logger)
    myL1.measure_L1_snr(counts_percentile=95, snr_percentile=95)
    for chip in chips:
        if chips == ['green', 'red']:
            try: 
                L1.header['PRIMARY']['FR452652'] = (round(myL1.GREEN_PEAK_FLUX[1,2]/myL1.RED_PEAK_FLUX[8,2],4), 
                                                    'Peak flux ratio (452nm/652nm) - SCI2')
                L1.header['PRIMARY']['FR548652'] = (round(myL1.GREEN_PEAK_FLUX[25,2]/myL1.RED_PEAK_FLUX[8,2],4), 
                                                    'Peak flux ratio (548nm/652nm) - SCI2')
            except Exception as e:
                logger.error(f"Problem with green L1 SNR measurements: {e}\n{traceback.format_exc()}")
        if chip == 'red':
            try:
                L1.header['PRIMARY']['FR747652'] = (round(myL1.RED_PEAK_FLUX[20,2]/myL1.RED_PEAK_FLUX[8,2],4), 
                                                    'Peak flux ratio (747nm/652nm) - SCI2')
                L1.header['PRIMARY']['FR852652'] = (round(myL1.RED_PEAK_FLUX[30,2]/myL1.RED_PEAK_FLUX[8,2],4), 
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
        print('Not a valid L1 KPF file.')
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


def add_headers_L2_barycentric(L2, logger=None):
    """
    Adds Barycentric RV correction to the L2 primary header
    
    Keywords:
        CCFBCV - Barycentric radial velocity correction (km/s), averaged
                 over the BCV values for each spectral order and weighted 
                 by the CCF Weights
        CCFBJD - Weighted avg of BJD values (days)

    Args:
        L2 - a KPF L2 object 

    Returns:
        L2 - a L2 file with header keywords added
    """

    if logger == None:
        logger = DummyLogger()

    data_products = get_data_products_L2(L2)
    chips = []
    if 'Green' in data_products: chips.append('green')
    if 'Red'   in data_products: chips.append('red')
    
    # Check that the input object is of the right type
    if str(type(L2)) != "<class 'kpfpipe.models.level2.KPF2'>" or chips == []:
        print('Not a valid L2 KPF file.')
        return L2
        
    # Use the AnalyzeL2 class to compute BCV
    myL2 = AnalyzeL2(L2, logger=logger)

    # Add value to header
    if hasattr(myL2, 'CCFBCV'):
        L2.header['PRIMARY']['CCFBCV'] = (myL2.CCFBCV, 'Weighted avg of barycentric RV correction (km/s)')
    # remove the two lines below when CCFBJD is computed when the RV table is assembled
    if hasattr(myL2, 'CCFBJD'):
        L2.header['PRIMARY']['CCFBJD']  = (myL2.CCFBJD, 'Weighted avg of BJD values (days)')

    return L2
