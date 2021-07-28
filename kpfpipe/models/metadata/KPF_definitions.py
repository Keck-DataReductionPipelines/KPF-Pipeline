
from kpfpipe.models.metadata.receipt_columns import RECEIPT_COL
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits

import numpy as np
import pandas as pd
from collections import OrderedDict

# refer to KPF PDR System Design manual v1, Nov 6, 2017
# and the table on the confluence page
# https://exoplanets.atlassian.net/wiki/spaces/shrek/pages/2142208120/Data+Format

# mapping between fits extension data types and Python object data types
FITS_TYPE_MAP = {fits.PrimaryHDU: OrderedDict,
                 fits.ImageHDU: np.array,
                 fits.BinTableHDU: pd.DataFrame}

## Header keywords required by all levels of data
LEVEL0_HEADER_KEYWORDS = {
    ## Temperal keywords
    'OBS-TIME': str,   # universal time of observation.
    'ELAPTIME': float,  # open shutter time (all coadds) (seconds)

    ## Observation location keywords
    'OBORIGIN':  str,    # Observatory name
    'TELESCOP':  str,    # telescope name
    'AIRMASS':  float,  # air mass
    'AZ':       str,  # telescope azimuth (deg)
    'EL':       str,  # telescope elevation (deg)
    'DEC':      str,  # declination (DD:MM:SS.SS or decimal degree)
    'EQUINOX':  float,  # telescope equinox (2000.0)
    'OBJECT':   str,    # observed object name
    'PA':       str,  # position angle from north
    'RA':       str,  # right ascension (HH:MM:SS.SS or decimal degree)
    'TARGNAME': str,    # target name from WMKO starlist

    ## Observation type keywords
    'FILTER':   float,  # any filter keywords --TODO--: what?
    'IMAGEMN':  float,  # image mean
    'IMAGEMD':  float,  # image median
    'IMAGESD':  float,  # image standard deviation
    'IMTYPE':   str,    # image type of the FITS file 
                       # (object: bias: dark: flatlamp: arclamps)
    'INSTRUME': str,    # instrument name
    'OBSMODE':  str,    # observation mode (imaging: spec)
    'POSTMN':   float,  # post scan mean
    'POSTMD':   float,  # post scan median
    'POSTSD':   float,  # post scan standard deviation
    'SPECRES':  float,  # approximate spectral resolution
    'SPATSCAL': float,  # CCD spetial pixel scale (arcsec/pixel)
    'WAVEBLUE': float,  # approximate blue wavelength of observation
    'WAVECNTR': float,  # approximate center wavelength of observation
    'WAVERED':  float,  # approximate red wavelength of observation

    ## File definition keywords
    'FRAMENO':  int,    # frame number of FITS file
    'OFNAME':   str,    # original filename written to disk (sep2i0001.fits)
    'OUTDIR':   str,    # location on disk where FITS file was written
    'FNDATE':   str,    # Date this file where this file is written

    ## Caliration association
    'DETMODE':  float,  # --TODO-- description
    'DISPERS':  float,  # --TODO-- description 
    'DISPPOS':  float,  # --TODO-- description 
    'ECHLPOS':  float,  # --TODO-- description 
    'FILNAME':  str,    # --TODO-- description 
    'SLIT-NAM':str,    # --TODO-- description 
    'STATEID':  int,    # unique ID assigned to all associated FITS file
    'STATENAM': str,    # user supplied name for STATEID

    ## Program identification keyword
    'PROGNAME': str,    # WMKO assigned program ID that FITS file

    ## World coordinate system keywords 
    'CD1_1':    float,  # WCS coordinate transform matrix [1, 1]
    'CD1_2':    float,  # WCS coordinate transform matrix [1, 2]
    'CD2_1':    float,  # WCS coordinate transform matrix [2, 1]
    'CD2_2':    float,  # WCS coordinate transform matrix [2, 2]
    'CRPIX1':   str,  # reference pixel (RA: degree)
    'CRPIX2':   str,  # reference pixel (DEC: degree)
    'CRVAL1':   str,  # reference pixel value (RA: degree)
    'CRVAL2':   str,  # reference pixel value (DEC: degree)
    'CTYPE1':   float,  # coordinate type and projection (RA-TAN)
    'CTYPE2':   float,  # coordiante type and projection (DEC-TAN)
    'RADECSYS': str,    # coordinate system
    'NAXIS': int,       # number of dimensions
    'NAXIS1':   int,    # number of pixels in axis 1
    'NAXIS2':   int,    # number of pixels in axis 2

    ## Spectrograph configurations
    # --TODO-- missing

    ## Detector configuration: gains: and bias
    # --TODO-- missing

    ## Exposure meter configuration
    # --TODO-- missing

    ## Telescope port telemetry
    # --TODO-- missing

    ## Observatory site weather telemetry 
    # --TODO-- missing
}

## Header keywords required by level 1 and level 2 data
LEVEL1_HEADER_KEYWORDS = {
    ## DRP configuration for extration
    # --TODO-- missing

    ## DRP configuration for wavelength calibration 
    # --TODO-- missing

    ## DRP configuration and reference files for instrument drift correction 
    # --TODO-- missing

    ## Derived mid-exposure time for different wavelength bins
    # --TODO-- missing
}

## Header keywords required by level 2 data only
LEVEL2_HEADER_KEYWORDS = {
    ## Reference mask used for cross-correlation
    # --TODO-- missing

    ## Orders used for final RV calculation
    # --TODO-- missing

    ## Estimated single-measure RV precision for each orderlet of
    ## science fiber
    # --TODO-- missing
}

# KPF level 0 extensions should be defined here
# as a dictionary with the name of the extensions as keys
# and the fits data type as the values
LEVEL0_EXTENSIONS = {'PRIMARY': fits.PrimaryHDU,
                     'GREEN_AMP1': fits.ImageHDU,
                     'GREEN_AMP2': fits.ImageHDU,
                     'GREEN_AMP3': fits.ImageHDU,
                     'GREEN_AMP4': fits.ImageHDU,
                     'GREEN_CCD': fits.ImageHDU,
                     'GREEN_VAR': fits.ImageHDU,
                     
                     'RED_AMP1': fits.ImageHDU,
                     'RED_AMP2': fits.ImageHDU,
                     'RED_AMP3': fits.ImageHDU,
                     'RED_AMP4': fits.ImageHDU,
                     'RED_CCD': fits.ImageHDU,
                     'RED_VAR': fits.ImageHDU,
                     
                     'CA_HK': fits.ImageHDU,
                     'EXPMETER': fits.ImageHDU,
                     'GUIDECAM': fits.ImageHDU,

                     'RECEIPT': fits.BinTableHDU,
                     
                     'SOLAR_IRRADIANCE': fits.BinTableHDU}

# KPF level 1 extensions should be defined here
# as a dictionary with the name of the extensions as keys
# and the fits data type as the values
LEVEL1_EXTENSIONS = {'PRIMARY': fits.PrimaryHDU,
                     'GREEN_SCI_FLUX1': fits.ImageHDU,
                     'GREEN_SCI_FLUX2': fits.ImageHDU,
                     'GREEN_SCI_FLUX3': fits.ImageHDU,
                     'GREEN_SKY_FLUX': fits.ImageHDU,
                     'GREEN_CAL_FLUX': fits.ImageHDU,
                     'GREEN_SCI_VAR1': fits.ImageHDU,
                     'GREEN_SCI_VAR2': fits.ImageHDU,
                     'GREEN_SCI_VAR3': fits.ImageHDU,
                     'GREEN_SKY_VAR': fits.ImageHDU,
                     'GREEN_CAL_VAR': fits.ImageHDU,
                     'GREEN_SCI_WAVE1': fits.ImageHDU,
                     'GREEN_SCI_WAVE2': fits.ImageHDU,
                     'GREEN_SCI_WAVE3': fits.ImageHDU,
                     'GREEN_SKY_WAVE': fits.ImageHDU,
                     'GREEN_CAL_WAVE': fits.ImageHDU,
                     'GREEN_TELLURIC': fits.BinTableHDU,
                     'GREEN_SKY': fits.BinTableHDU,
                     'GREEN_SEGMENTS': fits.BinTableHDU,
                     
                     'RED_SCI_FLUX1': fits.ImageHDU,
                     'RED_SCI_FLUX2': fits.ImageHDU,
                     'RED_SCI_FLUX3': fits.ImageHDU,
                     'RED_SKY_FLUX': fits.ImageHDU,
                     'RED_CAL_FLUX': fits.ImageHDU,
                     'RED_SCI_VAR1': fits.ImageHDU,
                     'RED_SCI_VAR2': fits.ImageHDU,
                     'RED_SCI_VAR3': fits.ImageHDU,
                     'RED_SKY_VAR': fits.ImageHDU,
                     'RED_CAL_VAR': fits.ImageHDU,
                     'RED_SCI_WAVE1': fits.ImageHDU,
                     'RED_SCI_WAVE2': fits.ImageHDU,
                     'RED_SCI_WAVE3': fits.ImageHDU,
                     'RED_SKY_WAVE': fits.ImageHDU,
                     'RED_CAL_WAVE': fits.ImageHDU,
                     'RED_TELLURIC': fits.BinTableHDU,
                     'RED_SKY': fits.BinTableHDU,
                     'RED_SEGMENTS': fits.BinTableHDU,
                     
                     'RECEIPT': fits.BinTableHDU,
                     'CONFIG': fits.BinTableHDU
                    }
# KPF level 2 extensions should be defined here
# as a dictionary with the name of the extensions as keys
# and the fits data type as the values
LEVEL2_EXTENSIONS = {'PRIMARY': fits.PrimaryHDU,
                     'GREEN_CCF': fits.ImageHDU,                     
                     'RED_CCF': fits.ImageHDU,
                     
                     'RECEIPT': fits.BinTableHDU,
                     'CONFIG': fits.BinTableHDU,
                     
                     'RV': fits.BinTableHDU,
                     'ACTIVITY': fits.BinTableHDU}

