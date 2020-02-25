
from astropy.time import Time
from astropy.coordinates import Angle

## refer to KPF PDR System Design manual v1, Nov 6, 2017

## Header keywords required by all levels of data
HEADER_KEY = {
    ## Temperal keywords
    'OBS-TIME': Time,   # universal time of observation.
    'ELAPTIME': float,  # open shutter time (all coadds) (seconds)

    ## Observation location keywords
    'AIRMASS':  float,  # air mass
    'AZ':       Angle,  # telescope azimuth (deg)
    'EL':       Angle,  # telescope elevation (deg)
    'DEC':      Angle,  # declination (DD:MM:SS.SS or decimal degree)
    'EQUINOX':  float,  # telescope equinox (2000.0)
    'OBJECT':   str,    # observed object name
    'PA':       Angle,  # position angle from north
    'RA':       Angle,  # right ascension (HH:MM:SS.SS or decimal degree)
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

    ## Caliration association
    'DETMODE':  float,  # --TODO-- description
    'DISPERS':  float,  # --TODO-- description 
    'DISPPOS':  float,  # --TODO-- description 
    'ECHLPOS':  float,  # --TODO-- description 
    'FILNAME':  str,    # --TODO-- description 
    'SLIT-NAME':str,    # --TODO-- description 
    'STATEID':  int,    # unique ID assigned to all associated FITS file
    'STATENAM': str,    # user supplied name for STATEID

    ## Program identification keyword
    'PROGNAME': str,    # WMKO assigned program ID that FITS file

    ## World coordinate system keywords 
    'CD1_1':    float,  # WCS coordinate transform matrix [1, 1]
    'CD1_2':    float,  # WCS coordinate transform matrix [1, 2]
    'CD2_1':    float,  # WCS coordinate transform matrix [2, 1]
    'CD2_2':    float,  # WCS coordinate transform matrix [2, 2]
    'CRPIX1':   Angle,  # reference pixel (RA: degree)
    'CRPIX2':   Angle,  # reference pixel (DEC: degree)
    'CRVAL1':   Angle,  # reference pixel value (RA: degree)
    'CRVAL2':   Angle,  # reference pixel value (DEC: degree)
    'CTYPE1':   float,  # coordinate type and projection (RA-TAN)
    'CTYPE2':   float,  # coordiante type and projection (DEC-TAN)
    'RADECSYS': str,    # coordinate system
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
LVL1_KEY = {
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
LVL2_KEY = {
    ## Reference mask used for cross-correlation
    # --TODO-- missing

    ## Orders used for final RV calculation
    # --TODO-- missing

    ## Estimated single-measure RV precision for each orderlet of
    ## science fiber
    # --TODO-- missing
}
