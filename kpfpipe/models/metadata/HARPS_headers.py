from astropy.time import Time
## Refer to HARPS DRS User Manual Chapter 4 for header descriptions
# https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/DRS.pdf

## Manual section 4.2.1
## Raw frame (equivalent of KPF lvl 0)
HAPRS_HEADER_RAW = {
    'NAXIS':     int,    # number of data axes 
    'NAXIS1':    int,    # length of data axis 1 (row)
    'NAXIS2':    int,    # length of data axis 2 (column)
    'ORIGIN':    str,    # observatory name (European Southern Observatory)
    'DATE':      Time,   # date this file was written
    'TELESCOPE': str,    # ESO telescope name
    'INSTRUME':  str,    # instrument name 
    'OBJECT':    str,    # target description
    'RA':        float,  # 17:57:47.9 RA (J2000) pointing
    'DEC':       float,  # 04:42:51.6 DEC (J2000) pointing 
    'EQUINOX':   float,  # standard FK5 (years)
    'RADECSYS':  str,    # coordinate reference frame
    'EXPTIME':   float,  # totdal integration time
    'MJD-OBS':   float,  # MJD start (2007-04-04T09:17:51.376)
    'DATE-OBS':  Time,   # date of observation
    'UTC':       float,  # 09:17:46.000 UTC
    'LST':       float,  # 17:24:05.503 LST
    'PI-COI':    str,    # PI-COI name
    'ORIGFILE':  str,    # original file name 
    
    ## DET
    'HIERARCH ESO DET READ SPEED': str,      # CCD readout mode (speed, port, and gain)
    'HIERARCH ESO DET OUT2 RON':   float,    # readout noise (e-) of Linda (float)
    'HIERARCH ESO DET OUT2 CONAD': float,    # conversion from ADUs to electrom of Linda (Blue)
    'HIERARCH ESO DET OUT4 RON':   float,    # readout noise (e-) of Jasmin (float)
    'HIERARCH ESO DET OUT4 CONAD': float,    # conversion from ADUs to electrom of Jasmin (Red)
    'HIERARCH ESO DET WIN1 DIT1':  float,    # actual sub-integration time 
    'HIERARCH ESO DET WINI DKTM':  float,    # dark current time (float)
    'HIERARCH ESO DET DPR CATG':   str,      # observation category
    'HIERARCH ESO DET DPR TYPE':   str,      # exposure type 

    ## INS 
    'HIERARCH ESO INS DET1 TMMEAN': float, # normalized mean exposure time
    'HIERARCH ESO INS DET1 CMMEAN': float, # average count PM on fiber A
    'HIERARCH ESO INS DET2 CMMEAN': float, # average count PM on fiber B

    ## TEL 
    'HIERARCH ESO OBS TARG NAME':       str,   # target name 
    'HIERARCH ESO TEL TARG EQUINOX':    float, # equinox
    'HIERARCH ESO TEL TARG PMA':        float, # proper motion alpha (arcsec/year) 
    'HIERARCH ESO TEL TARG PMD':        float, # proper motion delta (arcsec/year)
    'HIERARCH ESO TEL TARG RADVEL':     float, # radial velocity of target (km/s)
    'HIERARCH ESO TEL AMBI FWHM START': float, # seeing at start
    'HIERARCH ESO TEL AMBI FWHM END':   float, # seeing at end 
    'HIERARCH ESO TEL AIRM START':      float, # airmass at start 
    'HIERARCH ESO TEL AIRM END':        float, # airmass at end 
    
    ## TPL
    'HIERARCH ESO TPL NEXP':   int,  # TPL number of exposures 
    'HIERARCH ESO TPL EXPNO':  int,  # TPL exposure number within template
    'HIERARCH ESO TPL NAME':   str   # TPL name

    # --TODO-- Expand on this
}

## Manual section 4.2.6
## 2D extracted spectrums, equivalent of KPF lvl 1
HARPS_HEADER_E2DS = {
    'HIERARCH ESO CAL LOG FILE': str, # localization file used
    'HIERARCH ESO CAL EXT OPT':  int, # option extration 

    # --TODO-- finish this
}

