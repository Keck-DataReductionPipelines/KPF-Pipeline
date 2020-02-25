from astropy.time import Time
## Refer to HARPS DRS User Manual Chapter 4 for header descriptions
# https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/DRS.pdf

## Manual section 4.2.1
## Raw frame (equivalent KPF lvl 0)
HAPRS_HEADER_RAW = {
    # '%(HARPS_key):  (%(expected_value_type), %(equivalent_KPF_key))
    'NAXIS':     (int,    None),    # number of data axes (typically 2)
    'NAXIS1':    (int,    None),    # length of data axis 1 (row)
    'NAXIS2':    (int,    None),    # length of data axis 2 (column)
    'ORIGIN':    (str,    None),    # observatory name (European Southern Observatory)
    'DATE':      (Time,   None),   # date this file was written
    'TELESCOPE': (str,    None),    # ESO telescope name
    'INSTRUME':  (str,    None),    # instrument name 
    'OBJECT':    (str,    None),    # target description
    'RA':        (float,  None),  # 17:57:47.9 RA (J2000) pointing
    'DEC':       (float,  None),  # 04:42:51.6 DEC (J2000) pointing 
    'EQUINOX':   (float,  None),  # standard FK5 (years)
    'RADECSYS':  (str,    None),    # coordinate reference frame
    'EXPTIME':   (float,  None),  # totdal integration time
    'MJD-OBS':   (float,  None),  # MJD start (2007-04-04T09:17:51.376)
    'DATE-OBS':  (Time,   None),   # date of observation
    'UTC':       (float,  None),  # 09:17:46.000 UTC
    'LST':       (float,  None), # 17:24:05.503 LST
    'PI-COI':    (str,    None),   # PI-COI name
    'ORIGFILE':  (str,    None),   # original file name 

    ## DET
    'HIERARCH ESO DET READ SPEED': (str,   None),    # CCD readout mode (speed, port, and gain)
    'HIERARCH ESO DET OUT2 RON':   (float, None),    # readout noise (e-) of Linda (float)
    'HIERARCH ESO DET OUT2 CONAD': (float, None),    # conversion from ADUs to electrom of Linda (Blue)
    'HIERARCH ESO DET OUT4 RON':   (float, None),    # readout noise (e-) of Jasmin (float)
    'HIERARCH ESO DET OUT4 CONAD': (float, None),    # conversion from ADUs to electrom of Jasmin (Red)
    'HIERARCH ESO DET WIN1 DIT1':  (float, None),    # actual sub-integration time 
    'HIERARCH ESO DET WINI DKTM':  (float, None),    # dark current time (float)
    'HIERARCH ESO DET DPR CATG':   (str,   None),    # observation category
    'HIERARCH ESO DET DPR TYPE':   (str,   None),    # exposure type 

    ## INS 
    'HIERARCH ESO INS DET1 TMMEAN': (float, None), # normalized mean exposure time
    'HIERARCH ESO INS DET1 CMMEAN': (float, None), # average count PM on fiber A
    'HIERARCH ESO INS DET2 CMMEAN': (float, None), # average count PM on fiber B

    ## TEL 
    'HIERARCH ESO OBS TARG NAME':       (str,   None), # target name 
    'HIERARCH ESO TEL TARG EQUINOX':    (float, None), # equinox
    'HIERARCH ESO TEL TARG PMA':        (float, None), # proper motion alpha (arcsec/year) 
    'HIERARCH ESO TEL TARG PMD':        (float, None), # proper motion delta (arcsec/year)
    'HIERARCH ESO TEL TARG RADVEL':     (float, None), # radial velocity of target (km/s)
    'HIERARCH ESO TEL AMBI FWHM START': (float, None), # seeing at start
    'HIERARCH ESO TEL AMBI FWHM END':   (float, None), # seeing at end 
    'HIERARCH ESO TEL AIRM START':      (float, None), # airmass at start 
    'HIERARCH ESO TEL AIRM END':        (float, None), # airmass at end 
    
    ## TPL
    'HIERARCH ESO TPL NEXP':   (int,  None),  # TPL number of exposures 
    'HIERARCH ESO TPL EXPNO':  (int,  None),  # TPL exposure number within template
    'HIERARCH ESO TPL NAME':   (str,  None)  # TPL name

    # --TODO-- Expand on this
}

## Manual section 4.2.6
## 2D extracted spectrums ( equivalent of KPF lvl 1)
HARPS_HEADER_E2DS = {
    'HIERARCH ESO CAL LOG FILE': (str, None),  # localization file used
    'HIERARCH ESO CAL EXT OPT':  (int,  None)  # option extration 

    # --TODO-- finish this
}

