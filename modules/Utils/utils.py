# This file contains assorted utility functions that are mostly 
# for computing astronomical quantities associated with KPF data.

import numpy as np
from datetime import datetime
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_body

def get_sun_alt(UTdatetime):
    """
    Returns the altitude of the Sun at Maunakea on a specific UT datetime.

    Args:
        UTdatetime - an astropy Time object with the UT datetime

    Returns:
        alt - Altitude of the Sun in degrees at the UT datetime.  
              Negative values correspond to the Sun below the horizon.
    """
    
    sun = get_sun(UTdatetime).transform_to('icrs')
    maunakea = EarthLocation(lat='19d49m42.6s', lon='-155d28m48.9s', height=4205) 
    altaz_sun = sun.transform_to(AltAz(obstime=UTdatetime, location=maunakea))
    alt = altaz_sun.alt.deg

    return alt

from astropy.coordinates import EarthLocation, get_sun, AltAz
from astropy.time import Time
import astropy.units as u
import numpy as np
from datetime import datetime

# Define Maunakea location
maunakea = EarthLocation(lat='19d49m42.6s', lon='-155d28m48.9s', height=4205)

def get_sunrise_sunset_ut(utc_date):
    """
    Return sunrise and sunset times (UTC) as datetime.datetime objects for 
    the given UT date at Maunakea.

    Parameters:
        utc_date (str): Date string in 'YYYY-MM-DD' format

    Returns:
        (sunrise_dt, sunset_dt): Tuple of datetime.datetime objects (UT) or (None, None)
    """
    # Generate time samples (every 5 minutes over 24 hours)
    midnight = Time(utc_date + ' 00:00:00', scale='utc')
    delta_minutes = np.linspace(0, 1440, 288)  # 5-minute steps
    times = midnight + delta_minutes * u.minute

    # Compute Sun altitude
    altaz_frame = AltAz(obstime=times, location=maunakea)
    sun_altitudes = get_sun(times).transform_to(altaz_frame).alt

    # Find transitions across the horizon
    above_horizon = sun_altitudes > 0 * u.deg
    crossings = np.where(np.diff(above_horizon.astype(int)) != 0)[0]

    if len(crossings) >= 2:
        sunrise = times[crossings[0]].to_datetime()
        sunset = times[crossings[1]].to_datetime()
        return sunrise, sunset
    else:
        return None, None  # Polar day or night
   
    
def get_moon_sep(UTdatetime, RA, dec, observer_location=None):
    """
    Returns the separation in degrees between the Moon and an object with
    coordinates RA/dec at a specific UT datetime and observer location.

    Args:
        UTdatetime (Time): Astropy Time object specifying UT datetime.
        RA (str): Right ascension of the object (e.g., \"10:24:36.5\").
        dec (str): Declination of the object (e.g., \"+45:10:45.1\").
        observer_location (EarthLocation, optional): Observer's location. Defaults to Maunakea.

    Returns:
        float: Separation between Moon and object in degrees.
    """
    if observer_location is None:
        observer_location = EarthLocation(lat='19d49m42.6s', lon='-155d28m48.9s', height=4205)

    # Create SkyCoord object for the target
    target_coord = SkyCoord(RA, dec, unit=(u.hourangle, u.deg), frame='icrs')

    # Get Moon coordinates at given datetime and observer location
    moon_coord = get_body('moon', UTdatetime, observer_location).transform_to('icrs')

    # Compute separation
    separation = target_coord.separation(moon_coord)

    return separation.deg

class DummyLogger:
    """
    Make a dummy logger that prints messages in case a Python logger object is 
    not provided.  

    Usage:
       in def __init__(self, logger=None) for some class, include this line:
           self.logger = logger if logger is not None else DummyLogger()
    """
    def info(self, msg):
        print(f"INFO: {msg}")

    def debug(self, msg):
        print(f"DEBUG: {msg}")

    def warning(self, msg):
        print(f"WARNING: {msg}")

    def error(self, msg):
        print(f"ERROR: {msg}")

    def critical(self, msg):
        print(f"CRITICAL: {msg}")

def get_kpf_echelle_order(lambda_ang):
    """
    Returns the echelle order that a specified wavelength is in the free 
    spectral range of.  An example use case is to identify the spectral orders 
    in a wavelength solution when the "order number" in the L1 file is just the
    order index (starting from 0).  This function expects an input wavelength 
    in Angstroms, but will reluctantly convert to nm if the input is in the 
    range 350-900.  
    The wavelengths are from a Zemax-generated plot from Steve Gibson.

    Args:
        lambda_ang (Angstroms)

    Returns:
        order - echelle order - 137-103 on the green CCD and 102-71 on the red CCD
    """

    if lambda_ang < 900 and lambda_ang > 350: 
        lambda_ang *= 10
        print("Converting input wavelength from nm to Ang in get_kpf_echelle_order.")
        
    ech_order = -1 # default value if match not found
    fsr = {
        71:  [8709, 8587],
        72:  [8587, 8468],
        73:  [8468, 8353],
        74:  [8353, 8241],
        75:  [8241, 8132],
        76:  [8132, 8026],
        77:  [8026, 7922],
        78:  [7922, 7821],
        79:  [7821, 7723],
        80:  [7723, 7627],
        81:  [7627, 7533],
        82:  [7533, 7442],
        83:  [7442, 7353],
        84:  [7353, 7266],
        85:  [7266, 7181],
        86:  [7181, 7098],
        87:  [7098, 7017],
        88:  [7017, 6937],
        89:  [6937, 6860],
        90:  [6860, 6784],
        91:  [6784, 6710],
        92:  [6710, 6637],
        93:  [6637, 6566],
        94:  [6566, 6497],
        95:  [6497, 6429],
        96:  [6429, 6362],
        97:  [6362, 6297],
        98:  [6297, 6233],
        99:  [6233, 6171],
        100: [6171, 6109],
        101: [6109, 6049],
        102: [6049, 5990],
        103: [5990, 5932],
        104: [5932, 5875],
        105: [5875, 5820],
        106: [5820, 5765],
        107: [5765, 5711],
        108: [5711, 5659],
        109: [5659, 5607],
        110: [5607, 5556],
        111: [5556, 5506],
        112: [5506, 5458],
        113: [5458, 5409],
        114: [5409, 5362],
        115: [5362, 5316],
        116: [5316, 5270],
        117: [5270, 5225],
        118: [5225, 5181],
        119: [5181, 5138],
        120: [5138, 5095],
        121: [5095, 5053],
        122: [5053, 5012],
        123: [5012, 4971],
        124: [4971, 4931],
        125: [4931, 4892],
        126: [4892, 4854],
        127: [4854, 4815],
        128: [4815, 4778],
        129: [4778, 4741],
        130: [4741, 4705],
        131: [4705, 4669],
        132: [4669, 4634],
        133: [4634, 4599],
        134: [4599, 4565],
        135: [4565, 4531],
        136: [4531, 4498],
        137: [4498, 4465],
    }
    for key, wavs in fsr.items():
        if lambda_ang < wavs[0] and lambda_ang > wavs[1]:
            ech_order = key
            
    return ech_order

def styled_text(message, style="", color="", background=""):
    """
    Returns a message with the specified style, color, and background color using ANSI escape codes.

    Parameters:
    message (str): The message to be styled.
    style (str): The style to apply. Options are:
                 "Bold", "Dim", "Underline", "Blink", "Reverse", "Hidden"
    color (str): The text color to apply. Options are:
                 "Black", "Red", "Green", "Yellow", "Blue", "Magenta", "Cyan", "White"
    background (str): The background color to apply. Options are:
                      "BgBlack", "BgRed", "BgGreen", "BgYellow", "BgBlue", "BgMagenta", "BgCyan", "BgWhite"

    Returns:
    str: The styled message.

    Usage:
    print(styled_text("This is a bold message.", style="Bold"))
    print(styled_text("This is a red message.", color="Red"))
    print(styled_text("This is a green message with blue background.", color="Green", background="BgBlue"))
    """
    
    # Define ANSI escape codes
    codes = {
        "Reset": "\033[0m",
        "Bold": "\033[1m",
        "Dim": "\033[2m",
        "Underline": "\033[4m",
        "Blink": "\033[5m",
        "Reverse": "\033[7m",
        "Hidden": "\033[8m",
        "Black": "\033[30m",
        "Red": "\033[31m",
        "Green": "\033[32m",
        "Yellow": "\033[33m",
        "Blue": "\033[34m",
        "Magenta": "\033[35m",
        "Cyan": "\033[36m",
        "White": "\033[37m",
        "BgBlack": "\033[40m",
        "BgRed": "\033[41m",
        "BgGreen": "\033[42m",
        "BgYellow": "\033[43m",
        "BgBlue": "\033[44m",
        "BgMagenta": "\033[45m",
        "BgCyan": "\033[46m",
        "BgWhite": "\033[47m"
    }

    # Retrieve the ANSI codes from the dictionary
    style_code = codes.get(style, "")
    color_code = codes.get(color, "")
    background_code = codes.get(background, "")
    reset_code = codes["Reset"]

    # Construct the styled message
    styled_message = f"{style_code}{color_code}{background_code}{message}{reset_code}"
    return styled_message


def latex_number(number, sigfigs, min_exp=-2, max_exp=2):
    """
    Formats numbers into LaTeX-formatted strings, using scientific notation
    if the exponent is outside the specified range. Supports single floats and array inputs.

    Parameters:
    number (float or array-like): The number(s) to format.
    sigfigs (int): Number of significant figures.
    min_exp (int, optional): Minimum exponent for non-scientific notation. Defaults to -2.
    max_exp (int, optional): Maximum exponent for non-scientific notation. Defaults to 2.

    Returns:
    str or list of str: LaTeX-formatted string(s).

    Examples:
        >>> latex_number(1.236e-3, 3)
        '$1.24 \\times 10^{-3}$'

        >>> latex_number(1.236, 2)
        '1.2'

        >>> latex_number([123.6, 0.00456], 2)
        ['120', '$4.6 \\times 10^{-3}$']

        >>> latex_number(1.236e4, 3, -1, 3)
        '$1.24 \\times 10^{4}$'
    """
    from math import log10, floor

    def format_single(num):
        if np.isnan(num):
            return 'NaN'

        if num == 0:
            return '0'

        exponent = int(floor(log10(abs(num))))
        normalized = num / (10 ** exponent)

        normalized = round(normalized, sigfigs - 1)

        if normalized >= 10:
            normalized /= 10
            exponent += 1

        if min_exp <= exponent <= max_exp:
            final_number = round(num, sigfigs - 1 - exponent)
            return f'{final_number}'
        else:
            return fr'${normalized} \times 10^{{{exponent}}}$'

    if np.isscalar(number):
        return format_single(number)
    else:
        return [format_single(n) for n in np.atleast_1d(number)]

