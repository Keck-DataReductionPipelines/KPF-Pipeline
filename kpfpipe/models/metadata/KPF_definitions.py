
from kpfpipe.models.metadata.receipt_columns import RECEIPT_COL
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits

import numpy as np
import os
import pandas as pd
from collections import OrderedDict

# refer to KPF PDR System Design manual v1, Nov 6, 2017
# and the table on the confluence page
# https://exoplanets.atlassian.net/wiki/spaces/shrek/pages/2142208120/Data+Format

## Header keywords required by all levels of data
# defined in a series of CSV files
LEVEL0_HEADER_FILE = os.path.abspath(os.path.dirname(__file__)) + '/KPF_headers_L0.csv'
LEVEL1_HEADER_FILE = LEVEL0_HEADER_FILE.replace('L0', 'L1')
LEVEL2_HEADER_FILE = LEVEL0_HEADER_FILE.replace('L0', 'L2')

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
                     'CONFIG': fits.BinTableHDU,
                     
                     'SOLAR_IRRADIANCE': fits.BinTableHDU}

# KPF level 1 extensions should be defined here
# as a dictionary with the name of the extensions as keys
# and the fits data type as the values
LEVEL1_EXTENSIONS = {'PRIMARY': fits.PrimaryHDU,
                     'RECEIPT': fits.BinTableHDU,
                     'CONFIG': fits.BinTableHDU,

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

                     'CA_HK_FLUX1': fits.ImageHDU,
                     'CA_HK_FLUX2': fits.ImageHDU,
                     'CA_HK_WAVE': fits.ImageHDU,
                    }
# KPF level 2 extensions should be defined here
# as a dictionary with the name of the extensions as keys
# and the fits data type as the values
LEVEL2_EXTENSIONS = {'PRIMARY': fits.PrimaryHDU,
                     'RECEIPT': fits.BinTableHDU,
                     'CONFIG': fits.BinTableHDU,
                    
                     'GREEN_CCF': fits.ImageHDU,                     
                     'RED_CCF': fits.ImageHDU,

                     'RV': fits.BinTableHDU,
                     'ACTIVITY': fits.BinTableHDU}

# mapping between fits extension data types and Python object data types
FITS_TYPE_MAP = {fits.PrimaryHDU: OrderedDict,
                 fits.ImageHDU: np.array,
                 fits.BinTableHDU: pd.DataFrame}
