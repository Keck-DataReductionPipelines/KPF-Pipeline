import glob
import numpy as np
from astropy.io import fits

class FitsHeaders:

    """
    Description:
        This class contains functions to retrieve and act upon information
        in the headers of FITS files located within a given data directory.
        Typically, the functions will act upon all FITS files under a given date.

    Arguments:
        search_path (str, which can include file glob): Directory path of FITS files.
        header_keywords (str or list of str): FITS keyword(s) of interest.
        header_values (str or list of str): Value(s) of FITS keyword(s), in list order.

    Attributes:
        header_keywords (str or list of str): FITS keyword(s) of interest.
        header_values (str or list of str): Value(s) of FITS keyword(s), in list order.
        n_header_keywords (int): Number of FITS keyword(s) of interest.
        found_fits_files (list of str): Individual FITS filename(s) that match.

    """

    def __init__(self, search_path, header_keywords, header_values):
        self.n_header_keywords = np.size(header_keywords)
        if self.n_header_keywords == 1:
            header_keywords = [header_keywords]
            header_values = [header_values]
        self.header_keywords = header_keywords
        self.header_values = header_values
        self.found_fits_files = glob.glob(search_path)

    # Return list of files that each has lowercase string matches
    # to all input FITS kewords/values of interest.

    def match_headers_string_lower(self):
        matched_fits_files = []
        for fits_file in self.found_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()
                fits_value = (fits.getval(fits_file, self.header_keywords[i])).lower()

                if (fits_value == input_value): match_count += 1

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)
        return matched_fits_files

    # Return list of files that each has floating-point
    # values that are less than or equal to 
    # all input FITS kewords/values of interest.

    def match_headers_float_le(self):
        matched_fits_files = []
        for fits_file in self.found_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = float(self.header_values[i])
                fits_value = float(fits.getval(fits_file, self.header_keywords[i]))

                if (fits_value <= input_value): match_count += 1

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)
        return matched_fits_files
