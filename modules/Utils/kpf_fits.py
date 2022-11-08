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

    def __init__(self, search_path, header_keywords, header_values, logger=None):
        self.n_header_keywords = np.size(header_keywords)
        if self.n_header_keywords == 1:
            header_keywords = [header_keywords]
            header_values = [header_values]
        self.header_keywords = header_keywords
        self.header_values = header_values
        self.found_fits_files = glob.glob(search_path)
        if logger:
            self.logger = logger
            self.logger.debug('FitsHeaders class constructor: self.found_fits_files = {}'.format(self.found_fits_files))
        else:
            self.logger = None
            print('---->FitsHeaders class constructor: self.found_fits_files = {}'.format(self.found_fits_files))

    def match_headers_string_lower(self):

        """
        Return list of files that each has lowercase string matches
        to all input FITS kewords/values of interest.
        """

        matched_fits_files = []
        for fits_file in self.found_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()

                try:

                    val = fits.getval(fits_file, self.header_keywords[i])
                    fits_value = (val).lower()

                    if (fits_value == input_value): match_count += 1

                except KeyError as err:

                    print("KeyError:", err)

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)
        return matched_fits_files

    def match_headers_float_le(self):

        """
        Return list of files that each has floating-point
        values that are less than or equal to
        all input FITS kewords/values of interest.
        """

        matched_fits_files = []
        for fits_file in self.found_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = float(self.header_values[i])

                if self.logger:
                    self.logger.debug('FitsHeaders.match_headers_float_le(): file,i,keyword,value = {},{},{},{}'.\
                        format(fits_file,i,self.header_keywords[i],input_value))
                else:
                    print('---->FitsHeaders.match_headers_float_le(): file,i,keyword,value = {},{},{},{}'.\
                        format(fits_file,i,self.header_keywords[i],input_value))

                try:

                    val = fits.getval(fits_file, self.header_keywords[i])

                    try:
                        fits_value = float(val)

                        if (fits_value <= input_value): match_count += 1

                    except ValueError as err2:

                        print("ValueError:", err2)

                except KeyError as err:

                    print("KeyError:", err)

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)
        return matched_fits_files
