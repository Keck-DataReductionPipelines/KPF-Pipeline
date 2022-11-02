from modules.Utils.kpf_fits import FitsHeaders

from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

class FitsHeadersMatchFloatLe(KPF0_Primitive):

    """
    Description:
        This class sets up and executed the match_headers_float_le function of
        the FitsHeader class within the Keck framework. 

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

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.search_path = self.action.args[0]
        self.header_keywords = self.action.args[1]
        self.header_values = self.action.args[2]

        self.fh = FitsHeaders(self.search_path,self.header_keywords,self.header_values)

    """

    Return list of files that each has floating-point
    values that are less than or equal to 
    all input FITS kewords/values of interest.

    """

    def _perform(self):

        matched_fits_files = self.fh.match_headers_float_le()

        return Arguments(matched_fits_files)
