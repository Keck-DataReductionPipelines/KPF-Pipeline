import numpy as np
import pandas as pd
from os.path import exists
from configparser import ConfigParser
from modules.Utils.config_parser import ConfigHandler


class CaHKAlg:
    """Ca H&K spectrum extraction.

    This module defines class 'HKExtractionAlg' and methods to extract spectrum from H&K science data.

    Args:
        data (numpy.ndarray): Ca H&K 2D image data.
        fibers (list): List containing the interested fibers to be extracted.
        config (configparser.ConfigParser): config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.
        instrument (str): Imaging instrument.
        config_param (ConfigHandler): Related to 'PARAM' section or section associated with the instrument
            if it is defined in the config file.
        config_logger (ConfigHandler): Related to 'LOGGER' section defined in the config file.
        hk_data (numpy.ndarray): Numpy array storing 2d image data.
        fibers (list): List storing fibers to be processed.
        data_range (list): Index range of all pixels.
        debug_output (str): File path for the file that the debug information is printed to. The printing goes to
            standard output if it is an empty string or no printing is made if it is None.
        is_time_profile (bool): Print out the time status while running.
        is_debug (bool): Print out the debug information while running.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `data` or `config`.
        Exception: If the size of `data` is less than 20 pixels by 20 pixels.
    """

    FIT_ERROR_TH = 2.5      # default set per NEID flat
    UPPER = 1
    LOWER = 0
    LOC_X1 = 'x0'
    LOC_x2 = 'xf'
    LOC_y1 = 'y0'
    LOC_y2 = 'yf'

    def __init__(self, data, fibers, config=None, logger=None):
        if not isinstance(data, np.ndarray):
            raise TypeError('image data type error, cannot construct object from CaHKAlg')
        if not isinstance(config, ConfigParser):
            raise TypeError('config type error, cannot construct object from CaHKAlg')
        if not isinstance(fibers, list):
            raise TypeError('fiber content error, cannot construct object from CaHKAlg')

        self.logger = logger
        self.hk_data = data
        self.fibers = fibers
        ny, nx = np.shape(data)
        self.data_range = [0, ny - 1, 0, nx - 1]
        config_h = ConfigHandler(config, 'PARAM')
        self.instrument = config_h.get_config_value('instrument', '')
        ins = self.instrument.upper()
        self.config_param = ConfigHandler(config, ins, config_h)  # section of instrument or 'PARAM'
        self.config_logger = ConfigHandler(config, 'LOGGER')  # section of 'LOGGER'
        self.debug_output = None
        self.is_time_profile = False
        self.is_debug = True if self.logger else False
        self.trace_location = {fiber: None for fiber in self.fibers}
        self.order_buffer = np.zeros((1, nx), dtype=float)

    def enable_debug_print(self, to_print=True):
        """Enable or disable debug printing.

        Args:
            to_print (bool, optional): Print out the debug information while running. Defaults to True.

        """
        self.is_debug = to_print or bool(self.logger)

    def enable_time_profile(self, is_time=False):
        """Enable or disable time profiling printing.

        Args:
            is_time (bool, optional): Print out the time information while running. Defaults to False.
        """

        self.is_time_profile = is_time

    def get_config_value(self, param: str, default):
        """Get defined value from the config file.

        Search the value of the specified property from config section. The default value is returned if no found.

        Args:
            param (str): Name of the parameter to be searched.
            default (str/int/float): Default value for the searched parameter.

        Returns:
            int/float/str: Value for the searched parameter.

        """
        return self.config_param.get_config_value(param, default)

    def get_data_range(self):
        """Get image size range

        Returns:
            numpy.ndarray: image range in order of y1, y2, x1 and x2.
        """
        return self.data_range

    def get_instrument(self):
        """Get imaging instrument.

        Returns:
            str: Instrument name.

        """
        return self.instrument

    def get_trace_location(self, fiber=None):
        """Get the trace location on specified fibers

        Args:
            fiber (str): Fiber name. Defaults to None for all fibers

        Returns:
            dict: fiber location on one fiber or all fibers, like::

                {
                    <fiber name 1>: {<order_1>: {'x1': , 'x2', 'y1': 'y2' },
                                     <order_2>: {'x1': , 'x2', 'y1': 'y2' }, ...},
                    <fiber name n>: {<order_1>: {'x1': , 'x2', 'y1': 'y2' }, ...}
                }

        """
        if fiber is None:
            return self.trace_location
        else:
            return {fiber: self.trace_location[fiber] if fiber in self.trace_location else None}

    def load_trace_location(self, trace_path):
        """Load the file containing trace definition and record the trace information per order and per fiber

        Args:
            trace_path: the path to a file with order trace information.
                The file is assumed in csv format containing the header and
                the space as the delimiter for each row.
        Returns:
            dict: each item in dict object has the trace value for each fiber, like::

                {
                    <fiber name>: <fiber_trace>
                        # where <fiber trace> is a dict containing location for each order,
                        {<order_number_1>: {'x1': , 'y1':, 'x2': , 'y2': },
                         <order_number_2>: {'x1': , 'y1':, 'x2': , 'y2': }, .....,
                         <order_number_n>: {'x1': , 'y1':, 'x2': , 'y2': }}
                }


        """
        if not exists(trace_path):
            return None

        loc_result = pd.read_csv(trace_path, sep=' ')
        loc_vals = np.array(loc_result.values)
        loc_cols = np.array(loc_result.columns)

        order_col_name = 'order'
        fiber_col_name = 'fiber'
        loc_col_names = [self.LOC_X1, self.LOC_y1, self.LOC_x2, self.LOC_y2]

        loc_idx = {c: np.where(loc_cols == c)[0][0] for c in loc_col_names}
        order_idx = np.where(loc_cols == order_col_name)[0][0]
        fiber_idx = np.where(loc_cols == fiber_col_name)[0][0]

        for fiber in self.fibers:
            loc_for_fiber = loc_vals[np.where(loc_vals[:, fiber_idx] == fiber)[0], :]  # rows with the same fiber
            self.trace_location[fiber] = dict()
            for loc in loc_for_fiber:       # add each row from loc_for_fiber to trace_location for fiber
                self.trace_location[fiber][loc[order_idx]] = {'x1': loc[loc_idx[self.LOC_X1]],
                                                              'x2': loc[loc_idx[self.LOC_x2]],
                                                              'y1': loc[loc_idx[self.LOC_y1]],
                                                              'y2': loc[loc_idx[self.LOC_y2]]}
        return self.trace_location

    def d_print(self, *args, end='\n', info=False):
        """Print out running status to logger or debug information to a file.

        Args:
            *args: Variable length argument list to print.
            end (str, optional): Specify what to print at the end.
            info (bool): Print out for information level, not for debug level.

        Notes:
            This function handles the print-out to the logger defined in the config file or other file as specified in
            :func:`~alg.OrderTraceAlg.add_file_logger()`.

        """
        if self.is_debug:
            out_str = ' '.join([str(item) for item in args])
            if self.logger:
                if info:
                    self.logger.info(out_str)
                else:
                    self.logger.debug(out_str)
            if self.debug_output is not None and not info:
                if self.debug_output:
                    with open(self.debug_output, 'a') as f:
                        f.write(' '.join([str(item) for item in args]) + end)
                        f.close()
                else:
                    print(out_str, end=end)

    def t_print(self, *args):
        """Print time profiling information to the logger.

        Args:
             *args: Variable length argument list to print.

        """
        if self.is_time_profile and self.logger:
            out_str = ' '.join([str(item) for item in args])
            self.logger.info(out_str)

    def add_file_logger(self, filename: str = None):
        """Add file to log debug information.

        Args:
            filename (str, optional): Filename of the log file. Defaults to None.

        Returns:
            None.

        """
        self.enable_debug_print(filename is not None)
        self.debug_output = filename

    def get_spectral_data(self):
        """Get spectral information including data and dimension.

        Returns:
            tuple: Information of spectral data,

                * (*numpy.ndarray*): 2D spectral data.
                * **nx** (*int*): Width of the data.
                * **ny** (*int*): Height of the data.

        """
        ny, nx = np.shape(self.hk_data)

        return self.hk_data, nx, ny

    def get_order_buffer(self):
        """Get a pre-allocated buffer with all zeros to contain sum of  spectrum extraction

        Returns:
            numpy.ndarray: 1 x <spectrum width>  array with all zeros
        """

        self.order_buffer.fill(0.0)
        return self.order_buffer

    def summation_extraction_one_order(self, order_loc):
        """ Spectrum extraction and summation on extracted pixels per order location

        Args:
            order_loc (dict): Order location in terms of x1, x2, y1, y2 per order.

        Returns:
            dict: extracted and summed result with dimension 1xn where n is the image width, like::

            {
                    'extraction': numpy.ndarray   # summation of extraction
            }

        """

        x1, x2, y1, y2 = order_loc['x1'], order_loc['x2'], order_loc['y1'], order_loc['y2']
        p_result = self.get_order_buffer()
        extracted_img = self.hk_data[y1:y2+1, x1:x2+1]
        p_result[0, x1:x2+1] = np.sum(extracted_img, axis=0)
        return {'extraction': p_result}

    def summation_extraction(self, trace_location, selected_orders=None):
        """Extract spectrum for selected orders and perform summation on the extracted data

        Args:
            trace_location (dict): trace location of selected orders
            selected_orders (list): selected orders to be processed. Defaults to None for all orders.

        Returns:
            numpy.ndarray: 2D data containing the summation on selected orders.

        """

        _, nx, ny = self.get_spectral_data()

        if selected_orders is None:
            selected_orders = list(trace_location.keys())

        out_data = np.zeros((len(selected_orders), nx))

        for idx, ord_no in enumerate(selected_orders):
            sum_result = self.summation_extraction_one_order(trace_location[ord_no])
            out_data[idx] = 1.0 * sum_result['extraction']

        return out_data

    @staticmethod
    def write_data_to_dataframe(out_data, fiber_name, extraction_dim):
        """ Write Ca H&K extraction result to an instance of Pandas DataFrame.

        Args:
            out_data (numpy.ndarray): H&K spectrum extraction result.  Each row of the array corresponds to the reduced
                        1D data of one order.
            fiber_name (str): Fiber name.
            extraction_dim (dict): Dimension in orders for the specified fiber.

        Returns:
            Pandas.DataFrame: Instance of DataFrame containing the extraction result plus the following attributes:

            - *FIBER*: fiber name.
            - *Ordern*: Dimension data for order n.
        """

        df_result = pd.DataFrame(out_data)
        df_result.attrs['FIBER'] = fiber_name

        if isinstance(extraction_dim, dict) and bool(dict):
            for order_no in sorted(extraction_dim.keys()):
                order_coord = extraction_dim[order_no]
                order_dim = ','.join([str(coord) for coord in [order_coord['x1'], order_coord['y1'],
                                                               order_coord['x2'], order_coord['y2']]])

                df_result.attrs['ORDER'+str(order_no)] = (order_dim, "x1,y1,x2,y2")

        return df_result

    def extract_spectrum(self,
                         fiber_name,
                         order_set=None,
                         show_time=False,
                         print_debug=None):
        """ Spectrum extraction from 2D flux to 1D by doing summation on extracted data.

        Args:
            fiber_name (str, optional): Name of the fiber to be processed.
            order_set (numpy.ndarray, optional): Set of selected orders to extract. Defaults to None for all orders.
            show_time (bool, optional):  Show running time of the steps. Defaults to False.
            print_debug (str, optional): Print debug information to stdout if it is provided as empty string,
                a file with path `print_debug` if it is non empty string, or no print if it is None.
                Defaults to None.

        Returns:
            dict: spectral extraction result from 2D spectrum data, like::

                    {
                        'spectral_extraction_result':  Padas.DataFrame
                    }

        """

        self.add_file_logger(print_debug)
        self.enable_time_profile(show_time)

        if fiber_name not in self.fibers:
            return {'spectral_extraction_result': None, 'message:': 'invalid fiber name'}

        if not any(self.trace_location.values()):
            return {'spectral_extraction_result': None, 'message:': 'no trace location is loaded'}

        trace_loc = self.trace_location[fiber_name]
        if not trace_loc:
            return {'spectral_extraction_result': None, 'message:': 'no trace location for '+fiber_name}

        if order_set is None:
            order_set = list(trace_loc.keys()).sort()

        out_data = self.summation_extraction(trace_loc, order_set)

        df_data = self.write_data_to_dataframe(out_data, fiber_name, trace_loc)

        return {'spectral_extraction_result': df_data}
