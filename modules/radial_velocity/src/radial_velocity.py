# Standard dependencies
"""
    This module defines class `RadialVelocity` which inherits from `KPF1_Primitive` and provides methods to perform
    the event on radial velocity in the recipe.

    Attributes:
        RadialVelocity

    Description:
        * Method `__init__`:

            RadialVelocity constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocity` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data constructed by
                      optimal extraction.
                    - `action.args[1] (dict)`: Result from the init work made by `RadialVelocityInit` which makes
                      mask lines and velocity steps based on star and other module associated configuration for
                      radial velocity computation.
                    - `action.args['order_name'] (str, optional)`: Order name associated with the level 1 data.
                      Defaults to 'SCI1'.
                    - `action.args['start_order'] (int, optional)`: Index of the first order to be processed.
                      Defaults to None.
                    - `action.args['end_order'] (int, optional)`: Index of the last order to be processed.
                      Defaults to None.
                    - `action.args['input_ref'] (np.ndarray|str|pd.DataFrame, optional)`: Reference for
                      reweighting ccf orders. Defaults to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input (kpfpipe.models.level1.KPF1)`: Instance of `KPF1`, assigned by `action.args[0]`.
                - `rv_init (dict)`: Result from radial velocity init.
                - `sci (str)`: Name of the order to be processed.
                - `start_order (int)`: Index of the first order to be processed.
                - `end_order (int)`: Index of the last order to be processed.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `spectrum_data (numpy.ndarray)`: Reduced 1D data of all orders from optimal extraction, associated
                  with `action.args[0]`.
                - `wave_cal (numpy.ndarray)`: Wavelength calibration data, associated with `action.args[0]`.
                - `header (fits.header.Header)`: Fits header of HDU associated with `spectrum_data`.
                - `ref_ccf (numpy.ndarray)`: Reference or ratio of cross correlation values for scaling the computation
                  of cross correlation, associated with `action.args['input_ref']`.
                - `alg (RadialVelocityAlg)`: Instance of RadialVelocityAlg which has operation codes for the
                  computation of radial velocity.

        * Method `__perform`:

            RadialVelocity returns the result in `Arguments` object which contains the original input
            level 1 data object (`KPF1`) plus an extension with the radial velocity result.
            (the result will be put into a level 2 data object after level 2 data model is implemented.)

    Usage:
        For the recipe, the optimal extraction event is issued like::

            rv_init = RadialVelocityInit()
            :
            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_data = RadialVelocity(lev1_data, rv_init, order_name=order_name)
            :
"""

import configparser
import numpy as np
import os
import pandas as pd
# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg

DEFAULT_CFG_PATH = 'modules/radial_velocity/configs/default.cfg'


class RadialVelocity(KPF1_Primitive):

    default_agrs_val = {
        'order_name': 'SCI1'
    }

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.input = action.args[0]
        self.rv_init = action.args[1]
        self.ref_ccf = None

        if 'input_ref' in args_keys:
            if isinstance(action.args['input_ref'], np.ndarray):
                self.ref_ccf = action.args['input_ref']
            elif isinstance(action.args['input_ref'], pd.DataFrame):
                self.ref_ccf = action.arg['input_ref'].values
            elif isinstance(action.args['input_ref'], str) and os.path.exists(action.args['input_ref']):
                ratio_df = pd.read_csv(action.args['input_ref'])
                self.ref_ccf = ratio_df.values

        self.sci = action.args['order_name'] if 'order_name' in args_keys and action.args['order_name'] is not None\
            else self.default_args_val['order_name']
        self.start_order = action.args['start_order'] if 'start_order' in args_keys else None
        self.end_order = action.args['end_order'] if 'end_order' in args_keys else None
        is_kpf_type = action.args['is_kpf_type'] if 'is_kpf_type' in args_keys else True

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['radial_velocity']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        data = self.input.data if hasattr(self.input, 'data') else None
        self.spectrum_data = self.input.data[self.sci][0, :, :] if data and self.sci in data else None
        self.wave_cal = self.input.data[self.sci][1, :, :] if data and self.sci in data else None
        header = self.input.header if hasattr(self.input, 'header') else None
        self.header = None
        if header:
            if not is_kpf_type:
                self.header = header['PRIMARY']
            elif (self.sci + '_FLUX') in header:
                self.header = header[self.sci + '_FLUX']

        # Order trace algorithm setup
        self.alg = RadialVelocityAlg(self.spectrum_data, self.header, self.rv_init, wave_cal=self.wave_cal,
                                     config=self.config, logger=self.logger)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input, KPF1) and \
            (self.start_order is None  or self.end_order is None or self.start_order <= self.end_order) and \
            (self.ref_ccf is None or isinstance(self.ref_ccf, np.ndarray))

        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform radial velocity computation by calling method 'compute_rv_by_cc' from RadialVelocityAlg.

        Returns:
            Level 1 data from the input plus an extension with the cross correlation results from all orders.
            (this part will be updated after level 2 data model is made.)
        """

        _, nx, ny = self.alg.get_spectrum()
        s_order = -1
        e_order = -1
        s_x_pos = -1
        e_x_pos = -1

        if self.alg.get_instrument() == 'NEID':
            if self.rv_init['data']['rv_config']['starname'] != 'HD 127334':
                s_order = 3
                e_order = min(82, np.shape(self.spectrum_data)[0]-1)
            s_x_pos = 600
            e_x_pos = nx - s_x_pos
        elif self.alg.get_instrument() == 'HARPS':
            s_order = 0
            e_order = 69
            s_x_pos = 500
            e_x_pos = 3500

        if s_order == -1:
            if self.start_order is not None:
                s_order = self.start_order
            if self.end_order is not None:
                e_order = self.end_order
        else:
            if self.start_order is not None and e_order >= self.start_order > s_order:
                s_order = self.start_order
            if self.end_order is not None and e_order > self.end_order >= s_order:
                e_order = self.end_order

        if self.logger:
            self.logger.info("RadialVelocity: Start crorss correlation to find radial velocity... ")
        rv_results = self.alg.compute_rv_by_cc(start_order=s_order, end_order=e_order,
                                               start_x=s_x_pos, end_x=e_x_pos, ref_ccf=self.ref_ccf)
        output_df = rv_results['ccf_df']
        assert(not output_df.empty and output_df.values.any())

        self.input.create_extension('CCF')
        self.input.extension['CCF'] = output_df
        for att in output_df.attrs:
            self.input.header['CCF'][att] = output_df.attrs[att]
        self.input.receipt_add_entry('RadialVelocity', self.__module__, f'config_path={self.config_path}', 'PASS')

        if self.logger:
            self.logger.info("RadialVelocity: Receipt written")

        if self.logger:
            self.logger.info("RadialVelocity: Done!")

        return Arguments(self.input)



