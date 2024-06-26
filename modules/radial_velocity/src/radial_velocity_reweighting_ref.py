# Standard dependencies
"""
    This module defines class `RadialVelocityReweightingRef` which inherits from `KPF_Primitive` and provides
    methods to perform the event on radial velocity reweighting ratio table creation or loading in the recipe.

    Description:
        * Method `__init__`:

            RadialVelocityReweightingRef constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocityReweightingRef` event issued in the recipe:

                    - `action.args[0] (list|str)`: List of KPF2 or one single KPF2 for reweighting. None means to load
                      a predefined ratio table file.
                    - `action.args[1] (str)`: Reweighting method.
                    - `action.args[2] (int)`: Total order from the ccf data to build the reweighting template.
                    - `action.args['ccf_hdu_name'] (str)`: The HDU name in fits file for the HDU with ccf data.
                      Defaults to 'ccf'.
                    - `action.args['ccf_start_index'] (int)`: The order index that the first row of ccf_data is
                      associated with. Defaults to 0.
                    - `action.args['is_ratio_data'] (boolean)`: If the file is a csv file containing the ratio for
                      reweighting ccf orders. Defaults to False.
                    - `action.args['ccf_ratio_file'] (str)`: path of the ratio file.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `files (List)`: list of KPF2 objects to find the reference template for reweighting.
                - `reweighting_method (str)`: Reweighting method.
                - `total_segment (int)`: Total segments to build the reweighting reference.
                - `ccf_hdu_name (str)`: name of hdu containing ccf.
                - `ccf_start_index (int)`: The order index that the first row of ccf_data is associated with.
                - `is_ratio_data (bool)`: If the file is a csv file containing the ratio for reweighting.
                - `ccf_ratio_file (str)`: output file containing ccf ratio.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.

        * Method `__perform`:

            RadialVelocityReweightingRef returns the result in `Arguments` object containing an instance of
            numpy.ndarray to represent the ratio for the reweighting and the ratio result is written into a file in csv
            format if `ccf_ratio_file` is set and `is_ratio_data` is False.

    Usage:
        For the recipe, the make reweighting ratio table is issued like::

            :
            # load the table file with the weighs
            reweighting_ref =  RadialVelocityReweightingRef(None,  'ccf_max',
                                35,                             # total segment
                                ccf_hdu_nane='GREEN_CCF',
                                is_ratio_data=Tue,
                                ccf_ratio_file=<ratio data file path>
                                ccf_start_index=0
                            )
            :

        where `reweighting_ref` is Pandas DataFrame object wrapped in `Arguments` class object.
"""

import configparser
import numpy as np
import os.path
import pandas as pd

# Pipeline dependencies
from kpfpipe.primitives.level2 import KPF2_Primitive
from kpfpipe.models.level2 import KPF2

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg

DEFAULT_CFG_PATH = 'modules/radial_velocity/configs/default.cfg'


class RadialVelocityReweightingRef(KPF2_Primitive):
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF2_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.reweighting_method = action.args[1]
        self.total_segment = action.args[2]

        self.ccf_hdu_name = action.args['ccf_hdu_name'] if 'ccf_hdu_name' in args_keys else 'CCF'
        self.ccf_start_index = action.args['ccf_start_index'] if 'ccf_start_index' in args_keys else 0
        self.is_ratio_data = action.args['is_ratio_data'] if 'is_ratio_data' in args_keys else False
        self.ccf_ratio_file = action.args['ccf_ratio_file'] if 'ccf_ratio_file' in args_keys else ''
        # self.rv_ext_idx = action.args['rv_ext_idx'] if 'rv_ext_idx' in args_keys else 1

        file_list = action.args[0] if isinstance(action.args[0], list) else [action.args[0]]
        self.files = []

        f_size = -1
        for one_file in file_list:
            if isinstance(one_file, str):
                f_lev2 = KPF2.from_fits(one_file)
            elif isinstance(one_file, KPF2):
                f_lev2 = one_file
            else:
                f_lev2 = None
            if f_lev2 is not None:
                self.files.append(f_lev2)
                # find the total_segment if not set yet
                if self.total_segment is None and self.ccf_hdu_name in f_lev2.header \
                        and 'NAXIS' in f_lev2.header[self.ccf_hdu_name]:
                    nx = f_lev2.header[self.ccf_hdu_name]['NAXIS']    # 2 or 3
                    if f_size >= 0:
                        f_size = min(f_size, np.shape(f_lev2[self.ccf_hdu_name])[nx-2])
                    else:
                        f_size = np.shape(f_lev2[self.ccf_hdu_name])[nx-2]

        if self.total_segment is None and f_size != -1:
            self.total_segment = f_size
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

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # check reweighting method, and using ratio file or a collection of kpf2 files
        success = (self.reweighting_method in RadialVelocityAlg.CCF_Methods) and \
                  ((self.is_ratio_data and self.ccf_ratio_file) or
                   (isinstance(self.files, list) and (len(self.files) > 0)))

        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action - find radial velocity rewighting reference

        Returns:
            pandas.DataFrame as a reweighting ratio table or a ccf reference from observation template
        """

        def get_template_observation(f, hdu_idx_name, msg, is_ratio=False):
            if is_ratio:
                assert isinstance(f, str), msg + ':' + 'ratio file type is wrong'
                assert os.path.exists(f) == 1, msg + ':' + f + " doesn't exist"
                r_ccf = pd.read_csv(f, sep='\s+')
            else:
                r_ccf = f[hdu_idx_name]

            assert r_ccf is not None, msg
            return r_ccf

        def valid_total_segment(ccf_v):
            return (np.shape(ccf_v)[0] - self.ccf_start_index) \
                if self.ccf_start_index + self.total_segment > np.shape(ccf_v)[0] else self.total_segment

        if self.logger:
            self.logger.info("RadialVelocityReweightingRef: find reweighting reference by method " +
                                self.reweighting_method)

        if self.reweighting_method == 'ccf_steps':
            ccf_ref = get_template_observation(self.files[0], self.ccf_hdu_name,
                                               'template observation error')

        elif self.is_ratio_data:                    # get ratio from a csv file directly or 2d array
            ccf_ref = get_template_observation(self.ccf_ratio_file, self.ccf_hdu_name,
                                               "ratio table from template file error", is_ratio=True)
        else:
            m_file = []
            m_ccf_ref = []

            # assume the first row of ccf data from each file is for order of self.ccf_start_idx
            # and in total min(self.total_order, total rows of ccf data) are selected for making the ratio table

            for ccf_file in self.files:
                ccf_ref = get_template_observation(ccf_file, self.ccf_hdu_name, "observation with ccf error")
                if ccf_ref.size == 0:
                    continue

                header = ccf_file.header[self.ccf_hdu_name]
                total_orderlet = np.shape(ccf_ref)[0] if header['NAXIS'] == 3 else 1

                for o in range(total_orderlet):
                    if header['NAXIS'] == 3:
                        m_ccf_ref.append(ccf_ref[o, :, :])
                    else:
                        m_ccf_ref.append(ccf_ref)
                    one_ccf = m_ccf_ref[-1]
                    t_segment = valid_total_segment(one_ccf)
                    # t_segment = min(np.shape(one_ccf)[0], self.total_segment)
                    seg_range = range(self.ccf_start_index, self.ccf_start_index + t_segment)
                    # pick the max among all segments for each file
                    if self.reweighting_method == 'ccf_max':
                        m_file.append(np.max([np.nanpercentile(one_ccf[od, :], 95) for od in seg_range]))
                    elif self.reweighting_method == 'ccf_mean':
                        m_file.append(np.max([np.nanmean(one_ccf[od, :]) for od in seg_range]))

            # find the maximum among all sci orderlets and get the ccf data of the file with the maximum ccf
            tmp_idx = np.where(m_file == np.nanmax(m_file))[0]
            ccf_ref = m_ccf_ref[tmp_idx[0]]
            t_segment = valid_total_segment(ccf_ref)
            # t_segment = min(np.shape(ccf_ref)[0], self.total_segment)      # total segment for processing

            ccf_df = RadialVelocityAlg.make_reweighting_ratio_table(ccf_ref, self.ccf_start_index,
                                                                    self.ccf_start_index + t_segment - 1,
                                                                    self.reweighting_method, max_ratio = 1.0,
                                                                    output_csv=self.ccf_ratio_file)
            ccf_ref = ccf_df.values

        assert ccf_ref is not None, 'no reference table is made'

        if self.logger:
            self.logger.info("RadialVelocityReweightingRef: done")

        return Arguments(ccf_ref)



