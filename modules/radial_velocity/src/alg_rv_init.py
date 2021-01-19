import configparser
import numpy as np
from astropy.coordinates import Angle
import os
import os.path
import json
from dotenv import load_dotenv
from modules.radial_velocity.src.alg_rv_base import RadialVelocityBase
from modules.radial_velocity.src.alg_rv_mask_line import RadialVelocityMaskLine
from modules.radial_velocity.src.alg_barycentric_vel_corr import RVBaryCentricVelCorrection
from modules.Utils.config_parser import ConfigHandler

# Pipeline dependencies
# from kpfpipe.logger import start_logger
# from kpfpipe.primitives.level0 import KPF0_Primitive
# from kpfpipe.models.level0 import KPF0

mask_file_map = {'G2_espresso': 'G2.espresso.mas',
                 'G2_harps': 'G2.harps.mas',
                 'G2_neid_v1': 'G2.neid.v1.mas',
                 'G2_neid_v2': 'G2.neid.v2.mas'}


class RadialVelocityAlgInit(RadialVelocityBase):
    """ Radial velocity Init.

    This module defines class 'RadialVelocityAlgInit' and methods to do init work for making mask lines and velocity
    steps based on star and module associated configuration for further radial velocity computation.

    Args:
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        test_data_dir (str): KPF test data directory defined in .env to be loaded by the use of `load_dotenv`.
        rv_config (dict): A dict instance containing the values defined in radial velocity configuration file or star
            configuration file if there is. The instance includes the following keys (these are constants defined
            in the source):
                `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, `PARALLAX`, `T`,
                `OBSLON`, `OBSLAT`, `OBSALT`, `STEP`, `MASK_WID`, `AIR_TO_VACUUM`, `STEP_RANGE`.
        mask_path (str): Mask file path.
        velocity_loop (numpy.ndarray): Evenly spaced velocity steps.
        velocity_steps (int): Total steps in `velocity_loop`.
        zb_range (list): Redshift at a single time or redshift range over a period of time. The list contains
            one number for a single time or two numbers representing the minimum and the maximum during a period
            of time.
        mask_line (dict): A dict instance containing mask line information. Please refer to `Returns` section in
            :func:`~alg_rv_init.RadialVelocityAlgInit.get_mask_line()` for the information detail.
        reweighting_ccf_method (str): Method of reweighting ccf orders. The possible methods include
            `ccf_max` or `ccf_mean` which scales the ccf of each order based on the ratio of the maximum (or 95
            percentile) or the mean ccf among selected orders from  the template observation and `ccf_steps`
            which scales the ccf of each order based on the mean of the ratio between current ccf over all velocity
            steps and that of the same order from the template observation.


    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        Exception: If there is no configuration.
        Exception: If test data directory is not found.

    """

    # defined in configuration file
    STARNAME = 'starname'
    SPEC = 'instrument'
    STAR_CONFIG_FILE = 'star_config_file'
    START_RV = 'start_rv'         # km/s
    OBSLON = 'obslon'           # degree
    OBSLAT = 'obslat'           # degree
    OBSALT = 'obsalt'           # meters
    STEP = 'step'               # km/s
    STEP_RANGE = 'step_range'   # in format of list
    MASK_WID = 'mask_width'     # km/s
    AIR_TO_VACUUM = 'air_to_vacuum'    # True or False
    REWEIGHTING_CCF = 'reweighting_ccf_method'         # ratio, ccf, or None

    # defined in configuration file or star config for NEID
    RA = 'ra'                   # hours, like "01:44:04.0915236842"
    DEC = 'dec'                 # degree, like "-15:56:14.934780748"
    PMRA = 'pmra'               # mas/yr
    PMDEC = 'pmdec'             # mas/yr
    PARALLAX = 'parallax'       # mas
    DEF_MASK = 'mask'

    # defined for attribute access
    RV_CONFIG = 'rv_config'
    VELOCITY_LOOP = 'velocity_loop'
    VELOCITY_STEPS = 'velocity_steps'
    MASK_LINE = 'mask_line'

    def __init__(self, config=None, logger=None):
        RadialVelocityBase.__init__(self, config, logger)
        if self.config_param is None or self.config_param.get_section() is None:
            raise Exception("No config is set")

        load_dotenv()
        self.test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
        if not os.path.isdir(self.test_data_dir):
            raise Exception('no test data directory found')

        # ra, dec, pm_ra, pm_dec, parallax, def_mask, obslon, obslan, obsalt, start_rv, step,
        # air_to_vacuum, step_range, mask_width
        self.rv_config = dict()
        self.mask_path = None
        self.velocity_loop = None   # loop of velocities for rv finding
        self.velocity_steps = None  # total steps in velocity_loop
        self.zb_range = None
        self.mask_line = None       # def_mask,
        self.reweighting_ccf_method = None  # reweighting ccf orders method

    @staticmethod
    def ret_status(msg='ok'):
        ret = dict()
        ret['status'] = (msg == 'ok')
        ret['msg'] = msg if msg != 'ok' else ''

        return ret

    def init_star_config(self):
        """ Data initialization from star related configuration, including ra, dec, pmra, pmdec, parallax and
        the mask file.

        Returns:
            dict: status of getting data from star related configuration, like::

                {
                    'status': True|False,
                    'msg': <error message> # if status is False
                }

            Attribute `mask_path` is updated and the values of the following keys in `rv_config`,
            `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, and `PARALLAX`, are updated.

        """

        not_defined = ' not defined in config'
        star_name = self.get_value_from_config(self.STARNAME, default=None)
        if star_name is None:
            return self.ret_status(self.STARNAME + not_defined)

        self.rv_config[self.STARNAME] = star_name
        self.rv_config[self.SPEC] = self.instrument or 'neid'
        star_config_file = self.get_value_from_config(self.STAR_CONFIG_FILE, default=None)

        config_star = None
        if star_config_file is not None:
            f_config = configparser.ConfigParser()
            if len(f_config.read(self.test_data_dir + star_config_file)) == 1:
                config_star = ConfigHandler(f_config, star_name)

        star_info = (self.RA, self.DEC, self.PMRA, self.PMDEC, self.PARALLAX)

        for star_key in star_info:
            k_val = self.get_rv_config_value(star_key, config_star)
            if k_val is None:
                return self.ret_status(star_key + not_defined)
            else:
                if star_key == self.RA:
                    val = Angle(k_val+"hours").deg
                elif star_key == self.DEC:
                    val = Angle(k_val+"degrees").deg
                else:
                    val = float(k_val)
                self.rv_config[star_key] = val

        star_key = self.DEF_MASK
        default_mask = self.get_rv_config_value(star_key, config_star)
        if default_mask is None:
            return self.ret_status(star_key + not_defined)
        else:
            quote = ['"', '\'']
            for q in quote:
                if default_mask.startswith(q) and default_mask.endswith(q):
                    default_mask = default_mask.strip(q)
                    break
            if default_mask not in mask_file_map:
                return self.ret_status('default mask of '+default_mask + ' is not defined')

            self.mask_path = self.test_data_dir + 'rv_test/stellarmasks/'+mask_file_map[default_mask]
        return self.ret_status('ok')

    def init_calculation(self):
        """  Initial data setup for radial velocity computation based on the setting in `PARAM` section of the
        configuration file.

        Returns:
            dict: status of data initialization, like::

                {
                    'status': True|False,
                    'msg': <error message> # if status is False
                }

            The following attributes and values are updated,

                * `rv_config`: values of `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, `PARALLAX`,
                  `START_RV`, `OBSLON`, `OBSLAT`, `OBSALT`, `STEP`, `MASK_WID`,  `AIR_TO_VACUUM`, `STEP_RANGE`.
                * `velocity_steps`
                * `velocity_loop`
                * `zb_range`
                * `mask_line`
                * `mask_path`
                * `reweighting_ccf_method`

        """
        ret = self.init_star_config()
        if not ret['status']:
            return self.ret_status(ret['msg'])

        rv_keys = (self.START_RV, self.OBSLON, self.OBSLAT, self.OBSALT, self.STEP, self.MASK_WID)
        for rv_k in rv_keys:
            val = self.get_rv_config_value(rv_k)
            if val is None:
                return self.ret_status(rv_k + ' not defined in config')
            else:
                self.rv_config[rv_k] = float(val)

        self.rv_config[self.AIR_TO_VACUUM] = self.get_rv_config_value(self.AIR_TO_VACUUM, default=False)
        self.get_reweighting_ccf_method()
        self.get_step_range()
        self.get_velocity_loop()   # based on step_range and step, start_rv in rv_config
        self.get_velocity_steps()  # based on velocity_loop
        self.get_redshift_range(self.test_data_dir)         # get redshift from barycentric velocity correction
        self.get_mask_line()       # based on mask_path, velocity loop and mask_width/vacuum_to_air
        return self.ret_status()

    def get_rv_config_value(self, prop, star_config=None, default=None):
        """ Get value of specific parameter from the config file or star config file.

        Check the value from the configuration file first, then from the star configuration file if it is available.
        The default is set if it is not defined in any configuration file.

        Args:
            prop (str): Name of the parameter to be searched.
            star_config (configparser.SectionProxy): Section of designated star in star configuration file.
            default (Union[int, float, str, bool], optional): Default value for the searched parameter.
                Defaults to None.

        Returns:
            Union[int, float, str, bool]: Value for the searched parameter.

        """

        val = self.get_value_from_config(prop, default=None)

        # not exist in module config, check star config if there is or return default
        if val is None:
            if star_config is not None:
                return self.get_value_from_config(prop, config=star_config, default=default)
            else:
                return default

        if type(val) != str:
            return val

        # check if getting the value further from star config
        tag = 'star/'     # to find value from star config file
        if val.startswith(tag):
            if star_config is not None:
                attr = val[len(tag):]
                return self.get_value_from_config(attr, config=star_config, default=default)
            else:
                return default

        return val

    def get_value_from_config(self, prop, default='', config=None):
        """ Get value of specific parameter from the configuration file.

        Get defined value from the specific section of a configuration file. The configuration file is either defined
        with the module or some other configuration associated with the observation.

        Args:
            prop (str): Name of the parameter to be searched.
            default (Union[int, float, str, bool], optional): Default value for the searched parameter.
            config (ConfigHandler): External config, such as star config for NEID.
        Returns:
            Union[int, float, str, bool]: Value for the searched parameter.

        """
        if config is None:
            config = self.config_param

        return config.get_config_value(prop, default)


    def get_reweighting_ccf_method(self, default_method='ccf_max'):
        """ Get the ccf reweighting method.

        Args:
            default_method (str): Default ccf reweighting method.

        Returns:
            str: ccf reweighting method.

        """

        if self.reweighting_ccf_method is None:
            self.reweighting_ccf_method = self.get_value_from_config(self.REWEIGHTING_CCF, default=default_method)
        return self.reweighting_ccf_method

    def get_step_range(self, default='[-80, 81]'):
        """ Get the step range for the velocity.

        Args:
            default (list): Default step range in string format. Defaults to '[-80, 81]'.

        Returns:
            list: Step range. `step_range` in attribute `rv_config` is updated.

        """
        if self.STEP_RANGE not in self.rv_config:
            self.rv_config[self.STEP_RANGE] = json.loads(self.get_rv_config_value(self.STEP_RANGE, default=default))
        return self.rv_config[self.STEP_RANGE]

    def get_velocity_loop(self):
        """ Get array of velocities based on step range, step interval, and estimated star radial velocity.

        Returns:
            numpy.ndarray: Array of evenly spaced velocities. Attribute `velocity_loop` is updated.

        """
        if self.velocity_loop is None:
            v_range = self.get_step_range()
            self.velocity_loop = np.arange(v_range[0], v_range[1]) * self.rv_config[self.STEP] + \
                self.rv_config[self.START_RV]
        return self.velocity_loop

    def get_velocity_steps(self):
        """ Total velocity steps.

        Returns:
            int: Total velocity steps based on attribute `velocity_steps` of the class. Attribute `velocity_steps` is
            updated.

        """
        if self.velocity_steps is None:
            vel_loop = self.get_velocity_loop()
            self.velocity_steps = len(vel_loop)
        return self.velocity_steps

    def get_redshift_range(self, data_dir=None, jd_time=2458591.5, period=380):
        """ Get redshift range by using Barycentric velocity correction over a period of time.

        Args:
            data_dir (str): Test data directory. Defaults to None.
            jd_time (float, optional): Starting time for the period in Julian Date format.
                Defaults to 2458591.5 (Apr-18-2019).
            period (int, optional): Period of days. Defaults to 380 (days).

        Returns:
            numpy.ndarray: Minimum and maximum redshift over a period of time. The first number
            in the array is the minimum and the second one is the maximum. Attributes `zb_range` is updated.

        """
        rv_config_bc_key = [self.RA, self.DEC, self.PMRA, self.PMDEC, self.PARALLAX, self.OBSLAT,
                            self.OBSLON, self.OBSALT, self.START_RV]

        if self.zb_range is None:
            rv_config_bc = dict()
            for k in rv_config_bc_key:
                rv_config_bc[k] = self.rv_config[k]
            rv_bc_corr = RVBaryCentricVelCorrection()
            self.zb_range = rv_bc_corr.get_zb_long(rv_config_bc, jd_time, period,
                                                   instrument=self.rv_config[self.SPEC], data_dir=data_dir)
        return self.zb_range

    def get_mask_line(self):
        """ Get mask coverage per mask width and the mask centers read from the mask file.

        Returns:
            dict: Mask information, like::

                {
                    'start' : numpy.ndarray            # start points of masks
                    'end' : numpy.ndarray              # end points of masks
                    'center': numpy.ndarray            # center of masks
                    'weight': numpy.ndarray            # weight of masks
                    'bc_corr_start': numpy.ndarray     # adjusted start points of masks
                    'bc_corr_end': numpy.ndarray       # adjusted end points of masks
                }

             Attribute `mask_line` is updated.

        """

        if self.mask_line is None:
            zb_range = self.get_redshift_range()
            rv_maskline = RadialVelocityMaskLine()
            self.mask_line = rv_maskline.get_mask_line(self.mask_path, self.get_velocity_loop(),
                                                       zb_range, self.rv_config[self.MASK_WID],
                                                       self.rv_config[self.AIR_TO_VACUUM])

        return self.mask_line

    def collect_init_data(self):
        """ Collect init data for radial velocity analysis.

        Returns:
            dict: Collected init data.

        Note:
            This data set is passed to the instance of `RadialVelocityAlg` and used for radial velocity computation.

        """
        init_data = dict()
        # star, spectrograph, ra, dec, pm_ra, pm_dec, parallax, obslat, obslon, obsalt,
        # start_rv in rv_config, mask_width, step, step_range
        collection = [self.RV_CONFIG, self.MASK_LINE, self.VELOCITY_STEPS, self.VELOCITY_LOOP, self.REWEIGHTING_CCF]

        attrs = self.__dict__.keys()
        for c in collection:
            if c in attrs:
                init_data[c] = getattr(self, c)

        return init_data

    def start(self, print_debug=None):
        """ Start the data initialization for radial velocity computation.

        Args:
            print_debug (str, optional):  Print debug information to stdout if it is provided as empty string
                or to a file path,  `print_debug`, if it is non empty string, or no print is made  if it is None.
                Defaults to None.

        Returns:
            dict: Result of init data, like::

                {
                    'status': True|False
                    'msg': <error message if status is False>
                    'data': <init data>     # Please see Returns of function collect_init_data
                }

        """

        self.add_file_logger(print_debug)

        self.d_print("init ... ")
        if self.logger:
            self.logger.info('starting init...')
        init_status = self.init_calculation()

        if init_status['status']:
            init_status['data'] = self.collect_init_data()
            self.d_print('init data is: ', init_status['data'])

        if self.logger:
            self.logger.info('collecting init done')

        return init_status
