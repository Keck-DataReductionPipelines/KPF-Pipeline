import configparser
import numpy as np
from astropy.coordinates import Angle
import os
import os.path
from dotenv import load_dotenv
from modules.radial_velocity.src.alg_rv_base import RadialVelocityBase
from modules.radial_velocity.src.alg_rv_mask_line import RadialVelocityMaskLine
from modules.barycentric_correction.src.alg_barycentric_corr import BarycentricCorrectionAlg
from modules.Utils.config_parser import ConfigHandler
from astropy.time import Time

# Pipeline dependencies
# from kpfpipe.logger import start_logger
# from kpfpipe.primitives.level0 import KPF0_Primitive
# from kpfpipe.models.level0 import KPF0

mask_file_map = {'G2_espresso': ('G2.espresso.mas', 'air'),
                 'G2_harps': ('G2.harps.mas', 'air'),
                 'G2_neid_v1': ('G2.neid.v1.mas', 'air'),
                 'G2_neid_v2': ('G2.neid.v2.mas', 'air'),
                 'thar': ('Thorium_mask_031921.mas', 'vac'),
                 'lfc': ('kpf_lfc_mask_1025.mas', 'vac')}


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

                `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, `EPOCH`, `PARALLAX`, `STAR_RV`,
                `OBSLON`, `OBSLAT`, `OBSALT`, `STEP`, `MASK_WID`, `STEP_RANGE`.

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
        bc_jd (float): Observation time in Julian Date format. Defaults to None.
        bc_corr_path (str, optional): Path of csv file storing a list of redshift data from Barycentric correction
            over a period of time. Defaults to None.
        bc_corr_output (str, optional): Path of csv file to contain the redshift results from Barycentric correction
            over a period of time. Defaults to None for no output.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        Exception: If there is no configuration.
        Exception: If test data directory is not found.

    """
    
    # defined in configuration file
    STARNAME = 'starname'
    SPEC = 'instrument'
    STAR_CONFIG_FILE = 'star_config_file'
    STAR_RV = 'star_rv'        # km/s    # star rv (could be an estimation)
    OBSLON = 'obslon'           # degree
    OBSLAT = 'obslat'           # degree
    OBSALT = 'obsalt'           # meters

    STEP = 'step'               # km/s
    STEP_RANGE = 'step_range'   # in format of list
    MASK_WID = 'mask_width'     # km/s
    REWEIGHTING_CCF = 'reweighting_ccf_method'         # ratio, ccf, or None
    CCF_CODE = 'ccf_engine'     # ccf code language
    START_VEL = 'start_vel'     # start velocity
    STELLAR_DIR = 'stellarmask_dir' # stellarmask directory

    # defined in configuration file or star config for NEID
    RA = 'ra'                   # hours, like "01:44:04.0915236842"
    DEC = 'dec'                 # degree, like "-15:56:14.934780748"
    PMRA = 'pmra'               # mas/yr
    PMDEC = 'pmdec'             # mas/yr
    PARALLAX = 'parallax'       # mas
    EPOCH = 'epoch'
    DEF_MASK = 'mask'

    # defined for attribute access
    RV_CONFIG = 'rv_config'
    VELOCITY_LOOP = 'velocity_loop'
    VELOCITY_STEPS = 'velocity_steps'
    MASK_LINE = 'mask_line'
    ZB_RANGE = 'zb_range'
    MASK_TYPE = 'mask_type'

    def __init__(self, config=None, logger=None, l1_data=None, bc_time=None,  bc_period=380, bc_corr_path=None, bc_corr_output=None,
                test_data=None):
        RadialVelocityBase.__init__(self, config, logger)
        if self.config_ins is None or self.config_ins.get_section() is None:
            raise Exception("No config is set")

        load_dotenv()
        self.test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/' if test_data is None else test_data
        if not os.path.isdir(self.test_data_dir):
            raise Exception('no test data directory found')

        # instrument, starname, ra, dec, pm_ra, pm_dec, parallax, obslon, obslan, obsalt, star_rv, step,
        # step_range, mask_width
        # star_config_file, default_mask
        self.rv_config = dict()
        self.mask_path = None       # from init_star_config()
        self.mask_type = None
        self.velocity_loop = None   # loop of velocities for rv finding, from get_velocity_loop(), get_step_range()
        self.velocity_steps = None  # total steps in velocity_loop, from get_velocity_steps(), get_velocity_loop()
        self.zb_range = None        # redshift min and max, from get_redshift_range()
        self.mask_line = None       # def_mask, from get_mask_line(), get_redshift_range()
        self.reweighting_ccf_method = None  # reweighting ccf orders method
        self.bc_jd = bc_time       # starting time for Barycentric correction calculation in Julian Data format.
        self.bc_corr_path = bc_corr_path
        self.bc_corr_output = bc_corr_output
        self.bc_period = bc_period
        self.ccf_engine = None
        self.pheader = l1_data.header['PRIMARY'] if l1_data is not None else None
        self.star_config_file = None

    @staticmethod
    def ret_status(msg='ok'):
        ret = dict()
        ret['status'] = (msg == 'ok')
        ret['msg'] = msg if msg != 'ok' else ''

        return ret

    def init_star_from_header(self):
        if self.pheader is None:
            return self.ret_status('fits header is None')

        not_defined = " not defined in config"
        not_key_defined = " not defined in header"

        star_name_key = self.get_value_from_config(self.STARNAME, default=None)
        if star_name_key is None:
            return self.ret_status(star_name_key + not_defined)
        elif not star_name_key in self.pheader:
            return self.ret_status(star_name_key + not_key_defined)
        else:
            self.rv_config[self.STARNAME] = self.pheader[star_name_key]

        self.d_print("RadialVelocityAlgInit: get star info from header")
        self.rv_config[self.SPEC] = self.instrument or 'neid'
        star_info = (self.RA, self.DEC, self.PMRA, self.PMDEC, self.EPOCH,  self.PARALLAX)

        for s_key in star_info:
            h_key = self.get_value_from_config(s_key, None)
            if h_key is None:
                return self.ret_status(s_key + not_defined)
            elif h_key not in self.pheader:
                return self.ret_status(h_key + not_key_defined)
            else:
                h_val = self.pheader[h_key]
                if s_key == self.RA:
                    val = Angle(h_val + "hours").deg
                elif s_key == self.DEC:
                    val = Angle(h_val + "degrees").deg
                elif s_key == self.EPOCH:
                    year_days = 365.25
                    val = (float(h_val) - 2000.0) * year_days + Time("2000-01-01T12:00:00").jd  # to julian date
                else:
                    val = float(h_val)
                self.rv_config[s_key] = val

        #s_key = 'mask'
        #default_mask = self.get_rv_config_value(s_key, None)
        skyobj = self.pheader['SKY-OBJ']
        sciobj = self.pheader['SCI-OBJ']
        calobj = self.pheader['CAL-OBJ']
        if (skyobj==sciobj) and (sciobj==calobj) and (calobj=='Th_gold'):
            default_mask = 'thar'
        elif (skyobj==sciobj) and (sciobj==calobj) and (calobj=='LFCFiber'):
            default_mask = 'lfc'
        else:
            default_mask = 'G2_espresso'

        if default_mask is None:
            return self.ret_status(s_key + not_defined)
        quote = ['"', '\'']
        for q in quote:
            if default_mask.startswith(q) and default_mask.endswith(q):
                default_mask = default_mask.strip(q)
                break
        if default_mask not in mask_file_map:
            return self.ret_status('default mask of ' + default_mask + ' is not defined')

        stellar_dir = self.get_value_from_config(self.STELLAR_DIR, default=None)
        if stellar_dir is None:
            return self.ret_status(self.STELLAR_DIR + not_defined)

        self.mask_path = self.test_data_dir + stellar_dir + mask_file_map[default_mask][0]
        self.mask_type = default_mask
        self.mask_wavelengths = mask_file_map[default_mask][1]

        self.d_print("RadialVelocityAlgInit: mask config file: ", self.mask_path)

        return self.ret_status('ok')


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
            `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, `EPOCH`, and `PARALLAX`, are updated.

        """

        self.star_config_file = self.get_value_from_config(self.STAR_CONFIG_FILE, default=None)

        if self.star_config_file is not None and self.star_config_file.lower() == 'fits_header':
            return self.init_star_from_header()

        not_defined = ' not defined in config'
        star_name = self.get_value_from_config(self.STARNAME, default=None)
        if star_name is None:
            return self.ret_status(self.STARNAME + not_defined)

        self.rv_config[self.STARNAME] = star_name                       # in rv_config
        self.rv_config[self.SPEC] = self.instrument or 'neid'           # in rv_config

        config_star = None
        if self.star_config_file is not None:
            f_config = configparser.ConfigParser()
            self.d_print("RadialVelocityAlgInit: star config file: ", self.test_data_dir + self.star_config_file)
            if len(f_config.read(self.test_data_dir + self.star_config_file)) == 1:
                config_star = ConfigHandler(f_config, star_name)

        star_info = (self.RA, self.DEC, self.PMRA, self.PMDEC, self.EPOCH,  self.PARALLAX)  # in rv_config

        stellar_dir = self.get_value_from_config(self.STELLAR_DIR, default='rv_test/stellarmasks/')
        for star_key in star_info:
            k_val = self.get_rv_config_value(star_key, config_star)
            if k_val is None:
                return self.ret_status(star_key + not_defined)
            else:
                if star_key == self.RA:
                    val = Angle(k_val+"hours").deg
                elif star_key == self.DEC:
                    val = Angle(k_val+"degrees").deg
                elif star_key == self.EPOCH:
                    year_days = 365.25
                    val = (float(k_val) - 2000.0) * year_days + Time("2000-01-01T12:00:00").jd  # to julian date
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

            self.mask_path = self.test_data_dir + stellar_dir + mask_file_map[default_mask]
            self.d_print("RadialVelocityAlgInit: mask config file: ", self.mask_path)
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

                * `rv_config`: values of `SPEC`, `STARNAME`, `RA`, `DEC`, `PMRA`, `PMDEC`, `PARALLAX`, `EPOCH`,
                  `STAR_RV`, `OBSLON`, `OBSLAT`, `OBSALT`, `STEP`, `MASK_WID`, `STEP_RANGE`.
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

        # in rv_config
        if self.star_config_file.lower() == 'fits_header':
            rv_keys = (self.STAR_RV, self.OBSLON, self.OBSLAT, self.OBSALT,  self.STEP, self.MASK_WID, self.START_VEL)
        else:
            rv_keys = (self.STAR_RV, self.OBSLON, self.OBSLAT, self.OBSALT, self.STEP, self.MASK_WID, self.START_VEL)

        for rv_k in rv_keys:
            val = self.get_rv_config_value(rv_k)
            if val is None:
                if rv_k == self.START_VEL:   # optional
                    self.rv_config[self.START_VEL] = val
                else:
                    return self.ret_status(rv_k + ' not defined in config')
            else:
                self.rv_config[rv_k] = float(val)

        self.get_reweighting_ccf_method()
        self.get_step_range()
        self.get_velocity_loop()   # based on step_range and step, star_rv in rv_config
        self.get_velocity_steps()  # based on velocity_loop
        self.get_redshift_range()  # get redshift from barycentric velocity correction
        self.get_mask_line()       # based on mask_path, velocity loop and mask_width/vacuum_to_air
        self.get_ccf_version()     # get ccf engine in either 'python' or 'c'
        return self.ret_status()

    def get_rv_config_value(self, prop, star_config=None, default=None):
        """ Get value of specific parameter from the config file or star config file.

        Check the value from the configuration file first, then from the star configuration file if it is available.
        The default is set if it is not defined in any configuration file.

        Args:
            prop (str): Name of the parameter to be searched.
            star_config (ConfigHandler): Section of designated star in star configuration file.
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
            config = self.config_ins

        return config.get_config_value(prop, default)

    def get_ccf_version(self, default_code='python'):
        """ Get the ccf code language

        Args:
            default_code (str): Default ccf code language, 'python' or 'c'

        Returns:
            str: ccf code language

        """

        if self.ccf_engine is None:
            self.ccf_engine = self.get_value_from_config(self.CCF_CODE, default=default_code)
        return self.ccf_engine

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
            self.rv_config[self.STEP_RANGE] = self.get_rv_config_value(self.STEP_RANGE, default=default)
        return self.rv_config[self.STEP_RANGE]

    def get_velocity_loop(self):
        """ Get array of velocities based on step range, step interval, and estimated star radial velocity.

        Returns:
            numpy.ndarray: Array of evenly spaced velocities. Attribute `velocity_loop` is updated.

        """
        if self.velocity_loop is None:
            v_range = self.get_step_range()
            if self.rv_config[self.START_VEL] is not None:
                self.velocity_loop = np.arange(0, v_range[1]-v_range[0]) * self.rv_config[self.STEP] + \
                                     self.rv_config[self.START_VEL]
            else:
                self.velocity_loop = np.arange(v_range[0], v_range[1]) * self.rv_config[self.STEP] + \
                                     self.rv_config[self.STAR_RV]
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

    def get_redshift_range(self, bc_path=None, jd_time=None, period=None, bc_output=None):
        """ Get redshift range by using Barycentric velocity correction over a period of time.

        Args:
            bc_path (str): The path of barycentric correction data over a period. Defaults to None.
            jd_time (float, optional): Starting time for the period in Julian Date format. Defaults to None.
                For example,  2458591.5 is for Apr-18-2019.
            period (int, optional): Period of days. Defaults to 380 (days).
            bc_output (str): The path of the output for Barycentric correction results.

        Returns:
            numpy.ndarray: Minimum and maximum redshift over a period of time. The first number
            in the array is the minimum and the second one is the maximum. Attributes `zb_range` is updated.

        """
        rv_config_bc_key = [self.RA, self.DEC, self.PMRA, self.PMDEC, self.PARALLAX, self.EPOCH, self.OBSLAT,
                            self.OBSLON, self.OBSALT, self.STAR_RV, self.SPEC, self.STARNAME]

        if self.zb_range is None:
            rv_config_bc = {k: self.rv_config[k] for k in rv_config_bc_key}
            rv_bc_corr = BarycentricCorrectionAlg(rv_config_bc, logger=self.logger, logger_name=RadialVelocityBase.name)
            bc_path = bc_path or self.bc_corr_path
            bc_output = bc_output or self.bc_corr_output
            jd_time = jd_time or self.bc_jd
            period = period or self.bc_period
            self.zb_range = rv_bc_corr.get_zb_long(jd_time, period, data_path=bc_path, save_to_path=bc_output)

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
            rv_mask_line = RadialVelocityMaskLine()
            if self.mask_wavelengths == 'air':
                air2vac = True
            elif self.mask_wavelengths == 'vac':
                air2vac = False
            self.mask_line = rv_mask_line.get_mask_line(self.mask_path, self.get_velocity_loop(),
                                                        zb_range, self.rv_config[self.MASK_WID],
                                                        air2vac)

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
        # star_rv in rv_config, mask_width, step, step_range
        collection = [self.RV_CONFIG, self.MASK_LINE, self.VELOCITY_STEPS,
                      self.VELOCITY_LOOP, self.REWEIGHTING_CCF,
                      self.ZB_RANGE, self.CCF_CODE, self.MASK_TYPE]

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

        self.d_print("RadialVelocityAlgInit: starting ... ")
        init_status = self.init_calculation()

        if init_status['status']:
            init_status['data'] = self.collect_init_data()
            self.d_print('RadialVelocityAlgInit: result data is ', init_status['data'])

        return init_status
