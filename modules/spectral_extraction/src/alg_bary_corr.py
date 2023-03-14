import numpy as np
import pandas as pd

from astropy.time import Time
from modules.radial_velocity.src.midpoint_photon_arrival import MidpointPhotonArrival
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
from modules.Utils.alg_base import ModuleAlgBase
from astropy.io import fits
from astropy.coordinates import Angle
from modules.barycentric_correction.src.alg_barycentric_corr import BarycentricCorrectionAlg
from modules.Utils.config_parser import ConfigHandler


class BaryCorrTableAlg(ModuleAlgBase):
    """
    This module defines class 'BaryCorrAlg' and methods to build barycentric correction related information in L1 data

    Args:
        df_em (pd.DataFrame): EXPMETER_SCI table
        df_bc (pd.DataFrame): BARY_CORR table
        pheader (fits.header.Header): fits primary header of level 0 data.
        wls_data (np.ndarray): wavelength solution data
        total_order (int): total order of all ccds
        ccd_order (int): orders of the ccd to be processed.
        start_bary_index (int): start index in bary corr table
        config (configparser.ConfigParser): config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        df_EM (pd.DataFrame): EXPMETER_SCI data
        lev_header(fits.header.Header): primary fits header.
        instrument (str): instrument name
        wls_data (np.ndarray): wavelength solution data
        total_table_order (int): total rows of bary corr table.
        start_bary_index (int): start row index for bary corr table.
        end_bary_index (int): end row index for bary corr table.
        bary_corr_table (dict): dict instance contains data on
                        geometric midpoint exposure time (UTC),
                        geometric midpoint exposure time (BJD),
                        weighted-photon midpoint time (BJD),
                        barycentric velocity (m/sec)

    Raises:
        TypeError: If there is type error for `wls_data`

    """
    
    DATE_MID = 'DATE-MID'
    DATE_BEG = 'DATE-BEG'
    EXPTIME = 'EXPTIME'
    IMTYPE = 'IMTYPE'
    TARGFRAM = 'TARGFRAM'
    year_days = 365.25
    name = 'BaryCorrTable'
    BC_col1 = 'GEOMID_UTC'
    BC_col2 = 'GEOMID_BJD'
    BC_col3 = 'PHOTON_BJD'
    BC_col4 = 'BARYVEL'
    BARY_TABLE_KEYS = {BC_col1: "geometric midpoint UTC",
                       BC_col2: "geometric midpoint BJD",
                       BC_col3: "weighted-photon midpoint BJD",
                       BC_col4: "barycentric velocity(m/sec)"}

    def __init__(self, df_em, df_bc, pheader, wls_data, total_order, ccd_order,
                 start_bary_index=0, config=None, logger=None):
        if wls_data is not None and not isinstance(wls_data, np.ndarray):
            raise TypeError('wls data type error, cannot construct object from BaryCorrTableAlg')
        if pheader is None:
            raise TypeError('primary header error, cannot construct object from BaryCorrTableAlg')
        if total_order <= 0:
            raise TypeError('total order should be greater than 0')

        if ccd_order is not None and ccd_order > 0:
            total_ccd_orders = ccd_order
        elif wls_data is not None:
            total_ccd_orders = np.shape(wls_data)[0]
        else:
            raise TypeError('ccd orders error, cannot construct object from BaryCorrTableAlg')

        ModuleAlgBase.__init__(self, self.name, config, logger)
        ins = self.config_param.get_config_value('instrument', '') if self.config_param is not None else ''
        self.instrument = ins.lower() if ins else (pheader['INSTRUME'] if 'INSTRUME' in pheader else '')
        self.config_ins = ConfigHandler(config, ins, self.config_param)  # section of instrument or 'PARAM'
        self.df_EM = df_em
        self.lev_header = pheader
        self.wls_data = wls_data
        self.total_table_order = total_order
        self.start_bary_index = start_bary_index
        self.end_bary_index = start_bary_index + total_ccd_orders - 1
        if df_bc is not None and np.shape(df_bc.values)[0] == self.total_table_order:
            columns = df_bc.columns
            self.bary_corr_table = dict()
            for cl in [BaryCorrTableAlg.BC_col1, BaryCorrTableAlg.BC_col2,
                       BaryCorrTableAlg.BC_col3, BaryCorrTableAlg.BC_col4]:
                if cl == BaryCorrTableAlg.BC_col1:
                    self.bary_corr_table[cl] = np.array(df_bc[cl]) if cl in columns \
                        else np.empty(self.total_table_order, dtype=object)
                else:
                    self.bary_corr_table[cl] = np.array(df_bc[cl]) if cl in columns \
                        else np.zeros(self.total_table_order, dtype=float)
        elif self.instrument == 'kpf':
            self.bary_corr_table = {
                        BaryCorrTableAlg.BC_col1: np.empty(self.total_table_order, dtype=object),
                        BaryCorrTableAlg.BC_col2: np.zeros(self.total_table_order, dtype=float),
                        BaryCorrTableAlg.BC_col3: np.zeros(self.total_table_order, dtype=float),
                        BaryCorrTableAlg.BC_col4: np.zeros(self.total_table_order, dtype=float)}
        else:
            self.bary_corr_table = None

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

    def get_segment_wavelength(self):
        """ Get minimum and maximum wavelength of all orders

        Returns:
            tuple: two arrays containing minimum and maximum wavelength for each order
        """
        if self.wls_data is None:
            return None, None
        wls_h, wls_w = np.shape(self.wls_data)
        seg_min = list()
        seg_max = list()
        for r in range(wls_h):
            seg_min.append(min(self.wls_data[r, 0], self.wls_data[r, -1]))
            seg_max.append(max(self.wls_data[r, 0], self.wls_data[r, -1]))

        return np.array(seg_min), np.array(seg_max)

    def get_index_range(self):
        """ Get the starting and ending index in BARY_CORR table

        Returns:
            tuple: starting and ending index
        """
        return self.start_bary_index, self.end_bary_index

    def get_epoch(self, e_data):

        """Get and evaluate epoch value from the primary header

        Args:
            e_data(float): Epoch value.

        Returns:
            float: epoch value.
        """
        """
        targfram_key = 'TARGFRAM'
        if e_data == 0.0:
            try:
                targfram = self.lev_header[targfram_key]
                if targfram.lower() == 'fk4':
                    e_data = 1950.0
                elif targfram.lower() in ['fk5', 'apparent']:
                    e_data = 2000.0
            except KeyError:
                e_data = None

        if e_data is not None:
            year_days = 365.25
            val = (e_data - 2000.0) * year_days + Time("2000-01-01T12:00:00").jd  # to julian date
        else:
            val = None
        print("from check_epoch", val)
        """

        val = RadialVelocityAlgInit.check_epoch(float(e_data), self.lev_header)

        return val

    def get_rv_col(self, is_single_time=False):
        """ Barycentric correction velocity for each order

        Args:
            is_single_time (bool): the midpoint photon arrival time of each order is the same or not.

        Returns:
            numpy.ndarray: Barycentric correction velocity of all orders
        """
        bc_config = dict()
        header_keys = [RadialVelocityAlgInit.RA, RadialVelocityAlgInit.DEC,
                       RadialVelocityAlgInit.PMRA, RadialVelocityAlgInit.PMDEC,
                       RadialVelocityAlgInit.EPOCH, RadialVelocityAlgInit.PARALLAX,
                       RadialVelocityAlgInit.STARNAME, RadialVelocityAlgInit.STAR_RV]
        for h_key in header_keys:
            h_key_val = self.get_value_from_config(h_key, None)
            if h_key_val is None:
                return None
            h_key_data = self.lev_header[h_key_val] if h_key_val in self.lev_header else None
            if h_key_data is None:
                return None
            if h_key == RadialVelocityAlgInit.RA:
                h_key_data = Angle(h_key_data + "hours").deg
            elif h_key == RadialVelocityAlgInit.DEC:
                h_key_data = Angle(h_key_data + "degrees").deg
            elif h_key == RadialVelocityAlgInit.EPOCH:
                h_key_data = self.get_epoch(h_key_data)
            elif h_key != RadialVelocityAlgInit.STARNAME:
                h_key_data = float(h_key_data)
            bc_config[h_key] = h_key_data

        not_header_keys = [RadialVelocityAlgInit.OBSLAT, RadialVelocityAlgInit.OBSLON, RadialVelocityAlgInit.OBSALT]
        for n_key in not_header_keys:
            h_key_data = self.get_value_from_config(n_key, 0.0)
            bc_config[n_key] = h_key_data
        bc_config[RadialVelocityAlgInit.SPEC] = self.instrument

        for od in range(self.start_bary_index, self.end_bary_index+1):
            if is_single_time and od > self.start_bary_index:
                self.bary_corr_table[BaryCorrTableAlg.BC_col4][od] \
                    = self.bary_corr_table[BaryCorrTableAlg.BC_col4][self.start_bary_index]
            else:
                if RadialVelocityAlgInit.is_unknown_target(self.instrument, bc_config[RadialVelocityAlgInit.STARNAME],
                                                           bc_config[RadialVelocityAlgInit.EPOCH]):
                    # if bc_config[RadialVelocityAlgInit.EPOCH] is not None:
                    #    import pdb;pdb.set_trace()
                    self.bary_corr_table[BaryCorrTableAlg.BC_col4][od] = 0.0
                else:
                    bc_corr = BarycentricCorrectionAlg.get_zb_from_bc_corr(bc_config,
                                                                self.bary_corr_table[BaryCorrTableAlg.BC_col3][od])[0]
                    self.bary_corr_table[BaryCorrTableAlg.BC_col4][od] = bc_corr   # m/sec

        return self.bary_corr_table[BaryCorrTableAlg.BC_col4]

    def get_obs_utc(self, default=None):
        """ Get observation exposure time in UTC in string format

        Args:
            default (str): a default UTC time. Defaults to None.

        Returns:
            str: UTC in string format.

        """
        if self.instrument == 'kpf' and BaryCorrTableAlg.DATE_MID in self.lev_header:
            return self.lev_header[BaryCorrTableAlg.DATE_MID]
        return default

    def get_expmeter_science(self):
        """ Get the table from EXPMETER_SCI extension

        Returns:
            pandas.DataFrame: table from EXPMETER_SCI extension.
        """
        return self.df_EM

    def is_bc_calculated(self):
        """If Barycentric correction is to be calculated or skipped.

        If the value of keyword, `IMTYPE`, in the primary is `object`, the BC is calculated.

        Returns:
            bool: the BC is to be calculated if it is True.

        """
        if BaryCorrTableAlg.IMTYPE in self.lev_header:
            if self.lev_header[BaryCorrTableAlg.IMTYPE].lower() == 'object':
                return True
        else:
            return False

    def build_bary_corr_table(self):
        """Compute the BARY_CORR table

        Returns:
            Pandas.DataFrame: BARY_CORR table containing columns of

                * geometric midpoint in UTC string,
                * geometric midpoint in BJD,
                * weighted-photon midpoint BJD,
                * barycentric velocity(m/sec)

        """
        if self.bary_corr_table is None:
            return None

        # fill in "GEOMID_UTC" and "GEOMID_BJD" columns
        mid_utc = self.get_obs_utc()

        if mid_utc is None:
            return pd.DataFrame(self.bary_corr_table)
        else:
            self.bary_corr_table[BaryCorrTableAlg.BC_col1][self.start_bary_index:self.end_bary_index+1] = mid_utc
            self.bary_corr_table[BaryCorrTableAlg.BC_col2][self.start_bary_index:self.end_bary_index+1] = Time(mid_utc).jd

        df_em = self.get_expmeter_science()

        is_single_mid = False

        # key_sets = ['IMTYPE', 'TARGFRAM', 'TARGNAME', 'TARGEPOC', 'SCI-OBJ', 'SKY-OBJ', 'CAL-OBJ']
        # build BARY_CORR table based on EXPMETER_SCI table
        if df_em is not None and np.shape(df_em.values)[0] != 0:
            # get begin and finish time from primary header DATE_BEG + EXPTIME or from df_em table
            if BaryCorrTableAlg.DATE_BEG in self.lev_header and BaryCorrTableAlg.EXPTIME in self.lev_header:
                date_begin = np.datetime64(self.lev_header[BaryCorrTableAlg.DATE_BEG]).astype('<M8[ms]')
                delta_ms = np.timedelta64(int(self.lev_header[BaryCorrTableAlg.EXPTIME]*1000.0), 'ms')
                date_end = date_begin + delta_ms
            else:
                date_begs = np.array(df_em["Date-Beg"], dtype=np.datetime64)
                date_ends = np.array(df_em["Date-End"], dtype=np.datetime64)
                date_begin = date_begs[0]
                date_end = date_ends[-1]
            seg_min, seg_max = self.get_segment_wavelength()
            mid_photon_arr = MidpointPhotonArrival(df_em, date_begin, date_end, segmentMin=seg_min, segmentMax=seg_max)
            _, midphoton = mid_photon_arr.orderedMidPoint()
        else:  # using observation time for midpoint arrival time if no df_em exists
            midphoton = self.bary_corr_table[BaryCorrTableAlg.BC_col1][self.start_bary_index:self.end_bary_index+1]
            is_single_mid = True
        for i in range(midphoton.size):
            self.bary_corr_table[BaryCorrTableAlg.BC_col3][i+self.start_bary_index] = Time(midphoton[i]).jd

        if self.is_bc_calculated():
            self.get_rv_col(is_single_mid)

        bc_df_res = pd.DataFrame(self.bary_corr_table)
        bc_df_res.attrs['BCV_UNIT'] = 'm/sec'
        return bc_df_res
