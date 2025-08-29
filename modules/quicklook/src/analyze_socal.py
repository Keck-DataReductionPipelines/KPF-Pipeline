import os
import time
import datetime
import pvlib as pv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from modules.Utils.utils import DummyLogger
from astroplan import Observer
from astropy.time import Time
from astropy import units as u
from datetime import datetime, timedelta
import astropy.coordinates as coord
from modules.Utils.kpf_parse import HeaderParse

def dateformat(ax):
    locator   = mpl.dates.AutoDateLocator()
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

class AnalyzePyr:

    """
    Description:
        This class contains functions to analyze irradiance time series and 
        compute the clearsky index.

    Arguments:
        L0 - an L0 object
    """

    def __init__(self, datecode, basedir='/data/pyr', inst='kpf', logger=None, verbose=False):

        assert inst in ['kpf', 'neid'], "inst must be one of ['kpf', 'neid']" 
        self.inst = inst
        self.datecode = datecode
        self.basedir = basedir
        self.logger = logger if logger is not None else DummyLogger()
        self.verbose = verbose
        self.irr_fn = f'{self.basedir}/irradiance/{self.datecode[:4]}/pyr_irrad_{self.datecode}.csv'
        self.clearsky_dir = f'{self.basedir}/clearness/{self.datecode[:4]}'
        self.clearsky_fn = f'{self.clearsky_dir}/{self.datecode}_clearsky_metrics.csv'

        wmko = coord.EarthLocation.of_site('Keck Observatory')
        kpf = pv.location.Location(wmko.lat.to(u.deg).value, wmko.lon.to(u.deg).value,
                                   altitude=wmko.height.to(u.m).value, tz='UTC', name='KPF SoCal')
        
        kittpeak = coord.EarthLocation.of_site('Kitt Peak')
        neid = pv.location.Location(kittpeak.lat.to(u.deg).value, kittpeak.lon.to(u.deg).value,
                                    altitude=kittpeak.height.to(u.m).value, tz='UTC', name='NEID')
        
        locations = {'kpf': kpf, 'neid': neid}
        observers = {'kpf': Observer(latitude=wmko.lat, longitude=wmko.lon, elevation=wmko.height),
                     'neid': Observer(latitude=kittpeak.lat, longitude=kittpeak.lon, elevation=kittpeak.height)}
        utc_offsets = {'kpf': -10*u.hour, 'neid': -7*u.hour}
        self.loc = locations[inst]
        self.observer = observers[self.inst]
        self.utc_offset = utc_offsets[inst]
        
        self.irr_fn_exists = None # this is set to True/False by self.read_irradiance()
        self.pyrdata = self.read_irradiance()
        if type(self.pyrdata) == type(None):
            pass
        

    def read_irradiance(self):
        """
        Return dataframe for table of pyrheliometer irradiance
        measurements for the given instrument on the given date 
        in the file self.irr_fn.
        """

        if self.verbose:
            self.logger.debug(f'Irradiance file read: {self.irr_fn}')
        try:
            if os.path.isfile(self.irr_fn):
                self.irr_fn_exists = True
                pyrdata = pd.read_csv(self.irr_fn, comment='#')
                pyrdata = pyrdata.rename(columns={'       Date-Time        ': 'time', 'PYRIRRAD': 'irrad'})
                strfmt = '%Y-%m-%dT%H:%M:%S' # https://strftime.org/
                # old, non-vectorized and slow method:
                #dts = [(Time(timestr)-self.utc_offset).to_datetime() for timestr in pyrdata['time'].values] # this step takes some time
                # faster, vectorized method:
                arr = np.array(pyrdata['time'].values, dtype='U')   # or dtype='S' for bytes
                t = Time(arr, format='isot', scale='utc') - self.utc_offset
                dts = t.to_datetime()
                pyrdata['datetime'] = dts
                return pyrdata
            else:
                self.irr_fn_exists = False
                return None
        except FileNotFoundError:
            self.logger.error(f'Irradiance file not found: {self.irr_fn}')
            return None


    def compute_clearness_on_date(self, plot=False, save_output=True, **kwargs):
        """
        Compute the clearness index from the pyrheliometer data.
        """
                
        # to-do: add checks to make sure that data is present.
        self.dni = self.pyrdata.irrad.values
        self.dts = self.pyrdata.datetime.values
        self.jd = Time(self.dts).jd
        self.pdts = pd.DatetimeIndex(self.dts) # Pandas DatetimeIndex format for pvlib
    
        # Theoretical DNI 
        self.clear_irrad = self.loc.get_clearsky(self.pdts, model='ineichen')
        self.dni0 = self.clear_irrad['dni'].values #* (np.max(dni)/np.max(irrad0))
        self.sun_pos = self.loc.get_solarposition(self.pdts)
        self.sunalt = self.sun_pos['elevation'].values
    
        # UTC boundary for date
        strfmt = '%Y%m%d' # https://strftime.org/
        dt = datetime.strptime(self.datecode, strfmt)
        t1 = Time(dt, format='datetime') - self.utc_offset
    
        # Sunrise/set times (to not waste time calculating clearness when sun is down)
        sunrise = self.observer.sun_rise_time(t1, 'next')
        sunset5  = self.observer.sun_set_time(sunrise, 'next', horizon=5*u.deg)
        sunrise5 = self.observer.sun_rise_time(t1, 'next', horizon=5*u.deg)
        sun_up = (self.jd > sunrise5.jd) & (self.jd < sunset5.jd)
    
        # Compute clearness index
        self.clearness_index = self.compute_clearness_index(skip_times=~sun_up)
        self.clear_flag = (self.clearness_index < 4) & (self.dni >= 0.8 * self.dni0) & sun_up
        if self.verbose:
            self.logger.debug(f'{self.datecode} has {np.sum(self.clear_flag)/3600:.2f} hours of clear skies')
 
 
    def compute_clearness_index(self, time_window_size=300, time_slide_size=60, skip_times=None):
        """
        time_window_size [float]: window size in seconds
        time_slide_size [float] : length of time to slide windows by [sec] 

        self.jd [arr] : timestamps of irradiance timeseries [JD]
        self.dni [ arr] : direct solar irradiance measured at each timestamp
    
        Assume in a short time window (e.g., <30 min), DNI varies smoothly and 
        slowly (i.e., quadratic polynomial). DNI variations should follow this 
        smooth variation, up to some statistical noise. A quality factor computed 
        for that time window can be determined by fitting a quadratic model, and 
        then comparing either the chi^2 of the fit or the residual RMS to the 
        expected systematic noise floor (estimated from pre-sunrise data).
        """
    
        num_sub_windows = int(time_window_size/time_slide_size)
        clearness_values = np.zeros((num_sub_windows, len(self.jd))) + 9999
    
        chunks = []
        for n in range(num_sub_windows):
            start_t = self.jd.min() + n*time_slide_size/86400
            final_t = self.jd.max()
    
            cuts = []
            end_t = start_t + time_window_size/86400
            while end_t < final_t:
                if end_t > self.jd.max():
                    break
                cut = (self.jd >= start_t) & (self.jd < end_t)
    
                if skip_times is None or not np.all(skip_times[cut]):
                    cuts.append(cut)
                start_t = end_t
                end_t += time_window_size/86400
            chunks.append(cuts)
    
        for n in range(num_sub_windows):
            for c, chunk in enumerate(chunks[n]):
                tchunk = self.jd[chunk]
                ichunk = self.dni[chunk]
    
                if len(ichunk) < 100:
                    if self.verbose:
                        self.logger.debug('[{}] '.format(Time(np.mean(tchunk), format='jd').isot +\
                              'Irradiance chunk fewer than 100 points.  Declaring this chunk "not clear."'))
                    continue
    
                t = (tchunk - tchunk.min()) * 86400
                p = np.polyfit(t, ichunk, 2)
                model = np.poly1d(p)(t)
                residual = ichunk - model
    
                QF = np.sum(residual**2 / model)
                clearness_values[n][chunk] = QF
    
        clearness_index = np.min(clearness_values, axis=0)
        return clearness_index


    def return_clearsky_statistics(self, starttime, endtime):
        """
        Returns a tuple of the keywords below for an observation with that 
        starts on starttime and ends on stoptime (both are datetime objects).:
            CLEARSKY - Indicates clear-sky conditions for SoCal [bool]
            DNIMEAS  - Mean DNI from pyrheliometer during the exposure [W/m^s]
            DNICLR   - Theoretical DNI in perfect conditions [W/m^2]
            DNIRMS   - RMS of DNIMEAS during the exposure [W/m^2]
            CLEARIDX - SoCal clearness index (<4==CLEARSKY) [bool]
        """

        # Find timesteps between startime and stoptime
        mask = (self.pyrdata['datetime'] >= starttime) & (self.pyrdata['datetime'] <= endtime)

        # Get indices where condition is True
        indices = np.where(mask)[0]
        
        df = pd.DataFrame(np.array([self.jd, self.dni, self.dni0, self.clearness_index, self.sunalt, self.clear_flag]).T,
                          columns=['time', 'dni', 'dni0', 'clearness_index', 'sunalt', 'clearsky'])
        
        CLEARSKY = bool(np.prod(self.clear_flag[(indices)]))
        DNIMEAS  = float(np.mean(self.dni[(indices)]))
        DNICLR   = float(np.mean(self.dni0[(indices)]))
        DNIRMS   = float(np.std(self.dni[(indices)]))
        CLEARIDX = float(np.mean(self.clearness_index[(indices)]))
        return (CLEARSKY, DNIMEAS, DNICLR, DNIRMS, CLEARIDX)
    
    
    def save_clearsky_metrics_file(self, filename='auto'):
        """
        Save the clearsky_metrics file for self.datecode with the filename 
        self.clearsky_fn.  The columns are:
        'time', 'dni', 'dni0', 'clearness_index', 'sunalt', 'clearsky'
        """
        if filename == 'auto':
            filename = self.clearsky_fn
        
        df = pd.DataFrame(np.array([self.jd, self.dni, self.dni0, self.clearness_index, self.sunalt, self.clear_flag]).T,
                          columns=['time', 'dni', 'dni0', 'clearness_index', 'sunalt', 'clearsky'])

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        df.to_csv(filename, index=False)
        if self.verbose:
            self.logger.debug(f'Clearsky metrics file written: {filename}')


    def plot_clearness(self, very_clear_flag=None, sun_up=None, dni_extra=0,
                             overplot_clearness=False, fig_path=None, show_plot=False):
        """
        Generate a plot of observed and computed irradiance as well as the 
        clearness metric for a given date.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """
    
        pldts = mpl.dates.date2num(self.dts)-10/24 # for plotting in HST
        fig, ax = plt.subplots(1,1, figsize=(15,5))
    
        # Measured irradiance 
        ax.plot(pldts, self.dni - dni_extra, lw=3, zorder=2)
    
        # Clear sky irradiance
        ax.plot(pldts, self.dni0, lw=3, zorder=5)

        # Clearness index
        if overplot_clearness:
            ax2 = plt.gca().twinx()
            ax2.plot(pldts, self.clearness_index, color='k', ls=':', lw=1, zorder=0)
    
        # Shade clear/cloudy times
        notclear = (~self.clear_flag & sun_up) if not sun_up is None else ~self.clear_flag
        ax.fill_between(pldts, 0, 1200, where=notclear, alpha=0.1, color='r', zorder=-1)
        ax.fill_between(pldts, 0, 1200, where=self.clear_flag, alpha=0.1, color='g', zorder=-1)
        if not very_clear_flag is None:
            ax.fill_between(pldts, 0, 1200, where=very_clear_flag,  alpha=0.1, color='g', zorder=-1)
    
        # Color sunrise/sets
        if not sun_up is None:
            ax.fill_between(pldts, 0, 1200, where=~sun_up, color='grey', alpha=1.0, zorder=-2)
    
        # Plot settings
        if overplot_clearness:
            title = f'Measured and Computed Irradiance + Clearness Index on {self.datecode}'
        else:
            title = f'Measured and Computed Irradiance on {self.datecode}'
        ax.set_title(title, fontsize=16)
        dateformat(ax)
        ax.set(xlim=[pldts.min(), pldts.max()], ylim=[0,1200],
               xlabel='Time [HST]', ylabel='Irradiance [W/m$^2$]');
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        if overplot_clearness:
            ax2.set(ylabel='Clearness index');
            ax2.xaxis.label.set_size(12)
            ax2.yaxis.label.set_size(14)
        
        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close('all')
