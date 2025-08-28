#import time
#import math
import pvlib as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.lines as mlines
from modules.Utils.utils import DummyLogger
from astroplan import Observer
from astropy.time import Time
from astropy import units as u
from datetime import datetime
import astropy.coordinates as coord
from modules.Utils.kpf_parse import HeaderParse

class AnalyzePyr:

    """
    Description:
        This class contains functions to analyze irradiance time series and 
        compute the clearsky index.

    Arguments:
        L0 - an L0 object
    """

    def __init__(self, datecode, basedir='/data/pyr', inst='kpf', logger=None):

        self.datecode = datecode
        self.basedir = basedir
        self.logger = logger if logger is not None else DummyLogger()
        self.inst = inst
   
        wmko = coord.EarthLocation.of_site('Keck Observatory')
        kpf = pv.location.Location(wmko.lat.to(u.deg).value, wmko.lon.to(u.deg).value,
                                   altitude=wmko.height.to(u.m).value, tz='UTC', name='KPF SoCal')
        
        kittpeak = coord.EarthLocation.of_site('Kitt Peak')
        neid = pv.location.Location(kittpeak.lat.to(u.deg).value, kittpeak.lon.to(u.deg).value,
                                    altitude=kittpeak.height.to(u.m).value, tz='UTC', name='NEID')
        
        self.locations = {'kpf': kpf, 'neid': neid}
        self.observers = {'kpf': Observer(latitude=wmko.lat, longitude=wmko.lon, elevation=wmko.height),
                         'neid': Observer(latitude=kittpeak.lat, longitude=kittpeak.lon, elevation=kittpeak.height)}
        self.utc_offsets = {'kpf': -10*u.hour, 'neid': -7*u.hour}
        
        self.pyrdata = self.read_irrad_for_date()
        if type(self.pyrdata) == type(None):
            pass
        

    def read_irrad_for_date(self):
        '''
        Return dataframe for table of pyrheliometer irradiance
        measurements for the given instrument on the given date
        '''
        utc_offset = self.utc_offsets[self.inst]
        pfile = f'{self.basedir}/irradiance/{self.datecode[:4]}/pyr_irrad_{self.datecode}.csv'
        print(pfile)
        try:
            pyrdata = pd.read_csv(pfile, comment='#')
            pyrdata = pyrdata.rename(columns={'       Date-Time        ': 'time', 'PYRIRRAD': 'irrad'})
            strfmt = '%Y-%m-%dT%H:%M:%S' # https://strftime.org/
            dts = [(Time(timestr)-utc_offset).to_datetime() for timestr in pyrdata['time'].values] # this step takes some time
            # dts = list(map(lambda timestr: (Time(timestr)-utc_offset).to_datetime(), pyrdata['time'].values))
            pyrdata['datetime'] = dts
            return pyrdata
        except FileNotFoundError:
            self.logger(f'Irradiance file not found: {pfile}')
            return None
   
    def compute_clearness_on_date(date, plot=False, save_output=True, **kwargs):
        '''
        For a specificied date (in %Y%m%d format), 
        compute the clearness index from the pyrheliometer data
        '''
        assert self.inst in ['kpf', 'neid'], "inst must be one of ['kpf', 'neid']" 
    
        loc = self.locations[inst]
        utc_offset = self.utc_offsets[inst]
        
#        # Get irradiance timeseries
#        pyrdata = get_irrad_for_date(date, inst)
#        if pyrdata is None:
#            print(f'No Pyrheliometer data for {inst} on {date}')
#            return None
        self.dni = self.pyrdata.irrad.values
        self.dts = self.pyrdata.datetime.values
        self.jd = Time(dts).jd
        self.pdts = pd.DatetimeIndex(dts) # Pandas DatetimeIndex format for pvlib
        # pldts = mpl.dates.date2num(dts)-10/24 # for plotting in HST
    
        # Theoretical DNI 
        self.clear_irrad = loc.get_clearsky(pdts, model='ineichen')
        self.dni0 = self.clear_irrad['dni'].values #* (np.max(dni)/np.max(irrad0))
        self.sun_pos = loc.get_solarposition(self.pdts)
        self.sunalt = self.sun_pos['elevation'].values
    
        # UTC boundary for date
        strfmt = '%Y%m%d' # https://strftime.org/
        dt = datetime.datetime.strptime(date, strfmt)
        t1 = Time(dt, format='datetime') - utc_offset
        # t2 = t1 + 1*u.day
    
        # Sunrise/set times (to not waste time calculating clearness when sun is down)
        observer = self.observers[inst]
        sunrise = self.observer.sun_rise_time(t1, 'next')
        # sunset  = observer.sun_set_time(sunrise, 'next')
        # sun_below_horizon = (jd < sunrise.jd) | (jd > sunset.jd)
        sunset5  = self.observer.sun_set_time(sunrise, 'next', horizon=5*u.deg)
        sunrise5 = self.observer.sun_rise_time(t1, 'next', horizon=5*u.deg)
        sun_up = (self.jd > sunrise5.jd) & (self.jd < sunset5.jd)
    
        # Compute clearness index
        self.clearness_index = self.compute_clearness_index(self.jd, self.dni, skip_times=~sun_up)
        self.clear_flag = (self.clearness_index < 4) & (self.dni >= 0.8 * self.dni0) & sun_up
        self.logger.info(f'{datecode} has {np.sum(self.clear_flag)/3600:.2f} hours of clear skies')
        # very_clear_flag = (clearness_index < 5) & clear_flag
        # print('Week of {} is {:.1f}% very clear'.format(date, 100*np.sum(very_clear_flag)/len(very_clear_flag)))
    
#        if plot:
#            plot_savedir = f'{BASEDIR}/clearsky_metrics/{inst}/plots/{date[:4]}'
#            if not os.path.exists(plot_savedir):
#                os.makedirs(plot_savedir)
#            fig, ax = plt.subplots(1,1, figsize=(15,5))
#            make_clearness_plot(jd, dni, dni0,
#                                clearness_index, clear_flag,
#            #                     very_clear_flag=very_clear_flag,
#                                sun_up=sun_up, ax=ax,
#                                ylim=[0,5], **kwargs,
#            #                     yticks=[0,0.5,1],
#                                savefile=f'{plot_savedir}/{date}_clearsky_plot.png'
#                               )
    
#        if save_output:
#            df = pd.DataFrame(np.array([jd, dni, dni0, clearness_index, sunalt, clear_flag]).T,
#                              columns=['time', 'dni', 'dni0', 'clearness_index', 'sunalt', 'clearsky'])
#            savedir = f'{BASEDIR}/clearsky_metrics/{inst}/{date[:4]}'
#            if not os.path.exists(savedir):
#                os.makedirs(savedir)
#            datestr = date.replace('-', '')
#            df.to_csv(f'{savedir}/{datestr}_clearsky_metrics.csv', index=False)
 
