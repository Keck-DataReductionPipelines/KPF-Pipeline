import os
import datetime
import argparse
import pvlib as pv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord

from astroplan import Observer
from astropy.time import Time
from astropy import units as u

from multiprocessing import Pool
from utils import isot_to_date

import warnings
from datetime import datetime, timedelta

import astropy.constants as apc
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import legval

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler

class Socal:
    """
    This module defines 'Socal' and methods to compute the clearsky metric

    Args:
        kpfobject (KPF1): A KPF L1 object
        config (configparser.ConfigParser): Config context
        logger (logging.Logger): Instance of logging.Logger
    
    Attributes:
        none

    """
    def __init__(self, 
                 kpfobject, 
                 default_config_path,
                 logger=None
                ):
        """
        Inits Socal class with raw data, order mask, config, logger.
        
        Args:
            kpfobject (KPF1): A KPF L1 object
            config (configparser.ConfigParser): Config context
            logger (logging.Logger): Instance of logging.Logger
        """

        # Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('Socal', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')

        self.kpfobject = kpfobject        


   


        
        

mpl.rc('font', family='sans serif', size=16)

_second = 1/86400
_minute = 60/86400

BASEDIR = os.environ.get('BASEDIR')

wmko = coord.EarthLocation.of_site('Keck Observatory')
kpf = pv.location.Location(wmko.lat.to(u.deg).value, wmko.lon.to(u.deg).value,
                           altitude=wmko.height.to(u.m).value, tz='UTC', name='KPF SoCal')

kittpeak = coord.EarthLocation.of_site('Kitt Peak')
neid = pv.location.Location(kittpeak.lat.to(u.deg).value, kittpeak.lon.to(u.deg).value,
                            altitude=kittpeak.height.to(u.m).value, tz='UTC', name='KPF SoCal')

locations = {'kpf': kpf, 'neid': neid}
observers = {'kpf': Observer(latitude=wmko.lat, longitude=wmko.lon, elevation=wmko.height),
             'neid':Observer(latitude=kittpeak.lat, longitude=kittpeak.lon, elevation=kittpeak.height)}
utc_offsets = {'kpf': -10*u.hour, 'neid': -7*u.hour}

def dateformat(ax):
    locator   = mpl.dates.AutoDateLocator()
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def compute_clearness_on_date(date, inst='kpf', plot=False, save_output=True, **kwargs):
    '''
    For a specificied date (in %Y%m%d format), 
    compute the clearness index from the pyrheliometer data
    '''
    assert inst in ['kpf', 'neid'], "inst must be one of ['kpf', 'neid']" 

    loc = locations[inst]
    utc_offset = utc_offsets[inst]
    
    # Get irradiance timeseries
    pyrdata = get_irrad_for_date(date, inst)
    if pyrdata is None:
        print(f'No Pyrheliometer data for {inst} on {date}')
        return None
    dni = pyrdata.irrad.values
    dts = pyrdata.datetime.values
    jd = Time(dts).jd
    pdts = pd.DatetimeIndex(dts) # Pandas DatetimeIndex format for pvlib
    # pldts = mpl.dates.date2num(dts)-10/24 # for plotting in HST

    # Theoretical DNI 
    clear_irrad = loc.get_clearsky(pdts, model='ineichen')
    dni0 = clear_irrad['dni'].values #* (np.max(dni)/np.max(irrad0))
    sun_pos = loc.get_solarposition(pdts)
    sunalt = sun_pos['elevation'].values

    # UTC boundary for date
    strfmt = '%Y%m%d' # https://strftime.org/
    dt = datetime.datetime.strptime(date, strfmt)
    t1 = Time(dt, format='datetime') - utc_offset
    # t2 = t1 + 1*u.day

    # Sunrise/set times (to not waste time calculating clearness when sun is down)
    observer = observers[inst]
    sunrise = observer.sun_rise_time(t1, 'next')
    # sunset  = observer.sun_set_time(sunrise, 'next')
    # sun_below_horizon = (jd < sunrise.jd) | (jd > sunset.jd)
    sunset5  = observer.sun_set_time(sunrise, 'next', horizon=5*u.deg)
    sunrise5 = observer.sun_rise_time(t1, 'next', horizon=5*u.deg)
    sun_up = (jd > sunrise5.jd) & (jd < sunset5.jd)

    # Compute clearness index
    clearness_index = compute_clearness_index(jd, dni, skip_times=~sun_up)
    clear_flag = (clearness_index < 4) & (dni >= 0.8 * dni0) & sun_up
    print(f'{date} has {np.sum(clear_flag)/3600:.2f} hours of clear skies')
    # very_clear_flag = (clearness_index < 5) & clear_flag
    # print('Week of {} is {:.1f}% very clear'.format(date, 100*np.sum(very_clear_flag)/len(very_clear_flag)))

    if plot:
        plot_savedir = f'{BASEDIR}/clearsky_metrics/{inst}/plots/{date[:4]}'
        if not os.path.exists(plot_savedir):
            os.makedirs(plot_savedir)
        fig, ax = plt.subplots(1,1, figsize=(15,5))
        make_clearness_plot(jd, dni, dni0,
                            clearness_index, clear_flag,
        #                     very_clear_flag=very_clear_flag,
                            sun_up=sun_up, ax=ax,
                            ylim=[0,5], **kwargs,
        #                     yticks=[0,0.5,1],
                            savefile=f'{plot_savedir}/{date}_clearsky_plot.png'
                           )

    if save_output:
        df = pd.DataFrame(np.array([jd, dni, dni0, clearness_index, sunalt, clear_flag]).T,
                          columns=['time', 'dni', 'dni0', 'clearness_index', 'sunalt', 'clearsky'])
        savedir = f'{BASEDIR}/clearsky_metrics/{inst}/{date[:4]}'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        datestr = date.replace('-', '')
        df.to_csv(f'{savedir}/{datestr}_clearsky_metrics.csv', index=False)

def get_irrad_for_date(date, inst='kpf'):
    '''
    Return dataframe for table of pyrheliometer irradiance
    measurements for the given instrument on the given date
    '''
    utc_offset = utc_offsets[inst]
    pyrdir = f'{BASEDIR}/data/{inst}/pyr'
    pfile = f'{pyrdir}/{date[:4]}/pyr_irrad_{date}.csv'
    try:
        pyrdata = pd.read_csv(pfile, comment='#')
        pyrdata = pyrdata.rename(columns={'       Date-Time        ': 'time', 'PYRIRRAD': 'irrad'})
        strfmt = '%Y-%m-%dT%H:%M:%S' # https://strftime.org/
        dts = [(Time(timestr)-utc_offset).to_datetime() for timestr in pyrdata['time'].values] # this step takes some time
        # dts = list(map(lambda timestr: (Time(timestr)-utc_offset).to_datetime(), pyrdata['time'].values))
        pyrdata['datetime'] = dts
        return pyrdata
    except FileNotFoundError:
        return None
    
def compute_clearness_index(jd, dni, time_window_size=300, time_slide_size=60, skip_times=None):
    '''
    jd [arr] : timestamps of irradiance timeseries [JD]
    dni [ arr] : direct solar irradiance measured at each timestamp
    time_window_size [float]: window size in seconds
    time_slide_size [float] : length of time to slide windows by [sec] 

    Assume in a short time window (e.g., <30 min), DNI varies smoothly and slowly (i.e., quadratic polynomial). 
    DNI variations should follow this smooth variation, up to some statistical noise. A quality factor computed 
    for that time window can be determined by fitting a quadratic model, and then comparing either the chi^2 
    of the fit or the residual RMS to the expected systematic noise floor (estimated from pre-sunrise data).
    '''

    num_sub_windows = int(time_window_size/time_slide_size)
    clearness_values = np.zeros((num_sub_windows, len(jd))) + 9999

    chunks = []
    for n in range(num_sub_windows):
        start_t = jd.min() + n*time_slide_size/86400
        final_t = jd.max()

        cuts = []
        end_t = start_t + time_window_size/86400
        while end_t < final_t:
            if end_t > jd.max():
                break
            cut = (jd >= start_t) & (jd < end_t)

            if skip_times is None or not np.all(skip_times[cut]):
                cuts.append(cut)
            start_t = end_t
            end_t += time_window_size/86400
        chunks.append(cuts)

    for n in range(num_sub_windows):
        for c, chunk in enumerate(chunks[n]):
            tchunk = jd[chunk]
            ichunk = dni[chunk]

            if len(ichunk) < 100:
                print('[{}] '.format(Time(np.mean(tchunk), format='jd').isot +\
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


def make_clearness_plot(jd, dni, dni0, clearness_index, clear_flag,
                        very_clear_flag=None, ax=None, sun_up=None,
                        dni_extra=0, savefile=None, **kwargs):

    dts = Time(jd, format='jd').to_datetime()
    pldts = mpl.dates.date2num(dts)-10/24 # for plotting in HST
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(15,5))

    # Measured irradiance 
    ax.plot(pldts, dni - dni_extra, zorder=2)

    # Clear sky irradiance
    ax.plot(pldts, dni0, zorder=5)

    # Shade clear/cloudy times
    ax2 = plt.gca().twinx()
    ax2.plot(pldts, clearness_index, color='k', ls=':', lw=1, zorder=0)
    notclear = (~clear_flag & sun_up) if not sun_up is None else ~clear_flag
    ax.fill_between(pldts, 0, 1200, where=notclear,   alpha=0.1, color='r', zorder=-1)
    ax.fill_between(pldts, 0, 1200, where=clear_flag, alpha=0.1, color='g', zorder=-1)
    if not very_clear_flag is None:
        ax.fill_between(pldts, 0, 1200, where=very_clear_flag,  alpha=0.1, color='g', zorder=-1)

    # Color sunrise/sets
    if not sun_up is None:
        ax.fill_between(pldts, 0, 1200, where=~sun_up, color='grey', alpha=1.0, zorder=-2)

    # Plot settings
    dateformat(ax)
    ax.set(xlim=[pldts.min(), pldts.max()], ylim=[0,1200],
           xlabel='Time [HST]', ylabel='Irradiance [W/m$^2$]');
    ax2.set(**kwargs, ylabel='Clearness index');
    if not savefile is None:
        plt.savefig(savefile, dpi=300, bbox_inches='tight', format=savefile.split('.')[-1])


def match_clearness_to_timestamps(jd, exptime, date, inst, logger):
    '''
    parameters:
        jd [day]        : UTC midpoints of the exposures
        exptime [sec]   : exposure length of the exposures
        date [YYYYMMDD] : date to get pyrheliometer data for
        logger          : logging object to print with

    returns: 
        clear, dni, dni0, dnirms, clearidx
    '''

    #################  Get clearness metrics during date  #################
    clear_dir = f'{BASEDIR}/clearsky_metrics/{inst}/{date[:4]}'
    pyrfile = f'{clear_dir}/{date}_clearsky_metrics.csv'
    if not os.path.exists(pyrfile):
        logger.info('Getting pyrheliometer data during exposures...')
        #pyrdate = date[:4] + '-' + date[4:6] + '-' + date[6:8] 
        try:
            compute_clearness_on_date(date, plot=True)
        except Exception as e:
            logger.error(e)
            logger.error(f'No Pyrheliometer data for {date}')

    ################# Crossmatch clearness to observation times  #################
    clear  = np.zeros_like(jd).astype(bool)
    dni    = np.zeros_like(jd)
    dni0   = np.zeros_like(jd)
    dnirms = np.zeros_like(jd)
    clearidx = np.zeros_like(jd)
    tbuffer = 1*_minute
    threshold_window = 5*_minute # make sure we can at least average over p-modes

    try:
        df = pd.read_csv(pyrfile)
        day = (jd > df['time'].min()) & (jd < df['time'].max())
        for i in np.argwhere(day)[:,0]:
            texp = exptime[i]*u.s.to(u.day)
            in_exposure = (df['time'] >= (jd[i] - texp/2)) & (df['time'] <= (jd[i] + texp/2))
#            clear[i]  = np.all(df[in_exposure]['clearsky']==1)
            if ~np.any(in_exposure):
                logger.warning(f'No simultaneous pyrheliometer for {date}')
                dni[i] = dni0[i] = dnirms[i] = clearidx[i] = np.nan
            else:
                dni[i]    = np.nanmean(df[in_exposure]['dni'])
                dni0[i]   = np.nanmean(df[in_exposure]['dni0'])
                dnirms[i] = np.nanstd(df[in_exposure]['dni'])
                clearidx[i] = np.nanmedian(df[in_exposure]['clearness_index'])

        cleartimes = np.sort(df[df['clearsky']==1]['time'].values) 
        idxs = np.where(np.diff(cleartimes) > 10*_second)[0] # gaps in clearness
    
        # Loop through clear "chunks" to flag corresponding RVs 
        if len(idxs) == 0:
            if len(cleartimes)>0:
                # Fully clear day!  
                logger.info('  Fully clear day.')
                clear_start = cleartimes.min()+tbuffer
                clear_stop  = cleartimes.max()-tbuffer
                clear[(jd > clear_start) & (jd < clear_stop)] = True#(clear_stop-clear_start) > threshold_window 
            else:
                logger.info('Fully cloudy day.')
        else:
            istart = 0
            # This day has some cloudy patches. Loop through clear sections...
            lengths = []
            for iend in np.append(idxs, -1):
                clearchunk = cleartimes[istart:iend]
                clear_start = clearchunk.min()+tbuffer
                clear_stop  = clearchunk.max()-tbuffer
                lengths.append( (clear_stop-clear_start)*u.day.to(u.min) ) 
                # This chunk is considered clear if it is at least the threshold in duration 
                clear[(jd > clear_start) & (jd < clear_stop)] = (clear_stop-clear_start) > threshold_window 
                istart = iend + 1
            logger.info('  {} clear-sky chunks: {} min'.format(len(lengths), np.round(lengths, 1)))

    except Exception as e:
        logger.error(e)
        logger.error(f'No clearness metrics for {date}')

    return clear, dni, dni0, dnirms, clearidx


if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Process SoCal data for a given date")
    p.add_argument("-d", "--date", dest="date", type=str, required=False,
                    default=None,   help="Date to run SoCal pipeline on")
    p.add_argument("-s", "--datestart", dest="datestart", type=str, required=False,
                    default=None,   help="Start date for reprocess")
    p.add_argument("-e", "--dateend", dest="dateend", type=str, required=False,
                    default=None,   help="End date for reprocess")
    p.add_argument("-p", "--doplots",dest="doplots", action=argparse.BooleanOptionalAction,
                    default=False,   help="Whether to create and save clearness plots")
    p.add_argument("-i", "--inst",dest="inst", type=str, required=False,
                    default='kpf',   help="Which instrument to run the calculation on [kpf, neid]")
    args = p.parse_args()

    #################  Dates to run  #################
    if args.date is None:
        ds = args.datestart if not args.datestart is None else '20230425' # SoCal first light and Ryan's birthday
        de = args.dateend   if not args.dateend   is None else isot_to_date(Time.now().isot)
        date_start = Time(ds[:4]+'-'+ds[4:6]+'-'+ds[6:8], format='isot')
        date_end   = Time(de[:4]+'-'+de[4:6]+'-'+de[6:8], format='isot')
        all_dates = [isot_to_date(d) for d in Time(np.arange(date_start.jd, date_end.jd, 1), format='jd').isot]
    
        # parallelize
        ncores = min(16, len(all_dates))
        def wrapper(date):
            compute_clearness_on_date(date, inst=args.inst, plot=args.doplots)
        with Pool(ncores) as p:
            p.map(wrapper, all_dates)
   
        # for date in all_dates:
        #     #################  Compute clearness metrics during date  #################
        #     try:
        #         compute_clearness_on_date(date, inst=args.inst, plot=args.doplots)
        #     except Exception as e:
        #         print(e)
    
    else:
        try:
            compute_clearness_on_date(args.date, inst=args.inst, plot=args.doplots)
        except Exception as e:
            print(e)
