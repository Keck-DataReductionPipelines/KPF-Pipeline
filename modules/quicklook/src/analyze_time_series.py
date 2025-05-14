import os
import ast
import time
#import glob
#import copy
import json
import yaml
#import sqlite3
#import calendar
import numpy as np
import pandas as pd
#from tqdm import tqdm
#from tqdm.notebook import tqdm_notebook
#from astropy.table import Table
#from astropy.io import fits
from datetime import datetime, timedelta
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from modules.Utils.utils import DummyLogger, get_sunrise_sunset_ut
from modules.Utils.kpf_parse import get_datecode
from collections import Counter
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from matplotlib.dates import HourLocator, DayLocator, MonthLocator, YearLocator, AutoDateLocator, DateFormatter
from IPython.display import display, HTML

from modules.Utils.utils import DummyLogger
from database.modules.utils.tsdb import TSDB

import sys
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

import cProfile
import pstats
from io import StringIO

class AnalyzeTimeSeries:

    """
    Description:
        This class contains a set of methods to create a database of data associated 
        with KPF observations, as well as methods to ingest data, query the database, 
        print data, and made time series plots.  An elaborate set of standard time series 
        plots can be made over intervals of days/months/years/decades spanning a date 
        range.  
        
        The ingested data comes from L0/2D/L1/L2 keywords and the TELEMETRY extension 
        in L0 files.  With the current version of this code, all TELEMETRY keywords are 
        added to the database an a small subset of the L0/2D/L1/L2 keywords are added. 
        These lists can be expanded, but will require re-ingesting the data (which takes 
        about half a day for all KPF observations).  RVs are currently not ingested, but 
        that capability should be added.

    Arguments:
        db_path (string) - path to database file
        base_dir (string) - L0 directory
        drop (boolean) - if true, the database at db_path is dropped at startup
        logger (logger object) - a logger object can be passed, or one will be created

    Attributes:

    Related Commandline Scripts:
        'generate_time_series_plots.py' - creates standard time series plots
        
    To-do:
        * Make standard correlation plots.
        * Make standard phased plots (by day)
        * Make plot of correlation between per-order RVs and RVs per-chip and overall RVs.
        * Method to return the avg, std., etc. for a DB column over a time range, with conditions (e.g., fast-read mode only)
        * Make plots of temperature vs. RV for various types of RVs (correlation plots)
        * Add standard plots of flux vs. time for cals (all types?), stars, and solar -- highlight Junked files
        * Augment statistics in legends (median and stddev upon request)
        * Make a standard plot type that excludes outliers using ranges set 
          to, say, +/- 4-sigma where sigma is determined by aggressive outlier
          rejection.  This should be in Delta values.
        * For time series state plots, include the number of points in each state 
          in the legend.
        * Specify the yrange in the yaml files
    """

    def __init__(self, db_path='kpf_ts.db', base_dir='/data/L0', backend='sqlite', logger=None, verbose=False):
       
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Starting AnalyzeTimeSeries')
        self.db = TSDB(backend=backend, db_path=db_path, base_dir=base_dir, logger=logger, verbose=verbose)


    def plot_time_series_multipanel(self, plotdict, 
                                    start_date=None, end_date=None, 
                                    clean=False, 
                                    fig_path=None, show_plot=False, 
                                    log_savefig_timing=False):
        """
        Generate a multi-panel plot of data in a KPF TSDB.  The data to be 
        plotted and attributes are stored in an array of dictionaries, which 
        can be read from YAML configuration files.  

        Args:
            panel_dict makes panel_arr ...
            panel_arr (array of dictionaries) - each dictionary in the array has keys:
                panelvars: a dictionary of matplotlib attributes including:
                    ylabel - text for y-axis label
                paneldict: a dictionary containing:
                    col: name of DB column to plot
                    plot_type: 'plot' (points with connecting lines), 
                               'scatter' (points), 
                               'step' (steps), 
                               'state' (for non-floats, like DRPTAG)
                    plot_attr: a dictionary containing plot attributes for a scatter plot, 
                        including 'label', 'marker', 'color'
                    not_junk: if set to 'True', only files with NOTJUNK=1 are included; 
                              if set to 'False', only files with NOTJUNK=0 are included
                    on_sky: if set to 'True', only on-sky observations will be included; 
                            if set to 'False', only calibrations will be included
                    only_object (not implemented yet): if set, only object names in the keyword's value will be queried
                    object_like (not implemented yet): if set, partial object names matching the keyword's value will be queried
            start_date (datetime object) - start date for plot
            end_date (datetime object) - end date for plot
            fig_path (string) - set to the path for the file to be generated
            show_plot (boolean) - show the plot in the current environment
            These are now part of the dictionaries:
                only_object (string or list of strings) - object names to include in query
                object_like (string or list of strings) - partial object names to search for
                on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
            
        To do:
            * Make a standard plot type that excludes outliers using ranges set 
              to, say, +/- 4-sigma where sigma is determined by aggressive outlier
              rejection.  This should be in Delta values.
            * Make standard correlation plots.
            * Make standard phased plots (by day)
        """
        import warnings
        warnings.filterwarnings("ignore", message=".*tight_layout.*")

        def num_fmt(n: float, sf: int = 3) -> str:
            """
            Returns number as a formatted string with specified number of significant figures
            :param n: number to format
            :param sf: number of sig figs in output
            """
            r = f'{n:.{sf}}'  # use existing formatter to get to right number of sig figs
            if 'e' in r:
                exp = int(r.split('e')[1])
                base = r.split('e')[0]
                r = base + '0' * (exp - sf + 2)
            return r
    
        def format_func(value, tick_number):
            """ For formatting of log plots """
            return num_fmt(value, sf=2)

        # Retrieve the appropriate standard plot dictionary
        if type(plotdict) == type('str'):
            plotdict_str = plotdict
            import static.tsdb_plot_configs
            all_yaml = static.tsdb_plot_configs.all_yaml # an attribute from static/tsdb_plot_configs/__init__.py        
            base_filenames = [os.path.basename(y) for y in all_yaml]
            base_filenames = [str.split(f,'.')[0] for f in base_filenames]
            try:
                ind = base_filenames.index(plotdict_str)
                plotdict = self.yaml_to_dict(all_yaml[ind])
                self.logger.info(f'Plotting from config: {all_yaml[ind]}')
            except Exception as e:
                self.logger.info(f"Couldn't find the file {plotdict_str}.  Error message: {e}")
                return
        
        panel_arr = plotdict['panel_arr']
        
        npanels = len(panel_arr)
        unique_cols = set()
        unique_cols.add('DATE-MID')
        unique_cols.add('FIUMODE')
        unique_cols.add('OBJECT')
        unique_cols.add('NOTJUNK')
        for panel in panel_arr:
            for d in panel['panelvars']:
                unique_cols.add(d['col'])
                if 'col_err' in d:
                    unique_cols.add(d['col_err'])
                if 'col_subtract' in d:
                    unique_cols.add(d['col_subtract'])
        # add this?
        #if 'only_object' in thispanel['paneldict']:
        #if 'object_like' in thispanel['paneldict']:

        fig, axs = plt.subplots(npanels, 1, sharex=True, figsize=(15, npanels*2.5+1), tight_layout=True)
        if npanels == 1:
            axs = [axs]  # Make axs iterable even when there's only one panel
        if npanels > 1:
            plt.subplots_adjust(hspace=0)
        #plt.tight_layout() # this caused a core dump in scripts/generate_time_series_plots.py

        overplot_night_box = False
        no_data = True # for now; will be set to False when data is detected
        for p in np.arange(npanels):
            thispanel = panel_arr[p]            
            not_junk = thispanel['paneldict'].get('not_junk', plotdict.get('not_junk', None))
            if isinstance(not_junk, str):
                not_junk = True if not_junk.lower() == 'true' else False if not_junk.lower() == 'false' else not_junk
            only_object = thispanel['paneldict'].get('only_object', plotdict.get('only_object', None))
            object_like = thispanel['paneldict'].get('object_like', plotdict.get('object_like', None))
            if object_like is not None:
                if isinstance(object_like, str):
                    object_like = [object_like]
                elif isinstance(object_like, list):
                    flattened = []
                    for item in object_like:
                        if isinstance(item, list):
                            flattened.extend(item)
                        else:
                            flattened.append(item)
                    object_like = flattened
            only_source = thispanel['paneldict'].get('only_source', plotdict.get('only_source', None))

            if start_date == None:
                start_date = datetime(2020, 1,  1)
                start_date_was_none = True
            else:
                start_date_was_none = False
            if end_date == None:
                end_date = datetime(2300, 1,  1)
                end_date_was_none = True
            else:
                end_date_was_none = False

            # Get data from database
            df = self.db.dataframe_from_db(unique_cols, 
                                           start_date=start_date, 
                                           end_date=end_date, 
                                           not_junk=not_junk, 
                                           only_object=only_object, 
                                           object_like=object_like,
                                           only_source=only_source, 
                                           verbose=False)

        	# Check if the resulting dataframe has any rows
            empty_df = (len(df) == 0) # True if the dataframe has no rows
            if not empty_df:
                df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?
                if start_date_was_none == True:
                    start_date = min(df['DATE-MID'])
                if end_date_was_none == True:
                    end_date = max(df['DATE-MID'])
                df = df.sort_values(by='DATE-MID')
    
                # Remove outliers
                if clean:
                    df = self.db.clean_df(df)
    
                # Filter using on_sky criterion
                if 'on_sky' in thispanel['paneldict']:
                    if str(thispanel['paneldict']['on_sky']).lower() == 'true':
                        df = df[df['FIUMODE'] == 'Observing']
                    elif str(thispanel['paneldict']['on_sky']).lower() == 'false':
                        df = df[df['FIUMODE'] == 'Calibration']
                    
            thistitle = ''
            if ((end_date - start_date).days <= 1.05) and ((end_date - start_date).days >= 0.95):
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                xtitle = start_date.strftime('%B %d, %Y') + ' (UT; HST=UT-10 hours)'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 3600)
                axs[p].xaxis.set_major_locator(ticker.MultipleLocator(2))  # major tick every 2 hr
                axs[p].xaxis.set_minor_locator(ticker.MultipleLocator(1))  # minor tick every 1 hr
                def format_HHMM(x, pos):
                    try:
                        date = start_date + timedelta(hours=x)
                        return date.strftime('%H:%M') + ' UT \n' + (date-timedelta(hours=10)).strftime('%H:%M') + ' HST'
                    except:
                        return ''
                axs[p].xaxis.set_major_formatter(ticker.FuncFormatter(format_HHMM))
                overplot_night_box = True
                
                sunrise, sunset = get_sunrise_sunset_ut("2025-04-12")
                sunrise_h = sunrise.hour + sunrise.minute/60 + sunrise.second/3600
                sunset_h  = sunset.hour  + sunset.minute/60  + sunset.second/3600
                axs[p].axvspan(sunset_h, sunrise_h, facecolor='lightgray', alpha=0.2, hatch='', edgecolor='silver')
                axs[p].annotate("Night", xy=((sunset_h+sunrise_h)/48, 1), xycoords='axes fraction', 
                                fontsize=10, color="silver", ha="center", va="top",
                                xytext=(0, -5), 
                                textcoords='offset points')
            elif abs((end_date - start_date).days) <= 1.2:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                xtitle = 'Hours since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = str(thispanel['paneldict']['title']) + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 3600)
                if not empty_df:
                    if 'narrow_xlim_daily' in thispanel['paneldict']:
                        if str(thispanel['paneldict']['narrow_xlim_daily']).lower() == 'true':
                            if len(t) > 1:
                                axs[p].set_xlim(min(t), max(t))
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif abs((end_date - start_date).days) <= 3:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif 28 <= (end_date - start_date).days <= 31:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = start_date.strftime('%B %Y') + ' (UT Times)'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                
                axs[p].set_xlim(0, (end_date - start_date).days)
                axs[p].xaxis.set_major_locator(ticker.MultipleLocator(1))  # tick every 1 day
            
                # Custom formatter to convert "days since start" into actual calendar labels
                def format_mmdd(x, pos):
                    try:
                        date = start_date + timedelta(days=int(x))
                        return date.strftime('%d')
                    except:
                        return ''
                axs[p].xaxis.set_major_formatter(ticker.FuncFormatter(format_mmdd))
            elif abs((end_date - start_date).days) < 32:
                if not empty_df:
                     t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=3, prune=None))
            elif 360 <= (end_date - start_date).days <= 370:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = start_date.strftime('%Y')
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].set_xlim(0, (end_date - start_date).days)

                # Set major ticks at month boundaries
                month_starts = []
                current = start_date.replace(day=1)
                while current < end_date:
                    delta = (current - start_date).days
                    month_starts.append(delta)
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)

                axs[p].xaxis.set_major_locator(ticker.FixedLocator(month_starts))
                def format_mmdd(x, pos):
                    try:
                        date = start_date + timedelta(days=int(x))
                        return date.strftime('%m-%d')
                    except:
                        return ''
                axs[p].xaxis.set_major_formatter(ticker.FuncFormatter(format_mmdd))
            else:
                if not empty_df:
                    t = df['DATE-MID'] # dates
                xtitle = 'Date'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(7, prune=None))
            if p == npanels-1: 
                axs[p].set_xlabel(xtitle, fontsize=14)
                axs[0].set_title(thistitle, fontsize=18)
            if 'ylabel' in thispanel['paneldict']:
                axs[p].set_ylabel(thispanel['paneldict']['ylabel'], fontsize=14)
            axs[p].grid(color='lightgray')        
            if 'yscale' in thispanel['paneldict']:
                if thispanel['paneldict']['yscale'] == 'log':
                    formatter = FuncFormatter(format_func)  # this doesn't seem to be working
                    axs[p].minorticks_on()
                    axs[p].grid(which='major', axis='x', color='darkgray',  linestyle='-', linewidth=0.5)
                    axs[p].grid(which='both',  axis='y', color='lightgray', linestyle='-', linewidth=0.5)
                    axs[p].set_yscale('log')
                    axs[p].yaxis.set_minor_locator(plt.AutoLocator())
                    axs[p].yaxis.set_major_formatter(formatter)
            else:
                axs[p].grid(color='lightgray')        
            ylim=False
            if 'ylim' in thispanel['paneldict']:
                if type(ast.literal_eval(thispanel['paneldict']['ylim'])) == type((1,2)):
                    ylim = ast.literal_eval(thispanel['paneldict']['ylim'])

            makelegend = True
            if 'nolegend' in thispanel['paneldict']:
                if str(thispanel['paneldict']['nolegend']).lower() == 'true':
                    makelegend = False

            subtractmedian = False
            if 'subtractmedian' in thispanel['paneldict']:
                if str(thispanel['paneldict']['subtractmedian']).lower() == 'true':
                    subtractmedian = True

            nvars = len(thispanel['panelvars'])
            if not empty_df:
                df_initial = df
                for i in np.arange(nvars):
                    df = df_initial # start fresh for each panel in case NaN values were removed.
                    if 'plot_type' in thispanel['panelvars'][i]:
                        plot_type = thispanel['panelvars'][i]['plot_type']
                    else:
                        plot_type = 'scatter'
                    
                    # Extract data from df and manipulate
                    col_name = thispanel['panelvars'][i]['col']
                    # Filter out invalid values in col_name
                    df = df[~df[col_name].isin(['NaN', 'null', 'nan', 'None', None, np.nan])]
                    col_data = df[col_name]
                    col_data_replaced = col_data  # default, no subtraction
                    
                    if 'col_subtract' in thispanel['panelvars'][i]:
                        col_subtract_name = thispanel['panelvars'][i]['col_subtract']
                        # Now filter out invalid values in col_subtract_name,
                        # and also re-filter col_name because removing rows re-indexes the DataFrame.
                        df = df[~df[col_subtract_name].isin(['NaN', 'null', 'nan', 'None', None, np.nan])]
                        # Re-grab the series after dropping rows
                        col_data = df[col_name]
                        col_subtract_data = df[col_subtract_name]
                        col_data_replaced = col_data - col_subtract_data
    
                    if 'col_multiply' in thispanel['panelvars'][i]:
                        col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') * thispanel['panelvars'][i]['col_multiply']
    
                    if 'col_offset' in thispanel['panelvars'][i]:
                        col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') + thispanel['panelvars'][i]['col_offset']
    
                    if 'col_err' in thispanel['panelvars'][i]:
                        col_data_err = df[thispanel['panelvars'][i]['col_err']]
                        col_data_err_replaced = col_data_err.replace('NaN',  np.nan)
                        col_data_err_replaced = col_data_err.replace('null', np.nan)
                        if 'col_multiply' in thispanel['panelvars'][i]:
                            col_data_err_replaced = pd.to_numeric(col_data_err_replaced, errors='coerce') * thispanel['panelvars'][i]['col_multiply']
    
                    if 'normalize' in thispanel['panelvars'][i]:
                        if thispanel['panelvars'][i]['normalize'] == True:
                            col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') / np.nanmedian(pd.to_numeric(col_data_replaced, errors='coerce'))
                    
                    if plot_type == 'state':
                        states = np.array(col_data_replaced)
                    else:
                        data = np.array(col_data_replaced, dtype='float')
                        if plot_type == 'errorbar':
                            data_err = np.array(col_data_err_replaced, dtype='float')
    
                    if abs((end_date - start_date).days) <= 1.2:
                        t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                    elif abs((end_date - start_date).days) <= 3:
                        t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                    elif abs((end_date - start_date).days) < 32:
                        t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                    elif 360 <= (end_date - start_date).days <= 370:
                        if not empty_df:
                            t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]

                    else:
                        t = df['DATE-MID'] # dates
    
                    # Set plot attributes
                    plot_attributes = {}
                    if plot_type != 'state':
                        if np.count_nonzero(~np.isnan(data)) > 0:
                            if subtractmedian:
                                data -= np.nanmedian(data)
                            if 'plot_attr' in thispanel['panelvars'][i]:
                                if 'label' in thispanel['panelvars'][i]['plot_attr']:
                                    label = thispanel['panelvars'][i]['plot_attr']['label']
                                    try:
                                        if makelegend:
                                            if len(~np.isnan(data)) > 0:
                                                median = np.nanmedian(data)
                                            else:
                                                median = 0.
                                            if len(~np.isnan(data)) > 2:
                                                std_dev = np.nanstd(data)
                                                if std_dev != 0 and not np.isnan(std_dev):
                                                    decimal_places = max(1, 2 - int(np.floor(np.log10(abs(std_dev)))) - 1)
                                                else:
                                                    decimal_places = 1
                                            else:
                                                decimal_places = 1
                                                std_dev = 0.
                                            formatted_median = f"{median:.{decimal_places}f}"
                                            if len(~np.isnan(data)) > 2:
                                                formatted_std_dev = f"{std_dev:.{decimal_places}f}"
                                                label += ' (' + formatted_std_dev 
                                                if 'unit' in thispanel['panelvars'][i]:
                                                    label += ' ' + str(thispanel['panelvars'][i]['unit'])
                                                label += ' rms)'
                                    except Exception as e:
                                        self.logger.error(e)
                                plot_attributes = thispanel['panelvars'][i]['plot_attr']
                                if 'label' in plot_attributes:
                                    plot_attributes['label'] = label
                            else:
                               plot_attributes = {}
    
                    # Plot type: scatter plot
                    if plot_type == 'scatter':
                        axs[p].scatter(t, data, **plot_attributes)
                    
                    # Plot type: scatter plot with error bars
                    if plot_type == 'errorbar':
                        axs[p].errorbar(t, data, yerr=data_err, **plot_attributes)
                    
                    # Plot type: connected points
                    if plot_type == 'plot':
                        axs[p].plot(t, data, **plot_attributes)
                    
                    # Plot type: stepped lines
                    if plot_type == 'step':
                        axs[p].step(t, data, **plot_attributes)
                    
                    # Plot type: scatter plots for non-float 'states'
                    if plot_type == 'state':
                        # Plot states (e.g., DRP version number or QC result)
                        # First, convert states to a consistent type for comparison
                        states = [float(s) if is_numeric(s) else s for s in states]
                        # Separate numeric and non-numeric states for sorting
                        numeric_states = sorted(s for s in states if isinstance(s, float))
                        non_numeric_states = sorted(s for s in states if isinstance(s, str))
                        unique_states = sorted(set(states), key=lambda x: (not isinstance(x, float), x))
                        unique_states = list(set(unique_states))
                        # Check if unique_states contains only 0, 1, and None - QC test
                        if set(unique_states).issubset({0.0, 1.0, 'None'}):
                            states = ['Pass' if s == 1.0 else 'Fail' if s == 0.0 else s for s in states]
                            unique_states = sorted(set(states), key=lambda x: (not isinstance(x, float), x))
                            unique_states = list(set(unique_states))
                            if (unique_states == ['Pass', 'Fail']) or (unique_states == ['Pass']) or (unique_states == ['Fail']):
                                 unique_states = ['Fail', 'Pass']  # put Pass on the top of the plot
                            state_to_color = {'Fail': 'indianred', 'Pass': 'forestgreen', 'None': 'cornflowerblue'}
                            if thispanel['paneldict']['ylabel'] == 'Junk Status':
                                states = ['Not Junk' if s == 'Pass' else 'Junk' if s == 'Fail' else s for s in states]
                                unique_states = ['Junk', 'Not Junk']
                                state_to_color = {'Junk': 'indianred', 'Not Junk': 'forestgreen', 'None': 'cornflowerblue'}
                            mapped_states = [unique_states.index(state) if state in unique_states else None for state in states]
                            colors = [state_to_color[state] if state in state_to_color else 'black' for state in states]
                            color_map = {state: state_to_color[state] for state in unique_states if state in state_to_color}
                        else:
                            unique_states = sorted(list(set(unique_states)))
                            state_to_num = {state: i for i, state in enumerate(unique_states)}
                            mapped_states = [state_to_num[state] for state in states]
                            colors = plt.cm.jet(np.linspace(0, 1, len(unique_states)))
                            color_map = {state: colors[i] for i, state in enumerate(unique_states)}
                        try:
                            # check for a set of conditions that took forever to figure out
                            if (hasattr(t, 'tolist') and callable(getattr(t, 'tolist'))):
                                t = t.tolist()
                            else:
                                t = list(t)
                        except Exception as e:
                            self.logger.info(f"Error converting to a list: {e}")
                        try:
                            if (hasattr(states, 'tolist') and callable(getattr(states, 'tolist'))):
                                states = states.tolist()
                            else:
                                states = list(states)
                        except Exception as e:
                            self.logger.info(f"Error converting to a list: {e}")
                        if len(states) != len(t):
                            # Handle the mismatch
                            print(f"Length mismatch: states has {len(states)} elements, t has {len(t)}")
                        for state in unique_states:
                            color = color_map[state]
                            indices = [i for i, s in enumerate(states) if s == state]
                            label_text = f"{state} ({len(indices)})"
                            axs[p].scatter([t[i] for i in indices], [mapped_states[i] for i in indices], color=color, label=label_text)
                        axs[p].set_yticks(range(len(unique_states)))
                        axs[p].set_yticklabels(unique_states)
                    
                    # Print 'no data' if appropriate
                    if len(t) > 1:
                        no_data = False # this is checked for each variable
                    if (no_data and i == nvars-1):
                        axs[p].text(0.5, 0.5, 'No Data', 
                                    horizontalalignment='center', verticalalignment='center', 
                                    fontsize=24, transform=axs[p].transAxes)
    
                    axs[p].xaxis.set_tick_params(labelsize=10)
                    axs[p].yaxis.set_tick_params(labelsize=10)

            # Draw translucent boxes
            if 'axhspan' in thispanel['paneldict']:
                # This is needed so that ylim autoscaling is based on data and not the boxes below
                axs[p].relim()            
                axs[p].autoscale_view()   
#                ymin, ymax = axs[p].get_ylim()
#                axs[p].set_ylim(ymin, ymax)
                axs[p].set_autoscale_on(False)
                # Draw boxes
                for key, axh in thispanel['paneldict']['axhspan'].items():
                     ymin = axh['ymin']
                     ymax = axh['ymax']
                     clr  = axh['color']
                     alp  = axh['alpha']
                     axs[p].axhspan(ymin, ymax, color=clr, alpha=alp)
                


            # Make legend
            if makelegend:
                if not no_data:
                    if len(t) > 0:
                        if 'legend_frac_size' in thispanel['paneldict']:
                            legend_frac_size = thispanel['paneldict']['legend_frac_size']
                        else:
                            legend_frac_size = 0.20
                        handles, labels = axs[p].get_legend_handles_labels()
                        sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1], reverse=True)
                        handles, labels = zip(*sorted_pairs)
                        axs[p].legend(handles, labels, loc='upper right', bbox_to_anchor=(1+legend_frac_size, 1))

            # Set y limits
            if ylim:
                axs[p].set_ylim(ylim)

            # Add background grid
            axs[p].grid(color='lightgray')

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time} UT"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    #xytext=(100, -32), 
                    xytext=(0, -38), 
                    textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot or make a PNG
        try:
            if fig_path != None:
                t0 = time.process_time()
                plt.savefig(fig_path, dpi=300, facecolor='w')
                if log_savefig_timing:
                    self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
            if show_plot == True:
                plt.show()
            plt.close('all')
        except Exception as e:
            self.logger.info(f"Error saving file or showing plot: {e}")

    def plot_rv_per_fiber_wavelength(self, rv, chip, fiber, start_date=None, end_date=None, only_object=None, only_source=None, 
                                    object_like=None, fig_path=None, show_plot=True, 
                                    log_savefig_timing=False):
        """
        Generate a timeseries showing every orderlet of a specific fiber (SCI1, SCI2, or SCI3) for either green or red. 

        Args:
            rv (string) - string describing what rv type to plot (etalon, lfc, etc)
            chip (string) - green or red
            fiber (string) - SCI1, SCI2, or SCI3
            start_date (datetime object) - start date for plot
            end_date (datetime object) - end date for plot
            only_object (string or list of strings) - object names to include in query
            only_source (string or list of strings) - source names to include in query (e.g., 'Star')
            object_like (string or list of strings) - partial object names to search for
            fig_path (string) - set to the path for the file to be generated
            show_plot (boolean) - show the plot in the current environment

        Returns:
            PNG plot in fig_path or shows the plot in the current environment
            (e.g., in a Jupyter Notebook).
        """
        unique_cols = set()
        unique_cols.add('DATE-MID')
        unique_cols.add('NOTJUNK')
        unique_cols.add('ObsID')
        if chip.lower() == 'green':
            start = 0
            end = 35
            unique_cols.add('CCFW00')
            unique_cols.add('CCFW01')
            unique_cols.add('CCFW02')
            unique_cols.add('CCFW03')
            unique_cols.add('CCFW04')
            unique_cols.add('CCFW05')
            unique_cols.add('CCFW06')
            unique_cols.add('CCFW07')
            unique_cols.add('CCFW08')
            unique_cols.add('CCFW09')
            for i in range(10, 35):
                unique_cols.add(f'CCFW{i}')
            if fiber.lower() == 'sci1':
                fib = 100
                for i in range (100, 135):
                    unique_cols.add(f'RV{i}')
            elif fiber.lower() == 'sci2':
                fib = 200
                for i in range (200, 235):
                    unique_cols.add(f'RV{i}')
            elif fiber.lower() == 'sci3':
                fib = 300
                for i in range (300, 335):
                    unique_cols.add(f'RV{i}')
            else:
                self.logger.error("Need to specify 'fiber'")
        elif chip.lower() == 'red':
            start = 35
            end = 67
            for i in range(35, 67):
                unique_cols.add(f'CCFW{i}')
            if fiber.lower() == 'sci1':
                fib = 100
                for i in range (135, 167):
                    unique_cols.add(f'RV{i}')
            elif fiber.lower() == 'sci2':
                fib = 200
                for i in range (235, 267):
                    unique_cols.add(f'RV{i}')
            elif fiber.lower() == 'sci3':
                fib = 300
                for i in range (335, 367):
                    unique_cols.add(f'RV{i}')
            else:
                self.logger.error("Need to specify 'fiber'")
        else:
            self.logger.error("Need to specify 'chip'")

        rv_df = self.dataframe_from_db(unique_cols, start_date=start_date, end_date=end_date, only_object=only_object, only_source=only_source, 
                                    object_like=object_like, not_junk=True)
        rv_df = rv_df.drop(columns=['NOTJUNK'])
        rv_df = rv_df[['DATE-MID'] + ['ObsID'] + [col for col in rv_df.columns if (col != 'DATE-MID') and (col != 'ObsID')]]
        rv_df.iloc[:, 2:] = rv_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        rv_columns = sorted([col for col in rv_df.columns if col.startswith('RV')], key=lambda x: int(x[2:]))
        ccfw_columns = sorted([col for col in rv_df.columns if col.startswith('CCFW')], key=lambda x: int(x[4:]))
        rv_df = pd.concat([rv_df['DATE-MID'], rv_df['ObsID'],rv_df[rv_columns], rv_df[ccfw_columns]], axis=1)
        rv_df['DATE-MID'] = pd.to_datetime(rv_df['DATE-MID'])
        
        plt.figure(figsize=(10, 20)) 
        for i in range(start, end):
            rv_col = f"RV{fib + i}"
            weight_col = f"CCFW{str(i).zfill(2)}"
            
            valid_indices = rv_df[weight_col] != 0
            times = rv_df.loc[valid_indices, 'DATE-MID']
            rv_values = rv_df.loc[valid_indices, rv_col]

            if not rv_values.empty:
                plt.scatter(times, rv_values + i * 0.01, label=rv_col, alpha=0.7, s=2)

        plt.xlabel("Time")
        plt.ylabel(f"RV OF {rv.upper()}")
        plt.title(f"{rv.upper()} RV FOR {fiber.upper()} {chip.upper()}")
        plt.legend(fontsize=8) 
        plt.grid(True)

        try:
            if fig_path != None:
                t0 = time.process_time()
                plt.savefig(fig_path, dpi=300, facecolor='w')
                if log_savefig_timing:
                    self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
            if show_plot == True:
                plt.show()
            plt.close('all')
        except Exception as e:
            self.logger.info(f"Error saving file or showing plot: {e}")


    def plot_nobs_histogram(self, plot_dict=None, 
                            interval='full', date=None, exclude_junk=False, 
                            only_sources=['all'], only_autocal=False,
                            plot_junk=False, plot_source=False, 
                            fig_path=None, show_plot=False):
        """
        Plot a histogram of the number of observations per day or hour, 
        optionally color-coded by 'NOTJUNK' or 'Source'.
    
        Args:
            interval (string) - time interval over which plot is made
                                default: 'full',
                                possible values: 'full', 'decade', 'year', 'month', 'day'
            date (string) - one date in the interval (format: 'YYYYMMDD' or 'YYYY-MM-DD')
            only_sources (array of strings) - only observations whose Source name matches an element of only_strings are used
                                              possible sources = 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star'
            only_autocal - only observations OBJECT name includes 'autocal' are used
            exclude_junk (boolean) - if True, observations with NOTJUNK=False are removed
            plot_junk (boolean) - if True, will color-code based on 'NOTJUNK' column
            plot_source (boolean) - if True, will color-code based on 'Source' column
            fig_path (string) - set to the path for the file to be generated
            show_plot (boolean) - show the plot in the current environment
            
        Returns:
            PNG plot in fig_path or shows the plot in the current environment
            (e.g., in a Jupyter Notebook). 
        
        To-do: 
            Add highlighting of QC tests
        """
        
        # Use plotting dictionary, if provided
        #   (inspired by dictionaries for plot_time_series_multipanel)
        dict_ylabel = ''
        if plot_dict != None:
            panel_arr = plot_dict['panel_arr']
            if 'ylabel' in ['paneldict']:
                 dict_ylabel = panel_arr[0]['paneldict']['ylabel']
            if 'not_junk' in panel_arr[0]['paneldict']:
                 exclude_junk = bool(panel_arr[0]['paneldict']['not_junk'])
            if 'only_sources' in panel_arr[0]['paneldict']:
                 only_sources = panel_arr[0]['paneldict']['only_sources']
            if 'plot_source' in panel_arr[0]['paneldict']:
                 plot_source = panel_arr[0]['paneldict']['plot_source']

        # Define the source categories and their colors
        source_order = ['Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        source_colors = {
            'Bias':      'gray',
            'Dark':      'black',
            'Flat':      'gainsboro',
            'Wide Flat': 'silver',
            'LFC':       'gold',
            'Etalon':    'chocolate',
            'ThAr':      'orange',
            'UNe':       'forestgreen',
            'Sun':       'cornflowerblue',
            'Star':      'royalblue'
        }
    
        # Load data
        columns = ['DATE-BEG', 'NOTJUNK', 'Source', 'OBJECT']
        df = self.db.dataframe_from_db(columns)
        df['DATE-BEG'] = pd.to_datetime(df['DATE-BEG'], errors='coerce')
        #df['DATE-END'] = pd.to_datetime(df['DATE-END'], errors='coerce')
        df = df.dropna(subset=['DATE-BEG'])
        #df = df.dropna(subset=['DATE-END'])
        start_date = df['DATE-BEG'].dt.date.min()
        end_date   = df['DATE-BEG'].dt.date.max()

        if exclude_junk:
            df = df[df['NOTJUNK'] == 1.0]      
    
        if not ('all' in only_sources):
            df = df[df['Source'].isin(only_sources)]
            
        if only_autocal:
            df = df[df['OBJECT'].str.contains('autocal', na=False)]
        
        # Parse the date string into a timestamp
        if date is not None:
            date = pd.to_datetime(date, format='%Y%m%d', errors='coerce')  # Handle YYYYMMDD format
            if pd.isna(date):
                date = pd.to_datetime(date, errors='coerce')  # Handle other formats like YYYY-MM-DD
            if pd.isna(date):
                raise ValueError(f"Invalid date format: {date}")
    
        # Filter data based on interval
        if interval == 'decade':
            start_date = pd.Timestamp(f"{date.year // 10 * 10}-01-01")
            end_date = pd.Timestamp(f"{date.year // 10 * 10 + 9}-12-31")
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DATE'] = df['DATE-BEG'].dt.date
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = YearLocator()
            major_formatter = DateFormatter("%Y")
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Decade: {start_date.year}-{end_date.year}"
    
        elif interval == 'year':
            start_date = pd.Timestamp(f"{date.year}-01-01")
            end_date = pd.Timestamp(f"{date.year}-12-31")
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DATE'] = df['DATE-BEG'].dt.date
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = MonthLocator()
            major_formatter = DateFormatter("%b")  # Format ticks as month names (Jan, Feb, etc.)
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Year: {date.year}"
    
        elif interval == 'month':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-01")
            end_date = (start_date + pd.offsets.MonthEnd(0) + timedelta(days=1) - timedelta(seconds=0.1))
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DAY'] = df['DATE-BEG'].dt.day    
            #full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            full_range = range(1, end_date.day + 1) 
            entry_counts = df['DAY'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = DayLocator()
            major_formatter = lambda x, _: f"{int(x)}" if 1 <= x <= end_date.day else ""
            minor_locator = None
            column_to_count = 'DAY'
            plot_title = f"Observations (Month: {date.year}-{date.month:02d}"
    
        elif interval == 'day':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-{date.day:02d}")
            end_date = start_date + timedelta(days=1)# - timedelta(seconds=1)
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['HOUR'] = df['DATE-BEG'].dt.hour
            entry_counts = df['HOUR'].value_counts().sort_index()
            hourly_range = pd.Index(range(24))  # 0 through 23 hours
            entry_counts = entry_counts.reindex(hourly_range, fill_value=0)
            major_locator = plt.MultipleLocator(1)  # Tick every hour
            major_formatter = lambda x, _: f"{int(x):02d}:00" if 0 <= x <= 23 else ""
            minor_locator = None
            column_to_count = 'HOUR'
            plot_title = f"Observations (Day: {date.year}-{date.month:02d}-{date.day:02d}"
        else: # Default: 'full' interval
            df['DATE'] = df['DATE-BEG'].dt.date
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = AutoDateLocator()
            major_formatter = DateFormatter("%Y-%m")
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Full Range: {start_date.year}-{start_date.month:02d}-{start_date.day:02d} - {end_date.year}-{end_date.month:02d}-{end_date.day:02d}"
            
            # Ensure full date range is displayed
            full_range = pd.date_range(start=start_date, end=end_date, freq='D')
            entry_counts = entry_counts.reindex(full_range, fill_value=0)

        if not ('all' in only_sources):
            plot_title = plot_title + " - " + ', '.join(only_sources)
        if only_autocal:
            plot_title = plot_title + " - only autocal"
        if exclude_junk:
            plot_title = plot_title + " - junk excluded"
        plot_title = plot_title + ")"
        
        # Ensure all index values are datetime.date for consistent processing
        if isinstance(entry_counts.index, pd.DatetimeIndex):
            entry_counts.index = entry_counts.index.map(lambda x: x.date())
    
        # Adjust bar positions and plot edges for proper alignment
        bar_positions = entry_counts.index.map(
            lambda x: x.toordinal() if type(x) == type(datetime(2024, 1, 1, 1, 1, 1)) else x
        )

        if interval == 'decade':
            x_min = datetime(bar_positions[0].year // 10 * 10, 1, 1)
            x_max = datetime(bar_positions[0].year // 10 * 10 + 10, 1, 1)
        elif interval == 'year':
            x_min = datetime(bar_positions[0].year, 1, 1)
            x_max = datetime(bar_positions[0].year+1, 1, 1)
        elif interval == 'month':
            x_min = 0.5
            x_max = (datetime(date.year, date.month % 12 + 1, 1) - datetime(date.year, date.month, 1)).days + 0.5
        elif interval == 'day':
            x_min = 0 
            x_max = 24
        else: # Default: 'full' interval
            x_min = bar_positions.min() 
            x_max = bar_positions.max() 

        if plot_source and interval == 'day':
            plt.figure(figsize=(12, 4))
        else:
            plt.figure(figsize=(15, 4))
    
        # Plot stacked source data
        if plot_source:
            bottom_values = [0] * len(bar_positions)
            legend_labels = []  # Store labels with counts for the legend
            for source in source_order:
                source_counts = df[df['Source'] == source][column_to_count].value_counts().sort_index()
                source_counts = source_counts.reindex(entry_counts.index, fill_value=0)
    
                if interval == 'day':
                    plt.bar(bar_positions, source_counts.values, width=1, align='edge',
                            color=source_colors[source], label=source, bottom=bottom_values, zorder=3)
                else: 
                    plt.bar(bar_positions, source_counts.values, width=1, align='center',
                            color=source_colors[source], label=source, bottom=bottom_values, zorder=3)
                bottom_values = [b + s for b, s in zip(bottom_values, source_counts.values)]

                # Add source label with count for 'day' interval
                if interval in ['month', 'day']:
                    total_source_count = source_counts.sum()
                    legend_labels.append(f"{source} ({int(total_source_count)})")
                else:
                    legend_labels.append(source)

            # Place legend outside of the plot on the right
            handles, _ = plt.gca().get_legend_handles_labels()
            handles = handles[::-1]  # Reverse handles to match legend_labels order
            legend_labels = legend_labels[::-1]  # Reverse labels for proper order

            plt.legend(
                handles, legend_labels,  # Use updated labels
                title="Sources",
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),  # Adjust legend position (to the right of the plot)
                fontsize=10
            )
            plt.gcf().set_size_inches(15, 4)  # Increase the figure width
        elif plot_junk:
            notjunk_counts = df[df['NOTJUNK'] == True][column_to_count].value_counts().sort_index()
            junk_counts    = df[df['NOTJUNK'] == False][column_to_count].value_counts().sort_index()
            notjunk_counts = notjunk_counts.reindex(entry_counts.index, fill_value=0)
            junk_counts    = junk_counts.reindex(entry_counts.index, fill_value=0)

            if interval == 'day':
                plt.bar(bar_positions, notjunk_counts.values, width=1, align='edge', color='green', label='Not Junk', zorder=3)
                plt.bar(bar_positions, junk_counts.values,    width=1, align='edge', color='red',   label='Junk',     zorder=3, bottom=notjunk_counts.values)
            else: 
                plt.bar(bar_positions, notjunk_counts.values, width=1, align='center', color='green', label='Not Junk', zorder=3)
                plt.bar(bar_positions, junk_counts.values,    width=1, align='center', color='red',   label='Junk',     zorder=3, bottom=notjunk_counts.values)

            plt.legend()
        else:
            if interval == 'day':
                plt.bar(bar_positions, entry_counts.values, width=1, align='edge', zorder=3)
            else: 
                plt.bar(bar_positions, entry_counts.values, width=1, align='center', zorder=3)
        if interval == 'day':
            plt.xlabel("Hour of Day (UT)", fontsize=14)
            
        elif interval == 'month':
            plt.xlabel("Day of Month", fontsize=14)
        else:
            plt.xlabel("Date", fontsize=14)
        if dict_ylabel != '':
            plt.ylabel(dict_ylabel, fontsize=14)
        else:
            plt.ylabel("Number of Observations", fontsize=14)
        plt.title(plot_title, fontsize=18)
    
        ax = plt.gca()
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        if interval == 'day':
            ax.xaxis.set_tick_params(labelsize=9)
        else:
            ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        if minor_locator:
            ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(visible=True, which='major', axis='both', linestyle='--', color='lightgray', zorder=1)
        ax.set_axisbelow(True)
        ax.set_xlim(x_min, x_max)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot area to leave space for the legend
    
        # Add black box around the axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
            spine.set_zorder(4)
            spine.set_visible(True)
    
        # Save or show the plot
        if fig_path is not None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot:
            plt.show()
        plt.close('all')


    def yaml_to_dict(self, yaml_or_path):
        """
        Read a plotting configuration from either a YAML file or a YAML string.
        
        1) If `yaml_or_path` is a valid file path, open that file and parse.
        2) Otherwise, treat `yaml_or_path` as a YAML string and parse it directly.
        """
        if os.path.isfile(yaml_or_path):
            # It's an actual file path on disk
            with open(yaml_or_path, 'r') as f:
                plotdict = yaml.safe_load(f)
        else:
            # It's a (multi-line) YAML string
            plotdict = yaml.safe_load(yaml_or_path)
    
        return plotdict
    
    
    def plot_all_quicklook(self, start_date=None, interval=None, clean=True, 
                           last_n_days=None, 
                           fig_dir=None, show_plot=False, 
                           print_plot_names=False, verbose=False):
        """
        Generate all of the standard time series plots for the quicklook.  
        Depending on the value of the input 'interval', the plots have time ranges 
        that are daily, weekly, yearly, or decadal.

        Args:
            start_date (datetime object) - start date for plot
            interval (string) - 'day', 'month', 'year', or 'decade'
            last_n_days (int) - overrides start_date and makes a plot over the last n days
            fig_path (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.
            print_plot_names (boolean) - prints the names of possible plots and exits

        Returns:
            PNG plot in fig_path or shows the plots it the current environment
            (e.g., in a Jupyter Notebook).
        """

        plots = {}

        import static.tsdb_plot_configs
        yaml_paths = static.tsdb_plot_configs.all_yaml # an attribute from static/tsdb_plot_configs/__init__.py
        
        for this_yaml_path in yaml_paths:
            thisplotconfigdict = self.yaml_to_dict(this_yaml_path)
            plot_name = str.split(str.split(this_yaml_path,'/')[-1], '.')[0]
            subdir = str.split(os.path.dirname(this_yaml_path),'/')[-1]
            tempdict = {
                "plot_name": plot_name,
                "plot_type": thisplotconfigdict.get("plot_type", ""),
                "subdir": subdir,
                "description": thisplotconfigdict.get("description", ""),
                "panel_arr": thisplotconfigdict["panel_arr"],
            }
            plots[plot_name] = tempdict

        if print_plot_names:
            print("Plots available:")
            for p in plots:
                print("    '" + plots[p]["plot_name"] + "': " + plots[p]["description"])
            return

        if (last_n_days != None) and (type(last_n_days) == type(1)):
            now = datetime.now()
            if last_n_days > 3:
                end_date = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                end_date = now
            start_date = end_date - timedelta(days=last_n_days)

        if not isinstance(start_date, datetime):
            self.logger.error("'start_date' must be a datetime object.")
            return        
        
        for p in plots:
            plot_name = plots[p]["plot_name"]
            if verbose:
                self.logger.info(f"AnalyzeTimeSeries.plot_all_quicklook: making {plot_name}")

            # Set filename 
            if plots[p]['plot_type'] == 'time_series_multipanel':
                type_string = '_ts_'
            elif plots[p]['plot_type'] == 'nobs_histogram':
                type_string = '_nobs_'

            if interval == 'day':
                end_date = start_date + timedelta(days=1)
                filename = 'kpf_' + start_date.strftime("%Y%m%d") + type_string + plot_name + '.png' 
            elif interval == 'month':
                end_date = add_one_month(start_date)
                filename = 'kpf_' + start_date.strftime("%Y%m") + type_string + plot_name + '.png' 
            elif interval == 'year':
                end_date = datetime(start_date.year+1, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y") + type_string + plot_name + '.png' 
            elif interval == 'decade':
                end_date = datetime(start_date.year+10, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y")[0:3] + '0' + type_string + plot_name + '.png' 
            elif (last_n_days != None) and (type(last_n_days) == type(1)):
                filename = 'kpf_last' + str(last_n_days) + 'days' + type_string + plot_name + '.png'                 
            else:
                self.logger.error("The input 'interval' must be 'daily', 'weekly', 'yearly', or 'decadal'.")
                return

            if fig_dir != None:
                if not fig_dir.endswith('/'):
                    fig_dir += '/'
                savedir = fig_dir + plots[p]["subdir"] + '/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                fig_path = savedir + filename
                self.logger.info('Making ' + fig_path)
            else:
                fig_path = None

            # Make Plot
            plot_dict = plots[p]
            if plot_dict['plot_type'] == 'time_series_multipanel':
                try:
                    self.plot_time_series_multipanel(plot_dict, 
                                                     start_date=start_date, 
                                                     end_date=end_date, 
                                                     fig_path=fig_path, 
                                                     show_plot=show_plot, 
                                                     clean=clean)
                except Exception as e:
                    self.logger.error(f"Error while plotting {plot_name}: {e}")
                    continue  # Skip to the next plot
            elif plot_dict['plot_type'] == 'nobs_histogram':        
                try:
                    self.plot_nobs_histogram(plot_dict=plot_dict, 
                                             date=start_date.strftime('%Y%m%d'), 
                                             interval=interval,
                                             fig_path=fig_path, 
                                             show_plot=show_plot)
                except Exception as e:
                    self.logger.error(f"Error while plotting {plot_name}: {e}")
                    continue  # Skip to the next plot


    def print_df_with_obsid_links(self, df, url_stub='https://jump.caltech.edu/observing-logs/kpf/', nrows=None):
        '''
        Print a dataframe with links to a web page. 
        The default page is set to "Jump", the portal used by the KPF Science Team.
        The printed table will be sortable by clicking on column headers.
        '''
        df_copy = df.copy()  # Make a copy to avoid modifying the original DataFrame
        
        # Convert ObsID into clickable links
        df_copy['ObsID'] = df_copy['ObsID'].apply(
            lambda obsid: f'<a href="{url_stub}{obsid}" target="_blank">{obsid}</a>'
        )
        
        # Limit number of rows if requested
        if nrows is None:
            limited_df = df_copy
        else:
            limited_df = df_copy.head(nrows)
        
        # Generate the HTML for the table
        html = limited_df.to_html(escape=False, index=False, classes='sortable')
        
        # JavaScript for making the table sortable
        sortable_script = """
        <script>
          function sortTable(table, col, reverse) {
            const tb = table.tBodies[0],
              tr = Array.from(tb.rows),
              i = col;
            reverse = -((+reverse) || -1);
            tr.sort((a, b) => reverse * (a.cells[i].textContent.trim().localeCompare(b.cells[i].textContent.trim(), undefined, {numeric: true})));
            for(let row of tr) tb.appendChild(row);
          }
          document.querySelectorAll('table.sortable th').forEach(th => th.addEventListener('click', (() => {
            const table = th.closest('table');
            Array.from(table.querySelectorAll('th')).forEach((th, idx) => th.addEventListener('click', (() => sortTable(table, idx, this.asc = !this.asc))));
          })));
        </script>
        """
        
        # Display the combined table + script
        display(HTML(html + sortable_script))


    def print_log_error_report(self, df, log_dir='/data/logs/', aggregated_summary=False):
        '''
        For each ObsID in the dataframe, open the corresponding log file,
        find all lines containing [ERROR]:, and print either:
        - aggregated error report (if aggregated_summary=True)
        - individual ObsID error reports (if aggregated_summary=False)
        '''
        error_counter = Counter()  # Collect error bodies for aggregation
    
        for obsid in df['ObsID']:
            log_path = os.path.join(log_dir, f'{get_datecode(obsid)}/{obsid}.log')
            
            if not os.path.isfile(log_path):
                if not aggregated_summary:
                    print(f"ObsID: {obsid}")
                    print(f"Log file: {log_path}")
                    print(f"Log file not found.\n")
                continue
            
            mod_time = datetime.utcfromtimestamp(os.path.getmtime(log_path)).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            error_lines = []
            with open(log_path, 'r') as file:
                for line in file:
                    if '[ERROR]:' in line:
                        error_line = line.strip()
                        error_lines.append(error_line)
    
                        # Extract only the part after [ERROR]:
                        parts = error_line.split('[ERROR]:', 1)
                        if len(parts) > 1:
                            error_body = parts[1].strip()
                            error_counter[error_body] += 1
                        else:
                            error_counter[error_line] += 1
            
            if not aggregated_summary:
                # Print individual ObsID report
                print(f"ObsID: {obsid}")
                print(f"Log file: {log_path}")
                print(f"Log modification date: {mod_time}")
                print(f"Errors in log file:")
                
                if error_lines:
                    for error in error_lines:
                        print(f"    {error}")
                else:
                    print(f"    No [ERROR] lines found.")
                
                print("\n" + "-" * 50 + "\n")
        
        # After processing all ObsIDs, print the aggregated summary if requested
        if aggregated_summary:
            if error_counter:
                print("\nAggregated Error Summary:\n")
                
                summary_df = pd.DataFrame(
                    [(count, error) for error, count in error_counter.items()],
                    columns=['Count', 'Error Message']
                ).sort_values('Count', ascending=False).reset_index(drop=True)
                
                # Set wide display options for Pandas
                pd.set_option('display.max_colwidth', None)
                pd.set_option('display.width', 0)
                
                # Force all table cells to left-align with inline CSS
                html = summary_df.to_html(index=False, escape=False)
                html = html.replace('<td>', '<td style="text-align: left; white-space: normal; word-wrap: break-word;">')
                html = html.replace('<th>', '<th style="text-align: left; white-space: normal; word-wrap: break-word;">')
                
                display(HTML(html))
            else:
                print("No [ERROR] lines found across all logs.")


def add_one_month(inputdate):
    """
    Add one month to a datetime object, accounting for the number of days per month.
    """
    year, month, day = inputdate.year, inputdate.month, inputdate.day
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    if month in [4, 6, 9, 11] and day > 30:
        day = 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            if day > 29:
                day = 29
        else:
            if day > 28:
                day = 28
    
    outputdate = datetime(year, month, day)
    return outputdate

def is_numeric(value):
    if value is None:  # Explicitly handle NoneType
        return False
    try:
        float(value)  # Attempt to convert to float
        return True
    except (ValueError, TypeError):
        return False
