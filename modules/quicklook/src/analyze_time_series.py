import os
import re
import ast
import time
import copy
from astropy.time import Time
import yaml
import numpy as np
import pandas as pd
import operator
from datetime import datetime, timedelta, date
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D 
from matplotlib.ticker import FuncFormatter, LogLocator
from modules.Utils.utils import get_sunrise_sunset_ut
from modules.Utils.kpf_parse import get_datecode
from collections import Counter
from matplotlib.dates import DayLocator, MonthLocator, YearLocator, AutoDateLocator, DateFormatter

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
        tables_prefix (str) - prefix of the table names; default = 'tsdb_'.
        backend (string; 'sqlite' or 'psql') - database format 
        credentials (dictionary or None; optional) - optionally pass credentials for a PostgreSQL database
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
    """

    def __init__(self, db_path='kpf_ts.db', base_dir='/data/L0', tables_prefix='tsdb_', backend='sqlite', credentials=None, logger=None, verbose=False):
       
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Starting AnalyzeTimeSeries')
        self.db = TSDB(backend=backend, 
                       db_path=db_path, 
                       base_dir=base_dir, 
                       tables_prefix=tables_prefix, 
                       credentials=credentials, 
                       logger=logger, 
                       verbose=verbose)


    def plot_time_series_multipanel(self, plotdict, 
                                    start_date=None, end_date=None, 
                                    hatch_service_missions=True,
                                    clean=False, 
                                    fig_path=None, show_plot=False, 
                                    log_savefig_timing=False):
        """
        Generate a multi-panel time series plot from a KPF TSDB. Each subplot is configured
        via a dict (or YAML file path), enabling control over filters, transforms, and style.
    
        Parameters
        ----------
        plotdict : str or dict
            Path to a named YAML config or a dict with key 'panel_arr' (list of panel dicts).
        start_date, end_date : datetime, optional
            Query window (UT). Defaults if None: start=2020-01-01, end=2040-01-01. The code
            may tighten to the data’s min/max timestamps.
        hatch_service_missions : bool, default=True
            Overlay hatched spans from self.get_service_mission_df() (UT_start_date, UT_end_date).
        clean : bool, default=False
            Apply self.db.clean_df() to remove outliers.
        fig_path : str, optional
            Full output path (PNG).
        show_plot : bool, default=False
            Show the figure interactively.
        log_savefig_timing : bool, default=False
            Log CPU time for savefig().
    
        Plot Configuration (via panel_dict or YAML)
        -------------------------------------------
        plotdict['panel_arr'] : list[dict]
            Each element defines one panel and includes:
    
            - 'paneldict' : dict   # panel-level behavior/filters
                - 'only_object' : str | list[str]          # exact OBJECT names
                - 'object_like' : str | list[str]          # LIKE patterns (nested lists ok)
                - 'only_source' : str                      # passed to DB filter layer
                - 'not_junk' : bool | {'true','false'}     # filter on NOTJUNK
                - 'qc_pass' : str | list[str]              # columns that must be True
                - 'qc_fail' : str | list[str]              # columns that must be False
                - 'qc_not_pass' : str | list[str]          # columns that are not True (False/NaN)
                - 'qc_not_fail' : str | list[str]          # columns that are not False (True/NaN)
                - 'on_sky' : bool | {'true','false'}       # True→FIUMODE=='Observing', False→'Calibration'
                - 'ylabel' : str                           # label for vertical axis
                - 'ylim' : tuple | str                     # (ymin, ymax) or a string that evals to that
                - 'ymin', 'ymax' : float                   # override parts of ylim
                - 'yscale' : str                           # e.g., 'log'
                - 'subtractmedian' : bool                  # subtract per-variable median before plotting
                - 'nolegend' : bool                        # suppresses legend
                - 'labelrms' : bool                        # add legend text like ""(0.001 C rms)"
                - 'legend_frac_size' : float               # legend anchor offset
                - 'axhspan' : dict                         # {key: {'ymin','ymax','color','alpha'}}
                - 'title' : str                            # title for a set of panels
                - 'narrow_xlim_daily' : bool               # shrink x-limits to data for day-scale plots
    
            - 'panelvars' : list[dict]   # variables drawn in this panel
                - 'col' : str                                # main data column
                - 'col_err' : str                            # symmetric error column (optional)
                - 'col_subtract' : str                       # subtract this column from 'col'
                - 'col_multiply' : float                     # scalar multiplier
                - 'col_offset' : float                       # scalar offset
                - 'normalize' : bool                         # divide by median after transforms
                - 'plot_type' : {                            # determine plot type
                     'scatter',                              # scatter plot (default)
                     'errorbar',                             # errorbar plot; must include 'col_err'
                     'plot',                                 # line plot
                     'step',                                 # step plot
                     'state',                                # state value plot with distinct values, usually strings or booleans
                     'vlines'                                # plot with vertical lines; must include 'col_min' and 'col_max'
                     }
                - 'plot_attr' : dict                         # matplotlib kwargs (marker, label, etc.)
                - 'unit' : str                               # used when augmenting legend label with RMS
                - 'col_min','col_max' : str                  # required for plot_type='vlines'
                - 'vline_pt_color' : str                     # optional color for vline end points
    
        Returns
        -------
        None
            Saves to fig_path (if given), optionally shows, and logs as configured.
    
        Notes
        -----
        - Time axis adapts to span:
            * ~1 day: hours since start (UT/HST labels possible), optional “Night” shading
            * <3 days or <32 days: days since start
            * 28–31 days: month view (day numbers)
            * ~1 year: month tick marks with MM-DD labels
            * longer: calendar dates (YYYY-MM-DD)
        - 'state' plots render categorical levels; {0,1,None}→{Fail,Pass,None}. If ylabel=='Junk Status',
          {Pass,Fail}→{Not Junk,Junk}.
        - Empty selections are annotated “No Data”.
        - Labels may be augmented with “(X unit rms)” when legend shown and enough points exist.
        - Data come from self.db.dataframe_from_db(...); DATE-MID parsed and sorted; clean_df() optional.
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

        def normalize_list_input(value):
            if value is None:
                return None
            if isinstance(value, str):
                return [v.strip() for v in value.split(',')]
            if isinstance(value, list):
                return value
            return [value]

        def log_tick_formatter(y, pos):
            """
            Format log-scale ticks:
              - For y >= 1e-2: plain numeric (1, 0.1, 0.01)
              - For y < 1e-2: mathtext, e.g. 10^{-4}
            """
            if y <= 0:
                return ""
        
            exp = int(np.round(np.log10(y)))
        
            # If y is exactly a power of 10, use 10^{exp} notation for small values
            if np.isclose(y, 10**exp):
                if exp >= -2:
                    # show as plain number (strip trailing zeros)
                    return f"{y:g}"
                else:
                    # show as 10^{exp} for very small values
                    return rf"$10^{{{exp}}}$"
            else:
                # For non-powers-of-10 ticks (e.g. minor ticks), usually show nothing
                return ""


        # Helper: convert absolute datetimes to current axis coordinates
        def _to_axis_x(dt, mode):
            if mode == 'hours':
                return (dt - start_date).total_seconds() / 3600.0
            elif mode in ('days', 'days32', 'year'):
                return (dt - start_date).total_seconds() / 86400.0
            else:  # 'datetime'
                return dt

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
                if 'col' in d:
                    unique_cols.add(d['col'])
                if 'col_err' in d:
                    unique_cols.add(d['col_err'])
                if 'col_subtract' in d:
                    unique_cols.add(d['col_subtract'])
                if 'col_min' in d:
                    unique_cols.add(d['col_min'])
                if 'col_max' in d:
                    unique_cols.add(d['col_max'])

        fig, axs = plt.subplots(npanels, 1, sharex=True, figsize=(15, npanels*2.5+1), tight_layout=True)
        if npanels == 1:
            axs = [axs]  # Make axs iterable even when there's only one panel
        if npanels > 1:
            plt.subplots_adjust(hspace=0)

        overplot_night_box = False
        no_data = True # for now; will be set to False when data is detected
        for p in np.arange(npanels):
            thispanel = panel_arr[p]            
            not_junk = thispanel['paneldict'].get('not_junk', plotdict.get('not_junk', None))
            if isinstance(not_junk, str):
                not_junk = True if not_junk.lower() == 'true' else False if not_junk.lower() == 'false' else not_junk
            only_object = thispanel['paneldict'].get('only_object', plotdict.get('only_object', None))
            only_source = thispanel['paneldict'].get('only_source', plotdict.get('only_source', None))
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

            qc_pass      = normalize_list_input(thispanel['paneldict'].get('qc_pass', None))
            qc_fail      = normalize_list_input(thispanel['paneldict'].get('qc_fail', None))
            qc_not_pass  = normalize_list_input(thispanel['paneldict'].get('qc_not_pass', None))
            qc_not_fail  = normalize_list_input(thispanel['paneldict'].get('qc_not_fail', None))
            
            if start_date == None:
                start_date = datetime(2020, 1,  1)
                start_date_was_none = True
            else:
                start_date_was_none = False
            if end_date == None:
                end_date = datetime(2040, 1,  1)
                end_date_was_none = True
            else:
                end_date_was_none = False

            # Get data from database
            df = self.db.dataframe_from_db(list(unique_cols), 
                                           start_date=start_date, 
                                           end_date=end_date, 
                                           not_junk=not_junk, 
                                           only_object=only_object, 
                                           object_like=object_like,
                                           only_source=only_source, 
                                           qc_pass=qc_pass,
                                           qc_fail=qc_fail,
                                           qc_not_pass=qc_not_pass,
                                           qc_not_fail=qc_not_fail,
                                           verbose=False)

        	# Check if the resulting dataframe has any rows
            empty_df = (len(df) == 0) # True if the dataframe has no rows
            if not empty_df:
                df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?
                df = df.dropna(subset=['DATE-MID'])
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
                    
            # Determine how to display time
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
                time_mode = 'hours'
                
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
                time_mode = 'hours'
            elif abs((end_date - start_date).days) <= 3:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
                time_mode = 'days'
            elif 28 <= (end_date - start_date).days <= 31:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = start_date.strftime('%B %Y') + ' (UT Times)'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                
                axs[p].set_xlim(0, (end_date - start_date).days)
                axs[p].xaxis.set_major_locator(ticker.MultipleLocator(1))  # tick every 1 day
                time_mode = 'days'
            
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
                time_mode = 'days32'
            elif 360 <= (end_date - start_date).days <= 370:
                if not empty_df:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = start_date.strftime('%Y')
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].set_xlim(0, (end_date - start_date).days)
                time_mode = 'year'

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
                time_mode = 'datetime'
            if p == npanels-1: 
                axs[p].set_xlabel(xtitle, fontsize=14)
                axs[0].set_title(thistitle, fontsize=18)
            if 'ylabel' in thispanel['paneldict']:
                axs[p].set_ylabel(thispanel['paneldict']['ylabel'], fontsize=14)
            axs[p].grid(color='lightgray')        

            # Annotate panel with QC filters
            x_offset = -2  # starting horizontal offset in points
            y_offset = 2  # starting vertical offset in points
            for label, qc_value in [('QC_pass', qc_pass), ('QC_fail', qc_fail), 
                                    ('QC_not_pass', qc_not_pass), ('QC_not_fail', qc_not_fail)]:
                if qc_value is not None:
                    axs[p].annotate(f"{label}: {qc_value} ", xy=(1, 0), xycoords='axes fraction',
                                    fontsize=8, color='darkslategray', ha='right', va='bottom',
                                    xytext=(x_offset, y_offset), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, edgecolor='none'))
                    y_offset += 10  # stack next line above
            if not_junk is True:
                axs[p].annotate("Junk excluded", xy=(1, 0), xycoords='axes fraction',
                                fontsize=8, color='darkslategray', ha='right', va='bottom',
                                xytext=(x_offset, y_offset), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, edgecolor='none'))
                    
            if 'yscale' in thispanel['paneldict']:
                if thispanel['paneldict']['yscale'] == 'log':
                    axs[p].set_yscale('log')
            
                    # Major ticks: powers of 10
                    axs[p].yaxis.set_major_locator(LogLocator(base=10.0))
            
                    # Minor ticks: 2–9 * 10^exp
                    axs[p].yaxis.set_minor_locator(
                        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
                    )
            
                    # Use the custom formatter
                    axs[p].yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            
                    axs[p].minorticks_on()
                    axs[p].grid(which='major', axis='x', color='darkgray', linestyle='-', linewidth=0.5)
                    axs[p].grid(which='both',  axis='y', color='lightgray', linestyle='-', linewidth=0.5)
            else:
                axs[p].grid(color='lightgray')

            # Set y-axis limits
            ylim=False
            if 'ylim' in thispanel['paneldict']:
                if type(ast.literal_eval(thispanel['paneldict']['ylim'])) == type((1,2)):
                    ylim = ast.literal_eval(thispanel['paneldict']['ylim'])
            if ('ymin' in thispanel['paneldict']) or ('ymax' in thispanel['paneldict']):
                ymin_current, ymax_current = axs[p].get_ylim()
                ymin = thispanel['paneldict'].get('ymin', ymin_current)
                ymax = thispanel['paneldict'].get('ymax', ymax_current)
                ylim = (ymin, ymax)

            # Determine if legend should be made
            makelegend = True
            if 'nolegend' in thispanel['paneldict']:
                if str(thispanel['paneldict']['nolegend']).lower() == 'true':
                    makelegend = False

            # Determine if RMS values should be included in legend
            labelrms = False
            if 'labelrms' in thispanel['paneldict']:
                if str(thispanel['paneldict']['labelrms']).lower() == 'true':
                    labelrms = True

            # Subtract median from data
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
                    # for vlines plots:
                    if 'col_min' in thispanel['panelvars'][i]:
                        col_name_min = thispanel['panelvars'][i]['col_min']
                        col_name_max = thispanel['panelvars'][i]['col_max']
                        col_data_min = df[col_name_min]
                        col_data_max = df[col_name_max]
                    
                    # Subtract one column from another
                    if 'col_subtract' in thispanel['panelvars'][i]:
                        col_subtract_name = thispanel['panelvars'][i]['col_subtract']
                        # Now filter out invalid values in col_subtract_name,
                        # and also re-filter col_name because removing rows re-indexes the DataFrame.
                        df = df[~df[col_subtract_name].isin(['NaN', 'null', 'nan', 'None', None, np.nan])]
                        # Re-grab the series after dropping rows
                        col_data = df[col_name]
                        col_subtract_data = df[col_subtract_name]
                        col_data_replaced = col_data - col_subtract_data
    
                    # Multiply one column by a common factor
                    if 'col_multiply' in thispanel['panelvars'][i]:
                        col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') * thispanel['panelvars'][i]['col_multiply']
    
                    # Add a common offset to a column
                    if 'col_offset' in thispanel['panelvars'][i]:
                        col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') + thispanel['panelvars'][i]['col_offset']
    
                    # Use error bars
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

                    elif plot_type == 'vlines':
                        data_min = np.array(col_data_min, dtype='float')
                        data_max = np.array(col_data_max, dtype='float')
                        data = data_min # so that some logic below works

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
                                            if labelrms:
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
                    
                    # Plot type: vertical lines
                    if plot_type == 'vlines':
                        if 'vline_pt_color' in thispanel['panelvars'][i]:
                            vline_pt_color = thispanel['panelvars'][i]['vline_pt_color']
                        else:
                            vline_pt_color = plot_attributes.pop('color', 'black')
                        lw = 0.5
                        sz = 5
                        if not empty_df:
                            if abs((end_date - start_date).days) <= 3:
                                lw = 1
                                sz = 15
                            elif abs((end_date - start_date).days) < 32:
                                lw = 2
                                sz = 10
                        # sanitize unsupported attributes
                        for k in ['marker', 'markersize', 'linestyle']:
                            plot_attributes.pop(k, None)
                        plot_attributes.setdefault('colors', vline_pt_color)
                        plot_attributes.setdefault('linewidths', lw)
                        axs[p].vlines(t, data_min, data_max, **plot_attributes)

                        # Add points to the tops of the lines
                        axs[p].scatter(t, data_max, color=plot_attributes.get('colors', 'black'), s=sz, zorder=3)
                        
                        # Optionally add points to the bottoms of the lines
                        axs[p].scatter(t, data_min, color=plot_attributes.get('colors', 'black'), s=sz, zorder=3)
                    
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
                            # Unique categories for generic state plots
                            unique_states = list(set(unique_states))
                        
                            # --- Version-aware ordering for git tags like v2.10.3 ---
                            # Handles labels that *start* with vMAJ.MIN.PATCH (ignore any trailing text)
                            version_pattern = re.compile(r"^v(\d+)\.(\d+)\.(\d+)")
                        
                            def parse_version(state):
                                """
                                Return (major, minor, patch) as ints if state looks like 'vX.Y.Z',
                                otherwise return None.
                                """
                                s = str(state)
                                m = version_pattern.match(s)
                                if not m:
                                    return None
                                return tuple(int(part) for part in m.groups())
                        
                            # If *all* states look like git-style tags, sort numerically; otherwise lexicographically.
                            if unique_states and all(parse_version(s) is not None for s in unique_states):
                                # Numeric development order: v2.5.3 < v2.6.0 < v2.8.2 < v2.9.1 < v2.10.3 < ...
                                unique_states = sorted(unique_states, key=parse_version)
                            else:
                                # Fallback: normal alphabetical order
                                unique_states = sorted(unique_states)
                        
                            # Map states -> y positions according to the chosen order
                            state_to_num = {state: i for i, state in enumerate(unique_states)}
                            mapped_states = [state_to_num[state] for state in states]
                        
                            # Color map for generic state plots
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
                            self.logger.info(f"Length mismatch: states has {len(states)} elements, t has {len(t)}")
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

            # Hatch service mission intervals on the x-axis 
            if hatch_service_missions:
                try:
                    df_sm = self.get_service_mission_df()
                except Exception as _e:
                    df_sm = None
                if df_sm is not None and len(df_sm) > 0:
                    # Clip to visible window and draw spans
                    for _, row in df_sm.iterrows():
                        s = pd.to_datetime(row['UT_start_date'])
                        e = pd.to_datetime(row['UT_end_date'])
                        # clip to axis window
                        s_clip = max(s, start_date)
                        e_clip = min(e, end_date)
                        if e_clip <= s_clip:
                            continue
                        x0 = _to_axis_x(s_clip, time_mode)
                        x1 = _to_axis_x(e_clip, time_mode)
                        mid_x = pd.to_datetime(row['UT_start_date']) + (pd.to_datetime(row['UT_end_date'])-pd.to_datetime(row['UT_start_date']))/2
                        if ylim:
                            ymin = ylim[0]
                            ymax = ylim[1]
                        else:
                            ymin, ymax = axs[p].get_ylim()
                        y0_9 = ymin + 0.9 * (ymax - ymin)
                        if 'yscale' in thispanel['paneldict']:
                            if thispanel['paneldict']['yscale'] == 'log':
                                y0_9 = 0.6 * ymax  
                        # Hatched vertical span
                        axs[p].axvspan(
                            x0, x1,
                            facecolor='none',           # keep data visible
                            hatch='////',
                            edgecolor='dimgray',
                            linewidth=0.0,
                            alpha=0.4,
                            zorder=0.2
                        )
                        label = row.get('name', '')
                        if pd.notna(label):
                            axs[p].text(mid_x, y0_9, 
                                        str(label), 
                                        ha='center', 
                                        va='center', 
                                        fontsize=8, 
                                        color='darkgray', 
                                        zorder=0.3,
                                        bbox=dict(facecolor='white', 
                                                  edgecolor='none', 
                                                  boxstyle='round,pad=0.2'
                                        )
                            )

            # Draw translucent boxes
            if 'axhspan' in thispanel['paneldict']:
                # This is needed so that ylim autoscaling is based on data and not the boxes below
                axs[p].relim()            
                axs[p].autoscale_view()   
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
                        pairs = list(zip(labels, handles))  # (label, handle)
                        
                        # Match vMAJ.MIN.PATCH at the start of the label, ignore anything after that
                        version_pattern = re.compile(r"^v(\d+)\.(\d+)\.(\d+)")
                        
                        def version_components(label):
                            """
                            Return (major, minor, patch) as ints if label starts with 'vX.Y.Z',
                            otherwise return None.
                            """
                            m = version_pattern.match(label)
                            if not m:
                                return None
                            return tuple(int(part) for part in m.groups())
                        
                        def sort_key(label):
                            vc = version_components(label)
                            if vc is not None:
                                # (0, major, minor, patch, label) → versions first, numeric order
                                return (0, vc[0], vc[1], vc[2], label.lower())
                            # non-version labels go after, alphabetically
                            return (1, label.lower())
                        
                        # Sort (label, handle) pairs using the mixed key
                        pairs.sort(key=lambda lh: sort_key(lh[0]))
                        
                        sorted_labels, sorted_handles = zip(*pairs)
                        
                        axs[p].legend(
                            sorted_handles,
                            sorted_labels,
                            loc="upper right",
                            bbox_to_anchor=(1 + legend_frac_size, 1),
                        )

            # Set y-axis limits
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


    def get_service_mission_df(self, sm_csv='/code/KPF-Pipeline/static/service_mission_definitions.csv', debug=False):
        """
        Return a dataframe with start and stop times of the service missions.
        This is used to make hatched regions in time series plots.
        """

        if os.path.exists(sm_csv):
            try:
                df_sm = pd.read_csv(sm_csv)
                if debug:
                    self.logger.debug(f'Read the Service Missions file {sm_csv}.')
                df_sm['UT_start_date'] = pd.to_datetime(df_sm['UT_start_date'])
                df_sm['UT_end_date']   = pd.to_datetime(df_sm['UT_end_date'])
                return df_sm

            except Exception as e:
                self.logger.error(f"Exception: {e}")
                return None
        else:
            self.logger.error(f"The file {sm_csv} does not exist.")
            return None

    
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


    def plot_observing_rv_time_series(self, starname=None,
                                      start_date=None, end_date=None, 
                                      panels=['rv'], annotate=None,
                                      hatch_service_missions=True,
                                      qc_not_fail='auto', not_junk=True, clean=False, 
                                      log_savefig_timing=False, plot_timestamp=False,
                                      fig_path=None, show_plot=False):
        """
        Plot observing-time RV series with optional auxiliary panels on a shared UTC x-axis.
    
        Panels (top->bottom in the order provided via `panels`):
          - 'rv'      : CCFRV  CCFERV (km/s -> m/s), median-subtracted.
          - 'guiding' : GDRXRMS & GDRYRMS (scatter), labels: "X-errors", "Y-errors".
          - 'seeing'  : GDRSEEV (scatter), label: "Seeing (V-band)".
          - 'snr'     : SNRSC452/548/652/747/852 (scatter), labels: "452 nm", ..., "852 nm".
          - 'sun'     : SUNALT (scatter), y fixed to [-90, 0] deg; background bands:
                           -12..0  red (alpha 0.3),  -18..-12 orange (alpha 0.3).
          - 'moon'    : MOONSEP (scatter), y fixed to [0, 180] deg; background band:
                            0..30  orange (alpha 0.3).
          - 'el'      : EL (scatter), y fixed to [0, 90] deg; background band:
                            0..30  orange (alpha 0.1).
    
        Colors:
          rv='tab:blue'; guiding_x='mediumslateblue'; guiding_y='cornflowerblue';
          seeing='mediumseagreen';
          snr_452='royalblue', snr_548='forestgreen', snr_652='crimson',
          snr_747='darkorange', snr_852='goldenrod';
          sun='slategray'; moon='tab:blue'; el='tab:purple'.
    
        Date handling:
          - `start_date`/`end_date` accept str/date/datetime/None.
          - If only one bound is supplied, the other is mirrored to the same value
            (so the DB is asked for that single day/instant; see your DB semantics).
          - date-only -> UT midnight; tz-aware datetimes -> converted to naive UTC.
          - If start > end, they are swapped.
    
        X-axis bounds:
          - Locked to the min/max of the returned datas `time_utc` (with a small pad),
            so decorative spans (e.g., service missions) do not stretch the view.
          - If there is only one timestamp, 12 hours are shown.
    
        Panel heights:
          - `panel_heights` defines *actual* relative axes heights (gridspec height_ratios).
          - Figure height is computed as: sum(height_ratios) + EXTRA_PAD_INCH,
            where EXTRA_PAD_INCH0.7 provides room for title and bottom x-label.
    
        Other:
          - `hatch_service_missions=True` shades service windows using
            `self.get_service_mission_df()` (expects UT_start_date/UT_end_date).
          - Title shows either a single UT date or startend date range derived
            from the data actually plotted.
          - Saves to `fig_path` if provided; optionally shows the figure.
          - Dependencies expected: pandas as pd, numpy as np, matplotlib.pyplot as plt,
            matplotlib.dates as mdates, astropy.time.Time, and Pythons datetime/time modules.
    
        Parameters
        ----------
        starname : str or None
            Object filter for DB and string shown in title.
        start_date, end_date : str | datetime.date | datetime.datetime | None
            See Date handling above.
        panels : list[str] | str
            Any combination of: 'rv','guiding','seeing','snr','sun','moon','el'.
            List order controls vertical order. Unknown names are ignored.
            Default = ['rv'].
        annotate : list[str] or None, optional
            If a non-empty list, adds a small annotation box in the upper-right.
            Recognized keys:
              - 'rv_rms' : standard deviation of the RVs (m/s), computed from the
                           median-centered series.
        hatch_service_missions : bool
            If True, hatch service-mission intervals.
        qc_not_fail, not_junk, clean : misc
            Passed to the DB accessor as-is.
        log_savefig_timing : bool
            If True, logs time spent in savefig using time.process_time().
        plot_timestamp : bool
            If True, annotates "Plotted <UTC>" at the bottom-left of the final panel.
        fig_path : str | Path | None
            If not None, path to save the figure (dpi=300).
        show_plot : bool
            If True, calls plt.show().
    
        Returns
        -------
        None
            (Side effects: saves/shows a matplotlib figure.)
        """
    
        # ---- Standard colors ----
        COLORS = {
            'rv': 'tab:blue',
            'guiding_x': 'mediumslateblue',
            'guiding_y': 'cornflowerblue',
            'seeing': 'mediumseagreen',
            'snr_452': 'royalblue',
            'snr_548': 'forestgreen',
            'snr_652': 'crimson',
            'snr_747': 'darkorange',
            'snr_852': 'goldenrod',
            'sun': 'slategray',   # neutral marker/line; background bands carry meaning
            'moon': 'tab:blue',
            'el': 'tab:purple',
        }
    
        def _expand_ylim_to_fit_overlays(ax, artists, min_extra_frac=0.02):
            """
            Expand the top of ax's y-limits by the combined vertical size of artists.
            Sizes are measured in display coords then converted to an axes-fraction
            height and finally to data units. Skips panels with autoscale disabled.
            """
            if not artists or not ax.get_autoscaley_on():
                return
            fig = ax.figure
            # Ensure artists have a layout and real sizes
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axbox = ax.get_window_extent(renderer=renderer)
            if axbox.height <= 0:
                return
        
            # Sum overlay heights in axes fraction
            total_frac = 0.0
            for art in artists:
                try:
                    bbox_disp = art.get_window_extent(renderer=renderer)
                    total_frac += bbox_disp.height / axbox.height
                except Exception:
                    continue
        
            if total_frac <= 0:
                return
        
            # Convert to data-units and expand top limit
            y0, y1 = ax.get_ylim()
            dy = (y1 - y0) * max(min_extra_frac, total_frac + 0.01)  # small cushion
            ax.set_ylim(y0, y1 + dy)

        # ---------- Start/end normalization (mirror if only one bound is given) ----------
        def _coerce_datetime_like(x):
            """Return a naive UTC datetime from string/date/datetime, or None."""
            if x is None:
                return None
            if isinstance(x, str):
                ts = pd.to_datetime(x, utc=True, errors='coerce')
                if pd.isna(ts):
                    return None
                if ts.tzinfo is not None:
                    return ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
                return ts.to_pydatetime()
            if isinstance(x, date) and not isinstance(x, datetime):
                return datetime.combine(x, datetime.min.time())
            if isinstance(x, datetime) and x.tzinfo is not None:
                return x.astimezone(timezone.utc).replace(tzinfo=None)
            return x  # already a naive datetime
    
        start_date = _coerce_datetime_like(start_date)
        end_date   = _coerce_datetime_like(end_date)

        # If exactly one bound is provided, mirror it
        if (start_date is None) ^ (end_date is None):
            if start_date is None:
                start_date = end_date
            else:
                end_date = start_date
    
        # Ensure start <= end
        if start_date is not None and end_date is not None and start_date > end_date:
            start_date, end_date = end_date, start_date
        # -------------------------------------------------------------------------------
    
        if qc_not_fail == 'auto':
            qc_not_fail = ['GOODREAD']
    
        # Columns (include guiding/seeing/SNR/sun/moon/el fields)
        cols = ['OBJECT', 'DATE-MID',
                'CCFRV', 'CCFERV', 'CCD1BJD', 'CCD1RV', 'CCD2RV', 'CCD1ERV', 'CCD2ERV',
                'GDRXRMS', 'GDRYRMS', 'GDRSEEV', 'SUNALT', 'MOONSEP', 'EL',
                'SNRSC452', 'SNRSC548', 'SNRSC652', 'SNRSC747', 'SNRSC852']
    
        # Retrieve data
        time_col = "CCD1BJD"
        rv_col   = "CCFRV"
        err_col  = "CCFERV"
    
        d = self.db.dataframe_from_db(columns=cols, 
                                      start_date=start_date, 
                                      end_date=end_date, 
                                      only_object=starname, 
                                      not_junk=not_junk,
                                      qc_not_fail=qc_not_fail)
    
        # Time column to numeric & sort
        d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col]).sort_values(time_col)
        if d.empty:
            raise ValueError("No rows after time parsing. Check CCD1BJD values.")
    
        # BJD(TDB) -> UTC datetimes
        tvals = d[time_col].to_numpy(np.float64)
        t = Time(tvals, format="jd", scale="tdb")
        d["time_utc"] = pd.to_datetime(t.utc.to_datetime())
    
        # Panels registry & selection (heights are real, proportional axes heights)
        panel_heights = {
            'rv': 2.5,
            'snr': 1.25,
            'seeing': 1.25,
            'guiding': 1.25,
            'el': 1.25,
            'sun': 1.25,
            'moon': 1.25,
        }
    
        if panels is None:
            selected_panels = ['rv']  # default and on top
        else:
            if isinstance(panels, str):
                selected_panels = [p.strip().lower() for p in panels.replace('|', ',').replace(';', ',').split(',') if p.strip()]
            else:
                selected_panels = [str(p).lower() for p in panels]
            selected_panels = [p for p in selected_panels if p in panel_heights] or ['rv']
    
        # Panel-specific prep
        d_rv = None
        if 'rv' in selected_panels:
            d[[rv_col, err_col]] = d[[rv_col, err_col]].apply(pd.to_numeric, errors="coerce")
            d_rv = d.dropna(subset=[rv_col, err_col]).copy()
            if not d_rv.empty:
                d_rv["rv_ms"]  = d_rv[rv_col].astype(np.float64)  * 1000.0
                d_rv["err_ms"] = d_rv[err_col].astype(np.float64) * 1000.0
                rv_median_ms = float(np.nanmedian(d_rv["rv_ms"]))
                d_rv["rv_centered_ms"] = d_rv["rv_ms"] - rv_median_ms
    
        d_gx = d_gy = None
        if 'guiding' in selected_panels:
            d["GDRXRMS"] = pd.to_numeric(d["GDRXRMS"], errors="coerce")
            d["GDRYRMS"] = pd.to_numeric(d["GDRYRMS"], errors="coerce")
            d_gx = d.dropna(subset=["GDRXRMS"]).copy()
            d_gy = d.dropna(subset=["GDRYRMS"]).copy()
    
        d_see = None
        if 'seeing' in selected_panels:
            d["GDRSEEV"] = pd.to_numeric(d["GDRSEEV"], errors="coerce")
            d_see = d.dropna(subset=["GDRSEEV"]).copy()
    
        snr_specs = [
            ("SNRSC452", "452 nm", "snr_452"),
            ("SNRSC548", "548 nm", "snr_548"),
            ("SNRSC652", "652 nm", "snr_652"),
            ("SNRSC747", "747 nm", "snr_747"),
            ("SNRSC852", "852 nm", "snr_852"),
        ]
        d_snr = {}
        if 'snr' in selected_panels:
            for col, _, key in snr_specs:
                d[col] = pd.to_numeric(d[col], errors="coerce")
                dfc = d.dropna(subset=[col]).copy()
                if not dfc.empty:
                    d_snr[key] = dfc
    
        d_sun = None
        if 'sun' in selected_panels:
            d["SUNALT"] = pd.to_numeric(d["SUNALT"], errors="coerce")
            d_sun = d.dropna(subset=["SUNALT"]).copy()
    
        d_moon = None
        if 'moon' in selected_panels:
            d["MOONSEP"] = pd.to_numeric(d["MOONSEP"], errors="coerce")
            d_moon = d.dropna(subset=["MOONSEP"]).copy()
    
        d_el = None
        if 'el' in selected_panels:
            d["EL"] = pd.to_numeric(d["EL"], errors="coerce")
            d_el = d.dropna(subset=["EL"]).copy()
    
        # At least one panel has usable data?
        panels_with_data = []
        if 'rv' in selected_panels and d_rv is not None and not d_rv.empty:
            panels_with_data.append('rv')
        if 'snr' in selected_panels and any(k in d_snr for k in [k for _,_,k in snr_specs]):
            panels_with_data.append('snr')
        if 'seeing' in selected_panels and d_see is not None and not d_see.empty:
            panels_with_data.append('seeing')
        if 'guiding' in selected_panels and ((d_gx is not None and not d_gx.empty) or (d_gy is not None and not d_gy.empty)):
            panels_with_data.append('guiding')
        if 'el' in selected_panels and d_el is not None and not d_el.empty:
            panels_with_data.append('el')
        if 'sun' in selected_panels and d_sun is not None and not d_sun.empty:
            panels_with_data.append('sun')
        if 'moon' in selected_panels and d_moon is not None and not d_moon.empty:
            panels_with_data.append('moon')
        if not panels_with_data:
            raise ValueError("No valid rows for the selected panels after cleaning.")
    
        # -------- Figure sizing with true per-panel heights --------
        height_ratios = [panel_heights[p] for p in selected_panels]
        EXTRA_PAD_INCH = 0.7
        fig_height = EXTRA_PAD_INCH + sum(height_ratios)
    
        fig, axs = plt.subplots(
            len(selected_panels), 1, sharex=True,
            figsize=(8, fig_height),
            gridspec_kw={'height_ratios': height_ratios},
            constrained_layout=True,
            squeeze=False
        )
        axs = axs.ravel()
        annotate = [] if annotate is None else list(annotate)
        # -----------------------------------------------------------
    
        # Title based on data span
        dates = d["time_utc"].dt.date
        date_start = dates.min()
        date_end   = dates.max()
        date_str = f"{date_start.isoformat()} UT" if date_start == date_end else f"{date_start.isoformat()} - {date_end.isoformat()} UT"
        title = f"{starname}: {date_str}" if starname else f"RVs: {date_str}"
        fig.suptitle(title, fontsize=12)
    
        # Draw panels
        for i, panel in enumerate(selected_panels):
            ax = axs[i]
    
            if panel == 'rv':
                if d_rv is None or d_rv.empty:
                    ax.text(0.5, 0.5, "No RV data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.errorbar(
                        d_rv["time_utc"],
                        d_rv["rv_centered_ms"],
                        yerr=d_rv["err_ms"],
                        fmt="o",
                        ms=2.5,
                        capsize=2,
                        elinewidth=0.75,
                        capthick=0.75,
                        alpha=0.9,
                        color=COLORS['rv'],
                        ecolor=COLORS['rv'],
                    )
                    ax.set_ylabel(r"RV ($\mathrm{m\,s^{-1}}$)", fontsize=12)
    
            elif panel == 'snr':
                plotted_any = False
                for col, label, key in snr_specs:
                    if key in d_snr:
                        dfc = d_snr[key]
                        ax.scatter(
                            dfc["time_utc"], dfc[col],
                            s=8, alpha=0.9,
                            color=COLORS[key],
                            label=label
                        )
                        plotted_any = True
                if not plotted_any:
                    ax.text(0.5, 0.5, "No SNR data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.set_ylabel("SNR", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, ncol=5, handletextpad=0.1, columnspacing=0.3)
                    leg = ax.legend(loc="upper right", fontsize=8, ncol=5, handletextpad=0.1, columnspacing=0.3)
                    legend_map = legend_map if 'legend_map' in locals() else {}
                    legend_map.setdefault(ax, []).append(leg)
    
            elif panel == 'seeing':
                if d_see is None or d_see.empty:
                    ax.text(0.5, 0.5, "No seeing data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.scatter(
                        d_see["time_utc"], d_see["GDRSEEV"],
                        s=10, alpha=0.9,
                        color=COLORS['seeing'],
                        label="Seeing (V-band)"
                    )
                    ax.set_ylabel("Seeing\n(V band, as)", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, handletextpad=0.1, columnspacing=0.3)
    
            elif panel == 'guiding':
                plotted_any = False
                if d_gx is not None and not d_gx.empty:
                    ax.scatter(
                        d_gx["time_utc"], d_gx["GDRXRMS"],
                        s=8, alpha=0.9,
                        color=COLORS['guiding_x'],
                        label="X-errors"
                    )
                    plotted_any = True
                if d_gy is not None and not d_gy.empty:
                    ax.scatter(
                        d_gy["time_utc"], d_gy["GDRYRMS"],
                        s=8, alpha=0.9,
                        color=COLORS['guiding_y'],
                        label="Y-errors"
                    )
                    plotted_any = True
                if not plotted_any:
                    ax.text(0.5, 0.5, "No guiding data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.set_ylabel("Guiding\n(RMS, mas)", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, ncol=2, handletextpad=0.1, columnspacing=0.3)
                    leg = ax.legend(loc="upper right", fontsize=8, ncol=2, handletextpad=0.1, columnspacing=0.3)
                    legend_map = legend_map if 'legend_map' in locals() else {}
                    legend_map.setdefault(ax, []).append(leg)
    
            elif panel == 'el':
                if d_el is None or d_el.empty:
                    ax.text(0.5, 0.5, "No elevation data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.scatter(
                        d_el["time_utc"], d_el["EL"],
                        s=8, alpha=0.9,
                        color=COLORS['el'],
                        label="Elevation (deg)"
                    )
                    # Background band: 0..30 (orange, subtle)
                    ax.axhspan( 0, 30, facecolor='orange', alpha=0.2,  zorder=0)
                    ax.axhspan(29, 30, facecolor='orange', alpha=0.25, zorder=1)
                    ax.set_ylim(0, 90)
                    ax.set_autoscaley_on(False)
                    ax.set_ylabel("Elevation\n(deg)", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, handletextpad=0.1, columnspacing=0.3)
    
            elif panel == 'sun':
                if d_sun is None or d_sun.empty:
                    ax.text(0.5, 0.5, "No Sun-altitude data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.scatter(
                        d_sun["time_utc"], d_sun["SUNALT"],
                        s=8, alpha=0.9,
                        color=COLORS['sun'],
                        label="Altitude of Sun (deg)"
                    )
                    ax.axhspan(-12,   0, facecolor='red',    alpha=0.2,  zorder=0)
                    ax.axhspan(-18, -12, facecolor='orange', alpha=0.2,  zorder=0)
                    ax.axhspan(-12, -11, facecolor='red',    alpha=0.25, zorder=1)
                    ax.axhspan(-18, -17, facecolor='orange', alpha=0.25, zorder=1)
                    ax.set_ylim(-90, 0)
                    ax.set_autoscaley_on(False)
                    ax.set_ylabel("Sun Altitude\n(deg)", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, handletextpad=0.1, columnspacing=0.3)
    
            elif panel == 'moon':
                if d_moon is None or d_moon.empty:
                    ax.text(0.5, 0.5, "No Moon-separation data", transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.scatter(
                        d_moon["time_utc"], d_moon["MOONSEP"],
                        s=8, alpha=0.9,
                        color=COLORS['moon'],
                        label="Moon Sep (deg)"
                    )
                    ax.axhspan(0,  30, facecolor='orange', alpha=0.2,  zorder=0)
                    ax.axhspan(29, 30, facecolor='orange', alpha=0.25, zorder=1)
                    ax.set_ylim(0, 180)
                    ax.set_autoscaley_on(False)
                    ax.set_ylabel("Moon Sep\n(deg)", fontsize=12)
                    #ax.legend(loc="upper right", fontsize=8, handletextpad=0.1, columnspacing=0.3)
    
            # Common formatting
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='both', labelsize=10)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
            # Optional hatching to highlight service missions
            if hatch_service_missions:
                try:
                    df_sm = self.get_service_mission_df()
                    if not df_sm.empty:
                        for _, row in df_sm.iterrows():
                            try:
                                x0 = pd.to_datetime(row['UT_start_date'])
                                x1 = pd.to_datetime(row['UT_end_date'])
                                mid_x = x0 + (x1 - x0) / 2
                                ymin, ymax = ax.get_ylim()
                                y0_9 = ymin + 0.9 * (ymax - ymin)
                                if pd.notna(x0) and pd.notna(x1):
                                    ax.axvspan(x0, x1, facecolor='none', edgecolor='none', hatch='////', alpha=0.15)
                                ax.text(mid_x, y0_9, row('name'), ha='center', va='center', fontsize=8, color=darkgray)
                            except Exception:
                                continue
                except Exception:
                    pass
    
            if i == len(selected_panels) - 1:
                ax.set_xlabel("Time (UTC)", fontsize=12)
                if plot_timestamp:
                    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    ax.annotate(f"Plotted {current_time} UT",
                                xy=(0, 0), xycoords='axes fraction',
                                fontsize=8, color="darkgray", ha="left", va="bottom",
                                xytext=(0, -30), textcoords='offset points')
    
        # --- Lock x-limits strictly to the returned data range (with a small pad) ---
        xmin = pd.to_datetime(d["time_utc"].min())
        xmax = pd.to_datetime(d["time_utc"].max())
        if not (pd.isna(xmin) or pd.isna(xmax)):
            if xmin == xmax:
                xmin_plot = xmin - pd.Timedelta(hours=12)
                xmax_plot = xmax + pd.Timedelta(hours=12)
            else:
                span = xmax - xmin
                pad  = max(pd.Timedelta(minutes=1), span * 0.02)  # 1 min, ~2% margin
                xmin_plot = xmin - pad
                xmax_plot = xmax + pad
            for ax in axs:
                ax.set_xlim(xmin_plot, xmax_plot)
    
        # --- Optional figure annotation (upper-right) ---
        legend_map   = legend_map   if 'legend_map'   in locals() else {}
        annot_map    = annot_map    if 'annot_map'    in locals() else {}
        from modules.Utils.utils import latex_number
        if annotate:
            lines = []
            if ('rv_rms' in annotate) and (d_rv is not None) and (not d_rv.empty):
                vals = d_rv["rv_centered_ms"].to_numpy(np.float64)
                rv_rms = float(np.nanstd(vals, ddof=1))
                if np.isfinite(rv_rms):
                    #lines.append(f"RV RMS: {rv_rms:.1f} m/s")  
                    str_rms = latex_number(rv_rms, 2, return_latex=False)
                    lines.append(f"RV RMS: {str_rms} m/s")  
        
            if lines:
                idx = selected_panels.index('rv') if 'rv' in selected_panels else 0
                ann = axs[idx].annotate(
                    "\n".join(lines),
                    xy=(0.99, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.6, edgecolor='gray')
                )
                annot_map.setdefault(axs[idx], []).append(ann)
        
        # Expand y-limits to fit any legends/annotations drawn inside axes
        # (Only affects panels with autoscale enabled; fixed-limit panels are left alone.)
        for ax, arts in (legend_map.items() if 'legend_map' in locals() else []):
            _expand_ylim_to_fit_overlays(ax, arts)
        for ax, arts in (annot_map.items() if 'annot_map' in locals() else []):
            _expand_ylim_to_fit_overlays(ax, arts)

        # Save/show
        try:
            if fig_path is not None:
                t0 = time.process_time()
                plt.savefig(fig_path, dpi=150, facecolor='w')
                if log_savefig_timing:
                    self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
            if show_plot is not None:
                plt.show()
            plt.close('all')
        except Exception as e:
            self.logger.info(f"Error saving file or showing plot: {e}")


    def plot_nightly_campaigns(self, UT_date, Nobs_min=2, show_plot=None, fig_dir=''):
        """
        Plot time-series campaign figures for all targets with at least
        Nobs_min observations on UT_date. If the night has  1 total 
        observation, no plots are produced. Figures are optionally saved with 
        filenames of the form ``{OBJECT}_{YYYYMMDD}.png``.
    
        Parameters
        ----------
        UT_date : datetime.date or datetime.datetime
            The UTC date to analyze. Observations are fetched with
            ``start_date=UT_date`` and ``end_date=UT_date``.
        Nobs_min : int, optional
            Minimum number of rows (observations) for a target (``OBJECT``) on
            the given night to be considered campaign-worthy. Default is 2.
        show_plot : bool or None, optional
            Passed through to ``myTS.plot_observing_rv_time_series``. If True, show
            the figure(s); if False, do not show; if None, defer to the plotting
            functions default behavior. Default is None.
        fig_dir : str or None, optional
            Directory to which figures are saved. If ``None``, figures are not
            written to disk. If a string, the path must exist (this function does
            not create directories). Default is ''.
    
        Returns
        -------
        None
            This function is called for its side effects (plotting and optional
            file output).
        """
        cols = ['DATE-BEG', 'DATE-MID', 'DATE-END', 'EXPTIME', 'UT DATE', 'OBJECT', 'PROGNAME']
        df = self.db.dataframe_from_db(columns=cols, start_date=UT_date, end_date=UT_date, on_sky=True)
        Nobs_night = df.shape[0]
        if Nobs_night > 1:
            eligible = df['OBJECT'].dropna().value_counts()
            stars = eligible[eligible >= Nobs_min].index.tolist()
            if len(stars) > 0:
                for starname in stars:
                    if fig_dir is None:
                        fig_path = None
                    else:
                        fig_path = os.path.join(fig_dir, f"{starname}_{UT_date:%Y%m%d}.png")
                    self.plot_observing_rv_time_series(starname=starname, 
                                                       start_date=UT_date, 
                                                       panels=['rv', 'snr', 'guiding', 'seeing', 'el', 'sun', 'moon'], 
                                                       annotate = ['rv_rms'], 
                                                       plot_timestamp=True,
                                                       show_plot=show_plot, 
                                                       fig_path=fig_path)

    def performance_by_datecode(
        self,
        df: pd.DataFrame,
        spec_config,
        columns_to_display=None,
        datecode_col: str = 'datecode',
        ignore_service_missions=True,
    ) -> pd.DataFrame:
        """
        For each datecode, determine if any row violates each spec criterion.
    
        Supports two spec_config styles:
    
        1) Simple comparison (backward compatible):
           {
               'col': 'kpfgreen.STA_CCD_T',
               'name': 'Green CCD > -99 K',
               'op': '>',
               'threshold': -99.0
           }
    
        2) Expression over multiple columns:
           {
               "name": "|ΔT| > 10 mK AND Nobs>20",
               "cols": ["kpfgreen.STA_CCD_T", "kpfred.STA_CCD_T", "Nobs"],
               "bool_expr": "(abs(c0 - c1) > 0.01) & (c2 > 20)",
           }
    
           where:
             - c0, c1, c2 ... are the Series for the listed cols in order
             - allowed functions: abs, min, max (elementwise), np (restricted)
             - logical operators: &, |, ~ with parentheses
        """
    
        _OP_MAP = {
            '>':  operator.gt,
            '<':  operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
        }
    
        df_work = df.copy()
    
        # Optionally drop datecodes that fall inside service missions
        if ignore_service_missions:
            df_sm = self.get_service_mission_df()
            if df_sm is not None and not df_sm.empty:
                # assume df has a DATE-MID or similar; if not, you can adapt
                if 'DATE-MID' in df_work.columns:
                    dates = pd.to_datetime(df_work['DATE-MID'])
                else:
                    dates = None
    
                if dates is not None:
                    keep = pd.Series(True, index=df_work.index)
                    for _, row in df_sm.iterrows():
                        x0 = pd.to_datetime(row['UT_start_date'])
                        x1 = pd.to_datetime(row['UT_end_date'])
                        if pd.notna(x0) and pd.notna(x1):
                            keep &= ~dates.between(x0, x1)
                    df_work = df_work[keep].copy()
    
        if df_work.empty:
            base_cols = [datecode_col] + (columns_to_display or [])
            return pd.DataFrame(columns=base_cols)
    
        if datecode_col not in df_work.columns:
            raise KeyError(f"datecode column '{datecode_col}' not found in dataframe.")
    
        group_key = df_work[datecode_col]
        grouped = df_work.groupby(datecode_col, sort=True)
    
        out = {}
    
        # --- helpers for expression-based specs ---
        def _eval_bool_expr(spec, df_local):
            """
            Evaluate spec['bool_expr'] over df_local using columns in spec['cols'].
            Returns a boolean Series aligned with df_local.
            """
            cols = spec.get('cols')
            expr = spec.get('bool_expr')
    
            if not cols or not isinstance(cols, (list, tuple)):
                raise ValueError(
                    f"Spec '{spec.get('name','<unnamed>')}' with bool_expr "
                    f"must define a non-empty 'cols' list."
                )
            if not isinstance(expr, str):
                raise ValueError(
                    f"Spec '{spec.get('name','<unnamed>')}' must have 'bool_expr' "
                    f"as a string."
                )
    
            missing = [c for c in cols if c not in df_local.columns]
            if missing:
                raise KeyError(
                    f"Columns {missing} (for criterion '{spec.get('name','<unnamed>')}') "
                    f"not found in dataframe."
                )
    
            # Build the local environment: c0, c1, ... mapped to columns
            local_env = {}
            for idx, col in enumerate(cols):
                local_env[f'c{idx}'] = df_local[col]
    
            # Allowed functions & names inside bool_expr
            local_env.update({
                'abs': np.abs,
                'min': np.minimum,   # elementwise min(series0, series1)
                'max': np.maximum,   # elementwise max(series0, series1)
                'np': np,
            })
    
            try:
                cond = eval(expr, {"__builtins__": {}}, local_env)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating bool_expr '{expr}' for criterion "
                    f"'{spec.get('name','<unnamed>')}': {e}"
                )
    
            # Normalize to boolean Series aligned with df_local
            if isinstance(cond, pd.Series):
                cond_series = cond
            else:
                cond_series = pd.Series(cond, index=df_local.index)
    
            cond_series = cond_series.astype(bool)
            return cond_series
    
        # --- main loop over spec_config ---
        for spec in spec_config:
            name = spec.get('name')
            if not name:
                raise ValueError("Each spec must have a 'name' key.")
    
            # Path 1: expression-based spec (bool_expr + cols)
            if 'bool_expr' in spec:
                cond = _eval_bool_expr(spec, df_work)
                out[name] = cond.groupby(group_key).any()
                continue
    
            # Path 2: simple comparison spec (backward compatible)
            # expects: col, op, threshold
            if 'col' not in spec or 'op' not in spec or 'threshold' not in spec:
                raise ValueError(
                    f"Spec '{name}' must define either "
                    f"('cols' + 'bool_expr') or ('col', 'op', 'threshold')."
                )
    
            col = spec['col']
            op_str = spec['op']
            threshold = spec['threshold']
    
            if col not in df_work.columns:
                raise KeyError(f"Column '{col}' (for criterion '{name}') not found in dataframe.")
    
            if op_str not in _OP_MAP:
                raise ValueError(f"Unsupported operator '{op_str}' in criterion '{name}'.")
    
            op_func = _OP_MAP[op_str]
            cond = op_func(df_work[col], threshold)
    
            # True if ANY row for that datecode meets the condition
            out[name] = cond.groupby(group_key).any()
    
        # Construct out-of-spec summary frame
        out_df = pd.DataFrame(out)
    
        # Add info/display columns (first value within each datecode)
        if columns_to_display:
            columns_to_display = list(dict.fromkeys(columns_to_display))  # de-duplicate
            cols_present = [c for c in columns_to_display if c in df_work.columns]
    
            if cols_present:
                info_df = grouped[cols_present].first()
                summary_df = info_df.join(out_df)
            else:
                summary_df = out_df
        else:
            summary_df = out_df
    
        summary_df = summary_df.reset_index()  # brings datecode back as a column
        return summary_df


    def plot_performance_by_datecode(
        self,
        summary_df: pd.DataFrame,
        spec_config,
        datecode_col: str = 'datecode',
        date_format: str = '%Y%m%d',
        plot_title=None,
        figsize='auto',
        excise_serice_missions=True,
        hatch_service_missions=True,
        plot_timestamp=False,
        fig_path=None, 
        show_plot=False,
    ):
        """
        Plot criteria (rows) vs time (x-axis) using datecode interpreted as YYYYMMDD.
    
        False  -> small, faint green dot
        True   -> larger, red dot
    
        Parameters
        ----------
        summary_df : pd.DataFrame
            Output of summarize_out_of_spec_by_date, one row per datecode.
        spec_config : list of dict
            Same spec_config used to generate summary_df. Uses spec['name']
            to find boolean columns.
        datecode_col : str, default 'datecode'
            Column in summary_df giving the datecode (YYYYMMDD).
        date_format : str, default '%Y%m%d'
            strftime-style format string to parse datecode.
        """
        # Remove dates during service missions
        if excise_serice_missions:
            df_sm = self.get_service_mission_df()
            if not df_sm.empty:
                dates = pd.to_datetime(summary_df['datecode'].astype(str), format='%Y%m%d')
            
                # Start with "keep everything"
                keep = pd.Series(True, index=summary_df.index)
            
                for _, row in df_sm.iterrows():
                    x0 = pd.to_datetime(row['UT_start_date'])
                    x1 = pd.to_datetime(row['UT_end_date'])
                    if pd.notna(x0) and pd.notna(x1):
                        # Drop anything between x0 and x1 (inclusive)
                        keep &= ~dates.between(x0, x1)
                summary_df = summary_df[keep].copy()

        # Parse datecode -> datetime
        dates = pd.to_datetime(
            summary_df[datecode_col].astype(str),
            format=date_format,
            errors='coerce'
        )

        # Determine limits
        start_date = dates.min()
        end_date   = dates.max()
        start_datecode = summary_df['datecode'].min()
        end_datecode   = summary_df['datecode'].max()

        if dates.isna().any():
            bad = summary_df.loc[dates.isna(), datecode_col]
            raise ValueError(f"Could not parse some {datecode_col} values as dates: {bad.tolist()}")

        # Determine criteria columns from spec_config, preserving spec_config order
        criteria_cols = [
            spec['name']
            for spec in spec_config
            if spec['name'] in summary_df.columns and summary_df[spec['name']].dtype == bool
        ]
    
        if not criteria_cols:
            raise ValueError("No valid boolean criteria columns found in summary_df for given spec_config.")

        if figsize == 'auto':
            figsize = (10, 1.0 + len(criteria_cols) * 0.15)
        fig, ax = plt.subplots(figsize=figsize)
        
        # x is the datetime index
        x = dates.values  # matplotlib can plot numpy datetime64 directly
        
        # blended transform for right-side annotations
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

        # x is the datetime index
        x = dates.values  # matplotlib can plot numpy datetime64 directly

        # blended transform for right-side annotations
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

        # precompute strings & lengths for annotations 
        row_data = []  # (crit, vals, red_str, green_str, tail_str)
        for crit in criteria_cols:
            vals = summary_df[crit].values
            Nred = int(vals.sum())          # True == 1, so sum gives Nred
            Ntotal = int(len(vals))
            Ngreen = Ntotal - Nred

            red_str   = f"{Nred}"
            green_str = f":{Ngreen}"
            tail_str  = f"/{Ntotal} days"

            row_data.append((crit, vals, red_str, green_str, tail_str))

        # Max lengths for each "column" of text
        max_red_len   = max(len(r[2]) for r in row_data)
        max_green_len = max(len(r[3]) for r in row_data)
        # tail length max not strictly needed for alignment, but kept for completeness
        max_tail_len  = max(len(r[4]) for r in row_data)

        # Approximate width per character in axes coords
        char_width = 0.012  # tweak if needed

        # Fixed x-positions for each column (in axes coords)
        x_base_red   = 1.01
        x_base_green = x_base_red   + char_width * max_red_len
        x_base_tail  = x_base_green + char_width * max_green_len - 0.8*char_width


        # Optional hatching to highlight service missions
        if hatch_service_missions:
            try:
                df_sm = self.get_service_mission_df()
                if not df_sm.empty:
                    for _, row in df_sm.iterrows():
                        try:
                            x0 = pd.to_datetime(row['UT_start_date'])
                            x1 = pd.to_datetime(row['UT_end_date'])
                            if pd.notna(x0) and pd.notna(x1):
                                ax.axvspan(
                                    x0, x1,
                                    facecolor='none',           # keep data visible
                                    hatch='////',
                                    edgecolor='dimgray',
                                    linewidth=0.0,
                                    alpha=0.4,
                                    zorder=0.2
                                )
                        except Exception:
                            continue
            except Exception:
                pass

        # plot rows & aligned annotations 
        for j, (crit, vals, red_str, green_str, tail_str) in enumerate(row_data):
            y = np.full_like(x, j, dtype=float)

            # False = small, faint green
            mask_false = ~vals
            ax.scatter(
                x[mask_false],
                y[mask_false],
                s=15,
                color='green',
                alpha=0.2,
                edgecolor='none',
            )

            # True = larger, bright red
            mask_true = vals
            ax.scatter(
                x[mask_true],
                y[mask_true],
                s=30,
                color='red',
                alpha=0.9,
                edgecolor='k',
                linewidth=0.3,
            )

            # Row index in data coords
            y_data = j

            # Column 1: red Nred
            ax.text(
                x_base_red, y_data, red_str,
                transform=trans,
                va='center', ha='left',
                fontsize=8,
                color='red',
                clip_on=False,
            )

            # Column 2: green :Ngreen
            ax.text(
                x_base_green, y_data, green_str,
                transform=trans,
                va='center', ha='left',
                fontsize=8,
                color='green',
                clip_on=False,
            )

            # Column 3: black /Ntotal days
            ax.text(
                x_base_tail, y_data, tail_str,
                transform=trans,
                va='center', ha='left',
                fontsize=8,
                color='black',
                clip_on=False,
            )
        
        # Y-axis: criteria labels
        ax.set_yticks(range(len(criteria_cols)))
        ax.set_yticklabels(criteria_cols, fontsize=8)
        ax.set_ylim(len(criteria_cols) - 0.5, -0.5)  # invert so 0 is at top
    
        # X-axis: time formatting
        ax.set_xlim(start_date - timedelta(days=1), end_date + timedelta(days=1))
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelsize=8)

        ax.set_xlabel('Date')
        if plot_title:
            ax.set_title(plot_title, fontsize=10)
        ax.grid(True, axis='x', alpha=0.2)
    
        plt.tight_layout()

        # Save/show
        try:
            if fig_path is not None:
                t0 = time.process_time()
                plt.savefig(fig_path, dpi=250, facecolor='w')
            if show_plot is not None:
                plt.show()
            plt.close('all')
        except Exception as e:
            self.logger.info(f"Error saving file or showing plot: {e}")


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

def timebin(time, meas, meas_err, binsize):
    """Bin in equal sized time bins

    This routine bins a set of times, measurements, and measurement errors
    into time bins. All inputs and outputs should be floats or double.
    binsize should have the same units as the time array.
    (from Andrew Howard, ported to Python by BJ Fulton)

    Args:
        time (array): array of times, can be datetime or numeric
        meas (array): array of measurements to be combined
        meas_err (array): array of measurement uncertainties
        binsize (float): width of bins in the same units as time array (fractional days)

    Returns:
        tuple: (bin centers, binned measurements, binned uncertainties) in Series format if inputs are Series
    """
    
    # Convert time to numeric (days since epoch) while preserving precision
    if isinstance(time, pd.Series) and pd.api.types.is_datetime64_any_dtype(time):
        time_numeric = (time.astype('datetime64[ns]').view('int64') / 1e9 / 86400.0)  # Convert to days
    else:
        time_numeric = np.array(time)

    # Create a DataFrame to sort and keep track of original indices
    data = pd.DataFrame({'time': time_numeric, 'meas': meas, 'meas_err': meas_err})
    data.sort_values(by='time', inplace=True)

    # Use sorted values for calculations
    time_numeric_sorted = data['time'].values
    meas_sorted = data['meas'].values
    meas_err_sorted = data['meas_err'].values

    time_out = []
    meas_out = []
    meas_err_out = []

    ct = 0
    while ct < len(time_numeric_sorted):
        ind = np.where((time_numeric_sorted >= time_numeric_sorted[ct]) & 
                       (time_numeric_sorted < time_numeric_sorted[ct] + binsize))[0]
        num = len(ind)
        if num == 0:  # No measurements in this bin
            ct += 1
            continue
            
        wt = (1. / meas_err_sorted[ind]) ** 2.  # weights based on errors
        wt /= np.sum(wt)               # normalized weights
        
        # Calculate weighted averages and errors
        bin_center_numeric = np.sum(wt * time_numeric_sorted[ind])
        binned_meas = np.sum(wt * meas_sorted[ind])
        binned_err = 1. / np.sqrt(np.sum(1. / (meas_err_sorted[ind]) ** 2))

        time_out.append(bin_center_numeric)
        meas_out.append(binned_meas)
        meas_err_out.append(binned_err)

        ct += num

    # Convert bin centers back to pandas datetime if the input was a pandas datetime
    if isinstance(time, pd.Series) and pd.api.types.is_datetime64_any_dtype(time):
        # Convert back from days since epoch to datetime with full precision
        time_out = pd.to_datetime(np.array(time_out) * 86400 * 1e9)  # Convert days back to nanoseconds

    # Return results as Series if input was Series
    if isinstance(time, pd.Series):
        return (pd.Series(time_out), 
                pd.Series(meas_out), 
                pd.Series(meas_err_out))

    return time_out, meas_out, meas_err_out
