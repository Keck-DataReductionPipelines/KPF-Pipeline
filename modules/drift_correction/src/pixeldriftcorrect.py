import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from astropy.table import Table
from astropy.time import Time
import helper
import os


class calDriftTimeseries:
    def __init__(self,caltype,chip,times,drifts,errs,clip=True,outlierplot=False):
        # May replace chip with order
        """
        Construct a time series of a given calibration type for a single chip.
        
        Parameters:
        - caltype (str): Calibration source (lfc,etalon,etc.)
        - chip (str): Chip name
        """
        if not len(times)==len(drifts)==len(errs):
            raise ValueError('Input data is jagged')
        
        self.caltype=caltype
        self.chip=chip

        # Order the data
        timeorder = np.argsort(times)
        self.raw_data = {'BJD':times[timeorder], 'PixelDrifts':drifts[timeorder], 'DriftErr':errs[timeorder]}
        
        self.merged_caltypes = [self.caltype]
        
        if clip:
            self.clip_outliers(outlierplot=outlierplot)
        else:
            self.clipped_data = self.raw_data
            self.outlier_mask = None
            self.outlier_spline = None

        self.calibrated = False
        self.calibrated_data = {'BJD':np.array([]), 'PixelDrifts':np.array([]), 'DriftErr':np.array([])}
        self.model = None

    def __repr__(self):
        return (f"Calibration Series, "
                f"chip='{self.chip}', type='{self.caltype}'")

    def combine_with(self, other):
        if self.chip != other.chip:
            raise ValueError('Time series chips must match')

        if self.calibrated != other.calibrated:
            raise ValueError('Can only combine two calibrated series')
        
        raw_times = np.concatenate((self.raw_data['BJD'],other.raw_data['BJD']))
        raw_drifts = np.concatenate((self.raw_data['PixelDrifts'],other.raw_data['PixelDrifts']))
        raw_errs = np.concatenate((self.raw_data['DriftErr'],other.raw_data['DriftErr']))

        cal_times = np.concatenate((self.calibrated_data['BJD'],other.calibrated_data['BJD']))
        cal_drifts = np.concatenate((self.calibrated_data['PixelDrifts'],other.calibrated_data['PixelDrifts']))
        cal_errs = np.concatenate((self.calibrated_data['DriftErr'],other.calibrated_data['DriftErr']))
        cal_order = np.argsort(cal_times)
        
        combined = calDriftTimeseries(self.caltype,self.chip,raw_times,raw_drifts,raw_errs,clip=False)
        combined.calibrated = True
        combined.clipped_data = {'BJD':cal_times[cal_order],'PixelDrifts':cal_drifts[cal_order],'DriftErr':cal_errs[cal_order]}
        combined.calibrated_data = {'BJD':cal_times[cal_order],'PixelDrifts':cal_drifts[cal_order],'DriftErr':cal_errs[cal_order]}

        combined.merged_caltypes.append(other.caltype)
        
        return combined

    def clip_outliers(self,outlierplot=False):

        tdata = self.raw_data['BJD']
        ydata = self.raw_data['PixelDrifts']
        yerrs = self.raw_data['DriftErr']
        caltype=self.caltype

        if caltype == 'etalon':
            threshold = 10
        if caltype == 'lfc':
            threshold = 6
        
        mask = np.ones_like(ydata, dtype=bool)
        old_outliers = 0
        for r in range(4):
            tround = tdata[mask]
            yround = ydata[mask]
            
            fit_spline = helper.fit_adaptive_spline(tround,yround,base_spacing=3.0, density_window=5.0, min_points_per_knot=5)
            residuals = ydata - fit_spline(tdata)
            mad = np.median(np.abs(residuals-np.median(residuals)))
            mask = np.abs(residuals) < threshold*mad
            new_outliers = len(np.where(mask==False)[0])
            if new_outliers == old_outliers:
                break
            old_outliers = new_outliers
        
        self.outlier_mask = mask
        self.outlier_spline = fit_spline
        if outlierplot:
            self.outlier_plot()

        self.clipped_data = {'BJD':tdata[mask], 'PixelDrifts':ydata[mask], 'DriftErr':yerrs[mask]}

    def outlier_plot(self,outputdir='testfigs'):
        
        tdata = self.raw_data['BJD']
        ydata = self.raw_data['PixelDrifts']
        mask = self.outlier_mask
        fit_spline = self.outlier_spline

        filename = f'outliers_{self.chip}_{self.caltype}.png'
        
        tfine = np.linspace(min(tdata),max(tdata),10000)
        plt.figure(figsize=(10, 5))
        plt.plot(tdata, ydata, '.', label='Original',c='white')
        plt.plot(tdata[~mask], ydata[~mask], '.', label='Outliers',c='orange')
        plt.plot(tfine, fit_spline(tfine), c=self.chip, linewidth=4, label='Spline',zorder=1000)
        plt.title(f"Knotted Spline Clipping - {self.caltype}")
        plt.xlabel("BJD")
        plt.ylabel("Pixels")
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(os.path.join(outputdir,filename),dpi=150)

    def plot(self,axs=None):

        if self.calibrated == False:
            tdata = self.clipped_data['BJD']
            ydata = self.clipped_data['PixelDrifts']
            yerrs = self.clipped_data['DriftErr']
        else:
            tdata = self.calibrated_data['BJD']
            ydata = self.calibrated_data['PixelDrifts']
            yerrs = self.calibrated_data['DriftErr']
        
        if axs == None:
            plt.figure(figsize=(10, 5))
            plt.errorbar(tdata, ydata, yerrs, fmt='.',c=self.chip)
            plt.title(f"Calibration Time Series - {self.caltype}")
            plt.xlabel("BJD")
            plt.ylabel("Pixels")
            #plt.legend()
            plt.grid(True)
            plt.tight_layout()
        else:
            axs.errorbar(tdata, ydata, yerrs, fmt='.',label=self.caltype)
        
class chipDriftModel:
    def __init__(self,chip,method='gp'):
        # May replace chip with order
        """
        Construct a time series model for a given chip using several calibration types
        
        Parameters:
        - caltype (str): Calibration source (lfc,etalon,etc.)
        - chip (str): Chip name
        """
        methods = ['gp','polynomial']
        if method not in methods:
            raise ValueError('Method must be from',methods)
        self.chip=chip
        self.method=method
        self.caltypes=[]
        self.all_series={}
        self.training_data = {}

    def __repr__(self):
        return (f"Chip Drift Model, "
                f"chip='{self.chip}', caltypes='{self.caltypes}'")


    def model_drifts(self):
        if 'all' not in self.caltypes:
            raise ValueError('Must calibrate time series before modeling')

        ser = self.all_series['all']
        tdata = ser.calibrated_data['BJD']
        ydata = ser.calibrated_data['PixelDrifts']
        yerrdata = ser.calibrated_data['DriftErr']

        self.model = helper.interpolate_drifts(tdata,ydata,yerrdata, method=self.method)

    def predict(self, t):
        if self.model:
            return self.model(t)  # Use the interpolant for prediction
        else:
            raise ValueError("Model is not set.")

    def add_timeseries(self,cal_timeseries):
        if self.chip != cal_timeseries.chip:
            raise ValueError('The chip identifier of the model and each calibration series must match!')

        self.all_series[cal_timeseries.caltype] = cal_timeseries
        self.caltypes.append(cal_timeseries.caltype)
        self.training_data[cal_timeseries.caltype] = cal_timeseries.clipped_data

    def calibrate_all(self):

        if 'lfc' not in self.caltypes:
            raise ValueError('Missing LFC data in model, cannot calibrate to absolute reference.')

        # If LFC needs recalibration, it should reside here
        self.all_series['lfc'].calibrated_data = self.all_series['lfc'].clipped_data
        self.all_series['lfc'].calibrated = True
        
        # Calibrate etalon (or others) to lfc        
        if 'etalon' in self.caltypes:
            self.fit_etalon_to_lfc()
            self.all_series['etalon'].calibrated = True

        # Build a new timeseries object which contains all the calibrated datatypes
        joint_series = calDriftTimeseries(caltype='all',
                                          chip=self.chip,
                                          times=np.array([]),
                                          drifts=np.array([]),
                                          errs=np.array([]),
                                          clip=False)

        # Can only combine two calibrated series, so need to specify
        joint_series.calibrated = True
        for caltype in self.caltypes:
            joint_series = joint_series.combine_with(self.all_series[caltype])

        self.add_timeseries(joint_series)
        #self.caltypes.append('all')
            
            
    def fit_etalon_to_lfc(self):
        
        params,etalon_spline = helper.fit_etalon_to_lfc(t_ref     =self.training_data['lfc']['BJD'],
                                                        y_ref     =self.training_data['lfc']['PixelDrifts'],
                                                        t_fit     =self.training_data['etalon']['BJD'],
                                                        y_fit     =self.training_data['etalon']['PixelDrifts'],
                                                        y_fit_err =self.training_data['etalon']['DriftErr'])

        tcorr    = self.training_data['etalon']['BJD']
        yuncorr  = self.training_data['etalon']['PixelDrifts']
        yerrcorr = self.training_data['etalon']['DriftErr']
        
        # Mapping params are offset, multiplier, drift
        offset, multiplier, drift = params
        ycorr = multiplier*yuncorr + offset + drift*(tcorr-tcorr[0])

        self.all_series['etalon'].calibrated_data = {'BJD':tcorr,'PixelDrifts':ycorr,'DriftErr':yerrcorr}


    def plot_timeseries(self,caltypes=None):
        if caltypes is None:
            caltypes = self.caltypes

        fig,axs = plt.subplots(1,1,figsize=(10, 5))
        
        for cal in caltypes:
            self.all_series[cal].plot(axs)

        plt.title(f"Calibration Time Series - {caltypes}")
        plt.xlabel("BJD")
        plt.ylabel("Pixels")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def plot_model(self,lb=None,ub=None):

        if self.model == None:
            raise ValueError('No incumbent drift model')
    
        all_cals = self.all_series['all']
        tdata = all_cals.calibrated_data['BJD']
        
        if lb == None:
            lb = tdata[0]
        if ub == None:
            ub = tdata[-1]
    
        fig,axs = plt.subplots(2,1,figsize=(7,7))
    
        all_window = np.where((tdata >= lb) & (tdata <= ub))[0]
        tfine = np.linspace(tdata[all_window][0],tdata[all_window][-1],500)
        axs[0].plot(tfine,self.predict(tfine)*-1000,color='white',label=self.method,zorder=1000)
        
        
        for caltype in self.caltypes:
            if caltype != 'all':
                ser = self.all_series[caltype]
                t = ser.calibrated_data['BJD']
                y = ser.calibrated_data['PixelDrifts']
                yerr = ser.calibrated_data['DriftErr']
                
                cal_window = np.where((t >= lb) & (t <= ub))[0]
                axs[0].errorbar(t[cal_window],y[cal_window]*-1000,yerr[cal_window]*1000,fmt='.', label=caltype)
                residuals = y - self.predict(t)
                axs[1].errorbar(t[cal_window],residuals[cal_window]*-1000,yerr[cal_window]*1000,fmt='.')
    
        axs[0].legend()
        axs[1].set_xlabel('BJD')
        axs[0].set_ylabel('SCI1 RV (m/s)')
        axs[1].set_ylabel('Residuals (m/s)')
        
        plt.tight_layout()
        plt.show()