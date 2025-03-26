import time
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from modules.Utils.kpf_parse import HeaderParse

class AnalyzeEM:

    """
    Description:
        This class contains functions to analyze Exposure Meter data 
        (storing them as attributes) and functions to plot the results.
        Some of the functions need to be filled in.

    Arguments:
        L0 - an L0 object

    Attributes:
        EM_gain - gain (e- per DN) of exposure meter detector
        SKY_SCI_ratio - estimated flux ratio of SKY to SCI flux in the EM based on twilight obs
        SKY_SCI_main_spectrometer - estimated flux ratio between SKY and SCI in main spectrometer based on EM
        tdur_sec - exposure duration in sec
        time_em - array of times since first exposures (sec)
        wav_SCI - SCI wavelength array (nm)
        wav_SKY - SKY wavelength array (nm)
        disp_SCI_smooth - SCI dispersion array (nm/pixel)
        disp_SKY_smooth - SCI dispersion array (nm/pixel)
        flux_SCI         - SCI counts array (445-870 nm; ADU) with timesteps from time_em
        flux_SCI_551m    - SCI counts array (445-551 nm; ADU) with timesteps from time_em
        flux_SCI_551_658 - SCI counts array (551-658 nm; ADU) with timesteps from time_em
        flux_SCI_658_764 - SCI counts array (658-764 nm; ADU) with timesteps from time_em
        flux_SCI_764p    - SCI counts array (764-870 nm; ADU) with timesteps from time_em
        flux_SKY         - SKY counts array (445-870 nm; ADU) with timesteps from time_em  
        flux_SKY_551m    - SKY counts array (445-551 nm; ADU) with timesteps from time_em  
        flux_SKY_551_658 - SKY counts array (551-658 nm; ADU) with timesteps from time_em 
        flux_SKY_658_764 - SKY counts array (658-764 nm; ADU) with timesteps from time_em  
        flux_SKY_764p    - SKY counts array (764-870 nm; ADU) with timesteps from time_em
        flux_SCI         - SCI flux array (445-870 nm; e-/nm/s) with timesteps from time_em
        flux_SCI_551m    - SCI flux array (445-551 nm; e-/nm/s) with timesteps from time_em
        flux_SCI_551_658 - SCI flux array (551-658 nm; e-/nm/s) with timesteps from time_em
        flux_SCI_658_764 - SCI flux array (658-764 nm; e-/nm/s) with timesteps from time_em
        flux_SCI_764p    - SCI flux array (764-870 nm; e-/nm/s) with timesteps from time_em
        flux_SKY         - SKY flux array (445-870 nm; e-/nm/s) with timesteps from time_em  
        flux_SKY_551m    - SKY flux array (445-551 nm; e-/nm/s) with timesteps from time_em  
        flux_SKY_551_658 - SKY flux array (551-658 nm; e-/nm/s) with timesteps from time_em 
        flux_SKY_658_764 - SKY flux array (658-764 nm; e-/nm/s) with timesteps from time_em  
        flux_SKY_764p    - SKY flux array (764-870 nm; e-/nm/s) with timesteps from time_em
    """

    def __init__(self, L0, logger=None):

        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeEM object.')
        else:
            self.logger = None
        self.L0 = copy.deepcopy(L0)
        primary_header = HeaderParse(self.L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()

        self.EM_gain = 1.48424 # np.float(self.config['EM']['gain'])
        self.SKY_SCI_ratio = 14.1 # ratio flux ratio of SKY to SCI flux based on bright twilight observations (e.g., KP.20230114.15771.04)
        self.wav0 = 445      # boundaries between wavelength bins
        self.wav1 = 551.25   # "
        self.wav2 = 657.50   # "
        self.wav3 = 763.75   # "
        self.wav4 = 870      # "

        # Read data tables
        self.dat_SCI = self.L0['EXPMETER_SCI']
        self.dat_SKY = self.L0['EXPMETER_SKY']
        self.df_SCI_EM = self.dat_SCI
        self.df_SKY_EM = self.dat_SKY
        i = 0
        for col in self.df_SCI_EM.columns:
            if col.lower().startswith('date'):
                i += 1
            else:
                break
        self.wav_SCI_str = self.df_SCI_EM.columns[i:]
        self.wav_SCI     = self.df_SCI_EM.columns[i:].astype(float)
        self.wav_SKY_str = self.df_SKY_EM.columns[i:]
        self.wav_SKY     = self.df_SKY_EM.columns[i:].astype(float)
      
        # Time arrays
        self.date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        self.n_samples = self.df_SCI_EM.shape[0] #self.df_SCI_EM_norm.shape[0]
        if 'Date-Beg-Corr' in self.df_SCI_EM.columns:
            self.date_beg = np.array(self.df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            self.date_end = np.array(self.df_SCI_EM["Date-End-Corr"], dtype=np.datetime64)
        else:
            self.date_beg = np.array(self.df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            self.date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        self.tdur_sec = (self.date_end-self.date_beg).astype(float)/1000. # exposure duration in sec [array]
        self.time_em  = (self.date_beg-self.date_beg[0]).astype(float)/1000. # seconds since beginning [array]
        self.total_duration = (self.date_end[-1]-self.date_beg[0]).astype(float)/1000. # used for setting x-axis limits
  
        # Exposure times
        if 'Date-End-Corr' in self.df_SCI_EM.columns:
            self.timediff = np.array(self.df_SCI_EM["Date-End-Corr"], dtype=np.datetime64) - np.array(self.df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            self.EM_texp = pd.Timedelta(np.median(self.timediff)).total_seconds() # EM exposure time in seconds
            self.EM_nexp = self.timediff.shape[0]
        elif 'Date-End' in self.df_SCI_EM.columns:
            self.timediff = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64) - np.array(self.df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            self.EM_texp = pd.Timedelta(np.median(self.timediff)).total_seconds()
            self.EM_nexp = self.timediff.shape[0]
        elif 'Date-End' in self.df_SCI_EM.columns:
            self.EM_texp = 0.
            self.EM_nexp = 0

        # Spectral dispersion
        self.disp_SCI = self.wav_SCI*0+np.gradient(self.wav_SCI,1)*-1
        self.disp_SKY = self.wav_SKY*0+np.gradient(self.wav_SKY,1)*-1
        self.disp_SCI_smooth = np.polyval(np.polyfit(self.wav_SCI,self.disp_SCI, deg=6),self.wav_SCI)
        self.disp_SKY_smooth = np.polyval(np.polyfit(self.wav_SKY,self.disp_SKY, deg=6),self.wav_SKY)
        
        # Flux arrays
        self.ind_551m    = np.where((self.wav_SCI <  self.wav1))
        self.ind_551_658 = np.where((self.wav_SCI >= self.wav1) & (self.wav_SCI < self.wav2))
        self.ind_658_764 = np.where((self.wav_SCI >= self.wav2) & (self.wav_SCI < self.wav3))
        self.ind_764p    = np.where((self.wav_SCI >= self.wav3))

        cols_to_exclude = ['Date-Beg', 'Date-Beg-Corr','Date-End', 'Date-End-Corr'] # for sums below
        self.counts_SCI         = self.df_SCI_EM.drop(columns=cols_to_exclude).sum(axis=1) # flux (ADU) vs. time (per sample)
        self.counts_SCI_551m    = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI <  self.wav1))]].sum(axis=1)
        self.counts_SCI_551_658 = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= self.wav1) & (self.wav_SCI < self.wav2))]].sum(axis=1)
        self.counts_SCI_658_764 = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= self.wav2) & (self.wav_SCI < self.wav3))]].sum(axis=1)
        self.counts_SCI_764p    = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= self.wav3))]].sum(axis=1)
        self.counts_SKY         = self.df_SKY_EM.drop(columns=cols_to_exclude).sum(axis=1) # flux (ADU) vs. time (per sample)
        self.counts_SKY_551m    = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY <  self.wav1))]].sum(axis=1)
        self.counts_SKY_551_658 = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= self.wav1) & (self.wav_SKY < self.wav2))]].sum(axis=1)
        self.counts_SKY_658_764 = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= self.wav2) & (self.wav_SKY < self.wav3))]].sum(axis=1)
        self.counts_SKY_764p    = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= self.wav3))]].sum(axis=1)
        self.flux_SCI         = self.counts_SCI         / (self.wav4-self.wav0) / self.tdur_sec
        self.flux_SCI_551m    = self.counts_SCI_551m    / (self.wav1-self.wav0) / self.tdur_sec
        self.flux_SCI_551_658 = self.counts_SCI_551_658 / (self.wav2-self.wav1) / self.tdur_sec
        self.flux_SCI_658_764 = self.counts_SCI_658_764 / (self.wav3-self.wav2) / self.tdur_sec
        self.flux_SCI_764p    = self.counts_SCI_764p    / (self.wav4-self.wav3) / self.tdur_sec
        self.flux_SKY         = self.counts_SKY         / (self.wav4-self.wav0) / self.tdur_sec
        self.flux_SKY_551m    = self.counts_SKY_551m    / (self.wav1-self.wav0) / self.tdur_sec
        self.flux_SKY_551_658 = self.counts_SKY_551_658 / (self.wav2-self.wav1) / self.tdur_sec
        self.flux_SKY_658_764 = self.counts_SKY_658_764 / (self.wav3-self.wav2) / self.tdur_sec
        self.flux_SKY_764p    = self.counts_SKY_764p    / (self.wav4-self.wav3) / self.tdur_sec

        # Spectra
        self.df_SCI_EM_norm  = self.df_SCI_EM[self.wav_SCI_str] * self.EM_gain / self.disp_SCI_smooth
        self.df_SKY_EM_norm  = self.df_SKY_EM[self.wav_SCI_str] * self.EM_gain / self.disp_SKY_smooth
        if self.n_samples > 3:
            self.int_SCI_spec = self.df_SCI_EM_norm[1:-1].sum(axis=0) / np.sum(self.tdur_sec[1:-1]) # flux vs. wavelength per sec (don't use first and last frames because a worry about timing)
            self.int_SKY_spec = self.df_SKY_EM_norm[1:-1].sum(axis=0) / np.sum(self.tdur_sec[1:-1]) 
        else:
            self.int_SCI_spec = self.df_SCI_EM_norm.sum(axis=0) / np.sum(self.tdur_sec) 
            self.int_SKY_spec = self.df_SKY_EM_norm.sum(axis=0) / np.sum(self.tdur_sec) 
            
        # SCI / SKY ratio in main spectrometer (estimated from EM)
        self.SKY_SCI_main_spectrometer = np.nansum(self.counts_SKY/self.SKY_SCI_ratio) / np.nansum(self.counts_SCI)


    def plot_EM_time_series(self, fiber='both', fig_path=None, show_plot=False):
        """
        Generate a time series plot of the Exposure Meter spectra.  

        Args:
            fiber (string) - ['both', 'sci', 'sky', 'ratio'] - determines which EM spectra are plotted
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
        
        if fiber not in ['both', 'sky', 'sci', 'ratio']:
            self.logger.error('plot_EM_time_series: fiber argument must be "both", "sky", "sci", or "ratio".')

        # Plot flux array time series
        plt.figure(figsize=(12, 6), tight_layout=True)
        if fiber in ['both', 'sci']:
            plt.plot(self.time_em, self.flux_SCI_764p,    marker='o', linewidth=2, color='r', label = '764-870 nm')
            plt.plot(self.time_em, self.flux_SCI_658_764, marker='o', linewidth=2, color='orange', label = '658-764 nm')
            plt.plot(self.time_em, self.flux_SCI_551_658, marker='o', linewidth=2, color='g', label = '551-658 nm')
            plt.plot(self.time_em, self.flux_SCI_551m,    marker='o', linewidth=2, color='b', label = '445-551 nm')
            plt.plot(self.time_em, self.flux_SCI,         marker='o', linewidth=4, color='k', label = '445-870')
    
        if fiber in ['both', 'sky']:
            plt.plot(self.time_em, self.flux_SKY_764p,    ':', marker='o', linewidth=2, color='r', label = '764-870 nm')
            plt.plot(self.time_em, self.flux_SKY_658_764, ':', marker='o', linewidth=2, color='orange', label = '658-764 nm')
            plt.plot(self.time_em, self.flux_SKY_551_658, ':', marker='o', linewidth=2, color='g', label = '551-658 nm')
            plt.plot(self.time_em, self.flux_SKY_551m,    ':', marker='o', linewidth=2, color='b', label = '445-551 nm')
            plt.plot(self.time_em, self.flux_SKY,         ':', marker='o', linewidth=4, color='k', label = 'SKY 445-870 nm')

        if fiber == 'ratio':
            plt.plot(self.time_em, self.flux_SKY_764p    /self.SKY_SCI_ratio/ self.flux_SCI_764p   , marker='o', alpha=0.4, linewidth=2, color='r', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 764-870 nm')
            plt.plot(self.time_em, self.flux_SKY_658_764 /self.SKY_SCI_ratio/ self.flux_SCI_658_764, marker='o', alpha=0.4, linewidth=2, color='orange', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 658-764 nm')
            plt.plot(self.time_em, self.flux_SKY_551_658 /self.SKY_SCI_ratio/ self.flux_SCI_551_658, marker='o', alpha=0.4, linewidth=2, color='g', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 551-658 nm')
            plt.plot(self.time_em, self.flux_SKY_551m    /self.SKY_SCI_ratio/ self.flux_SCI_551m   , marker='o', alpha=0.4, linewidth=2, color='b', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 445-551 nm')
            plt.plot(self.time_em, self.flux_SKY         /self.SKY_SCI_ratio/ self.flux_SCI        , marker='o',            linewidth=4, color='k', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 445-870 nm')
        plt.plot(0, 0, marker='o', markersize=0.1, color='white') # force the y-axis to go to zero
        if fiber == 'both':
            plottitle = 'Exposure Meter Time Series (SCI and SKY) - ' + str(self.EM_nexp)+ r' EM exposures $\times$ ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'sky':
            plottitle = 'Exposure Meter Time Series (SKY) - ' + str(self.EM_nexp)+ r' EM exposures $\times$  ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'sci':
            plottitle = 'Exposure Meter Time Series (SCI) - ' + str(self.EM_nexp)+ r' EM exposures $\times$  ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'ratio':
            median_flux_ratio = np.nanmedian(self.flux_SKY/self.SKY_SCI_ratio / self.flux_SCI)
            avg_flux_ratio = np.nansum(self.flux_SKY/self.SKY_SCI_ratio) / np.nansum(self.flux_SCI)
            coefficient, exponent = f"{avg_flux_ratio:.2e}".split("e")
            flux_ratio_latex = r"${} \times 10^{{{}}}$".format(coefficient, int(exponent))
            plottitle = r'EM: total(SKY$_{\mathrm{corrected}}$) / total(SCI) = ' + flux_ratio_latex
            if avg_flux_ratio > 0:
                plottitle += r' $\rightarrow$ ' + "{:.3g}".format(-2.5 * math.log10(avg_flux_ratio)) + ' mag' 
                plottitle += ': ' + str(self.ObsID) + ' - ' + self.name
            else:
                 plottitle = ''
        plt.title(plottitle, fontsize=14)
        plt.xlabel("Time (sec)",fontsize=14)
        if fiber == 'ratio':
            plt.ylim(min([0.0, 1.2*min(self.flux_SKY/self.SKY_SCI_ratio/self.flux_SCI)]), 1.4*max(self.flux_SKY/self.SKY_SCI_ratio/self.flux_SCI))
            plt.ylabel(r'SKY$_{\mathrm{corrected}}$ / SCI Exposure Meter Flux Ratio',fontsize=14)
        elif fiber == 'sci':
            plt.ylabel("Exposure Meter SCI Flux (e-/nm/s)",fontsize=14)
        elif fiber == 'sky':
            plt.ylabel("Exposure Meter SKY Flux (e-/nm/s, uncorrected)",fontsize=14)
        if fiber == 'both':
            plt.yscale('log')
        elif fiber == 'ratio':
            # plot Delta Magnitude lines
            ymax = plt.gca().get_ylim()[1]
            done = False
            for m in np.arange(16):
                fr = 100**(-m/5)
                if (fr < ymax) and (not done):
                    done = True
                    plt.axhline(y=fr, color='gray', linestyle='-', label=r'$\Delta$mag = ' + str(m))
                    plt.axhline(y=fr/2.5119, color='gray', linestyle='--', label=r'$\Delta$mag = ' + str(m+1))
                    plt.axhline(y=fr/2.5119**2, color='gray', linestyle='-.', label=r'$\Delta\,$mag = ' + str(m+2))
                    plt.axhline(y=fr/2.5119**3, color='gray', linestyle=':', label=r'$\Delta\,$mag = ' + str(m+3))

        plt.xlim([-self.total_duration*0.03,self.total_duration*1.03])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if fiber == 'ratio':
            custom_line = mlines.Line2D([], [], color='white', label='')
            lines, labels = plt.gca().get_legend_handles_labels()
            lines.append(custom_line)
            labels.append(r'SKY / SKY$_{\mathrm{corrected}}$ = ' + str(self.SKY_SCI_ratio)) # ratio of measured sky to actual sky in SCI
            plt.legend(loc='upper left', ncol=2, handles=lines, labels=labels, framealpha = 0.8)
        else:
            plt.legend(framealpha=0.7)#loc='lower right')
     
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -30), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     
        
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_EM_spectrum(self, fig_path=None, show_plot=False):
        """
        Generate spectra from the Exposure Meter of the SCI and SKY fibers.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        # Plot spectra
        fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
        plt.axvspan(self.wav0, self.wav1, alpha=0.5, color='b')
        plt.axvspan(self.wav1, self.wav2, alpha=0.5, color='g')
        plt.axvspan(self.wav2, self.wav3, alpha=0.5, color='orange')
        plt.axvspan(self.wav3, self.wav4, alpha=0.5, color='red')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(self.wav0,self.wav4)
        lns1 = ax1.plot(self.wav_SCI, self.int_SCI_spec, marker='.', color='k', label ='SCI',zorder = 1)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(self.wav_SKY, self.int_SKY_spec, marker='.', color='brown', label = 'SKY',zorder = 0, alpha = 0.5)
        ax1.set_ylim(0,np.nanpercentile(self.int_SCI_spec,99.9)*1.1)
        ax2.set_ylim(0,np.nanpercentile(self.int_SKY_spec,99.9)*1.1)
        plt.title('Exposure Meter Spectrum: ' + str(self.EM_nexp)+ r' EM exposures $\times$ ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.yticks(fontsize=14, color='brown')
        ax1.set_xlabel("Wavelength (nm)",fontsize=14)
        ax1.set_ylabel("Exposure Meter SCI Flux (e-/nm/s)",fontsize=14)
        ax2.set_ylabel("Exposure Meter SKY Flux (e-/nm/s)",fontsize=14, color='brown')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0,fontsize=14)
     
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -30), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
