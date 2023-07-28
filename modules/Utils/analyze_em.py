import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.table import Table
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
        TBD
    """

    def __init__(self, L0, logger=None):

        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeEM object.')
        else:
            self.logger = None
        self.L0 = L0 
        primary_header = HeaderParse(L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()

        self.EM_gain = 1.48424 #np.float(self.config['EM']['gain'])

        # Read data tables
        self.dat_SCI = L0['EXPMETER_SCI']
        self.dat_SKY = L0['EXPMETER_SKY']
        self.df_SCI_EM = self.dat_SCI
        self.df_SKY_EM = self.dat_SKY
        #self.dat_SCI = Table.read(self.L0, format='fits',hdu='EXPMETER_SCI')
        #self.dat_SKY = Table.read(self.L0, format='fits',hdu='EXPMETER_SKY')
        #self.df_SKY_EM = self.dat_SKY.to_pandas() 
        #self.df_SCI_EM = self.dat_SCI.to_pandas()
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
        
        #self.logger.info(help(self.df_SKY_EM))
        # Exposure time
        if 'Date-End-Corr' in self.df_SCI_EM.columns:
            timediff = np.array(self.df_SCI_EM["Date-End-Corr"], dtype=np.datetime64) - np.array(self.df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            self.EM_texp = pd.Timedelta(np.median(timediff)).total_seconds() # EM exposure time in seconds
            self.EM_nexp = timediff.shape[0]
        elif 'Date-End' in self.df_SCI_EM.columns:
            timediff = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64) - np.array(self.df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            self.EM_texp = pd.Timedelta(np.median(timediff)).total_seconds()
            self.EM_nexp = timediff.shape[0]
        elif 'Date-End' in self.df_SCI_EM.columns:
            self.EM_texp = 0.
            self.EM_nexp = 0

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

        # Define time arrays
        date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        if 'Date-Beg-Corr' in self.df_SCI_EM.columns:
            date_beg = np.array(self.df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            date_end = np.array(self.df_SCI_EM["Date-End-Corr"], dtype=np.datetime64)
        else:
            date_beg = np.array(self.df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        tdur_sec = (date_end-date_beg).astype(float)/1000. # exposure duration in sec
        time_em  = (date_beg-date_beg[0]).astype(float)/1000. # seconds since beginning
        
        # Define flux arrays
        ind_550m    = np.where((self.wav_SCI <  550))
        ind_550_650 = np.where((self.wav_SCI >= 550) & (self.wav_SCI < 650))
        ind_650_750 = np.where((self.wav_SCI >= 650) & (self.wav_SCI < 750))
        ind_750p    = np.where((self.wav_SCI >= 750))
        int_SCI_flux         = self.df_SCI_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
        int_SCI_flux_550m    = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI <  550))]].sum(axis=1)
        int_SCI_flux_550_650 = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= 550) & (self.wav_SCI < 650))]].sum(axis=1)
        int_SCI_flux_650_750 = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= 650) & (self.wav_SCI < 750))]].sum(axis=1)
        int_SCI_flux_750p    = self.df_SCI_EM[self.wav_SCI_str[np.where((self.wav_SCI >= 750))]].sum(axis=1)
        int_SKY_flux         = self.df_SKY_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
        int_SKY_flux_550m    = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY <  550))]].sum(axis=1)
        int_SKY_flux_550_650 = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= 550) & (self.wav_SKY < 650))]].sum(axis=1)
        int_SKY_flux_650_750 = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= 650) & (self.wav_SKY < 750))]].sum(axis=1)
        int_SKY_flux_750p    = self.df_SKY_EM[self.wav_SKY_str[np.where((self.wav_SKY >= 750))]].sum(axis=1)

        SKY_SCI_ratio = 14.1 # ratio of flux in SKY to SCI fibers based on bright twilight observations (e.g., KP.20230114.15771.04)

        # Plot time series
        plt.figure(figsize=(12, 6), tight_layout=True)
        total_duration = (date_end[-1]-date_beg[0]).astype(float)/1000.
        if fiber in ['both', 'sci']:
            plt.plot(time_em, int_SCI_flux_750p    / (870-750) / tdur_sec, marker='o', linewidth=2, color='r', label = '750-870 nm')
            plt.plot(time_em, int_SCI_flux_650_750 / (750-650) / tdur_sec, marker='o', linewidth=2, color='orange', label = '650-750 nm')
            plt.plot(time_em, int_SCI_flux_550_650 / (650-550) / tdur_sec, marker='o', linewidth=2, color='g', label = '550-650 nm')
            plt.plot(time_em, int_SCI_flux_550m    / (550-445) / tdur_sec, marker='o', linewidth=2, color='b', label = '445-550 nm')
            plt.plot(time_em, int_SCI_flux         / (870-445) / tdur_sec, marker='o', linewidth=4, color='k', label = 'SCI 445-870 nm')
    
        if fiber in ['both', 'sky']:
            plt.plot(time_em, int_SKY_flux_750p    / (870-750) / tdur_sec,':', marker='o', linewidth=2, color='r', label = '750-870 nm')
            plt.plot(time_em, int_SKY_flux_650_750 / (750-650) / tdur_sec,':', marker='o', linewidth=2, color='orange', label = '650-750 nm')
            plt.plot(time_em, int_SKY_flux_550_650 / (650-550) / tdur_sec,':', marker='o', linewidth=2, color='g', label = '550-650 nm')
            plt.plot(time_em, int_SKY_flux_550m    / (550-445) / tdur_sec,':', marker='o', linewidth=2, color='b', label = '445-550 nm')
            plt.plot(time_em, int_SKY_flux         / (870-445) / tdur_sec,':', marker='o', linewidth=4, color='k', label = 'SKY 445-870 nm')

        if fiber == 'ratio':
            plt.plot(time_em, int_SKY_flux_750p    /SKY_SCI_ratio/ int_SCI_flux_750p    / tdur_sec, marker='o', alpha=0.4, linewidth=2, color='r', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 750-870 nm')
            plt.plot(time_em, int_SKY_flux_650_750 /SKY_SCI_ratio/ int_SCI_flux_650_750 / tdur_sec, marker='o', alpha=0.4, linewidth=2, color='orange', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 650-750 nm')
            plt.plot(time_em, int_SKY_flux_550_650 /SKY_SCI_ratio/ int_SCI_flux_550_650 / tdur_sec, marker='o', alpha=0.4, linewidth=2, color='g', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 550-650 nm')
            plt.plot(time_em, int_SKY_flux_550m    /SKY_SCI_ratio/ int_SCI_flux_550m    / tdur_sec, marker='o', alpha=0.4, linewidth=2, color='b', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 445-550 nm')
            plt.plot(time_em, int_SKY_flux         /SKY_SCI_ratio/ int_SCI_flux         / tdur_sec, marker='o',            linewidth=4, color='k', label = r'SKY$_{\mathrm{corrected}}$ / SCI - 445-870 nm')
        plt.plot(0, 0, marker='o', markersize=0.1, color='white') # force the y-axis to go to zero
        if fiber == 'both':
            plottitle = 'Exposure Meter Time Series (SCI and SKY) - ' + str(self.EM_nexp)+ ' EM exposures, ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'sky':
            plottitle = 'Exposure Meter Time Series (SKY) - ' + str(self.EM_nexp)+ ' EM exposures, ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'sci':
            plottitle = 'Exposure Meter Time Series (SCI) - ' + str(self.EM_nexp)+ ' EM exposures, ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name
        elif fiber == 'ratio':
            median_flux_ratio = np.nanmedian(int_SKY_flux/SKY_SCI_ratio / int_SCI_flux)
            avg_flux_ratio = np.nansum(int_SKY_flux/SKY_SCI_ratio) / np.nansum(int_SCI_flux)
            coefficient, exponent = f"{avg_flux_ratio:.2e}".split("e")
            flux_ratio_latex = r"${} \times 10^{{{}}}$".format(coefficient, int(exponent))
            plottitle = r'EM Time Series - total(SKY$_{\mathrm{corrected}}$) / total(SCI) = ' + flux_ratio_latex + r' $\rightarrow \geq$' + "{:.3g}".format(-2.5 * math.log10(avg_flux_ratio)) + ' mag: ' + str(self.ObsID) + ' - ' + self.name
        else:
             plottitle = ''
        plt.title(plottitle, fontsize=14)
        plt.xlabel("Time (sec)",fontsize=14)
        if fiber == 'ratio':
#            plt.ylim(0.8*min(int_SKY_flux/SKY_SCI_ratio/int_SCI_flux), 1.4*max(int_SKY_flux/SKY_SCI_ratio/int_SCI_flux))
            plt.ylim(min([0.0, 1.2*min(int_SKY_flux/SKY_SCI_ratio/int_SCI_flux)]), 1.4*max(int_SKY_flux/SKY_SCI_ratio/int_SCI_flux))
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

        plt.xlim([-total_duration*0.03,total_duration*1.03])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if fiber == 'ratio':
            custom_line = mlines.Line2D([], [], color='white', label='')
            lines, labels = plt.gca().get_legend_handles_labels()
            lines.append(custom_line)
            labels.append(r'SKY / SKY$_{\mathrm{corrected}}$ = ' + str(SKY_SCI_ratio)) # ratio of measured sky to actual sky in SCI
            plt.legend(loc='upper left', ncol=2, handles=lines, labels=labels, framealpha = 0.8)
        else:
            plt.legend(framealpha=0.7)#loc='lower right')
        
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

        # Define dispersion
        disp_SCI = self.wav_SCI*0+np.gradient(self.wav_SCI,1)*-1
        disp_SKY = self.wav_SKY*0+np.gradient(self.wav_SKY,1)*-1
        disp_SCI_smooth = np.polyval(np.polyfit(self.wav_SCI,disp_SCI, deg=6),self.wav_SCI)
        disp_SKY_smooth = np.polyval(np.polyfit(self.wav_SKY,disp_SKY, deg=6),self.wav_SKY)
        df_SCI_EM_norm  = self.df_SCI_EM[self.wav_SCI_str] * self.EM_gain /disp_SCI_smooth
        df_SKY_EM_norm  = self.df_SKY_EM[self.wav_SCI_str] * self.EM_gain /disp_SKY_smooth
        
        # Define time arrays
        date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        n_samples = df_SCI_EM_norm.shape[0]
        if 'Date-Beg-Corr' in self.df_SCI_EM.columns:
            date_beg = np.array(self.df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            date_end = np.array(self.df_SCI_EM["Date-End-Corr"], dtype=np.datetime64)
        else:
            date_beg = np.array(self.df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            date_end = np.array(self.df_SCI_EM["Date-End"], dtype=np.datetime64)
        tdur_sec = (date_end-date_beg).astype(float)/1000. # exposure duration in sec
        time_em  = (date_beg-date_beg[0]).astype(float)/1000. # seconds since beginning
        
        if n_samples > 3:
            int_SCI_spec = df_SCI_EM_norm[1:-1].sum(axis=0) / np.sum(tdur_sec[1:-1]) # flux vs. wavelength per sec (don't use first and last frames because a worry about timing)
            int_SKY_spec = df_SKY_EM_norm[1:-1].sum(axis=0) / np.sum(tdur_sec[1:-1]) 
        else:
            int_SCI_spec = df_SCI_EM_norm.sum(axis=0) / np.sum(tdur_sec) 
            int_SKY_spec = df_SKY_EM_norm.sum(axis=0) / np.sum(tdur_sec) 
        

        # Plot spectra
        #plt.figure(figsize=(10, 4), tight_layout=True)
        fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
        plt.axvspan(445, 550, alpha=0.5, color='b')
        plt.axvspan(550, 650, alpha=0.5, color='g')
        plt.axvspan(650, 750, alpha=0.5, color='orange')
        plt.axvspan(750, 870, alpha=0.5, color='red')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(445,870)
        lns1 = ax1.plot(self.wav_SCI, int_SCI_spec, marker='.', color='k', label ='SCI',zorder = 1)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(self.wav_SKY, int_SKY_spec, marker='.', color='brown', label = 'SKY',zorder = 0, alpha = 0.5)
        ax1.set_ylim(0,np.percentile(int_SCI_spec,99.9)*1.1)
        ax2.set_ylim(0,np.percentile(int_SKY_spec,99.9)*1.1)
        plt.title('Exposure Meter Spectrum: ' + str(self.EM_nexp)+ ' EM exposures, ' + str(self.EM_texp)+ ' sec - ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.yticks(fontsize=14, color='brown')
        ax1.set_xlabel("Wavelength (nm)",fontsize=14)
        ax1.set_ylabel("Exposure Meter SCI Flux (e-/nm/s)",fontsize=14)
        ax2.set_ylabel("Exposure Meter SKY Flux (e-/nm/s)",fontsize=14, color='brown')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0,fontsize=14)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
