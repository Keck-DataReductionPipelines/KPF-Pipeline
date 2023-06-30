import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

class AnalyzeEM:

    """
    Description:
        This class contains functions to analyze EM data 
        (storing them as attributes) and functions to plot the results.
        Some of the functions need to be filled in.

    Arguments:
        L0 - an L0 object

    Attributes:
        TBD
    """

    def __init__(self, L0, logger=None):
        self.L0 = L0 
        if logger:
            self.logger = logger
            self.logger.debug('AnalyzeEM class constructor')
        else:
            self.logger = None
            print('---->AnalyzeEM class constructor')

    def plot_EM_time_series(self, fig_path=None, show_plot=False):

        """
        Generate a time series plot of the Exposure Meter spectra.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        EM_gain = 1.48424 #np.float(self.config['EM']['gain'])
        starname = L0['PRIMARY'].header['TARGNAME']
        ObsID = L0['PRIMARY'].header['OFNAME']

        # Read data tables
        dat_SKY = Table.read(L0, format='fits',hdu='EXPMETER_SKY')
        dat_SCI = Table.read(L0, format='fits',hdu='EXPMETER_SCI')
        df_SKY_EM = dat_SKY.to_pandas()
        df_SCI_EM = dat_SCI.to_pandas()
        i = 0
        for col in df_SCI_EM.columns:
            if col.lower().startswith('date'):
                i += 1
            else:
                break
        wav_SCI_str = df_SCI_EM.columns[i:]
        wav_SCI     = df_SCI_EM.columns[i:].astype(float)
        wav_SKY_str = df_SKY_EM.columns[i:]
        wav_SKY     = df_SKY_EM.columns[i:].astype(float)

        # Define time arrays
        date_end = np.array(df_SCI_EM["Date-End"], dtype=np.datetime64)
        if 'Date-Beg-Corr' in df_SCI_EM.columns:
            date_beg = np.array(df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            date_end = np.array(df_SCI_EM["Date-End-Corr"], dtype=np.datetime64)
        else:
            date_beg = np.array(df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            date_end = np.array(df_SCI_EM["Date-End"], dtype=np.datetime64)
        tdur_sec = (date_end-date_beg).astype(float)/1000. # exposure duration in sec
        time_em     = (date_beg-date_beg[0]).astype(float)/1000. # seconds since beginning
        
        # Define flux arrays
        ind_550m    = np.where((wav_SCI <  550))
        ind_550_650 = np.where((wav_SCI >= 550) & (wav_SCI < 650))
        ind_650_750 = np.where((wav_SCI >= 650) & (wav_SCI < 750))
        ind_750p    = np.where((wav_SCI >= 750))
        int_SCI_flux         = df_SCI_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
        int_SCI_flux_550m    = df_SCI_EM[wav_SCI_str[np.where((wav_SCI <  550))]].sum(axis=1)
        int_SCI_flux_550_650 = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 550) & (wav_SCI < 650))]].sum(axis=1)
        int_SCI_flux_650_750 = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 650) & (wav_SCI < 750))]].sum(axis=1)
        int_SCI_flux_750p    = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 750))]].sum(axis=1)
        int_SKY_flux         = df_SKY_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
        int_SKY_flux_550m    = df_SKY_EM[wav_SKY_str[np.where((wav_SKY <  550))]].sum(axis=1)
        int_SKY_flux_550_650 = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 550) & (wav_SKY < 650))]].sum(axis=1)
        int_SKY_flux_650_750 = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 650) & (wav_SKY < 750))]].sum(axis=1)
        int_SKY_flux_750p    = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 750))]].sum(axis=1)

        # Plot time series
        plt.figure(figsize=(10, 4), tight_layout=True)
        total_duration = (date_end[-1]-date_beg[0]).astype(float)/1000.
        plt.plot(time_em, int_SCI_flux_750p    / (870-750) / tdur_sec, marker='o', color='r', label = '750-870 nm')
        plt.plot(time_em, int_SCI_flux_650_750 / (750-650) / tdur_sec, marker='o', color='orange', label = '650-750 nm')
        plt.plot(time_em, int_SCI_flux_550_650 / (650-550) / tdur_sec, marker='o', color='g', label = '550-650 nm')
        plt.plot(time_em, int_SCI_flux_550m    / (550-445) / tdur_sec, marker='o', color='b', label = '445-550 nm')
        plt.plot(time_em, int_SCI_flux         / (870-445) / tdur_sec, marker='o', color='k', label = 'SCI 445-870 nm')
        plt.plot(time_em, int_SKY_flux_750p    / (870-750) / tdur_sec,':', marker='o', color='r')
        plt.plot(time_em, int_SKY_flux_650_750 / (750-650) / tdur_sec,':', marker='o', color='orange')
        plt.plot(time_em, int_SKY_flux_550_650 / (650-550) / tdur_sec,':', marker='o', color='g')
        plt.plot(time_em, int_SKY_flux_550m    / (550-445) / tdur_sec,':', marker='o', color='b')
        plt.plot(time_em, int_SKY_flux         / (870-445) / tdur_sec,':', marker='o', color='k', label = 'SKY 445-870 nm')
        plt.title('Exposure Meter Time Series: ' + str(ObsID) + ' - ' + starname, fontsize=14)
        plt.xlabel("Time (sec)",fontsize=14)
        plt.ylabel("Exposure Meter Flux (e-/nm/s)",fontsize=14)
        plt.yscale('log')
        plt.xlim([-total_duration*0.03,total_duration*1.18])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc='best')
        
        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()

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

        EM_gain = 1.48424 #np.float(self.config['EM']['gain'])
        starname = L0['PRIMARY'].header['TARGNAME']
        ObsID = L0['PRIMARY'].header['OFNAME']

        # Read data tables
        dat_SKY = Table.read(L0, format='fits',hdu='EXPMETER_SKY')
        dat_SCI = Table.read(L0, format='fits',hdu='EXPMETER_SCI')
        df_SKY_EM = dat_SKY.to_pandas()
        df_SCI_EM = dat_SCI.to_pandas()
        i = 0
        for col in df_SCI_EM.columns:
            if col.lower().startswith('date'):
                i += 1
            else:
                break
        wav_SCI_str = df_SCI_EM.columns[i:]
        wav_SCI     = df_SCI_EM.columns[i:].astype(float)
        wav_SKY_str = df_SKY_EM.columns[i:]
        wav_SKY     = df_SKY_EM.columns[i:].astype(float)

        # Define dispersion
        disp_SCI = wav_SCI*0+np.gradient(wav_SCI,1)*-1
        disp_SKY = wav_SKY*0+np.gradient(wav_SKY,1)*-1
        disp_SCI_smooth = np.polyval(np.polyfit(wav_SCI,disp_SCI, deg=6),wav_SCI)
        disp_SKY_smooth = np.polyval(np.polyfit(wav_SKY,disp_SKY, deg=6),wav_SKY)
        df_SCI_EM_norm        = df_SCI_EM[wav_SCI_str] * EM_gain /disp_SCI_smooth
        df_SKY_EM_norm        = df_SKY_EM[wav_SCI_str] * EM_gain /disp_SKY_smooth
        
        # Define time arrays
        date_end = np.array(df_SCI_EM["Date-End"], dtype=np.datetime64)
        if 'Date-Beg-Corr' in df_SCI_EM.columns:
            date_beg = np.array(df_SCI_EM["Date-Beg-Corr"], dtype=np.datetime64)
            date_end = np.array(df_SCI_EM["Date-End-Corr"], dtype=np.datetime64)
        else:
            date_beg = np.array(df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            date_end = np.array(df_SCI_EM["Date-End"], dtype=np.datetime64)
        tdur_sec = (date_end-date_beg).astype(float)/1000. # exposure duration in sec
        time_em  = (date_beg-date_beg[0]).astype(float)/1000. # seconds since beginning
        int_SCI_spec         = df_SCI_EM_norm[:5].sum(axis=0) / np.sum(tdur_sec[:5]) # flux vs. wavelength per sec (use first five samples)
        int_SKY_spec         = df_SKY_EM_norm[:5].sum(axis=0) / np.sum(tdur_sec[:5]) # flux vs. wavelength per sec (use first five samples)

        # Plot spectra
        plt.figure(figsize=(10, 4), tight_layout=True)
        fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
        plt.axvspan(445, 550, alpha=0.5, color='b')
        plt.axvspan(550, 650, alpha=0.5, color='g')
        plt.axvspan(650, 750, alpha=0.5, color='orange')
        plt.axvspan(750, 870, alpha=0.5, color='red')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(445,870)
        lns1 = ax1.plot(wav_SCI, int_SCI_spec, marker='.', color='k', label ='SCI',zorder = 1)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(wav_SKY, int_SKY_spec, marker='.', color='brown', label = 'SKY',zorder = 0, alpha = 0.5)
        ax1.set_ylim(0,np.percentile(int_SCI_spec,99.9)*1.1)
        ax2.set_ylim(0,np.percentile(int_SKY_spec,99.9)*1.1)
        plt.title('Exposure Meter Spectrum: ' + str(ObsID) + ' - ' + starname, fontsize=14)
        plt.yticks(fontsize=14, color='brown')
        ax1.set_xlabel("Wavelength (nm)",fontsize=14)
        ax1.set_ylabel("SCI Exposure Meter Flux (e-/nm/s)",fontsize=14)
        ax2.set_ylabel("SKY Exposure Meter Flux (e-/nm/s)",fontsize=14, color='brown')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0,fontsize=14)

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()
