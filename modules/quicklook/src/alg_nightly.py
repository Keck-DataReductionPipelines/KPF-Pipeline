import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
import os
import pandas as pd
import glob
import math
from astropy import modeling
from astropy.time import Time
from datetime import datetime


class Nightly_summaryAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger




    def nightly_procedures(self,night):
        exposures_dir = self.config['Nightly']['exposures_dir']
        masters_dir = self.config['Nightly']['masters_dir']
        output_dir = self.config['Nightly']['output_dir']+'/'+night+'/nightly_summary'

        if not os.path.exists(self.config['Nightly']['output_dir']+'/'+night):
            os.makedirs(self.config['Nightly']['output_dir']+'/'+night)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #plot the master files for a particular night
        master_list = glob.glob(masters_dir+night+'/*master_dark*.fits')
        print(master_list)


        for i in range(len(master_list)):
            if master_list[i][-7:] == 'L1.fits' or master_list[i][-7:] == 'L2.fits': continue

            exposure_name = master_list[i][23:-5]
            print(master_list[i],exposure_name)


            L0_data = master_list[i]
            hdulist = fits.open(L0_data)
            hdr = hdulist[0].header

            exptime = hdr['EXPTIME']
            print(hdulist.info())

            #get ccd names
            ccd_color=[]
            ccd_list = self.config.items( "CCD_LIST")
            for key, path in ccd_list:
                ccd_color.append(path)


            if len(hdulist[ccd_color[0]].data)<1 and len(hdulist[ccd_color[1]].data)<1:
                print('skipping empty file')
                return
            print(ccd_color)

            #2d plots
            for i_color in range(len(ccd_color)):
                counts = np.array(hdulist[ccd_color[i_color]].data,'d')

                if master_list[i].find('dark'):#scale up dark exposures
                    counts*=exptime

                flatten_counts = np.ravel(counts)
                if len(flatten_counts)<1: continue
                #master_flatten_counts='None'


                #2D image
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
                plt.imshow(counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
                plt.xlabel('x (pixel number)')
                plt.ylabel('y (pixel number)')
                plt.title(ccd_color[i_color]+' '+exposure_name, fontsize = 8)
                plt.colorbar(label = 'Counts (e-)')


                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png')
                #print(output_dir+'/'+exposure_name+ccd_color[i_color]+'_zoomable.png')
                plt.savefig(output_dir+'/'+exposure_name+'_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
                #plt.close()

                #histogram
                plt.close()
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

                #print(np.percentile(flatten_counts,99.9),saturation_limit)
                plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
                #if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
                #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
                plt.xlabel('Counts (e-)')
                plt.ylabel('Number of Pixels')
                plt.yscale('log')
                plt.title(ccd_color[i_color]+' '+exposure_name, fontsize = 8)
                plt.legend(loc='lower right')
                #plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png')
                plt.savefig(output_dir+'/'+exposure_name+'_'+ccd_color[i_color]+'_histogram.png', dpi=200)





        #get all exposures taken on a particular night

        file_list = glob.glob(exposures_dir+night+'/*.fits')
        date_obs = []
        temp = []
        for i in range(len(file_list)):
            #file_list[i] = file_list[i][18:-8]

            hdulist = fits.open(file_list[i])
            hdr = hdulist[0].header
            print(hdr)
            date_obs.append(hdr['DATE-OBS'])
            temp.append(hdr['RELH'])

        date_obs = np.array(date_obs,'str')
        date_obs = Time(date_obs, format='isot', scale='utc')
        plt.scatter(date_obs.jd,temp, marker = '.')
        plt.xlabel('Time')
        plt.ylabel('Relative Humidity')
        plt.savefig(output_dir+'/'+night+'_Relative_Humidity_variation.png')
        plt.close()
