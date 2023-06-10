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
        master_master_date = self.config['Nightly']['master_master_date']


        if not os.path.exists(self.config['Nightly']['output_dir']+'/'+night):
            os.makedirs(self.config['Nightly']['output_dir']+'/'+night)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #plot the master files for a particular night
        master_list = glob.glob(masters_dir+night+'/*master*.fits')
        master_master_list = glob.glob(masters_dir+master_master_date+'/*master*.fits')
        print(master_list)
        print(masters_dir+master_master_date+'/*master*.fits',master_master_list)


        for i in range(len(master_list)):
            if master_list[i][-7:] == 'L1.fits' or master_list[i][-7:] == 'L2.fits': continue

            exposure_name = master_list[i][23:-5]
            version = master_list[i][35:-5]
            print(i,master_list[i],exposure_name,version)

            master_master_file = 'None'
            for j in range(len(master_master_list)):
                #print(j,master_master_list[i])
                if master_master_list[j][-7:] == 'L1.fits' or master_master_list[j][-7:] == 'L2.fits': continue
                #print('test j',version,master_master_list[j],master_master_list[j].find(version))
                if master_master_list[j].find(version)!=-1:
                    master_master_file = master_master_list[j]

            if master_master_file != 'None': hdulist1=fits.open(master_master_list[j])#identify master by the same type

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
            #print(ccd_color)

            #2d plots
            for i_color in range(len(ccd_color)):
                counts = np.array(hdulist[ccd_color[i_color]].data,'d')
                print(master_master_file)
                print(hdulist1.info())
                if master_master_file != 'None': master_counts = np.array(hdulist1[ccd_color[i_color]].data,'d')

                if master_list[i].find('flat')!=-1:

                    counts = np.array(hdulist[ccd_color[i_color]+'_STACK'].data,'d')
                    if master_master_file != 'None': master_counts = np.array(hdulist1[ccd_color[i_color]+'_STACK'].data,'d')
                if master_list[i].find('dark')!=-1:#scale up dark exposures
                    counts*=hdulist[0].header['EXPTIME']
                    if master_master_file != 'None':master_counts*=hdulist1[0].header['EXPTIME']

                flatten_counts = np.ravel(counts)
                if master_master_file != 'None': master_flatten_counts = np.ravel(master_counts)
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

                order_trace_file = self.config['L1']['order_trace']+ccd_color[i_color]+'.csv'
                order_trace = pd.read_csv(order_trace_file)
                #print(order_trace_file,order_trace)
                for j in range(np.shape(order_trace)[0]):#[50]:#range(np.shape(order_trace)[0])
                    #print(order_trace.iloc[i]['X1'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                    x_grid = np.linspace(order_trace.iloc[j]['X1'],order_trace.iloc[j]['X2'],int(order_trace.iloc[j]['X2']-order_trace.iloc[j]['X1']))
                    y_grid = order_trace.iloc[j]['Coeff0']+x_grid*order_trace.iloc[j]['Coeff1']+x_grid**2*order_trace.iloc[j]['Coeff2']+x_grid**3*order_trace.iloc[j]['Coeff3']
                    plt.plot(x_grid,y_grid,color ='magenta',linewidth = 0.2)
                    plt.plot(x_grid,y_grid-order_trace.iloc[j]['BottomEdge'],':',color ='white',linewidth = 0.2,alpha = 1)
                    plt.plot(x_grid,y_grid+order_trace.iloc[j]['TopEdge'],'--',color ='black',linewidth = 0.2,alpha = 1)
                    #plt.fill_between(x_grid,y_grid-order_trace.iloc[i]['BottomEdge'],y_grid+order_trace.iloc[i]['TopEdge'],color ='pink',alpha = 0.2)
                    #print(x_grid,y_grid)
                plt.xlim(3200,4000)
                plt.ylim(3200,4000)
                plt.title(ccd_color[i_color]+' Order Trace '+exposure_name, fontsize = 8)
                #plt.title(ccd_color[i_color]+' '+version+' Order Trace ' +exposure_name)
                #plt.savefig(output_dir+'fig/'+exposure_name+'_order_trace_'+ccd_color[i_color]+'.png')
                plt.savefig(output_dir+'/'+exposure_name+'_'+ccd_color[i_color]+'_order_trace.png', dpi=300)
                plt.close()

                if master_master_file != 'None':
                    plt.figure(figsize=(5,4))
                    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
                    #pcrint(counts,master_counts)
                    counts_norm = np.percentile(counts,99)
                    master_counts_norm = np.percentile(master_counts,99)
                    if np.shape(counts)!=np.shape(master_counts): continue
                    difference = counts/counts_norm-master_counts/master_counts_norm

                    plt.imshow(difference, vmin = np.percentile(difference,1),vmax = np.percentile(difference,99), interpolation = 'None',origin = 'lower')
                    plt.xlabel('x (pixel number)')
                    plt.ylabel('y (pixel number)')
                    plt.title(ccd_color[i_color]+' '+version+'- Master '+version+' '+exposure_name, fontsize =8)
                    plt.colorbar(label = 'Fractional Difference')
                    plt.savefig(output_dir+'/'+exposure_name+'_'+ccd_color[i_color]+'_2D_Difference_zoomable.png', dpi=1000)

                #histogram
                plt.close()
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

                #print(np.percentile(flatten_counts,99.9),saturation_limit)
                plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
                if master_master_file != 'None':
                    if len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
                #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
                plt.xlabel('Counts (e-)')
                plt.ylabel('Number of Pixels')
                plt.yscale('log')
                plt.title(ccd_color[i_color]+' '+exposure_name, fontsize = 8)
                plt.legend(loc='lower right')
                #plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png')
                plt.savefig(output_dir+'/'+exposure_name+'_'+ccd_color[i_color]+'_histogram.png', dpi=200)
                plt.close()







        #get all exposures taken on a particular night
        '''
        file_list = glob.glob(exposures_dir+night+'/*.fits')
        date_obs = []
        temp = []
        for i in range(len(file_list)):
            #file_list[i] = file_list[i][18:-8]

            hdulist = fits.open(file_list[i])
            hdr = hdulist[0].header
            print(hdr)
            date_obs.append(hdr['DATE'])
            temp.append(hdr['RELH'])

        date_obs = np.array(date_obs,'str')
        date_obs = Time(date_obs, format='isot', scale='utc')
        plt.close()
        plt.figure(figsize=(8,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
        plt.scatter(date_obs.jd-2460000,temp, marker = '.')
        plt.xlabel('Time (BJD-2460000)')
        plt.ylabel('Relative Humidity')
        #print(date_obs.jd,date_obs.utc)
        #plt.xlim(np.min(date_obs.jd),np.max(date_obs.jd))
        plt.savefig(output_dir+'/'+night+'_Relative_Humidity_variation.png')
        plt.close()
        '''
