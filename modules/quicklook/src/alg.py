import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
import os

class QuicklookAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger

    def plot_2d_frames(self,hdulist,output_dir):
        #print(self.config['output']['qlp_outdir'])
        saturation_limit = int(self.config['2D']['saturation_limit'])*1.
        plt.rcParams.update({'font.size': 8})
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

        #check if output location exist, if not create it

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir+'/fig')


        #print(hdulist)
        exposure_name = '1'
        #print(hdulist.header)
        hdr = hdulist.header
        version = hdr['PRIMARY']['IMTYPE']

        exposure_name = hdr['PRIMARY']['OFNAME'][:-5]
        print('working on',exposure_name)

        master_file = 'None'
        if version == 'Sol_All':
            master_file = self.onfig['2D']['master_socal']
        if version == 'Etalon_All':
            master_file = self.config['2D']['master_etalon']
        if version == 'Sol_All':
            master_file = self.config['2D']['master_socal']
        if version == 'Flat_All':
            master_file = self.config['2D']['master_flat']
        if version == 'Dark':
            master_file = self.config['2D']['master_dark']
        if version == 'Bias':
            master_file = self.config['2D']['master_bias']
        if version == 'Th_All':
            master_file = self.config['2D']['master_ThAr']
        if version == 'Une_All':
            master_file = self.config['2D']['master_Une']
        if version == 'LFC_SciCal':
            master_file = self.config['2D']['master_LFC']

        ccd_color = ['GREEN_CCD','RED_CCD']
        for i_color in range(len(ccd_color)):
            counts = np.array(hdulist[ccd_color[i_color]].data,'d')
            flatten_counts = np.ravel(counts)
            if len(flatten_counts)<1: continue
            master_flatten_counts='None'
            if master_file != 'None':
                hdulist1 = fits.open(master_file)
                master_counts = np.array(hdulist1[ccd_color[i_color]].data,'d')
                master_flatten_counts = np.ravel(master_counts)


            #2D image
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            plt.imshow(counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
            plt.xlabel('x (pixel number)')
            plt.ylabel('y (pixel number)')
            plt.title(ccd_color[i_color]+' '+version)
            plt.colorbar(label = 'Counts')
            plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.pdf')
            plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png', dpi=1000)
            #2D difference image
            plt.close()
            if master_file != 'None' and len(master_flatten_counts)>1:
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
                #pcrint(counts,master_counts)
                counts_norm = np.percentile(counts,99)
                master_counts_norm = np.percentile(master_counts,99)

                difference = counts/counts_norm-master_counts/master_counts_norm

                plt.imshow(difference, vmin = np.percentile(difference,1),vmax = np.percentile(difference,99), interpolation = 'None',origin = 'lower')
                plt.xlabel('x (pixel number)')
                plt.ylabel('y (pixel number)')
                plt.title(ccd_color[i_color]+' '+version+'- Master '+version)
                plt.colorbar(label = 'Fractional Difference')
                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'.pdf')
                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'.png', dpi=500)
             #Hisogram
            plt.close()
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

            #print(np.percentile(flatten_counts,99.9),saturation_limit)
            plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f' % np.nanmedian(flatten_counts)+'; Saturated? '+str(np.percentile(flatten_counts,99.9)>saturation_limit),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
            if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
            #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
            plt.xlabel('Counts')
            plt.ylabel('Number of Pixels')
            plt.yscale('log')
            plt.title(ccd_color[i_color]+' '+version+' Histogram')
            plt.legend()
            plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.pdf')
            plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png', dpi=200)

            #Column cut
            plt.close()
            plt.figure(figsize=(8,4))
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

            column_sum = np.nansum(counts,axis = 0)
            #print('which_column',np.where(column_sum==np.nanmax(column_sum))[0][0])
            which_column = np.where(column_sum==np.nanmax(column_sum))[0][0] #int(np.shape(master_counts)[1]/2)

            plt.plot(np.ones_like(counts[:,which_column])*saturation_limit,':',alpha = 0.5,linewidth =  1., label = 'Saturation Limit', color = 'gray')
            plt.plot(counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = ccd_color[i_color]+' '+version, color = 'Blue')
            if master_file != 'None' and len(master_flatten_counts)>1: plt.plot(master_counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = 'Master', color = 'Orange')
            plt.yscale('log')
            plt.ylabel('log(Counts)')
            plt.xlabel('Row Number')
            plt.title(ccd_color[i_color]+' '+version+' Column Cut Through Column '+str(which_column))#(Middle of CCD)
            plt.ylim(1,1.2*np.nanmax(counts[:,which_column]))
            plt.legend()
            plt.savefig(output_dir+'fig/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'.pdf')
            plt.savefig(output_dir+'fig/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'.png', dpi=200)

    def plot_1d_spectrum(self,hdulist_1d,output_dir):
        print('L1 runs', hdulist_1d, output_dir)
