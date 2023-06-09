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
        output_dir = self.config['Nightly']['output_dir']+'/'+night+'/nightly_summary/'

        if not os.path.exists(self.config['Nightly']['output_dir']+'/'+night):
            os.makedirs(self.config['Nightly']['output_dir']+'/'+night)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #plot the master files for a particular night
        master_list = glob.glob(masters_dir+night+'/*master*.fits')
        print(master_list)


        for i in range(len(master_list)):
            if master_list[i][-7:] == 'L1.fits' or master_list[i][-7:] == 'L2.fits': continue
            print(master_list[i])


            L0_data = masters_dir+night+master_list[i]
            hdulist = fits.open(L0_data)
            print(hdulist.info())
            '''
            #get ccd names
            ccd_color=[]
            ccd_list = self.config.items( "CCD_LIST")
            for key, path in ccd_list:
                ccd_color.append(path)


            if len(hdulist[ccd_color[0]].data)<1 and len(hdulist[ccd_color[1]].data)<1:
                print('skipping empty file')
                return


            #2d plots
            for i_color in range(len(ccd_color)):
                counts = np.array(hdulist[ccd_color[i_color]].data,'d')
                flatten_counts = np.ravel(counts)
                if len(flatten_counts)<1: continue
                master_flatten_counts='None'
                if master_file != 'None':
                    hdulist1 = fits.open(master_file)
                    master_counts = np.array(hdulist1[ccd_color[i_color]].data,'d')
                    master_flatten_counts = np.ravel(master_counts)
                    if version == 'Dark':#scale up dark exposures
                        master_flatten_counts*=hdr['EXPTIME']

                #2D image
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
                plt.imshow(counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
                plt.xlabel('x (pixel number)')
                plt.ylabel('y (pixel number)')
                plt.title(ccd_color[i_color]+' '+version +' '+exposure_name)
                plt.colorbar(label = 'Counts (e-)')


                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png')
                plt.savefig(output_dir+'/'+master_list[i][:-5]+ccd_color[i_color]+'_zoomable.png', dpi=1000)
                #plt.close()


            if master_list[i].find('bias') == True or master_list[i].find('dark'):
                exptime = hdr['EXPTIME']
                print('exptime',exptime)

                # Read telemetry
                from astropy.table import Table
                df_telemetry = Table.read(L0_data, format='fits', hdu=11).to_pandas() # need to refer to HDU by name
                num_columns = ['average', 'stddev', 'min', 'max']
                for column in df_telemetry:
                    df_telemetry[column] = df_telemetry[column].str.decode('utf-8')
                    df_telemetry = df_telemetry.replace('-nan', 0)# replace nan with 0
                    if column in num_columns:
                        df_telemetry[column] = pd.to_numeric(df_telemetry[column], downcast="float")
                    else:
                        df_telemetry[column] = df_telemetry[column].astype(str)
                df_telemetry.set_index("keyword", inplace=True)

                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    print(df_telemetry)
                if ccd_color[i_color] == 'GREEN_CCD':
                    coll_pressure_torr = df_telemetry.at['kpfgreen.COL_PRESS', 'average']
                    ech_pressure_torr  = df_telemetry.at['kpfgreen.ECH_PRESS', 'average']
                    coll_current_a     = df_telemetry.at['kpfgreen.COL_CURR',  'average']
                    ech_current_a      = df_telemetry.at['kpfgreen.ECH_CURR',  'average']
                if ccd_color[i_color] == 'RED_CCD':
                    coll_pressure_torr = df_telemetry.at['kpfred.COL_PRESS', 'average']
                    ech_pressure_torr  = df_telemetry.at['kpfred.ECH_PRESS', 'average']
                    coll_current_a     = df_telemetry.at['kpfred.COL_CURR',  'average']
                    ech_current_a      = df_telemetry.at['kpfred.ECH_CURR',  'average']

                frame = counts
                if exptime > 0:
                    exptype = 'dark'
                    timelabel = ' e$^-$ hr$^{-1}$'
                    frame *= (3600./exptime)  # convert to e- per hour
                # Bias frame
                else:
                    exptype = 'bias'
                    timelabel = ' e$^-$'
                reg = {'ref1': {'name': 'Reference Region 1',         'x1': 1690, 'x2': 1990, 'y1': 1690, 'y2': 1990, 'short':'ref1', 'med_elec':0, 'label':''},
                           'ref2': {'name': 'Reference Region 2',         'x1': 1690, 'x2': 1990, 'y1': 2090, 'y2': 2390, 'short':'ref2', 'med_elec':0, 'label':''},
                           'ref3': {'name': 'Reference Region 3',         'x1': 2090, 'x2': 2390, 'y1': 1690, 'y2': 1990, 'short':'ref3', 'med_elec':0, 'label':''},
                           'ref4': {'name': 'Reference Region 4',         'x1': 2090, 'x2': 2390, 'y1': 2090, 'y2': 2390, 'short':'ref4', 'med_elec':0, 'label':''},
                           'ref5': {'name': 'Reference Region 5',         'x1':   80, 'x2':  380, 'y1':  700, 'y2': 1000, 'short':'ref5', 'med_elec':0, 'label':''},
                           'ref6': {'name': 'Reference Region 6',         'x1':   80, 'x2':  380, 'y1': 3080, 'y2': 3380, 'short':'ref6', 'med_elec':0, 'label':''},
                           'amp1': {'name': 'Amplifier Region 1',         'x1':  300, 'x2':  500, 'y1':    5, 'y2':   20, 'short':'amp1', 'med_elec':0, 'label':''},
                           'amp2': {'name': 'Amplifier Region 2',         'x1': 3700, 'x2': 3900, 'y1':    5, 'y2':   20, 'short':'amp2', 'med_elec':0, 'label':''},
                           'coll': {'name': 'Ion Pump (Collimator side)', 'x1': 3700, 'x2': 4000, 'y1':  700, 'y2': 1000, 'short':'coll', 'med_elec':0, 'label':''},
                           'ech':  {'name': 'Ion Pump (Echelle side)',    'x1': 3700, 'x2': 4000, 'y1': 3080, 'y2': 3380, 'short':'ech',  'med_elec':0, 'label':''}
                          }
                for r in reg.keys():
                    current_region = frame[reg[r]['y1']:reg[r]['y2'],reg[r]['x1']:reg[r]['x2']]
                    reg[r]['med_elec'] = np.median(current_region)

                #print(reg[r]['name'] + ': ' + str(np.round(reg[r]['med_elec'],1)) + ' e- per hour')
                #print('Ion Pump pressure (Torr) - Collimator side: ' + f'{coll_pressure_torr:.1e}')
                #print('Ion Pump pressure (Torr) - Echelle side: '    + f'{ech_pressure_torr:.1e}')
                #print('Ion Pump current (A) - Collimator side: '     + f'{coll_current_a:.1e}')
                #print('Ion Pump current (A) - Echelle side: '        + f'{ech_current_a:.1e}')

                from matplotlib.patches import Rectangle
                plt.figure(figsize=(5, 4))
                plt.imshow(frame,
                           cmap='viridis',
                           origin='lower',
                           vmin=np.percentile(frame[300:3780,0:4080],5),
                           vmax=np.percentile(frame[300:3780,0:4080],95)
                          )
                for r in reg.keys():
                    plt.gca().add_patch(Rectangle((reg[r]['x1'],reg[r]['y1']),reg[r]['x2']-reg[r]['x1'],reg[r]['y2']-reg[r]['y1'],linewidth=1,edgecolor='r',facecolor='none'))
                    plt.text(((reg[r]['short'] == 'ref3') or
                              (reg[r]['short'] == 'ref4') or
                              (reg[r]['short'] == 'ref5') or
                              (reg[r]['short'] == 'ref6') or
                              (reg[r]['short'] == 'amp1'))*(reg[r]['x1'])+
                             ((reg[r]['short'] == 'ref1') or
                              (reg[r]['short'] == 'ref2') or
                              (reg[r]['short'] == 'ech')  or
                              (reg[r]['short'] == 'coll') or
                              (reg[r]['short'] == 'amp2'))*(reg[r]['x2']),
                             (((reg[r]['y1'] < 2080) and (reg[r]['y1'] > 100))*(reg[r]['y1']-30)+
                              ((reg[r]['y1'] > 2080) or  (reg[r]['y1'] < 100))*(reg[r]['y2']+30)),
                             str(np.round(reg[r]['med_elec'],1)) + timelabel,
                             weight='bold',
                             color='r',
                             ha=(((reg[r]['short'] == 'ref3') or
                                  (reg[r]['short'] == 'ref4') or
                                  (reg[r]['short'] == 'ref5') or
                                  (reg[r]['short'] == 'ref6') or
                                  (reg[r]['short'] == 'amp1'))*('left')+
                                 ((reg[r]['short'] == 'ref1') or
                                  (reg[r]['short'] == 'ref2') or
                                  (reg[r]['short'] == 'ech')  or
                                  (reg[r]['short'] == 'coll') or
                                  (reg[r]['short'] == 'amp2'))*('right')),
                             va=(((reg[r]['y1'] < 2080) and (reg[r]['y1'] > 100))*('top')+
                                 ((reg[r]['y1'] > 2080) or (reg[r]['y1'] < 100))*('bottom'))
                            )
                now = datetime.now()
                coll_text = 'Ion Pump (Coll): \n' + (f'{coll_pressure_torr:.1e}' + ' Torr, ' + f'{coll_current_a*1e6:.1f}' + ' $\mu$A')*(coll_pressure_torr > 1e-9) + ('Off')*(coll_pressure_torr < 1e-9)
                ech_text  = 'Ion Pump (Ech): \n'  + (f'{ech_pressure_torr:.1e}'  + ' Torr, ' + f'{ech_current_a*1e6:.1f}'  + ' $\mu$A')*(ech_pressure_torr  > 1e-9) + ('Off')*(ech_pressure_torr < 1e-9)
                #plt.text(4080, -250, now.strftime("%m/%d/%Y, %H:%M:%S"), ha='right', color='gray')
                plt.text(4220,  500, coll_text,  rotation=90, ha='center',fontsize = 6)
                plt.text(4220, 3000, ech_text, rotation=90, ha='center',fontsize = 6)
                plt.text(3950, 1500, 'Bench Side\n (blue side of orders)',  rotation=90, ha='center', color='white',fontsize = 6)
                plt.text( 150, 1500, 'Top Side\n (red side of orders)',    rotation=90, ha='center', color='white',fontsize = 6)
                plt.text(2040,   70, 'Collimator Side',                     rotation= 0, ha='center', color='white',fontsize = 6)
                plt.text(2040, 3970, 'Echelle Side',                        rotation= 0, ha='center', color='white',fontsize = 6)
                cbar = plt.colorbar()
                cbar.set_label(timelabel)#, fontsize=18
                cbar.ax.tick_params()#labelsize=18
                cbar.ax.tick_params()#size=18
                plt.title(ccd_color[i_color]+' '+version +' '+exposure_name)
                plt.xlabel('Column (pixel number)')
                plt.ylabel('Row (pixel number)')#fontsize=18
                plt.xticks()#KP.20230317.07770.97
                plt.yticks()
                plt.grid(False)
                plt.savefig(output_dir+'/'+master_list[i][:-5]+ccd_color[i_color]+'_zoomable.png', dpi=1000)
                plt.close()



        #get all exposures taken on a particular night

        file_list = glob.glob(exposures_dir+night+'/*.fits')
        date_obs = []
        temp = []
        for i in range(len(file_list)):
            file_list[i] = file_list[i][18:-8]

            hdulist = fits.open(file_list[i])
            hdr = hdulist[0].header
            date_obs.append(hdr['DATE-OBS'])
            temp.append(hdr['TEMP'])

        date_obs = np.array(date_obs,'str')
        date_obs = Time(date_obs, format='isot', scale='utc')
        plt.scatter(date_obs.jd,temp, marker = '.')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.savefig(output_dir+night+'_temperature_variation.png')
        plt.close()
        '''
