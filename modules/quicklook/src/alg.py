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


class QuicklookAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger




    def qlp_procedures(self,kpf0_file,output_dir,end_of_night_summary):



        saturation_limit = int(self.config['2D']['saturation_limit'])*1.
        plt.rcParams.update({'font.size': 8})
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

        #check if output location exist, if not create it

        exposure_name = kpf0_file.filename.replace('_2D.fits', '.fits')[:-5]
        date = exposure_name[3:11]
        print('test',exposure_name, date)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #if not os.path.exists(output_dir+'/fig'):
        #    os.makedirs(output_dir+'/fig')

        if not os.path.exists(output_dir+'/'+exposure_name+'/2D'):
            os.makedirs(output_dir+'/'+exposure_name+'/2D')

        if not os.path.exists(output_dir+'/'+exposure_name+'/2D_analysis'):
            os.makedirs(output_dir+'/'+exposure_name+'/2D_analysis')

        if not os.path.exists(output_dir+'/'+exposure_name+'/1D'):
            os.makedirs(output_dir+'/'+exposure_name+'/1D')

        if not os.path.exists(output_dir+'/'+exposure_name+'/ExpMeter'):
            os.makedirs(output_dir+'/'+exposure_name+'/ExpMeter')

        if not os.path.exists(output_dir+'/'+exposure_name+'/CaHK'):
            os.makedirs(output_dir+'/'+exposure_name+'/CaHK')

        if not os.path.exists(output_dir+'/'+exposure_name+'/CCF'):
            os.makedirs(output_dir+'/'+exposure_name+'/CCF')
        #print('working on',file_name)

        # try:
        #     exposure_name = kpf0_file.header['PRIMARY']['OFNAME'][:-5]#file_name[18:-5]#hdr['PRIMARY']['OFNAME'][:-5]
        # except:


        if end_of_night_summary == True:
            print('working on end of night summary of '+date)
            file_list = glob.glob(self.config['IO']['input_prefix_l0']+date+'/*.fits')
            #print(len(file_list),file_list)

            '''
            #pull temps from all the fits header and draw the temps
            date_obs = []
            temp = []
            for file in file_list:
                print(file)
                hdulist = fits.open(file)
                #print(hdulist.info())
                hdr = hdulist[0].header

                date_obs.append(hdr['DATE-OBS'])
                temp.append(hdr['TEMP'])
            date_obs = np.array(date_obs,'str')
            date_obs = Time(date_obs, format='isot', scale='utc')
            plt.scatter(date_obs.jd,temp, marker = '.')
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.savefig(output_dir+'fig/end_of_night_summary_temperature.png')
            plt.close()


            #plot how the order trace changes
            order_trace_master_file = '/data/order_trace/20220601/KP.20220601.37734.61_GREEN_CCD.csv'
            order_trace_list = glob.glob('/data/order_trace/*/*_GREEN_CCD.csv', recursive = True)
            for order_trace_file in order_trace_list:
                order_trace = pd.read_csv(order_trace_file)
                order_trace_master = pd.read_csv(order_trace_master_file)
                print(np.shape(order_trace),np.shape(order_trace_master))
                for i in range(1,np.shape(order_trace)[0]-2,1):#[50]:#range(np.shape(order_trace)[0])
                    x_grid_master = np.linspace(order_trace_master.iloc[i]['X1'],order_trace_master.iloc[i]['X2'],int(order_trace_master.iloc[i]['X2']-order_trace_master.iloc[i]['X1']))
                    y_grid_master = order_trace_master.iloc[i]['Coeff0']+x_grid_master*order_trace_master.iloc[i]['Coeff1']+x_grid_master**2*order_trace_master.iloc[i]['Coeff2']+x_grid_master**3*order_trace_master.iloc[i]['Coeff3']

                    x_grid = np.linspace(order_trace.iloc[i]['X1'],order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                    y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']

                    print(order_trace_file,'order',i,np.nanstd(y_grid_master-y_grid),abs(order_trace.iloc[i]['BottomEdge']-order_trace_master.iloc[i]['BottomEdge']),abs(order_trace.iloc[i]['TopEdge']-order_trace_master.iloc[i]['TopEdge']))

                    plt.plot(x_grid,y_grid,color ='red',linewidth = 0.2)
                    plt.plot(x_grid,y_grid-order_trace.iloc[i]['BottomEdge'],color ='white',linewidth = 0.2,alpha = 1)
                    plt.plot(x_grid,y_grid+order_trace.iloc[i]['TopEdge'],color ='black',linewidth = 0.2,alpha = 1)
            plt.xlim(3200,4000)
            plt.ylim(3200,4000)
            plt.savefig(output_dir+'fig/order_trace_evolution.png')
            '''
            return
        print('working on',date,exposure_name)







        #read ccd directly
        L0_data = self.config['IO']['input_prefix_l0']+date+'/'+exposure_name+'_2D.fits'
        hdulist = fits.open(L0_data)
        print(hdulist.info())

        #get ccd names
        ccd_color=[]
        ccd_list = self.config.items( "CCD_LIST")
        for key, path in ccd_list:
            ccd_color.append(path)


        if len(hdulist[ccd_color[0]].data)<1 and len(hdulist[ccd_color[1]].data)<1:
            print('skipping empty file')
            return

        #hdr = hdulist.header
        #version = hdr['PRIMARY']['IMTYPE']
        hdr = hdulist[0].header
        version = hdr['IMTYPE']
        Cal_Source = hdr['SCI-OBJ']
        #print('2d header',hdr,hdr['IMTYPE'],hdr['CAL-OBJ'],hdr['SCI-OBJ'],hdr['SKY-OBJ'])




        master_file = 'None'
        if version == 'Solar':
            master_file = self.config['2D']['master_solar']
        if version == 'Arclamp':
            if Cal_Source == 'Th_daily':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-thar-all-night.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_Th_daily']
            if Cal_Source == 'U_daily':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-une-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_U_daily']
            if Cal_Source == 'EtalonFiber':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-etalon-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_EtalonFiber']
            if Cal_Source == 'LFCFiber':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-lfc-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_LFCFiber']
        if version == 'Flatlamp':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_flat.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_flatlamp']
        if version == 'Dark':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_dark.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_dark']
        if version == 'Bias':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_bias.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_bias']
        if version == 'Sol_All':
            master_file = self.config['2D']['master_socal']
        if version == 'Etalon_All':
            master_file = self.config['2D']['master_etalon']
        if version == 'Th_All':
            master_file = self.config['2D']['master_ThAr']
        if version == 'Une_All':
            master_file = self.config['2D']['master_Une']
        if version == 'UNe_All':
            master_file = self.config['2D']['master_Une']
        if version == 'LFC_SciCal':
            master_file = self.config['2D']['master_LFC']




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
                    master_counts*=hdr['EXPTIME']/6.
                    master_flatten_counts*=hdr['EXPTIME']/6.
            print(version,hdr,hdr['EXPTIME'],type(hdr['EXPTIME']),hdulist1[0].header['EXPTIME'],Cal_Source,master_file,os.path.exists(master_file))
            #print(master_counts)
            #input("Press Enter to continue...")

            #looking at the fixed noise patterns
            '''
            if version =='Bias':
                a = np.copy(counts)
                a_med = np.nanmedian(a.ravel())
                a_std = np.nanstd(a.ravel())
                pdf, bin_edges = np.histogram(a.ravel(),bins=80, range = (a_med-8*a_std,a_med+8*a_std))
                print('bin edges',bin_edges)
                count_fit = (bin_edges[1:]+bin_edges[:-1])/2
                from astropy import modeling
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit-a_med, pdf)
                amp =fitted_model.amplitude.value
                gamma =fitted_model.mean.value+a_med
                std =fitted_model.stddev.value
                print('1st gaussian',amp,gamma,std)
                plt.close('all')
                plt.plot(count_fit,amp*np.exp(-0.5*(count_fit-gamma)**2/std**2),':',color = 'red', label = '1st component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10*amp)
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)]-a_med, pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)])
                amp2 =fitted_model.amplitude.value
                gamma2 =fitted_model.mean.value+a_med
                std2 =fitted_model.stddev.value
                print('2nd gaussian',amp2,gamma2,std2)

                plt.plot(count_fit,amp2*np.exp(-0.5*(count_fit-gamma2)**2/std2**2),':',color = 'green', label = '2nd component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10**7)
                plt.plot((bin_edges[1:]+bin_edges[:-1])/2,pdf, label = 'All')
                plt.scatter(np.array((bin_edges[1:]+bin_edges[:-1])/2,'d')[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)],pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)], label = 'Larger Var Component')
                plt.legend()
                plt.xlabel('Counts (e-)')
                plt.ylabel('Number of Pixels')
                plt.yscale('log')
                plt.savefig(output_dir+'fig/'+exposure_name+'_bias_'+ccd_color[i_color]+'.png')
                plt.close('all')
            '''




            #2D image
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            plt.imshow(counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
            plt.xlabel('x (pixel number)')
            plt.ylabel('y (pixel number)')
            plt.title(ccd_color[i_color]+' '+version +' '+exposure_name)
            plt.colorbar(label = 'Counts (e-)')


            #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png')
            plt.savefig(output_dir+'/'+exposure_name+'/2D/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
            #plt.close()



            #2D difference image

            #if the frame is a flat, let's plot the order trace
            if version != '':#if version == 'Flat_All':
                order_trace_file = self.config['L1']['order_trace']+ccd_color[i_color]+'.csv'
                order_trace = pd.read_csv(order_trace_file)
                print(order_trace_file,order_trace)
                for i in range(np.shape(order_trace)[0]):#[50]:#range(np.shape(order_trace)[0])
                    #print(order_trace.iloc[i]['X1'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                    x_grid = np.linspace(order_trace.iloc[i]['X1'],order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                    y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']
                    plt.plot(x_grid,y_grid,color ='magenta',linewidth = 0.2)
                    plt.plot(x_grid,y_grid-order_trace.iloc[i]['BottomEdge'],':',color ='white',linewidth = 0.2,alpha = 1)
                    plt.plot(x_grid,y_grid+order_trace.iloc[i]['TopEdge'],'--',color ='black',linewidth = 0.2,alpha = 1)
                    #plt.fill_between(x_grid,y_grid-order_trace.iloc[i]['BottomEdge'],y_grid+order_trace.iloc[i]['TopEdge'],color ='pink',alpha = 0.2)
                    #print(x_grid,y_grid)
                plt.xlim(3200,4000)
                plt.ylim(3200,4000)
                plt.title(ccd_color[i_color]+' '+version+' Order Trace ' +exposure_name)
                #plt.savefig(output_dir+'fig/'+exposure_name+'_order_trace_'+ccd_color[i_color]+'.png')
                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_order_trace_'+ccd_color[i_color]+'_zoomable.png', dpi=300)
            plt.close()

            #diagnostic for fixed noise patterns
            if version =='Bias' or version == 'Dark':
                #a plot that looks at the ion pump, overwrites existing 2-D frames

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

                print(reg[r]['name'] + ': ' + str(np.round(reg[r]['med_elec'],1)) + ' e- per hour')
                print('Ion Pump pressure (Torr) - Collimator side: ' + f'{coll_pressure_torr:.1e}')
                print('Ion Pump pressure (Torr) - Echelle side: '    + f'{ech_pressure_torr:.1e}')
                print('Ion Pump current (A) - Collimator side: '     + f'{coll_current_a:.1e}')
                print('Ion Pump current (A) - Echelle side: '        + f'{ech_current_a:.1e}')

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
                plt.savefig(output_dir+'/'+exposure_name+'/2D/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
                plt.close()
                #end of ion pump plot


                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
                threshold = 2
                high_var_counts = np.copy(counts)
                low_var_counts = np.copy(counts)
                high_var_counts[abs(counts-np.nanmedian(counts))<threshold*np.nanstd(counts)] = np.nan
                low_var_counts[abs(counts-np.nanmedian(counts))>threshold*np.nanstd(counts)] = np.nan

                plt.imshow(high_var_counts, vmin = np.percentile(flatten_counts,0.1),vmax = np.percentile(flatten_counts,99.9),interpolation = 'None',origin = 'lower',cmap = 'bwr')
                plt.xlabel('x (pixel number)')
                plt.ylabel('y (pixel number)')
                plt.title(ccd_color[i_color]+' '+version+' High Variance '+exposure_name)
                plt.colorbar(label = 'Counts (e-)')

                plt.text(2200,3600, 'Nominal STD: %5.1f' % np.nanstd(np.ravel(low_var_counts)))
                plt.text(2200,3300, 'Fixed Pattern STD: %5.1f' % np.nanstd(np.ravel(high_var_counts)))
                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png')
                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
                plt.close()
                '''
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

                plt.imshow(low_var_counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
                plt.xlabel('x (pixel number)')
                plt.ylabel('y (pixel number)')
                plt.title(ccd_color[i_color]+' '+version+' Low Variance')
                plt.colorbar(label = 'Counts (e-)')
                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_low_var_'+ccd_color[i_color]+'_zoomable.png')
                '''
            print('master file',version,i_color,master_file,len(master_flatten_counts))
            if master_file != 'None' and len(master_flatten_counts)>1:
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
                plt.title(ccd_color[i_color]+' '+version+'- Master '+version+' '+exposure_name)
                plt.colorbar(label = 'Fractional Difference')
                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png')
                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
             #Hisogram
            plt.close()
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

            #print(np.percentile(flatten_counts,99.9),saturation_limit)
            plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts)+'; Saturated? '+str(np.percentile(flatten_counts,99.99)>saturation_limit),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
            if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
            #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
            plt.xlabel('Counts (e-)')
            plt.ylabel('Number of Pixels')
            plt.yscale('log')
            plt.title(ccd_color[i_color]+' '+version+' Histogram '+exposure_name)
            plt.legend(loc='lower right')
            #plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png')
            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png', dpi=200)

            #Column cut
            plt.close()
            plt.figure(figsize=(8,4))
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

            column_sum = np.nansum(counts,axis = 0)
            #print('which_column',np.where(column_sum==np.nanmax(column_sum))[0][0])
            which_column = np.where(column_sum==np.nanmax(column_sum))[0][0] #int(np.shape(master_counts)[1]/2)

            plt.plot(np.ones_like(counts[:,which_column])*saturation_limit,':',alpha = 0.5,linewidth =  1., label = 'Saturation Limit: '+str(saturation_limit), color = 'gray')
            plt.plot(counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = ccd_color[i_color]+' '+version, color = 'Blue')
            if master_file != 'None' and len(master_flatten_counts)>1: plt.plot(master_counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = 'Master', color = 'Orange')
            plt.yscale('log')
            plt.ylabel('log(Counts/e-)')
            plt.xlabel('Row Number')
            plt.title(ccd_color[i_color]+' '+version+' Column Cut Through Column '+str(which_column) + ' '+exposure_name)#(Middle of CCD)
            plt.ylim(1,1.2*np.nanmax(counts[:,which_column]))
            plt.legend()
            '''
            #show the order order_trace
            if version == 'Flat_All':
                 for i in range(np.shape(order_trace)[0]):#[50]:#range(np.shape(order_trace)[0])
                     #print(order_trace.iloc[i]['X1'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                     x_grid = np.linspace(0,order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-0))
                     y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']
                     #print(i,len(y_grid),which_column,y_grid[which_column])
                     #plt.plot([y_grid[which_column],y_grid[which_column]],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
                     plt.plot([y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],y_grid[which_column]-order_trace.iloc[i]['BottomEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
                     plt.plot([y_grid[which_column]+order_trace.iloc[i]['TopEdge'],y_grid[which_column]+order_trace.iloc[i]['TopEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='magenta',linewidth = 0.2)
                     #plt.plot(x_grid[which_column],y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],color ='white',linewidth = 0.2,alpha = 1)
                     #plt.plot(x_grid[which_column],y_grid[which_column]+order_trace.iloc[i]['TopEdge'],color ='black',linewidth = 0.2,alpha = 1)
            '''
            #plt.savefig(output_dir+'fig/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'.png')
            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'_zoomable.png', dpi=200)
            plt.close()

        #exposure meter plots
        if 'EXPMETER_SCI' in hdulist and len(hdulist['EXPMETER_SCI'].data)>=1:
            print('working on exposure meter data')

            EM_gain = np.float(self.config['EM']['gain'])
            from astropy.table import Table
            from scipy.ndimage import gaussian_filter1d
            def gaussian_1d_apply(row):
                newrow = gaussian_filter1d(row,20)
                return newrow

            dat_SKY = Table.read(L0_data, format='fits',hdu='EXPMETER_SKY')
            dat_SCI = Table.read(L0_data, format='fits',hdu='EXPMETER_SCI')
            df_SKY_EM = dat_SKY.to_pandas()
            df_SCI_EM = dat_SCI.to_pandas()

            wav_SCI_str = df_SCI_EM.columns[2:]
            wav_SCI     = df_SCI_EM.columns[2:].astype(float)
            wav_SKY_str = df_SKY_EM.columns[2:]
            wav_SKY     = df_SKY_EM.columns[2:].astype(float)

            disp_SCI = wav_SCI*0+np.gradient(wav_SCI,1)*-1
            disp_SKY = wav_SKY*0+np.gradient(wav_SKY,1)*-1
            df_SCI_EM_norm        = df_SCI_EM[wav_SCI_str] * EM_gain /disp_SCI
            df_SCI_EM_norm_smooth = df_SCI_EM_norm
            df_SCI_EM_norm_smooth.apply(gaussian_1d_apply, axis=1)
            df_SKY_EM_norm        = df_SKY_EM[wav_SCI_str] * EM_gain /disp_SKY
            df_SKY_EM_norm_smooth = df_SKY_EM_norm
            df_SKY_EM_norm_smooth.apply(gaussian_1d_apply, axis=1)

            # define time arrays
            date_beg = np.array(df_SCI_EM["Date-Beg"], dtype=np.datetime64)
            date_end = np.array(df_SCI_EM["Date-End"], dtype=np.datetime64)
            tdur_sec = (date_end-date_beg).astype(float)/1000. # exposure duration in sec
            time_em     = (date_beg-date_beg[0])/1000 # seconds since beginning
            ind_550m    = np.where((wav_SCI <  550))
            ind_550_650 = np.where((wav_SCI >= 550) & (wav_SCI < 650))
            ind_650_750 = np.where((wav_SCI >= 650) & (wav_SCI < 750))
            ind_750p    = np.where((wav_SCI >= 750))
            int_SCI_spec         = df_SCI_EM_norm[:5].sum(axis=0) / np.sum(tdur_sec[:5]) # flux vs. wavelength per sec (use first five samples)
            int_SCI_flux         = df_SCI_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
            int_SCI_flux_550m    = df_SCI_EM[wav_SCI_str[np.where((wav_SCI <  550))]].sum(axis=1)
            int_SCI_flux_550_650 = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 550) & (wav_SCI < 650))]].sum(axis=1)
            int_SCI_flux_650_750 = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 650) & (wav_SCI < 750))]].sum(axis=1)
            int_SCI_flux_750p    = df_SCI_EM[wav_SCI_str[np.where((wav_SCI >= 750))]].sum(axis=1)

            int_SKY_spec         = df_SKY_EM_norm[:5].sum(axis=0) / np.sum(tdur_sec[:5]) # flux vs. wavelength per sec (use first five samples)
            int_SKY_flux         = df_SKY_EM.sum(axis=1)                         # flux (ADU) vs. time (per sample)
            int_SKY_flux_550m    = df_SKY_EM[wav_SKY_str[np.where((wav_SKY <  550))]].sum(axis=1)
            int_SKY_flux_550_650 = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 550) & (wav_SKY < 650))]].sum(axis=1)
            int_SKY_flux_650_750 = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 650) & (wav_SKY < 750))]].sum(axis=1)
            int_SKY_flux_750p    = df_SKY_EM[wav_SKY_str[np.where((wav_SKY >= 750))]].sum(axis=1)

            plt.style.use('seaborn-whitegrid')
            plt.figure(figsize=(12, 6), tight_layout=True)
            od_arr = [0.1, 0.4, 0.5, 0.6, 0.7, 0.8] # OD0.1, OD1.0, OD1.3, OD2.0, OD3.0, OD4.0
            total_duration = (date_end[-1]-date_beg[0]).astype(float)/1000.

            grid_width = math.ceil(total_duration*1.1/10/10)*10
            #print('grid_width',grid_width)
            #for i_grid in range(12):
            #    plt.axvspan(  i_grid*grid_width,  (i_grid+1)*grid_width, alpha=od_arr[i_grid%6], color='gray')

            plt.plot(time_em, int_SCI_flux_750p    / (870-750)           / tdur_sec, marker='o', color='r', label = '750-870 nm')
            plt.plot(time_em, int_SCI_flux_650_750 / (750-650)                   / tdur_sec, marker='o', color='orange', label = '650-750 nm')
            plt.plot(time_em, int_SCI_flux_550_650 / (650-550)                   / tdur_sec, marker='o', color='g', label = '550-650 nm')
            plt.plot(time_em, int_SCI_flux_550m    / (550-445)         / tdur_sec, marker='o', color='b', label = '445-550 nm')
            plt.plot(time_em, int_SCI_flux         / (870-445) / tdur_sec, marker='o', color='k', label = 'SCI 445-870 nm')

            plt.plot(time_em, int_SKY_flux_750p    / (870-750)           / tdur_sec,':', marker='o', color='r')
            plt.plot(time_em, int_SKY_flux_650_750 / (750-650)                   / tdur_sec,':', marker='o', color='orange')
            plt.plot(time_em, int_SKY_flux_550_650 / (650-550)                   / tdur_sec,':', marker='o', color='g')
            plt.plot(time_em, int_SKY_flux_550m    / (550-445)         / tdur_sec,':', marker='o', color='b')
            plt.plot(time_em, int_SKY_flux         / (870-445) / tdur_sec,':', marker='o', color='k', label = 'SKY 445-870 nm')
            plt.xlabel("Time (sec)",fontsize=12)
            plt.ylabel("Exposure Meter Flux (e-/nm/s)",fontsize=12)
            plt.title('Exposure Meter Time Series '+exposure_name,fontsize=12)
            plt.yscale('log')
            plt.xlim([-total_duration*0.1,total_duration*1.1])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12, loc='best')
            plt.savefig(output_dir+'/'+exposure_name+'/ExpMeter/'+exposure_name+'_Exposure_Meter_Time_Series.png', dpi=200)
            plt.close()

            #plt.style.use('seaborn-whitegrid')
            plt.figure(figsize=(12, 6))
            fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
            plt.axvspan(445, 550, alpha=0.5, color='b')
            plt.axvspan(550, 650, alpha=0.5, color='g')
            plt.axvspan(650, 750, alpha=0.5, color='orange')
            plt.axvspan(750, 870, alpha=0.5, color='red')
            lns1 = ax1.plot(wav_SCI, int_SCI_spec, marker='.', color='k', label ='SCI',zorder = 1)
            ax2 = ax1.twinx()
            lns2 = ax2.plot(wav_SKY, int_SKY_spec, marker='.', color='brown', label = 'SKY',zorder = 0, alpha = 0.5)
            ax1.set_ylim(0,np.percentile(int_SCI_spec,99.9)*1.1)
            ax2.set_ylim(0,np.percentile(int_SKY_spec,99.9)*1.1)
            ax1.set_xlabel("Wavelength (nm)",fontsize=12)
            ax1.set_ylabel("SCI Exposure Meter Flux (e-/nm/s)",fontsize=12)
            ax2.set_ylabel("SKY Exposure Meter Flux (e-/nm/s)",fontsize=12)
            plt.title('Exposure Meter Spectrum '+exposure_name,fontsize=12)
            #plt.yscale('log')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlim(445,870)
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc=0,fontsize=12)
            #plt.show()
            plt.savefig(output_dir+'/'+exposure_name+'/ExpMeter/'+exposure_name+'_Exposure_Meter_Spectrum.png', dpi=200)
            plt.close()
            plt.style.use('default')
            #input("Press Enter to continue...")
        #Ca HK data

        if 'CA_HK' in hdulist and len(hdulist['CA_HK'].data)>=1:
            print('working on Ca HK data')

            def plot_trace_boxes(data,trace_location,trace_location_sky):

                fig, ax = plt.subplots(figsize = (12,6),tight_layout=True)
                im = ax.imshow(data,vmin = np.percentile(data.ravel(),1),vmax = np.percentile(data.ravel(),99), interpolation = 'None',origin = 'lower',aspect='auto')
                for i in trace_location.keys():
                    height = trace_location[i]['x2'] - trace_location[i]['x1']
                    width = trace_location[i]['y2'] - trace_location[i]['y1']
                    ax.add_patch(patches.Rectangle((trace_location[i]['y1'], trace_location[i]['x1']),width,height,linewidth=0.5, edgecolor='r',facecolor='none'))
                    if i == 0: ax.add_patch(patches.Rectangle((trace_location[i]['y1'], trace_location[i]['x1']),width,height,linewidth=0.5, edgecolor='r',facecolor='none',label = 'Sci (Saturation at '+str(64232)+')'))

                for i in trace_location_sky.keys():
                    height = trace_location_sky[i]['x2'] - trace_location_sky[i]['x1']
                    width = trace_location_sky[i]['y2'] - trace_location_sky[i]['y1']
                    ax.add_patch(patches.Rectangle((trace_location_sky[i]['y1'], trace_location_sky[i]['x1']),width,height,linewidth=0.5, edgecolor='white',facecolor='none'))
                    if i == 0: ax.add_patch(patches.Rectangle((trace_location_sky[i]['y1'], trace_location_sky[i]['x1']),width,height,linewidth=0.5, edgecolor='white',facecolor='none',label = 'Sky'))
                fig.colorbar(im, orientation='vertical',label = 'Counts (ADU)')
                plt.xlabel('y (pixel number)')
                plt.ylabel('x (pixel number)')
                plt.title('Ca H&K 2D '+exposure_name)#
                plt.legend()
                plt.savefig(output_dir+'/'+exposure_name+'/CaHK/'+exposure_name+'_CaHK_2D_zoomable.png', dpi=1000)
                plt.close()


            def load_trace_location(fiber,trace_path,offset=0):
                loc_result = pd.read_csv(trace_path,header =0, sep = ' ')
                #print(loc_result)
                loc_vals = np.array(loc_result.values)
                loc_cols = np.array(loc_result.columns)
                #print(loc_cols)
                order_col_name = 'order'
                fiber_col_name = 'fiber'
                loc_col_names = ['y0', 'x0', 'yf','xf']#['x0', 'y0', 'xf','yf']

                loc_idx = {c: np.where(loc_cols == c)[0][0] for c in loc_col_names}
                order_idx = np.where(loc_cols == order_col_name)[0][0]
                fiber_idx = np.where(loc_cols == fiber_col_name)[0][0]
                loc_for_fiber = loc_vals[np.where(loc_vals[:, fiber_idx] == fiber)[0], :]  # rows with the same fiber
                trace_location = dict()
                for loc in loc_for_fiber:       # add each row from loc_for_fiber to trace_location for fiber
                    trace_location[loc[order_idx]] = {'x1': loc[loc_idx['y0']]-offset,'x2': loc[loc_idx['yf']]-offset,'y1': loc[loc_idx['x0']],'y2': loc[loc_idx['xf']]}

                return trace_location


            trace_file = self.config['CaHK']['trace_file']
            trace_location = load_trace_location('sky',trace_file,offset=-1)
            trace_location_sky = load_trace_location('sci',trace_file,offset=-1)
            plot_trace_boxes(hdulist['ca_hk'].data,trace_location,trace_location_sky)
            def extract_HK_spectrum(data,trace_location,rv_shift,wavesoln ):

                wave_lib = pd.read_csv(wavesoln,header =None, sep = ' ',comment = '#')
                wave_lib*=1-rv_shift/3e5
                print(trace_location)
                orders = np.array(wave_lib.columns)
                padding = 200

                plt.figure(figsize=(12,6),tight_layout=True)
                color_grid = ['purple','blue','green','yellow','orange','red']
                chk_bandpass  =  [384, 401.7]
                caK = [393.2,393.5]
                caH = [396.7,397.0]
                Vcont = [389.9,391.9]
                Rcont = [397.4,399.4]

                fig, ax = plt.subplots(1, 1, figsize=(9,4))
                ax.fill_between(chk_bandpass,y1=0,y2=1,facecolor='gray',alpha=0.3,zorder=-100)
                ax.fill_between(caH,y1=0,y2=1,facecolor='m',alpha=0.3)
                ax.fill_between(caK,y1=0,y2=1,facecolor='m',alpha=0.3)
                ax.fill_between(Vcont,y1=0,y2=1,facecolor='c',alpha=0.3)
                ax.fill_between(Rcont,y1=0,y2=1,facecolor='c',alpha=0.3)

                ax.text(np.mean(Vcont)-0.6,0.08,'V cont.')
                ax.text(np.mean(Rcont)-0.6,0.08,'R cont.')
                ax.text(np.mean(caK)-0.15,0.08,'K')
                ax.text(np.mean(caH)-0.15,0.08,'H')

                #ax.plot([chk_bandpass[0]-1, chk_bandpass[1]+1], [0.04,0.04],'k--',lw=0.7)
                #ax.text(385.1,0.041,'Requirement',fontsize=9)

                #ax.plot(x,t_all,label=label) instead iterate over spectral orders plottign
                ax.set_xlim(388,400)
                #ax.set_ylim(0,0.09)

                ax.set_xlabel('Wavelength (nm)',fontsize=10)
                ax.set_ylabel('Flux',fontsize=10)

                ax.plot([396.847,396.847],[0,1],':',color ='black')
                ax.plot([393.366,393.366],[0,1],':',color ='black')


                for i in range(len(orders)):
                    wav = wave_lib[i]
                    print(i,trace_location[i]['x1'],trace_location[i]['x2'])
                    flux = np.sum(hdulist['ca_hk'].data[trace_location[i]['x1']:trace_location[i]['x2'],:],axis=0)
                    ax.plot(wav[padding:-padding],flux[padding:-padding]/np.percentile(flux[padding:-padding],99.9),color = color_grid[i],linewidth = 0.5)
                plt.title('Ca H&K Spectrum '+exposure_name)#
                plt.legend()
                plt.savefig(output_dir+'/'+exposure_name+'/CaHK/'+exposure_name+'_CaHK_Spectrum.png', dpi=1000)
                plt.close()
            #print(np.shape(hdulist['ca_hk'].data))
            rv_shift = hdulist[0].header['TARGRADV']
            extract_HK_spectrum(hdulist['ca_hk'].data,trace_location,rv_shift,wavesoln = self.config['CaHK']['cahk_wav'])
        #moving on the 1D data
        L1_data = self.config['IO']['input_prefix_l1']+date+'/'+exposure_name+'_L1.fits'
        if os.path.exists(L1_data):
            print('working on', L1_data)
            hdulist = fits.open(L1_data)

            wav_green = np.array(hdulist['GREEN_CAL_WAVE'].data,'d')
            wav_red = np.array(hdulist['RED_CAL_WAVE'].data,'d')
            print('test wav_green',wav_green)
            '''
            wave_soln = self.config['L1']['wave_soln']
            if wave_soln!='None':#use the master the wavelength solution
                hdulist1 = fits.open(wave_soln)
                wav_green = np.array(hdulist1['GREEN_CAL_WAVE'].data,'d')
                wav_red = np.array(hdulist1['RED_CAL_WAVE'].data,'d')
                hdulist1.close()
            '''

            #print(hdulist1.info())

            flux_green = np.array(hdulist['GREEN_SCI_FLUX1'].data,'d')
            flux_red = np.array(hdulist['RED_SCI_FLUX1'].data,'d')#hdulist[40].data

            flux_green2 = np.array(hdulist['GREEN_SCI_FLUX2'].data,'d')
            flux_red2 = np.array(hdulist['RED_SCI_FLUX2'].data,'d')#hdulist[40].data

            flux_green3 = np.array(hdulist['GREEN_SCI_FLUX3'].data,'d')
            flux_red3 = np.array(hdulist['RED_SCI_FLUX3'].data,'d')#hdulist[40].data

            flux_green_cal = np.array(hdulist['GREEN_CAL_FLUX'].data,'d')
            flux_red_cal = np.array(hdulist['RED_CAL_FLUX'].data,'d')#hdulist[40].data

            flux_green_sky = np.array(hdulist['GREEN_SKY_FLUX'].data,'d')
            flux_red_sky = np.array(hdulist['RED_SKY_FLUX'].data,'d')#hdulist[40].data


            if np.shape(flux_green)==(0,):flux_green = wav_green*0.#place holder when there is no data
            if np.shape(flux_red)==(0,): flux_red = wav_red*0.#place holder when there is no data
            if np.shape(flux_green2)==(0,):flux_green2 = wav_green*0.#place holder when there is no data
            if np.shape(flux_red2)==(0,): flux_red2 = wav_red*0.#place holder when there is no data
            if np.shape(flux_green3)==(0,):flux_green3 = wav_green*0.#place holder when there is no data
            if np.shape(flux_red3)==(0,): flux_red3 = wav_red*0.#place holder when there is no data
            if np.shape(flux_green_cal)==(0,):flux_green_cal = wav_green*0.#place holder when there is no data
            if np.shape(flux_red_cal)==(0,): flux_red_cal = wav_red*0.#place holder when there is no data
            if np.shape(flux_green_sky)==(0,):flux_green_sky = wav_green*0.#place holder when there is no data
            if np.shape(flux_red_sky)==(0,): flux_red_sky = wav_red*0.#place holder when there is no data

            print(np.shape(flux_green),np.shape(flux_green)==(0,),np.shape(flux_red),np.shape(flux_green))

            wav = np.concatenate((wav_green,wav_red),axis = 0)
            print('test wave',np.shape(wav))
            #print(hdulist1.info())

            flux = np.concatenate((flux_green,flux_red),axis = 0)
            flux2 = np.concatenate((flux_green2,flux_red2),axis = 0)
            flux3 = np.concatenate((flux_green3,flux_red3),axis = 0)
            flux_cal = np.concatenate((flux_green_cal,flux_red_cal),axis = 0)
            flux_sky = np.concatenate((flux_green_sky,flux_red_sky),axis = 0)

            n = int(self.config['L1']['n_per_row']) #number of orders per panel
            cm = plt.cm.get_cmap('rainbow')

            from matplotlib import gridspec
            gs = gridspec.GridSpec(n,1 , height_ratios=np.ones(n))

            plt.rcParams.update({'font.size': 15})
            fig, ax = plt.subplots(int(np.shape(wav)[0]/n)+1,1, sharey=False,figsize=(24,16))

            plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
            fig.subplots_adjust(hspace=0.4)

            for i in range(np.shape(wav)[0]):
                if wav[i,0] == 0: continue
                low, high = np.nanpercentile(flux[i,:],[0.1,99.9])
                flux[i,:][(flux[i,:]>high) | (flux[i,:]<low)] = np.nan
                j = int(i/n)
                rgba = cm((i % n)/n*1.)
                #print(j,rgba)
                ax[j].plot(wav[i,:],flux[i,:], linewidth =  0.3,color = rgba)

            for j in range(int(np.shape(flux)[0]/n)):
                low, high = np.nanpercentile(flux[j*n:(j+1)*n,:],[.1,99.9])
                #print(j,high*1.5)
                ax[j].set_ylim(-high*0.1, high*1.2)

            low, high = np.nanpercentile(flux,[0.1,99.9])

            ax[int(np.shape(wav)[0]/n/2)].set_ylabel('Counts (e-) in SCI1',fontsize = 20)
            ax[0].set_title('1D Spectrum ' +exposure_name,fontsize = 20)
            plt.xlabel('Wavelength (Ang)',fontsize = 20)
            #plt.savefig(output_dir+'fig/'+exposure_name+'_1D_spectrum.png')
            plt.savefig(output_dir+'/'+exposure_name+'/1D/'+exposure_name+'_1D_spectrum_zoomable.png',dpi = 200)

            #make a comparison plot of the three science fibres
            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
            for i_orderlet in [1,2,3]:
                flux_tmp = np.array(hdulist['GREEN_SCI_FLUX'+str(i_orderlet)].data,'d')
                if np.shape(flux_tmp)==(0,): continue
                plt.plot(wav_green[10,:],flux_tmp[10,:], label = 'GREEN_SCI_FLUX'+str(i_orderlet), linewidth =  0.3)
            plt.legend()
            plt.title('Science Orderlets in GREEN '+exposure_name)
            plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/1D/'+exposure_name+'_3_science_fibres_GREEN_CCD.png',dpi = 200)
            plt.close()

            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
            for i_orderlet in [1,2,3]:
                flux_tmp = np.array(hdulist['RED_SCI_FLUX'+str(i_orderlet)].data,'d')
                if np.shape(flux_tmp)==(0,): continue
                plt.plot(wav_red[10,:],flux_tmp[10,:], label = 'RED_SCI_FLUX'+str(i_orderlet), linewidth =  0.3)
            plt.legend()
            plt.title('Science Orderlets in RED '+exposure_name)
            plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/1D/'+exposure_name+'_3_science_fibres_RED_CCD.png',dpi = 200)
            plt.close()


            #plot the ratio between orderlets all relative to the first order, plot as a function of wav, label by order number, red and green in the same plot
            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
            is_fiber_on =[np.nanmedian(flux_green2/flux_green)>0.2,np.nanmedian(flux_green3/flux_green)>0.2,np.nanmedian(flux_green_cal/flux_green)>0.05,np.nanmedian(flux_green_cal/flux_green)>0.05]
            print('test orderlets', np.nanmedian(flux_green2/flux_green),np.nanmedian(flux_green3/flux_green),np.nanmedian(flux_green_cal/flux_green),np.nanmedian(flux_green_sky/flux_green))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green2/flux_green,axis = 1),marker = 'o', color = 'green', label = 'Sci2/Sci1; On: ' +str(is_fiber_on[0]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green3/flux_green,axis = 1),marker = 'o', color = 'red', label = 'Sci3/Sci1; On: ' +str(is_fiber_on[1]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green_cal/flux_green,axis = 1),marker = 'o', color = 'blue', label = 'Cal/Sci1; On: ' +str(is_fiber_on[2]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green_sky/flux_green,axis = 1),marker = 'o', color = 'magenta', label = 'Sky/Sci1; On: ' +str(is_fiber_on[3]))

            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red2/flux_red,axis = 1),marker = 'D', color = 'green')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red3/flux_red,axis = 1),marker = 'D', color = 'red')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red_cal/flux_red,axis = 1),marker = 'D', color = 'blue')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red_sky/flux_red,axis = 1),marker = 'D', color = 'magenta')
            plt.legend()
            plt.title('Orderlets Flux Ratios '+exposure_name)
            #plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/1D/'+exposure_name+'_orderlets_flux_ratio.png',dpi = 200)
            plt.close()
        else: print('L1 file does not exist')

        #now onto the plotting of CCF
        #date = exposure_name[3:11]
        ccf_file = self.config['IO']['input_prefix_l2']+date+'/'+exposure_name+'_L2.fits'
        print(date,ccf_file)
        if os.path.exists(ccf_file):
            print('Working on L2 file')
            hdulist = fits.open(ccf_file)
            print(hdulist.info())


            ccf_color=[]
            ccf_list = self.config.items( "CCF_LIST")
            for key, path in ccf_list:
                ccf_color.append(path)
            #ccf_rv = ['GREEN_CCF','RED_CCF']

            ccf_rv=[]
            ccf_rv_list = self.config.items( "CCF_RV_LIST")
            for key, path in ccf_rv_list:
                ccf_rv.append(path)
            #ccf_rv = ['CCD1RV','CCD2RV']

            color_grid = ['Green','Red']

            plt.rcParams.update({'font.size': 8})
            plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
            fig, ax = plt.subplots(1,1, sharex=True,figsize=(5,4))
            ax = plt.subplot()
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
            for i_color in range(len(ccf_color)):
                ccf = np.array(hdulist[ccf_color[i_color]].data,'d')
                sci_mask = hdulist[ccf_color[i_color]].header['SCI_MASK']

                #step = float(self.config['RV']['step'])
                step = float(hdulist[ccf_color[i_color]].header['STEPV'])
                startv = float(hdulist[ccf_color[i_color]].header['STARTV'])

                #print('gamma',hdulist['GREEN_CCF'].header)
                #vel_grid = np.array(range(-int(np.shape(ccf)[2]/2),int(np.shape(ccf)[2]/2),1),'d')*step
                vel_grid = startv+np.array(range(np.shape(ccf)[2]),'d')*step

                '''
                # plot the individual orders?
                for kk in range(np.shape(ccf)[1]):
                    plt.plot(vel_grid,ccf[0,kk,:])
                    plt.plot(vel_grid,ccf[1,kk,:])
                    plt.plot(vel_grid,ccf[2,kk,:])
                '''

                if np.shape(ccf)==(0,): continue
                #print('ccf shape', np.shape(ccf))
                ccf = np.sum(ccf[1:2,:,:],axis =0)
                #ccf = np.sum(ccf[:-1,:,:],axis =0)#sum over orderlets
                #ccf = np.sum(ccf[1:,:,:],axis =0)
                #print('ccf shape', np.shape(ccf))


                #print('step',step,len(vel_grid))
                if i_color == 0: ccf_weights_file='/data/masters/static_green_ccf_ratio_2.csv'
                if i_color == 1: ccf_weights_file='/data/masters/static_red_ccf_ratio_2.csv'
                newdata = pd.read_csv(ccf_weights_file,sep = '\s+',header = 0)
                ccf_weights = np.array(newdata[sci_mask],'d')#np.ones(np.shape(ccf)[0])
                #if i_color == 0: ccf_weights[12] = 0

                mean_ccf = np.average(ccf,axis = 0,weights = ccf_weights)/np.percentile(np.average(ccf,axis = 0,weights = ccf_weights),[99.9])
                #print('test',np.shape(np.nanmean(ccf,axis = 0)))

                #mean_ccf = np.nanmedian(mean_ccf,axis = 0)
                plt.plot(vel_grid,mean_ccf,label = ccf_color[i_color],color = color_grid[i_color],linewidth = 0.5)



                #fit the center of the ccf
                '''
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model, vel_grid, 1.-mean_ccf)
                gamma =fitted_model.mean.value
                std =fitted_model.stddev.value
                #print(i_color,gamma,std)
                '''

                #read the RV from headers directly
                #print('gamma',hdulist['GREEN_CCF'].header)
                gamma = hdulist['RV'].header[ccf_rv[i_color]]
                plt.plot([gamma,gamma],[np.nanmin(mean_ccf),1.],':',color ='gray',linewidth = 0.5)
                ax.text(0.6,0.3+i_color*0.2,ccf_rv[i_color]+' $\gamma$ (km/s): %5.2f' % gamma,transform=ax.transAxes,color = color_grid[i_color])
                #ax.text(0.6,0.2+i_color*0.2,ccf_color[i_color]+' $\sigma$ (km/s): %5.2f' % std,transform=ax.transAxes)

            plt.xlabel('RV (km/s)')
            plt.ylabel('CCF')
            plt.title('Mean CCF '+exposure_name)
            plt.xlim(np.min(vel_grid),np.max(vel_grid))
            plt.legend()
            #plt.savefig(output_dir+'fig/'+exposure_name+'_simple_ccf.png')
            plt.savefig(output_dir+'/'+exposure_name+'/CCF/'+exposure_name+'_simple_ccf_zoomable.png')
            plt.close()

            #plot ccf in individual orders
            for i_color in range(len(ccf_color)):
                ccf = np.array(hdulist[ccf_color[i_color]].data,'d')[1:2,:,:]
                step = float(hdulist[ccf_color[i_color]].header['STEPV'])
                startv = float(hdulist[ccf_color[i_color]].header['STARTV'])
                vel_grid = startv+np.array(range(np.shape(ccf)[2]),'d')*step
                gamma = hdulist['RV'].header[ccf_rv[i_color]]

                if i_color == 0: ccf_weights_file='/data/masters/static_green_ccf_ratio_2.csv'
                if i_color == 1: ccf_weights_file='/data/masters/static_red_ccf_ratio_2.csv'
                newdata = pd.read_csv(ccf_weights_file,sep = '\s+',header = 0)
                sci_mask = hdulist[ccf_color[i_color]].header['SCI_MASK']
                ccf_weights = np.array(newdata[sci_mask],'d')#np.ones(np.shape(ccf)[0])
                if i_color == 0: ccf_weights[12] = 0

                fig, ax = plt.subplots(1,1,figsize=(5,15),tight_layout = True)
                ax = plt.subplot()
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
                for kk in range(np.shape(ccf)[1]):
                    if ccf_weights[kk] == 1: plt.plot(vel_grid,np.nanmean(ccf[:,kk,:],axis=0)/np.percentile(np.nanmean(ccf[:,kk,:],axis=0),[99.9])+kk*0.3)
                    if ccf_weights[kk] == 0: plt.plot(vel_grid,np.nanmean(ccf[:,kk,:],axis=0)/np.percentile(np.nanmean(ccf[:,kk,:],axis=0),[99.9])+kk*0.3,':')
                    plt.plot([gamma,gamma],[0,1+kk*0.3],':',color = 'gray')
                    plt.text(vel_grid[-1]+2,1+kk*0.3,str(kk),verticalalignment = 'center')
                plt.xlabel('RV (km/s)')
                plt.ylabel('CCF')
                plt.title(ccf_color[i_color]+' by Order '+exposure_name)
                plt.savefig(output_dir+'/'+exposure_name+'/CCF/'+exposure_name+'_ccf_'+ccf_color[i_color]+'_zoomable.png',dpi =200)
                plt.close()
        else: print('L2 file does not exist')

        hdulist.close()

        plt.close('all')
        #output the results to html
        '''
        f = open(output_dir+exposure_name+'_summary.html','w')

        message = """<html><head><title>""" +exposure_name+ """</title>
        <script type='text/javascript'>
        </script>
        <style>
        </style>
        <style type="text/css">
        .column {
        float: left;
        width: 33.33%;
        padding: 0px;
        }

        /* Clear floats after image containers */
        .row::after {
        content: "";
        clear: both;
        display: table;
        }
        .column2 {
        float: left;
        width: 49.5%;
        padding: 0px;
        }

        /* Clear floats after image containers */
        .row::after {
        content: "";
        clear: both;
        display: table;
        }

        .zoomleft img {
        }
        .zoomleft img:hover {
        z-index:0;
        display : inline;
        visibility : visible;
        position:relative;
        margin-top: 50%;
        margin-left: 70%;
        transform: scale(2.5);
        }
        .zoom img {
        }
        .zoom img:hover {
        z-index:0;
        display : inline;
        visibility : visible;
        position:relative;
        margin-top: 50%;
        margin-left: 0%;
        transform: scale(2.5);
        }
        .zoomright img {
        }
        .zoomright img:hover {
        z-index:0;
        display : inline;
        visibility : visible;
        position:relative;
        margin-top: 50%;
        margin-left: -70%;
        transform: scale(2.5);
        }
        .zoomleft2 img {
        }
        .zoomleft2 img:hover {
        z-index:0;
        display : inline;
        visibility : visible;
        position:relative;
        margin-top: 50%;
        margin-left: 50%;
        transform: scale(1.8);
        }

        .zoomright2 img {
        }
        .zoomright2 img:hover {
        z-index:0;
        display : inline;
        visibility : visible;
        position:relative;
        margin-top: 50%;
        margin-left: -50%;
        transform: scale(1.8);
        }
        </style>
        </head><center><h1> """ +exposure_name+ """ </h1><br><br></center>
        <div class="row">
        <div class="column">
        <div class="zoomleft">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png">
        <img src="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoom">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.png" >
        <img src="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoomright">
        <a target="_blank" href="fig/""" +exposure_name+ """_Histogram_GREEN_CCD.png" >
        <img src="fig/""" +exposure_name+ """_Histogram_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>
        <br>
        <br>
        <br>


        <div class="row">
        <div class="column">
        <div class="zoomleft">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png" >
        <img src="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoom">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.png" >
        <img src="fig/""" +exposure_name+ """_2D_Difference_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoomright">
        <a target="_blank" href="fig/""" +exposure_name+ """_Histogram_RED_CCD.png" >
        <img src="fig/""" +exposure_name+ """_Histogram_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>
        <br>
        <br>
        <br>

        <div class="row">
        <div class="column2">
        <div class="zoomleft2">
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png" >
        <img src="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column2">
        <div class="zoomright2">
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_RED_CCD.png" >
        <img src="fig/""" +exposure_name+ """_Column_cut_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>
        <br>
        <br>
        <br>

        <div class="row">
        <div class="column2">
        <div class="zoomleft2">
        <a target="_blank" href="fig/""" +exposure_name+ """_Exposure_Meter_Time_Series.png" >
        <img src="fig/""" +exposure_name+ """_Exposure_Meter_Time_Series.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column2">
        <div class="zoomright2">
        <a target="_blank" href="fig/""" +exposure_name+ """_Exposure_Meter_Spectrum.png" >
        <img src="fig/""" +exposure_name+ """_Exposure_Meter_Spectrum.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>

        <br>
        <br>
        <br>
        <div class="row">
        <div class="column2">
        <div class="zoomleft2">
        <a target="_blank" href="fig/""" +exposure_name+ """'_ccf_GREEN_CCF.png'" >
        <img src="fig/""" +exposure_name+ """_ccf_GREEN_CCF.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column2">
        <div class="zoomright2">
        <a target="_blank" href="fig/""" +exposure_name+ """_ccf_RED_CCF.png" >
        <img src="fig/""" +exposure_name+ """_ccf_RED_CCF.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>
        <br>
        <br>
        <br>

        <figure>
        <a target="_blank" href="fig/""" +exposure_name+ """_wavlength_calibration.png" >
        <img src="fig/""" +exposure_name+ """_wavlength_calibration.png" width="38%" alt="" title="">
        </a>
        <a target="_blank" href="fig/""" +exposure_name+ """_wav_drift.png" >
        <img src="fig/""" +exposure_name+ """_wav_drift.png" width="38%" alt="" title="">
        </a>
        <a target="_blank" href="fig/""" +exposure_name+ """_simple_ccf.png" >
        <img src="fig/""" +exposure_name+ """_simple_ccf.png" width="22%" alt="" title="">
        </a>
        </figure>
        <br>
        <figure>
        <a target="_blank" href="fig/""" +exposure_name+ """_sky.png" >
        <img src="fig/""" +exposure_name+ """_sky.png" width="22%" alt="" title="">
        </a>

        <a target="_blank" href="fig/""" +exposure_name+ """_guider_cam.png" >
        <img src="fig/""" +exposure_name+ """_guider_cam.png" width="35%" alt="" title="">
        </a>

        <a target="_blank" href="fig/""" +exposure_name+ """_Fiber_Light_Curve.png"  >
        <img src="fig/""" +exposure_name+ """_Fiber_Light_Curve.png" width="42%" alt="" title="">
        </a>
        </figure>

        <br>
        <a target="_blank" href="fig/""" +exposure_name+ """_1D_spectrum.png"  >
        <figure>
        <span><img src="fig/""" +exposure_name+ """_1D_spectrum.png" style="width:100%" alt="" title=""></span>
        </figure>
        </a>
        <br>

        <br>
        <a target="_blank" href="fig/""" +exposure_name+ """_orderlets_flux_ratio.png"  >
        <figure>
        <span><img src="fig/""" +exposure_name+ """_orderlets_flux_ratio.png" style="width:100%" alt="" title=""></span>
        </figure>
        </a>
        <br>

        <a target="_blank" href="fig/""" +exposure_name+ """_CaHK_2D.png"  >
        <figure>
        <img src="fig/""" +exposure_name+ """_CaHK_2D.png" style="width:100%" alt="" title="">
        </figure>
        </a>

        <a target="_blank" href="fig/""" +exposure_name+ """_CaHK_Spectrum.png"  >
        <figure>
        <img src="fig/""" +exposure_name+ """_CaHK_Spectrum.png" style="width:100%" alt="" title="">
        </figure>
        </a>

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png" >
        <img id="imgZoom" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn(event)" onmouseout="zoomOut()" src="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png');
        background-repeat: no-repeat;"
        id="overlay"
        onmousemove="zoomIn(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn(event) {
        var element = document.getElementById("overlay");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 10) + "px " + (-posY * 10) + "px";

        }

        function zoomOut() {
        var element = document.getElementById("overlay");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png" >
        <img id="imgZoom1" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn1(event)" onmouseout="zoomOut1()" src="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png');
        background-repeat: no-repeat;"
        id="overlay1"
        onmousemove="zoomIn1(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn1(event) {
        var element = document.getElementById("overlay1");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom1");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 10) + "px " + (-posY * 10) + "px";

        }

        function zoomOut1() {
        var element = document.getElementById("overlay1");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png" >
        <img id="imgZoom2" style="border: 1px solid black; align: right;" width="600px" height="300px" align="right" onmousemove="zoomIn2(event)" onmouseout="zoomOut2()" src="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png">
        <div style="border: 1px solid black;
        width: 400px;
        height: 300px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png');
        background-repeat: no-repeat;"
        id="overlay2"
        onmousemove="zoomIn2(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn2(event) {
        var element = document.getElementById("overlay2");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom2");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2.5) + "px " + (-posY * 2.5) + "px";

        }

        function zoomOut2() {
        var element = document.getElementById("overlay2");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_RED_CCD.png" >
        <img id="imgZoom3" style="border: 1px solid black; align: right;" width="600px" height="300px" align="right" onmousemove="zoomIn3(event)" onmouseout="zoomOut3()" src="fig/""" +exposure_name+ """_Column_cut_RED_CCD.png">
        <div style="border: 1px solid black;
        width: 400px;
        height: 300px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_Column_cut_RED_CCD.png');
        background-repeat: no-repeat;"
        id="overlay3"
        onmousemove="zoomIn3(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn3(event) {
        var element = document.getElementById("overlay3");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom3");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2.5) + "px " + (-posY * 2.5) + "px";

        }

        function zoomOut3() {
        var element = document.getElementById("overlay3");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_1D_spectrum.png" >
        <img id="imgZoom4" style="border: 1px solid black; align: right;" width="600px" height="400px" align="right" onmousemove="zoomIn4(event)" onmouseout="zoomOut4()" src="fig/""" +exposure_name+ """_1D_spectrum.png">
        <div style="border: 1px solid black;
        width: 400px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_1D_spectrum.png');
        background-repeat: no-repeat;"
        id="overlay4"
        onmousemove="zoomIn4(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn4(event) {
        var element = document.getElementById("overlay4");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom4");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 8) + "px " + (-posY * 8) + "px";

        }

        function zoomOut4() {
        var element = document.getElementById("overlay4");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_high_var_GREEN_CCD.png" >
        <img id="imgZoom5" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn5(event)" onmouseout="zoomOut5()" src="fig/""" +exposure_name+ """_2D_Frame_high_var_GREEN_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_2D_Frame_high_var_GREEN_CCD.png');
        background-repeat: no-repeat;"
        id="overlay5"
        onmousemove="zoomIn5(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn5(event) {
        var element = document.getElementById("overlay5");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom5");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 10) + "px " + (-posY * 10) + "px";

        }

        function zoomOut5() {
        var element = document.getElementById("overlay5");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_high_var_RED_CCD.png" >
        <img id="imgZoom6" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn6(event)" onmouseout="zoomOut6()" src="fig/""" +exposure_name+ """_2D_Frame_high_var_RED_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_2D_Frame_high_var_RED_CCD.png');
        background-repeat: no-repeat;"
        id="overlay6"
        onmousemove="zoomIn6(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn6(event) {
        var element = document.getElementById("overlay6");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom6");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 10) + "px " + (-posY * 10) + "px";

        }

        function zoomOut6() {
        var element = document.getElementById("overlay6");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_order_trace_GREEN_CCD.png" >
        <img id="imgZoom7" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn7(event)" onmouseout="zoomOut7()" src="fig/""" +exposure_name+ """_order_trace_GREEN_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_order_trace_GREEN_CCD.png');
        background-repeat: no-repeat;"
        id="overlay7"
        onmousemove="zoomIn7(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn7(event) {
        var element = document.getElementById("overlay7");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom7");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2.5) + "px " + (-posY * 2.5) + "px";

        }

        function zoomOut7() {
        var element = document.getElementById("overlay7");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_order_trace_RED_CCD.png" >
        <img id="imgZoom8" style="border: 1px solid black; align: right;" width="500px" height="400px" align="right" onmousemove="zoomIn8(event)" onmouseout="zoomOut8()" src="fig/""" +exposure_name+ """_order_trace_RED_CCD.png">
        <div style="border: 1px solid black;
        width: 500px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_order_trace_RED_CCD.png');
        background-repeat: no-repeat;"
        id="overlay8"
        onmousemove="zoomIn8(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn8(event) {
        var element = document.getElementById("overlay8");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom8");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2.5) + "px " + (-posY * 2.5) + "px";

        }

        function zoomOut8() {
        var element = document.getElementById("overlay8");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_3_science_fibres_GREEN_CCD.png" >
        <img id="imgZoom9" style="border: 1px solid black; align: right;" width="1000px" height="400px" align="right" onmousemove="zoomIn9(event)" onmouseout="zoomOut9()" src="fig/""" +exposure_name+ """_3_science_fibres_GREEN_CCD.png">
        <div style="border: 1px solid black;
        width: 400px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_3_science_fibres_GREEN_CCD.png');
        background-repeat: no-repeat;"
        id="overlay9"
        onmousemove="zoomIn9(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn9(event) {
        var element = document.getElementById("overlay9");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom9");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2) + "px " + (-posY * 2) + "px";

        }

        function zoomOut9() {
        var element = document.getElementById("overlay9");
        element.style.display = "inline-block";
        }
        </script>

        <hr />

        <hr />
        <a target="_blank" href="fig/""" +exposure_name+ """_3_science_fibres_RED_CCD.png" >
        <img id="imgZoom10" style="border: 1px solid black; align: right;" width="1000px" height="400px" align="right" onmousemove="zoomIn10(event)" onmouseout="zoomOut10()" src="fig/""" +exposure_name+ """_3_science_fibres_RED_CCD.png">
        <div style="border: 1px solid black;
        width: 400px;
        height: 400px;
        display: inline-block;
        background-image: url('fig/""" +exposure_name+ """_3_science_fibres_RED_CCD.png');
        background-repeat: no-repeat;"
        id="overlay10"
        onmousemove="zoomIn10(event)"></div>
        <p>&nbsp;</p>



        <script>
        function zoomIn10(event) {
        var element = document.getElementById("overlay10");
        element.style.display = "inline-block";
        var img = document.getElementById("imgZoom10");
        var posX = event.offsetX ? (event.offsetX) : event.pageX - img.offsetLeft;
        var posY = event.offsetY ? (event.offsetY) : event.pageY - img.offsetTop;
        element.style.backgroundPosition = (-posX * 2) + "px " + (-posY * 2) + "px";

        }

        function zoomOut10() {
        var element = document.getElementById("overlay10");
        element.style.display = "inline-block";
        }
        </script>

        <hr />
        """


        f.write(message)
        f.close()
        '''
