import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
import os
from astropy import modeling


class QuicklookAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger




    def qlp_procedures(self,kpf0_file,file_name,output_dir):


        saturation_limit = int(self.config['2D']['saturation_limit'])*1.
        plt.rcParams.update({'font.size': 8})
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

        #check if output location exist, if not create it

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir+'/fig')


        #print('working on',file_name)

        exposure_name = kpf0_file.header['PRIMARY']['OFNAME'][:-5]#file_name[18:-5]#hdr['PRIMARY']['OFNAME'][:-5]
        date = exposure_name[3:11]
        print('working on',date,exposure_name)



        #read ccd directly
        L0_data = self.config['IO']['input_prefix_l0']+date+'/'+exposure_name+'.fits'
        hdulist = fits.open(L0_data)
        print(hdulist[0].info())

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





        master_file = 'None'
        if version == 'Sol_All':
            master_file = self.config['2D']['master_socal']
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
                plt.xlabel('Counts')
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
            plt.title(ccd_color[i_color]+' '+version)
            plt.colorbar(label = 'Counts')
            plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.pdf')
            plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png', dpi=1000)
            #2D difference image
            plt.close()
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
                plt.title(ccd_color[i_color]+' '+version+'- Master '+version)
                plt.colorbar(label = 'Fractional Difference')
                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'.pdf')
                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'.png', dpi=500)
             #Hisogram
            plt.close()
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

            #print(np.percentile(flatten_counts,99.9),saturation_limit)
            plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts)+'; Saturated? '+str(np.percentile(flatten_counts,99.9)>saturation_limit),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
            if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
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



        #moving on the 1D data

        L1_data = self.config['IO']['input_prefix_l1']+date+'/'+exposure_name+'_L1.fits'
        print('working on', L1_data)
        hdulist = fits.open(L1_data)

        wav_green = np.array(hdulist['GREEN_CAL_WAVE'].data,'d')
        wav_red = np.array(hdulist['RED_CAL_WAVE'].data,'d')

        wave_soln = self.config['L1']['wave_soln']
        if wave_soln!='None':#use the master the wavelength solution
            hdulist1 = fits.open(wave_soln)
            wav_green = np.array(hdulist1['GREEN_CAL_WAVE'].data,'d')
            wav_red = np.array(hdulist1['RED_CAL_WAVE'].data,'d')




        flux_green = np.array(hdulist['GREEN_SCI_FLUX1'].data,'d')
        flux_red = np.array(hdulist['RED_SCI_FLUX1'].data,'d')#hdulist[40].data

        if np.shape(flux_green)==(0,):flux_green = wav_green*0.#place holder when there is no data
        if np.shape(flux_red)==(0,): flux_red = wav_red*0.#place holder when there is no data
        print(np.shape(flux_green),np.shape(flux_red))
        wav = np.concatenate((wav_green,wav_red),axis = 0)
        flux = np.concatenate((flux_green,flux_red),axis = 0)


        n = int(self.config['L1']['n_per_row']) #number of orders per panel
        cm = plt.cm.get_cmap('rainbow')

        from matplotlib import gridspec
        gs = gridspec.GridSpec(n,1 , height_ratios=np.ones(n))

        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(int(np.shape(wav)[0]/n)+1,1, sharey=False,figsize=(24,16))

        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        fig.subplots_adjust(hspace=0.4)

        for i in range(np.shape(wav)[0]):
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

        ax[int(np.shape(wav)[0]/n/2)].set_ylabel('Counts',fontsize = 20)
        ax[0].set_title('1D Spectrum',fontsize = 20)
        plt.xlabel('Wavelength (Ang)',fontsize = 20)
        plt.savefig(output_dir+'fig/'+exposure_name+'_1D_spectrum.pdf')
        plt.savefig(output_dir+'fig/'+exposure_name+'_1D_spectrum.png',dpi = 200)


        #now onto the plotting of CCF
        #date = exposure_name[3:11]
        ccf_file = self.config['IO']['input_prefix_l2']+date+'/'+exposure_name+'_L2.fits'
        print(date,ccf_file)
        if os.path.exists(ccf_file):
            print('Working on L2 file')
            hdulist = fits.open(ccf_file)



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
                print('ccf',np.shape(ccf))
                if np.shape(ccf)==(0,): continue
                ccf = np.sum(ccf,axis =0)#sum over orderlets
                #print(np.shape(ccf))

                step = float(self.config['RV']['step'])
                vel_grid = np.array(range(-int(np.shape(ccf)[1]/2),int(np.shape(ccf)[1]/2),1),'d')*step
                #print('step',step,len(vel_grid))


                mean_ccf = np.nanmean(ccf,axis = 0)/np.percentile(np.nanmean(ccf,axis = 0),[99.9])
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
                gamma = hdulist['RV'].header[ccf_rv[i_color]]
                plt.plot([gamma,gamma],[np.nanmin(mean_ccf),1.],':',color ='gray',linewidth = 0.5)
                ax.text(0.6,0.3+i_color*0.2,ccf_rv[i_color]+' $\gamma$ (km/s): %5.2f' % gamma,transform=ax.transAxes)
                #ax.text(0.6,0.2+i_color*0.2,ccf_color[i_color]+' $\sigma$ (km/s): %5.2f' % std,transform=ax.transAxes)

            plt.xlabel('RV (km/s)')
            plt.ylabel('CCF')
            plt.title('Mean CCF')
            plt.xlim(np.min(vel_grid),np.max(vel_grid))
            plt.legend()
            plt.savefig(output_dir+'fig/'+exposure_name+'_simple_ccf.pdf')
            plt.savefig(output_dir+'fig/'+exposure_name+'_simple_ccf.png')
            plt.close()
        else: print('L2 file does not exist')
        #output the results to html
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
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.pdf">
        <img src="fig/""" +exposure_name+ """_2D_Frame_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoom">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.pdf" >
        <img src="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoomright">
        <a target="_blank" href="fig/""" +exposure_name+ """_Histogram_GREEN_CCD.pdf" >
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
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.pdf" >
        <img src="fig/""" +exposure_name+ """_2D_Frame_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoom">
        <a target="_blank" href="fig/""" +exposure_name+ """_2D_Difference_GREEN_CCD.pdf" >
        <img src="fig/""" +exposure_name+ """_2D_Difference_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column">
        <div class="zoomright">
        <a target="_blank" href="fig/""" +exposure_name+ """_Histogram_RED_CCD.pdf" >
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
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.pdf" >
        <img src="fig/""" +exposure_name+ """_Column_cut_GREEN_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        <div class="column2">
        <div class="zoomright2">
        <a target="_blank" href="fig/""" +exposure_name+ """_Column_cut_RED_CCD.pdf" >
        <img src="fig/""" +exposure_name+ """_Column_cut_RED_CCD.png" style="width:100%" alt="" title="">
        </a>
        </div>
        </div>
        </div>
        <br>
        <br>
        <br>

        <figure>
        <a target="_blank" href="fig/""" +exposure_name+ """_wavlength_calibration.pdf" >
        <img src="fig/""" +exposure_name+ """_wavlength_calibration.png" width="38%" alt="" title="">
        </a>
        <a target="_blank" href="fig/""" +exposure_name+ """_wav_drift.pdf" >
        <img src="fig/""" +exposure_name+ """_wav_drift.png" width="38%" alt="" title="">
        </a>
        <a target="_blank" href="fig/""" +exposure_name+ """_simple_ccf.pdf" >
        <img src="fig/""" +exposure_name+ """_simple_ccf.png" width="22%" alt="" title="">
        </a>
        </figure>
        <br>
        <figure>
        <a target="_blank" href="fig/""" +exposure_name+ """_sky.pdf" >
        <img src="fig/""" +exposure_name+ """_sky.png" width="22%" alt="" title="">
        </a>

        <a target="_blank" href="fig/""" +exposure_name+ """_guider_cam.png" >
        <img src="fig/""" +exposure_name+ """_guider_cam.png" width="35%" alt="" title="">
        </a>

        <a target="_blank" href="fig/""" +exposure_name+ """_Fiber_Light_Curve.pdf"  >
        <img src="fig/""" +exposure_name+ """_Fiber_Light_Curve.png" width="42%" alt="" title="">
        </a>
        </figure>

        <br>
        <a target="_blank" href="fig/""" +exposure_name+ """_1D_spectrum.pdf"  >
        <figure>
        <span><img src="fig/""" +exposure_name+ """_1D_spectrum.png" style="width:100%" alt="" title=""></span>
        </figure>
        </a>
        <br>

        <a target="_blank" href="fig/""" +exposure_name+ """_CaHK.png"  >
        <figure>
        <img src="fig/""" +exposure_name+ """_CaHK.png" style="width:100%" alt="" title="">
        </figure>
        </a>

        <hr />
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
        """


        f.write(message)
        f.close()
