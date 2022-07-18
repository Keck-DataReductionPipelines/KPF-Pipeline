import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

class NoisePattern(KPF0_Primitive):
    """ This utility calculates the fixed-pattern noise patterns on L0 images (unasssembled),
    specifically bias and dark frames.
    """
    def __init__(self, action, context):
        KPF0_Primitive.__init__(action, context)
        self.L0_frames_files = self.action.args[0]
        self.master_frames_file = self.action.args[1]
        self.ccd_info = self.action.args[2]
        self.csv_output_dir = self.action.args[3]
        self.assembly_stage = self.action.args[4]
        
    def canonical_readout_noise(self,l0_file,exts):
        """Performs calculations relevant to fixed pattern noise.
        
        Args:
            l0_file(KPF L0 object):
            exts(list):
            
        Returns:

        """
        stds = []
        rmss = []
        for ext in exts:
            frame_data = np.array(l0_file[ext],'d')
            
            #standard dev
            std = np.nanstd(frame_data)
            
            #root mean sq
            x = frame_data - np.nanmean(frame_data)
            rms = np.sqrt(np.nanmean((x**2)))
            
            vmin = np.percentile(frame_data.ravel(),1)
            vmax = np.percentile(frame_data.ravel(),99)
            range = (np.median(frame_data) - (10*std),np.median(frame_data) + (10*std))
            
            #plt one
            plt.figure()
            plt.text(300,200,'STD:%3.1f' % std)
            plt.text(300,300,'RMS: %3.1f' % rms)
            plt.title('Bias Frame')
            plt.imshow(frame_data,vmin=vmin,vmax=vmax,interpolation='none')
            plt.colorbar(label='Counts')
            # plt.savefig()

            #plt two
            low_bound = np.median(frame_data) - (3*std)
            upp_bound = np.median(frame_data) + (3*std)
            plt.figure()
            plt.text(300,200,'STD: %3.1f' %std)
            xx = np.copy(frame_data)
            xx[(xx<low_bound)|(xx>upp_bound)] = np.nan
            #print(len(xx[(xx<low_bound)|(xx>upp_bound)]))
            plt.text(300,300,'RMS: %3.1f' % rms)
            plt.title('Bias Frame, Low Variance')
            plt.imshow(xx,vmin=vmin,vmax=vmax,interpolation='none')
            plt.colorbar(label='Counts')
            # plt.savefig
        
            #plt three
            xx = np.copy(frame_data)
            #print(len(xx[(xx>low_bound)&(xx<upp_bound)]))
            xx[(xx>low_bound)&(xx<upp_bound)] = np.nan
            plt.text(300,200,'STD: %3.1f' % std)
            plt.text(300,300,'RMS: %3.1f' % rms)
            plt.title('Bias Frame, High Variance')
            plt.imshow(x,vmin=vmin,vmax=vmax,interpolation='none')
            plt.colorbar(label = 'Counts')
            # plt.savefig()
            
            #plt four
            #print(np.nanmedian(frame_data),np.nanstd(frame_data))
            plt.hist(frame_data.ravel(),bins =100, range=range)
            plt.plot([low_bound,low_bound],[0.5,1e6],color = 'red')
            plt.plot([upp_bound,upp_bound],[0.5,1e6],color = 'red')
            plt.yscale('log')
            plt.xlabel('Counts')
            plt.ylabel('Number of Pixels')
            plt.title('Bias Histogram')
            # plt.savefig()
            stds.append(std)
            rmss.append(rms)
            return stds,rmss
        
    def make_csv(self,file,stds,rmss,exts):
        """Makes csv file for outputting std and rms.
        
        Args:
            file():
            stds():
            rmss():
            exts():
        """
        csv_vals = pd.DataFrame(data={'Extension':exts,'STD':stds,'RMS':rmss})
        name = os.path.split(file)[1]
        _,date,extone,exttwo,_ = name.split('.')
        csv_name = date+'.'+extone+'.'+exttwo+'.fpn'+'.csv'
        csv_vals.to_csv(self.csv_output_dir+csv_name,sep=',',index=False) #name will be changed
        
    def _perform(self):
        """Performs fixed pattern noise algorithms.
        """
        if self.assembly_stage == 'pre' or 'Pre':
            _,_,_,_,exts = self.ccd_txt_info
            for file in self.L0_frames_files: 
                l0_obj = KPF0.from_fits(file)
                stds,rmss = self.canonical_readout_noise(l0_obj,exts)
                self.make_csv(file,stds,rmss,exts)
                
        if self.assembly_stage == 'post' or 'Post':
            
            
            #may need to run this twice - before image assembly and after image assembly/bias/flat/overscan corrections
            #if statement: if before assembly and corrections, do x. if after assembly and corrections, do y
            #expect all quadrants to be affected similarly and simultaneously, so testing one quad may be enough
            #write recipe, fei will take over from there