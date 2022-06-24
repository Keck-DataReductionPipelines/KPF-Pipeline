import numpy as np
import matplotlib.pyplot as plt
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
        self.ccd_txt_info = self.action.args[2]
        
    def canonical_readout_noise(self,l0_file):
        """
        """
        _,_,_,_,exts = self.ccd_txt_info
        for ext in exts:
            frame_data = np.array(l0_file[ext],'d')
            
            #standard dev
            std = np.nanstd(frame_data)
            
            #root mean sq
            x = frame_data - np.nanmean(frame_data)
            rms = np.sqrt(np.nanmean((x)))
            
            #plt one
            plt.figure()
            plt.text(300,200,'STD:%3.1f' % std)
            plt.text(300,300,'RMS: %3.1f' % rms)
            plt.title('Bias Frame')
            plt.imshow(frame_data,vmin=260,vmax=340,interpolation='none')
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
            plt.imshow(xx,vmin=260,vmax=340,interpolation='none')
            plt.colorbar(label='Counts')
            # plt.savefig
        
            #plt three
            xx = np.copy(frame_data)
            #print(len(xx[(xx>low_bound)&(xx<upp_bound)]))
            xx[(xx>low_bound)&(xx<upp_bound)] = np.nan
            plt.text(300,200,'STD: %3.1f' % std)
            plt.text(300,300,'RMS: %3.1f' % rms)
            plt.title('Bias Frame, High Variance')
            plt.imshow(x,vmin=260,vmax= 340,interpolation='none')
            plt.colorbar(label = 'Counts')
            # plt.savefig()
            
            #plt four
            #print(np.nanmedian(frame_data),np.nanstd(frame_data))
            plt.hist(frame_data.ravel(),bins =100, range = (240,359))
            plt.plot([low_bound,low_bound],[0.5,1e6],color = 'red')
            plt.plot([upp_bound,upp_bound],[0.5,1e6],color = 'red')
            plt.yscale('log')
            plt.xlabel('Counts')
            plt.ylabel('Number of Pixels')
            plt.title('Bias Histogram')
            # plt.savefig()
        
    def _perform(self):
        for file in self.L0_frames_files:
            l0_obj = KPF0.from_fits(file)
            self.canonical_readout_noise(l0_obj)
            