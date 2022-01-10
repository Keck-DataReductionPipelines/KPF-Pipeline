import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import stats
import pandas as pd
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
"""
Averaging several frames together reduces the impact of readout
noise and gives a more accurate estimate of the bias level. 
The master bias frame produced from this averaging - Dealing with CCD Data

"""
class FrameCombinePrimitive(KPF0_Primitive):
    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)
        self.frame_type=self.action.args[0]
        self.L0_names=self.action.args[1]
        self.ffi_ext=self.action.args[2]
        self.data_type=self.action.args[3]

    def _perform(self):
        if self.logger:
            self.logger.info
            print("Frame Combine: frame type is",self.frame_type)
        if self.frame_type == 'bias':
            tester = KPF0.from_fits(self.L0_names[0])
            ext_list = []
            for i in tester.extensions.keys():
                if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                    ext_list.append(i)
            master_holder = tester
            for ffi in self.ffi_ext:
                frames_data=[]
                for path in self.L0_names:
                    obj = KPF0.from_fits(path)
                    frames_data.append(obj[ffi])
                frames_data = np.array(frames_data)
                medians = np.median(frames_data,axis=0)
                final_frame = medians
                ### kpf master file creation ###
                master_holder[ffi] = final_frame
            for ext in ext_list:#figure out where to open master_holder_path
                master_holder.del_extension(ext)

        if self.frame_type == 'flat':
            tester = KPF0.from_fits(self.L0_names[0])
            ext_list = []
            for i in tester.extensions.keys():
                if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                    ext_list.append(i)
            master_holder = tester
            for ffi in self.ffi_ext:
                frames_data=[]
                for path in self.L0_names:
                    obj = KPF0.from_fits(path) #check this
                    frames_data.append(obj[ffi])
                obj = KPF0.from_fits(self.L0_names[0]) #check this
                master_holder[ffi] = obj[ffi]
            for ext in ext_list:
                master_holder.del_extension(ext)
            for ffi in self.ffi_ext:
                rows = np.shape(master_holder[ffi])[0]
                cols = np.shape(master_holder[ffi])[1]
                norm_flat= np.ones((rows,cols))
                norm_flat = pd.DataFrame(norm_flat)
                master_holder.create_extension((ffi+'_NORMALIZED'),ext_type=np.array)
                master_holder[ffi+'_NORMALIZED'] = norm_flat
                print(np.shape(master_holder[ffi+'_NORMALIZED']))

            print (master_holder.info())
            # for ffi in self.ffi_ext:
            #     frames_data=[]
            #     for path in self.L0_names:
            #         obj = KPF0.from_fits(path) #check this
            #         frames_data.append(obj[ffi])
                # frames_data = np.array(frames_data)
                # medians = np.median(frames_data,axis=0) #taking pixel median
                # mmax,mmin = medians.max(),medians.min()
                # norm_meds = (medians - mmin)/(mmax - mmin) #normalizing
                # final_frame = norm_meds
                # ### kpf master file creation ###
                # master_holder[ffi] = final_frame
            # for ext in ext_list:
            #     master_holder.del_extension(ext)

            # plt.figure()
            # plt.imshow(final_frame)
            # plt.colorbar()
            # plt.savefig('final_frame_{}.pdf'.format(self.ffi_ext))
            # plt.close()
            # plt.figure()
            # plt.imshow(master_holder[self.ffi_ext])
            # plt.colorbar()
            # plt.savefig('master_holder_{}.pdf'.format(self.ffi_ext))
            # plt.close()
        return Arguments(master_holder)