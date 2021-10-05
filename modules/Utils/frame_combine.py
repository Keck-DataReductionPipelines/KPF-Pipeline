import numpy as np
from astropy.io import fits
from astropy import stats
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
"""
Averaging several frames together reduces the impact of readout
noise and gives a more accurate estimate of the bias level. 
The master bias frame produced from this averaging - Dealing with CCD Data

Frame combine Steps:
    - Takes 2D arrays (currently takes them written out)
    - Stacks arrays on 3rd axis (now 3d matrix)
    - Takes element-wise mean of each array
    - Returns 2d array of element-wise averages
"""
class FrameCombinePrimitive(KPF0_Primitive):
    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)
        self.L0_names=self.action.args[0]
        self.lev0_ffi_exts=self.action.args[1]
        self.data_type=self.action.args[2]

    def _perform(self):
        no_ffis = len(self.lev0_ffi_exts)
        self.logger.info(f'Number of masters to create: {no_ffis}')
        #loop here through L0 objects
        arrays_list=[]
        #overwrites first iterated file with the combination of it+the rest
        self.logger.info(f'L0_names: {self.L0_names}')
        for ext in self.lev0_ffi_exts:
            for name in self.L0_names:
                obj=fits.open(name)
                #issue here with 'NotImplementedError: memoryview: unsupported format >f' when using KPF0.from_fits
                arrays_list.append(obj[ext].data)
                #self.logger.info(f'file: {name}, obj.data_type is {type(obj.data)}')
        master_frames = []
        for frame in range(len(self.lev0_ffi_exts)):
            split = np.array_split(arrays_list,no_ffis)
            single_frame_data =split[frame]
            data=np.dstack(single_frame_data)
            master_frame = stats.sigma_clip(data,sigma=5,masked=True)
        #assuming all data will be 2D arrays
        #master_frame.data=np.mean(data,2)
            mast_mean_axis = len(np.array(master_frame).shape)-1
            master_mean = np.mean(master_frame,axis=mast_mean_axis)
            master_frames.append(master_mean)
            #master_frame.receipt_add_entry('frame_combine', self.__module__, f'input_files={self.L0_names}', 'PASS')
            if self.logger:
                self.logger.info("frame_combine: Receipt written")
        # for ext in self.lev0_ffi_exts:
        master_frames = np.array(master_frames)
        #master_frames.reshape(master_frames.shape[0])
            #master_frames = np.array(master_frames).reshape(obj[ext].shape[0],obj[ext].shape[1])

        master_file_HDU = fits.HDUList()
        master_file_HDU.append(fits.PrimaryHDU())
        for ffi_no in range(no_ffis):
            master_file_HDU.append(fits.ImageHDU(name='MASTER_'+str(ffi_no)))
        
        for ext in range(no_ffis): 
            single = np.array_split(master_frames,len(self.lev0_ffi_exts))[ext]
            single = np.squeeze(single)
            master_file_HDU[ext+1].data = single
        ## per number of ffi extentions, made hdu list with that many extensions
        ## populate those extensions with master frames split per iter of loop
        print(master_file_HDU.info())
        print (f'master_frame_type:{type(master_frame)}')
        ####################write workaround fits.writeto? 
        master_file_HDU.writeto('./examples/V1/FlatRecipe/FlatRecipeRes/test_masterflat.fits',overwrite=True)
        #######
        return Arguments(master_file_HDU)