import numpy as np
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
        self.data_type=self.action.args[1]

    def _perform(self):
        #loop here through L0 objects
        arrays_list=[]
        self.logger.info(f'L0_names: {self.L0_names}')
        for name in self.L0_names:
            obj=KPF0.from_fits(name,self.data_type)
            arrays_list.append(obj.data)
            self.logger.info(f'file: {name}, obj.data_type is {type(obj.data)}')
        data=np.dstack(arrays_list)
        #create level0 object, assign master_frame
        master_frame = KPF0()
        #assuming all data will be 2D arrays
        master_frame.data=np.mean(data,2)

        master_frame.receipt_add_entry('frame_combine', self.__module__, f'input_files={self.L0_names}', 'PASS')
        if self.logger:
            self.logger.info("frame_combine: Receipt written")

        print (f'master_frame_type:{type(master_frame)}')
        return Arguments(master_frame)