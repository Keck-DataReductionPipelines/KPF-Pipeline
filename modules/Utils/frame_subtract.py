from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

class FrameSubtract(KPF0_Primitive):
    """This utility subtracts one frame from another.

    """
    def __init__(self, action, context):
        "FrameSubtract constructor."
        
        #Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        #Input arguments
        self.principal_file = self.action.args[0]
        self.correcting_file = self.action.args[1]
        self.ffi_exts = self.action.args[2]
        self.sub_type = self.action.args[3] #dark,bias,background as options
        
    def subtraction(self):
        for ffi in self.ffi_exts:
            assert self.principal_file[ffi].data.shape==self.correcting_file[ffi].data.shape, "Frames' dimensions don't match. Check failed."
            if self.sub_type == 'bias':
                assert self.correcting_file.header['PRIMARY']['OBSTYPE'] == 'Bias', "Correcting file is not a master bias. Check failed."
            if self.sub_type == 'dark':
                assert self.correcting_file.header['PRIMARY']['OBSTYPE'] == 'Dark', "Correcting file is not a dark frame file. Check failed."
                assert self.principal_file.header['PRIMARY']['EXPTIME'] == self.correcting_file.header['PRIMARY']['EXPTIME'], "Frames' exposure times don't match. Check failed."
            subtracted = self.principal_file[ffi] - self.correcting_file[ffi]
                
    def _perform(self):
        
        subtracted = self.subtraction()
        return Arguments(subtracted)
