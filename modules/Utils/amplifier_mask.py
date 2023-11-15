import numpy as np
from kpfpipe.models.level1 import KPF1
from kpfpipe.primitives.level1 import KPF1_Primitive
from keckdrpframework.models.arguments import Arguments


class AmplifierMask(KPF1_Primitive):
    """This utility is used to mask the data from the bad amplifiers with
       high CTI on both the red and green chips when readout in 4 amplifier mode.

    """
    def __init__(self, action, context):

        """
        Arguments:
            input_l1 (KPF1): input L1 object to mask
        """

        #Initialize parent class
        KPF1_Primitive.__init__(self, action, context)

        self.logger = self.context.logger

        #Input arguments
        self.input_l1 = self.action.args[0]
        self.header = self.input_l1.header['PRIMARY']
        self.namps = {'GREEN': int(self.header['GRNAMPS']),
                      'RED': int(self.header['REDAMPS'])}

        # self.chips = ['RED', 'GREEN']
        self.chips = ['GREEN',]
        self.orderlets = ['CAL_FLUX', 'SCI_FLUX1', 'SCI_FLUX2', 'SCI_FLUX3', 'SKY_FLUX']
        self.bad_regions = {'GREEN': (slice(0, 20), slice(0, 2040)),
                            'RED': (slice(16, 32), slice(0, 2040))}
        
    def mask_amplifiers(self):
        for chip in self.chips:
            if self.namps[chip] > 2:
                for ol in self.orderlets:
                    ol_name = chip + '_' + ol
                    self.input_l1[ol_name][self.bad_regions[chip]] = np.nan
        
        return self.input_l1
                
    def _perform(self):
        
        masked = self.mask_amplifiers()
        return Arguments(masked)
