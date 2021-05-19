import numpy as np
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

class OrientationReference(KPF0_Primitive):
    """This utility reads in a reference .txt file with channel orientations. 
    """
    def __init__(self,action,context):
        """Initializes orientation-reference reading utility.
        """
        KPF0_Primitive.__init__(self, action, context)
        self.ref_path = self.action.args[0]

    def _perform(self):
        """Reads channel/image orientation .txt file and returns
        orientation key for each channel. File must be formatted in ASCII format.
        It should resemble the following:

        CHANNEL CHANNEL_KEYS
        1 4
        2 1
        3 3
        4 2

        where there are two columns, named CHANNEL and CHANNEL_KEYS, with 
        channel number and channel orientation key underneath, respectively.
        Channel orientation key is as follows:

        1=overscan on bottom and left
        2=overscan on left and top
        3=overscan on top and right
        4=overscan on right and bottom

        """
        channel_ref = open(ref_path,'r')
        keys = []
        for line in channel_ref:
            columns = line.split()
            key = columns[1]
            keys.append(key)
        del keys[0]
        return Arguments(keys)
