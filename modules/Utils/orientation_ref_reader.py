import numpy as np
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

class OrientationReference(KPF0_Primitive):
    """This utility reads in a reference .txt file with channel orientations. 
    """
    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)
        self.reference_path = self.action.args[0]

    def _perform(self):
        """Reads channel/image orientation .txt file and returns
        orientation key for each channel. File must be formatted in ASCII format.
        It should resemble the following:
        CHANNEL CHANNEL_KEYS CHANNEL_FFI_ROW CHANNEL_FFI_COLUMN CHANNEL_EXT
        1 4 1 1 1 1 2
        2 1 1 2 1 2 3
        3 3 2 1 2 1 4
        4 2 2 2 2 2 5
        where there are 5 columns, named CHANNEL, CHANNEL_KEYS, CHANNEL_FFI_ROW, CHANNEL_FFI_COLUMN, and CHANNEL_EXT, with 
        channel number, channel orientation key, intended row of image in FFI, intended column of image in FFI, and channel ext 
        underneath, respectively.
        Channel orientation key is as follows:
        1=overscan on bottom and left
        2=overscan on left and top
        3=overscan on top and right
        4=overscan on right and bottom
        """
        channel_ref = open(self.reference_path,'r')
        channels = []
        keys = []
        rows = []
        cols = []
        exts = []

        for line in channel_ref:
            columns = line.split()
            channel = columns[0]
            channels.append(channel)
            
            key = columns[1]
            keys.append(key)
            
            row = columns[2]
            rows.append(row)
            
            col = columns[3]
            cols.append(col)
            
            ext = columns[4]
            exts.append(ext)

        del channels[0]
        del keys[0]
        del rows[0]
        del cols[0]
        del exts[0]

        channels = list(map(int,channels))
        keys = list(map(int,keys))
        rows = list(map(int,rows))
        cols = list(map(int,cols))
        exts = list(map(str,exts))
        tot_frames = len(channels)/max(channels)

        if max(channels)==len(channels):
            print (f'CCD reference file appears to show {max(channels)} amplifiers per CCD, of which there is {int(tot_frames)}')

        elif max(channels)!=len(channels):
            if channels.count(max(channels)) == tot_frames:
                print (f'CCD reference file appears to show {max(channels)} amplifiers per CCD, of which there are {int(tot_frames)}')
         
        elif max(channels)!=len(channels):
            if channels.count(max(channels))!=tot_frames:
                raise TypeError('Irregular/incorrect channel list')

        all_output = channels,keys,rows,cols,exts
        return Arguments(all_output)