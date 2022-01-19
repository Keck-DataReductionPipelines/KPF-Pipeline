import matplotlib.pyplot as plt
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
import glob
import numpy as np
from fpdf import FPDF

class quicklook_pdf(KPF0_Primitive):
    
    def __init__(self,action,context):
        """Initializes quicklook_pdf utility."""
        KPF0_Primitive.__init__(self, action, context)
        self.png_list = self.action.args[0] #using find files in recipe
        self.ql_directory = self.action.args[1]
    
    def _perform(self):
        """Combines quicklook outputs into single pdf file.
        """
        print('png list:', self.png_list)
        #print(self.png_list[0])
        date_time,_ = (self.png_list[0].split('_')[-1]).split('.')
        pdf = FPDF(orientation = 'L',unit='mm', format='A3')
        # pnglist is the list with all image filenames
        x  = [10,100,10,100,10,100,0,200,200,230]
        y  = [10,10,80,80,160,160,230,10,230,160]
        w = [80,80,80,80,80,80,180,200,160,80]
        h = [80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,180*3/10,200*18/24,160*3/8,80*4/5]
        #print(self.png_list)
        pdf.add_page()
        for i in range(len(self.png_list)):
            image = self.png_list[i]
            pdf.image(image,x[i],y[i],w[i],h[i])
        # pdf.output("quicklook_" + date_time + ".pdf", "F")
        pdf.output(self.ql_directory+ 'quicklook_assembled_{}.pdf'.format(date_time), "F")