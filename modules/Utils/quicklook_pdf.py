import matplotlib.pyplot as plt
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
import glob
from fpdf import FPDF

class quicklook_pdf(KPF0_Primitive):
    
    def __init__(self,action,context):
        """Initializes quicklook_pdf utility."""
        KPF0_Primitive.__init__(self, action, context)
        self.ql_directory = self.action.args[0]
        self.date_time = self.action.args[1]

    #steps: 1. get all saved plots 2. combine into pdf
    def pull_plots(self):
        """Collects .pngs or .pdfs output by various modules for quicklook purposes.

        Returns:
            list: .png files to collate for resulting quicklook pdf.
        """
        #pdf_list = glob.glob(self.ql_directory + '*.pdf') #pdf or png
        png_list = glob.glob(self.ql_directory + '*.png')
        
        return png_list
    
    def quicklook_combine(self,png_list,pdf_list=None):
        """Combines quicklook outputs into single pdf file.

        Args:
            png_list (list, optional): List of .png files output by quicklook parts of modules to combine onto pdf. Defaults to None.
            pdf_list (list, optional): List of .pdf files output by quicklook parts of modules to combine onto pdf (if possible). Defaults to None.
        """
        pdf = FPDF(orientation = 'L',unit='mm', format='A3')
        # pnglist is the list with all image filenames
        x  = [10,100,10,100,10,100,0,200,200,230]
        y  = [10,10,80,80,160,160,230,10,230,160]
        w = [80,80,80,80,80,80,180,200,160,80]
        h = [80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,180*3/10,200*18/24,160*3/8,80*4/5]
        print(png_list)
        pdf.add_page()
        for i in range(len(png_list)):
            image = png_list[i]
            print(image)
            pdf.image(image,x[i],y[i],w[i],h[i])
        pdf.output("quicklook_" + self.date_time + ".pdf", "F")