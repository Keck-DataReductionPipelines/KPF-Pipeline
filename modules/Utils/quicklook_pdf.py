import matplotlib.pyplot as plt
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
import glob
import numpy as np
from fpdf import FPDF

class QuicklookPDF(KPF0_Primitive):
    
    def __init__(self,action,context):
        """Initializes quicklook_pdf utility."""
        KPF0_Primitive.__init__(self, action, context)
        # self.png_list = self.action.args[0] #using find files in recipe
        self.ql_directory = self.action.args[0]
    
    def pull_pngs(self):
        """Pulls pngs from quicklook directory as as to combine into pdf.

        Returns:
            list: .png file paths/names
        """
        pngs = glob.glob('{}/*.png'.format(self.ql_directory))
        return pngs
    
    def _perform(self):
        """Combines quicklook outputs into single pdf file.
        """
        png_list_full = self.pull_pngs()
        png_list = png_list_full[0:10]
        #print('png list:', png_list)
        date_time,_ = (png_list[0].split('_')[-1]).split('.')
        pdf = FPDF(orientation = 'L',unit='mm', format='A4')
        small_fig = 60
        big_fig = 100
        x_margin = 0
        y_margin = 10
        x  = [2+x_margin,x_margin+small_fig-3,x_margin+small_fig*2-3,x_margin+small_fig*3-3,x_margin+small_fig*4-3,x_margin,x_margin,x_margin,x_margin,x_margin+big_fig-2,x_margin+big_fig*1-5,x_margin+big_fig*1-5+big_fig*3/8*11/3]
        y  = [y_margin,y_margin,y_margin,y_margin+5,y_margin-3,y_margin+small_fig*4/5,y_margin+small_fig*4/5+big_fig*3/8,y_margin+small_fig*4/5+big_fig*3/8+big_fig*3/8,y_margin+small_fig*4/5+big_fig*3/8*3,y_margin+small_fig*4/5-3,y_margin+small_fig*4/5+big_fig*3/8*3,y_margin+small_fig*4/5+big_fig*3/8*3-5]
        w = [small_fig,small_fig,small_fig,small_fig,small_fig,big_fig,big_fig,big_fig,big_fig,200,big_fig*3/8*11/3,small_fig*0.9]
        h = [small_fig*4/5,small_fig*4/5,small_fig*4/5,small_fig*4/5*0.8,small_fig*4/5,big_fig*3/8,big_fig*3/8,big_fig*3/8,big_fig*3/8,200*3/5,big_fig*3/8,small_fig*4/5*0.9]
        print(png_list)
        pdf.add_page()
    
        for i in range(len(png_list)):
            image = png_list[i]
            print(image)
            pdf.image(image,x[i],y[i],w[i],h[i])
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 0, '11/29/2021 10:12:16  j4420001.fits j442 HD10700 n B5 10:15:12 900 1.2 2', 0)
        pdf.output(self.ql_directory+ '/quicklook_assembled_{}.pdf'.format(date_time), "F")
        
        pdf = FPDF(orientation = 'L',unit='mm', format='A3')
        # pnglist is the list with all image filenames
        x  = [10,100,10,100,10,100,0,200,200,230]
        y  = [10,10,80,80,160,160,230,10,230,160]
        w = [80,80,80,80,80,80,180,200,160,80]
        h = [80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,80*4/5,180*3/10,200*18/24,160*3/8,80*4/5]
        pdf.add_page()
        for i in range(len(png_list)):
            image = png_list[i]
            pdf.image(image,x[i],y[i],w[i],h[i])
        # pdf.output("quicklook_" + date_time + ".pdf", "F")
        pdf.output(self.ql_directory+ 'quicklook_assembled_{}.pdf'.format(date_time), "F")