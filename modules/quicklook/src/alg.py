import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class QuicklookAlg:
    """
    
    """
    
    def __init__(self,config=None,logger=None):
        
        """
        
        """
        self.config=config
        self.logger=logger
        
    def testrun(self,twod_image):
        "for testing"
        plt.figure()
        plt.imshow(twod_image['GREEN_CCD'])
        plt.savefig('/Users/paminabby/Desktop/green_ccd.png')
        
        plt.figure()
        plt.imshow(twod_image['RED_CCD'])
        plt.savefig('/Users/paminabby/Desktop/red_ccd.png')

