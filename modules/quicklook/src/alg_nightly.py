import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
import os
import pandas as pd
import glob
import math
from astropy import modeling
from astropy.time import Time
from datetime import datetime


class Nightly_summaryAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger




    def nightly_procedures(self,night):
        exposures_dir = self.config['Nightly']['exposures_dir']

        #get all exposures taken on a particular night
        file_list = glob.glob(exposures_dir+night+'/*.fits')
        for i in range(len(file_list)):
            file_list[i] = file_list[i][18:-8]
        print(file_list)