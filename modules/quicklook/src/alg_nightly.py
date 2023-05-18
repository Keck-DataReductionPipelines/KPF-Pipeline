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
        file_list = glob.glob(exposures_dir+night+'/*.fits')
        exposure_list = file_list[:][17:-9]
        print(exposure_list)
