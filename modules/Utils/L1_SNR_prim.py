# Standard dependencies
import configparser
import numpy as np
import pandas as pd
from astropy.io import fits

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
#from modules.continuum_normalization.src.alg import ContNormAlgg
from modules.Utils.analyze_l1 import AnalyzeL1

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/Utils/L1_SNR.cfg'

class L1_SNR(KPF1_Primitive):

    def __init__(self, action:Action, context:ProcessingContext) -> None:

        #Initialize parent class
        KPF1_Primitive.__init__(self,action,context)

        #input recipe arguments
        self.l1_obj=self.action.args[0]
        self.data_dir=self.action.args[1]
        # self.data_type=self.action.args[1]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['cont_norm']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # algorithm setup
        #self.alg=ContNormAlgg(self.config,self.logger)

    #Perform
    def _perform(self) -> None:
        L1_file = fits.open(self.data_dir+self.l1_obj.filename)

        L1_SNR = AnalyzeL1(L1_file)
        L1_SNR.measure_L1_snr(L1_file,snr_percentile=95)
        #print(L1_SNR.GREEN_SNR) #(orders number, orderlet number)

        self.l1_obj.header['PRIMARY']['SNRSC452'] = L1_SNR.GREEN_SNR[1,-1]
        self.l1_obj.header['PRIMARY']['SNRSK452'] = L1_SNR.GREEN_SNR[1,-2]
        self.l1_obj.header['PRIMARY']['SNRCL452'] = L1_SNR.GREEN_SNR[1,0]

        self.l1_obj.header['PRIMARY']['SNRSC548'] = L1_SNR.GREEN_SNR[25,-1]
        self.l1_obj.header['PRIMARY']['SNRSK548'] = L1_SNR.GREEN_SNR[25,-2]
        self.l1_obj.header['PRIMARY']['SNRCL548'] = L1_SNR.GREEN_SNR[25,0]


        self.l1_obj.header['PRIMARY']['SNRSC747'] = L1_SNR.RED_SNR[20,-1]
        self.l1_obj.header['PRIMARY']['SNRSK747'] = L1_SNR.RED_SNR[20,-2]
        self.l1_obj.header['PRIMARY']['SNRCL747'] = L1_SNR.RED_SNR[20,0]

        self.l1_obj.header['PRIMARY']['SNRSC865'] = L1_SNR.RED_SNR[-1,-1]
        self.l1_obj.header['PRIMARY']['SNRSK865'] = L1_SNR.RED_SNR[-1,-2]
        self.l1_obj.header['PRIMARY']['SNRCL865'] = L1_SNR.RED_SNR[-1,0]

        '''
        print('all science',L1_SNR.GREEN_SNR[1,-1]) #all science
        print('sky',L1_SNR.GREEN_SNR[1,-2]) #sky
        print('cal',L1_SNR.GREEN_SNR[1,0]) #cal
        print('all science',L1_SNR.GREEN_SNR[25,-1]) #all science
        print('sky',L1_SNR.GREEN_SNR[25,-2]) #sky
        print('cal',L1_SNR.GREEN_SNR[25,0]) #cal”

        #print(L1_SNR.G_SNR) #(orders number, orderlet number)
        print('all science',L1_SNR.GREEN_SNR[20,-1]) #all science
        print('sky',L1_SNR.GREEN_SNR[20,-2]) #sky
        print('cal',L1_SNR.GREEN_SNR[20,0]) #cal
        print('all science',L1_SNR.GREEN_SNR[-1,-1]) #all science
        print('sky',L1_SNR.GREEN_SNR[-1,-2]) #sky
        print('cal',L1_SNR.GREEN_SNR[-1,0]) #cal”
        '''

        self.l1_obj.to_fits(self.data_dir+self.l1_obj.filename)

        #print(self.l1_obj.header)


        #print(L1_SNR.RED_SNR)
