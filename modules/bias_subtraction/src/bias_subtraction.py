#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:19:46 2020

@author: paminabby
"""
# Standard dependencies
import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.bias_subtraction.src.alg import BiasSubtraction

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/bias_subtraction/configs/default.cfg'

class BiasSubtraction(KPF0_Primitive):
    def __init__(self, action:Action, context:ProcessingContext) -> None:

        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)
        self.logger=start_logger(self.__class__.__name__, config_path)

        #Input argument
        
        #Start logger
        self.logger=start_logger(self.__class__.__name__.,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        #Bias subtraction algorithm setup
        self.alg=BiasSubtraction(config=self.config,logger=self.logger)

        #Preconditions
        
        #Postconditions
        
        #Perform - primitive's action

        # 1) input bias files, get data from .fits
        if self.logger:
            self.logger.info("Bias Subtraction: inputting bias files...")
        #input 
        #biases_data= get .fits data

        # 2) stack bias files using util fxn, creates master bias
        if self.logger:
            self.logger.info("Bias Subtraction: creating master bias...")
        master_bias=stack_average(biases_data)

        # 3) output master bias
        if self.logger:
            self.logger.info("Bias Subtraction: outputting master bias .fits...")
        #output

        # 4) input raw science file, master bias
        if self.logger:
            self.logger.info("Bias Subtraction: inputting raw science and master bias frames...")
        #input
        #masterbias_data = get .fits data
        #raw_data = get .fits data

        # 5) subtract master bias from raw
        if self.logger:
            self.logger.info("Bias Subtraction: subtracting master bias from science...")
        bias_corrected_sci=self.alg.bias_subtraction(rawdata,masterbias_data)

        # 6) output bias-corrected science
        if self.logger:
            self.logger.info("Bias Subtraction: outputting bias-corrected science frame...")
        #output
        