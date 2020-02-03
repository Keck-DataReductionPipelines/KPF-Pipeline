

# Standard dependencies
import configparser as cp
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
import modules.TemplateFit.src.macro as mc
import modules.TemplateFit.src.alg as alg
import modules.TemplateFit.src.arg as arg
import modules.TemplateFit.src.primitives as prim

class TFAMakeTemplate(KPF1_Primitive):
    '''
    TFA Module: create a KFP1 template for followup RV calculation
    Input: 
        a list of KPF1 data models 
    
    '''
    abbr = 'TFATemp'
    default_config_path = 'modules/TemplateFit/configs/TFATemp.cfg'

    def __init__(self, 
            action: Action, 
            context: ProcessingContext) -> None:
        '''
        Initializer
        args: 
            arg: keckDRP defined object that contains input
            context: keckDRP defined object that keeps track of 
        '''

        # we assume that a ConfigParser class is included in the argument
        KPF1_Primitive.__init__(self, action, context)

        try: 
            in_arg = self.context.arg['tfa_input']
            self.flist = self.context.arg[in_arg]
        except AttributeError:
            raise IOError('Mandatory input missing')

        try: 
            config_file = self.context.config_path['tfa_config']
        except AttributeError:
            config_file = self.default_config_path

        # Read the config file with config parser
        self.logger = start_logger(self.abbr, config_file)
        self.cfg = cp.ConfigParser(comment_prefixes='#')
        res = self.cfg.read(config_file)
        if res == []:
            self.cfg.read(self.default_config_path)
        
        # Pre and post process

        self.res = arg.TFAFinalResult()

    def make_template(self) -> None:
        # Initialize the preliminary as the template
        prelim = alg.prob_for_prelim(self.flist, 'PRIMARY')
        # The file name, without path 
        self.logger.info('beginning to create template')
        self.logger.info('preliminary file used: {}'.format(prelim.filename))
        
        SP = alg.bary_correct(prelim)


        n_files = len(self.flist)
        # get the wavelength and specs of the preliminary  
        # as foundation to the template
        twave = copy.deepcopy(SP.spectrums['PRIMARY'].wave)
        tflux = copy.deepcopy(SP.spectrums['PRIMARY'].flux)

        # Currently just a average of all spectrum
        # should also be taking care of the outliers (3-sigma clipping)
        for i, data in enumerate(self.flist):
            print(data.get_order(28, 'PRIMARY'))
            data = alg.bary_correct(data)
            T = alg.SingleTFA(SP, data, self.cfg, self.logger)
            res = T.run()
            rel = res.res_df[['alpha', 'success']].to_records()
            for order in rel:
                if order[2]: #success
                    flamb, fflux = copy.deepcopy(data.get_order(order[0], 'PRIMARY'))
                    fflux2 = np.interp(twave[order[0], :], flamb, fflux)
                    tflux[order[0], :] += fflux2
                else: 
                    n_files -= 1
            self.logger.info('({}/{})[{}]processed {} after {} loop'.format(
                i+1, len(self.flist), 
                np.mean(res.res_df['success']), 
                data.filename, 
                np.mean(res.res_df['iteration'])))
        tflux = np.divide(tflux, n_files)
        self.logger.info('finised making templated')
        self.context.arg.tfa_out = KPF1from_array(
                result, SP.julian.jd, 'PRIMARY')
        self.context.arg.tfa_out.write_to('template.fits', 3)

    # def calc_velocity(self, temp: str, flist:list) -> arg.TFAFinalResult:
    #     '''

    #     '''
    #     self.logger.log('beginning to calculate radial velocity', 'info')
    #     for i, file in enumerate(flist):
    #         name = file.split('/')[-1]
    #         self.logger.log('({}/{}) processing {}'.format(i, len(flist), name), 'info')
    #         S = sp.Spec(filename=file)
    #         SS = self.pre.run(S)

    #         T = SingleTFA(temp, SS, self.cfg, self.logger)
    #         r = T.run()
    #         self.res.append(i, r.write_to_final())
    #     self.logger.log('finised calculating velocity', 'info')
    #     return self.res
    
    def _perform(self):
        self.make_template()

if __name__ == "__main__":
    pass

        