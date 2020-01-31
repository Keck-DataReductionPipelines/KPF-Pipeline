

# Standard dependencies
import configparser as cp
import logging
import numpy as np
import sys

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
            self.flist = self.context.arg['tfa_input']
        except AttributeError:
            raise IOError('Mandatory input missing')

        try: 
            config_file = self.context.config_path['tfa_config']
        except AttributeError:
            config_file = self.default_config_path

        # Read the config file with config parser
        self.logger = start_logger(self.abbr, config_file)
        cfg = cp.ConfigParser(comment_prefixes='#')
        res = cfg.read(config_file)
        if res == []:
            cfg.read(self.default_config_path)
        # Now parse the config 
        self.parse_config(cfg)
        
        # Pre and post process

        self.res = arg.TFAFinalResult()

    def parse_config(self, config: cp.ConfigParser) -> None:
        '''
        get all logging related configurations
        '''
        pass

    def make_template(self) -> None:
        # Initialize the preliminary as the template
        prelim = alg.prob_for_prelim(self.flist)
        # The file name, without path 
        fname = prelim.split('/')[-1]
        self.logger.info('beginning to create template')
        self.logger.info('preliminary file used: {}'.format(fname))
        
        SP = arg.TFASpec(filename=prelim)
        SP = bary_correct(SP) 

        n_files = len(self.flist)
        # get the wavelength and specs of the preliminary  
        # as foundation to the template
        twave = SP._wave
        tspec = SP._spec

        # Currently just a average of all spectrum
        # should also be taking care of the outliers (3-sigma clipping)
        for i, file in enumerate(self.flist):
            name = file.split('/')[-1]
            self.logger.info('({}/{}) processing {}'.format(
                i+1, len(self.flist), name))
            S = arg.TFASpec(filename=file)
            SS = bary_correct(S)
            T = alg.SingleTFA(SP, SS, None, self.logger)
            res = T.run()
            rel = res.res_df[['alpha', 'success']].to_records()
            for order in rel:
                if order[2]: #success
                    flamb, fspec = SS.get_order(order[0])
                    fspec2 = np.interp(twave[order[0], :], flamb, fspec)
                    tspec[order[0], :] += fspec2
                else: 
                    n_files -= 1
        tspec = np.divide(tspec, n_files)
        self.logger.info('finised making templated')
        self.context.arg.tfa_out = arg.TFASpec(data=(twave, tspec), jd=SP.julian_day)
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

        