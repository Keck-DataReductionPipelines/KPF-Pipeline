import os

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1
from kpfpipe.logger import start_logger

# local dependencies
from modules.TemplateFit.src.macro import findfiles

class ReadKPF1(KPF1_Primitive): 
    '''
    Input: a directory path to the input data, in KPF1 format
    Output: None

    Convert a directory of KPF1 .fits data files into a list of KPF1
    data types. The list of KPF1 will be stored to pipeline.context
    with a key specified in the configuration file. This primitive 
    is meant to start the pipeline
    '''

    def __init__(self, 
                 action: Action, 
                 context: ProcessingContext) -> None:
        '''
        Constructor
        '''
        KPF1_Primitive.__init__(self, action, context)
        # default_path = 'modules/TemplateFit/configs/ReadKPF1.cfg'

        self.fpath = self.context.arg['pipeline_input']
        try:
            config_path = self.context.config_path['ReadKPF1_path']
        except: 
            config_path = 'modules/TemplateFit/configs/ReadKPF1.cfg'
        self.logger = start_logger(self.__class__.__name__, config_path)
    
    # Optional
    def _pre_condition(self):
        try: 
            assert(os.path.isdir(self.fpath))
            return True
        except AssertionError:
            return False

    def _perform(self):
        self.logger.info('beginning {}'.format(self.__class__.__name__))
        flist = findfiles(self.fpath, '.fits')
        self.context.arg['KPF1'] = []
        for fn in flist: 
            self.context.arg['KPF1'].append(KPF1.from_fits(fn))
            self.logger.info('read {}'.format(os.path.basename(fn)))
        self.logger.info('finished {}'.format(self.__class__.__name__))



