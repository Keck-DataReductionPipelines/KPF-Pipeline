# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os
import importlib
import configparser as cp
import logging

from kpfpipe.logger import start_logger

# KeckDRPFramework dependencies
from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext


class KPFPipeline(BasePipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Attributes:
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
    """
    # Modification: 
    name = 'KPF-Pipe'
    event_table = {
        # action_name: (name_of_callable, current_state, next_event_name)
        'evaluate_recipe': ('evaluate_recipe', 'evaluating_recipe', None), 
        'exit': ('exit_loop', 'exiting...', None),
        'TFAMakeTemplate': ('TFAMakeTemplate', 'TEST', None)
        }
    

    def __init__(self, context):
        BasePipeline.__init__(self, context)

    def start(self, config: cp.ConfigParser) -> None:
        '''
        Initialize the customized pipeline.
        Customized in that it sets up logger and configurations differently 
        from how the BasePipeline does. 
        Args: 
            config: a ConfigParser object containing pipeline configuration 
        '''
        ## setup logger
        try: 
            self.logger = start_logger(self.name, config['LOGGER'])
        except KeyError: 
            raise IOError('cannot find [LOGGER] section in config')
        self.logger.info('Logger started')
        ## setup pipeline configuration 
        # Technically the pipeline's configuration is stored in self.context as 
        # a ConfigClass() defined by keckDRP. But we will be using configParser
        try:
            self.config =  config['PIPELINE']
        except KeyError:
            raise IOError('cannot find [PIPELINE] section in config')

        ## Setup argument
        try: 
            arg = config['ARGUMENT']
        except KeyError:
            raise IOError('cannot find [ARGUMENT] section in config')
        self.context.arg = arg

        ## Setup primitive-specific configs:
        self.context.tfa_config = config['MODULES'].get('tfa_config')
        self.logger.info('Finished initializting Pipeline')

    def evaluate_recipe(self, action, context):
        """
        Evaluates the recipe file (Python file) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        try: 
            recipe_file = action.args.recipe
            f = open(recipe_file)
            fstr = f.read()
            exec(fstr)
            f.close()
        except:
            print(sys.exit_info())

    def exit_loop(self, action, context):
        """
        Force the Keck DRP Framework to exit the infinite loop

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        self.logger.info("exiting pipeline...")
        os._exit(0)