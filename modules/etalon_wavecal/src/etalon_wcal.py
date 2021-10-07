# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

class EtalonWaveCal(KPF1_Primitive):
    """[summary]

    Args:
        KPF1_Primitive ([type]): [description]
    """
    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """Etalon Wavelength Calibration constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `EtalonWaveCal` event issued in recipe:
                    
                    `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file

            context (ProcessingContext): Contains path of config file defined for `etalon_wavecal` module in master config file associated with recipe.

        """

        KPF1_Primitive.__init__(self,action,context)

        self.l1_obj=self.action.args[0]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['etalon_wavecal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Etalon wavelength calibration algorithm setup
        self.alg=EtalonWaveCalAlg(self.config,self.logger)

    def _perform(self) -> None:
         if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Performing Etalon calibration')

        etalon_alg(self.l1_obj)

        #return Arguments(self.l1_obj)
