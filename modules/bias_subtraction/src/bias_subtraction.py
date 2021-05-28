
# Standard dependencies
import configparser
import numpy as np

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
from modules.Utils.overscan_subtract import OverscanSubtraction

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/bias_subtraction/configs/default.cfg'

class BiasSubtraction(KPF0_Primitive):
    """This module defines class `BiasSubtraction,` which inherits from `KPF0_Primitive` and provides methods
    to perform the event `bias subtraction` in the recipe.

    Args:
        KPF0_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `BiasSubtraction` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `bias_subtraction` module in master config file associated with recipe.

    Attributes:
        rawdata (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[0]`            
        masterbias (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[1]`
        data_type (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[2]`


        config_path (str): Path of config file for the computation of bias subtraction.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.bias_subtraction.src.alg.BiasSubtraction): Instance of `BiasSubtraction,` which has operation codes for bias subtraction.

    """
    def __init__(self, 
                action:Action, 
                context:ProcessingContext) -> None:
        """
        BiasSubtraction constructor.

        Args:
            action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `BiasSubtraction` event issued in recipe:

                `action.args[0]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing raw image data
                `action.args[1]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing frame orientation key 
                `action.args[2]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing master bias data
                `action.args[3]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing the instrument/data type

            context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `bias_subtraction` module in master config file associated with recipe.

        """
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.rawdata=self.action.args[0]
        self.masterbias=self.action.args[1]
        self.channel_no=self.action.args[2]
        self.channel_orients=self.action.args[3]
        self.ch_ffi_rows=self.action.args[4]
        self.ch_ffi_cols=self.action.args[5]
        self.channel_exts=self.action.args[6]
        self.overscan_region=self.action.args[7]
        self.overscan_method=self.action.args[8]
        self.data_type=self.action.args[9]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['bias_subtraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Bias subtraction algorithm setup

        self.alg=BiasSubtraction(self.rawdata,self.config,self.logger)

        #Preconditions
        
        #Postconditions
        
        #Perform - primitive's action
    def _perform(self) -> None:
        """Primitive action - 
        Performs bias subtraction by calling method 'bias_subtraction' from BiasSubtraction.
        Returns the bias-corrected raw data, L0 object.

        Returns:
            Arguments object(np.ndarray): Level 0, bias-corrected, raw observation data
        """
    # ~pull correct extensions from raw file
        if self.logger:
            self.logger.info("Bias Subtraction: Pulling bias frame extensions")
        
        frames_data = []
        for ext in self.channel_exts:
            data = self.rawdata[ext].data
            frames_data.append(data)
        frames_data = np.array(frames_data)

        ffi_exts = self.alg.get_ffi_exts()

    # ~overscan subtraction (includes subtraction,flipping frame, etc) and bias sub

        for frame_no in range(len(ffi_exts)):
            single_frame_data = np.array_split(frames_data,len(ffi_exts))[frame_no]
            if self.logger:
                self.logger.info(f"Bias Subtraction: performing overscan subtraction for {frame_no+1} of {len(ffi_exts)}...") 
            full_frame_img = OverscanSubtraction(single_frame_data, self.overscan_region, self.overscan_method, self.channel_orients)

            if self.logger:
                self.logger.info(f"Bias Subtraction: subtracting master bias from raw full frame image for {frame_no+1} of {len(ffi_exts)}...")
            bias_subbed = self.alg.bias_subtraction(full_frame_img, self.masterbias)

            self.rawdata[ffi_exts[frame_no]].data = bias_subbed

        # ~save new file to original L0 file
        return Arguments(self.rawdata)