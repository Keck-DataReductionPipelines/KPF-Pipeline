
# Standard dependencies
import configparser
import numpy as np
from astropy.io import fits

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.image_processing.src.alg import ImageProcessingAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/image_processing/configs/default.cfg'

class ImageProcessing(KPF0_Primitive):
    """
    Pipeline primitive for KPF Level 0 image processing.

    This class provides methods to perform image processing actions such as bias subtraction,
    dark subtraction, flat division, cosmic ray masking, background subtraction, and bad pixel masking
    on KPF0 data. The specific action is determined by the type of the correcting file or an action string.

    Available action types:
        - Bias: Subtracts a master bias frame from the raw image.
        - Dark: Subtracts a master dark frame from the raw image.
        - Flat: Divides the raw image by a flat field frame.
        - Remove_Cosmics: Masks cosmic rays using the astroscrappy algorithm.
        - Background_Subtraction: Subtracts background using an ordermask file.
        - pixelmask: Masks bad pixels using a provided mask.

    To add new actions, modify the `DEFINED_ACTIONS` list or set the IMTYPE in the header of the correcting file.

    Args:
        action (keckdrpframework.models.action.Action): Contains positional and keyword arguments
            passed by the `ImageProcessing` event issued in the recipe.
            - action.args[0]: KPF0 instance containing the raw image data.
            - action.args[1]: KPF0 instance with the correcting file (e.g., master bias, dark, flat)
              or a string specifying an action (e.g., "remove_cosmics").
            - action.args[2]: KPF0 instance containing FITS FFI extension(s) list.
            - action.args[3]: KPF0 instance specifying the instrument/data type.
            - action.args[4]: KPF0 instance indicating quicklook toggle (True/False).
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains
            the path to the config file for image processing, as defined in the master config file.

    Attributes:
        raw_file (KPF0): The raw image data.
        correcting_file_or_action (KPF0 or str): The correcting file or action string.
        ffi_exts (KPF0): FITS FFI extension(s) list.
        data_type (KPF0): Instrument/data type.
        quicklook (KPF0): Quicklook toggle.
        config_path (str): Path to the configuration file for image processing.
        config (configparser.ConfigParser): Parsed configuration.
        logger (logging.Logger): Logger instance.
        alg (ImageProcessingAlg): Image processing algorithm handler.
    """
    def __init__(self,
                action:Action,
                context:ProcessingContext) -> None:
        """
        ImageProcessing constructor.

        Args:
            action (keckdrpframework.models.action.Action): Contains positional
                arguments and keyword arguments passed by the `ImageProcessing`
                event issued in recipe:

                `action.args[0]`(kpfpipe.models.level0.KPF0)`: Instance of
                    `KPF0` containing target image data (e.g., a raw image)

                `action.args[1]`(kpfpipe.models.level0.KPF0)`: Instance of
                    `KPF0` containing either correcting file data (e.g., master
                    bias data, dark data) OR an action string, specifying the
                    in-place image processing action to perform.  Currently,
                    the only implemented action is "remove_cosmics".

                `action.args[2]`(kpfpipe.models.level0.KPF0)`: Instance of
                    `KPF0` containing FITS FFI extension(s) list

                `action.args[3]`(kpfpipe.models.level0.KPF0)`: Instance of
                    `KPF0` containing the instrument/data type

                `action.args[4]`(kpfpipe.models.level0.KPF0)`: Instance of
                    `KPF0` containing quicklook toggle (T/F)

            context
            (keckdrpframework.models.processing_context.ProcessingContext):
                Contains path of config file defined for `bias_subtraction`
                module in master config file associated with recipe.
        """
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.raw_file = self.action.args[0]
        self.correcting_file_or_action = self.action.args[1]
        #self.masterbias = self.action.args[1]
        self.ffi_exts = self.action.args[2]
        self.data_type = self.action.args[3]
        self.quicklook = self.action.args[4]

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['image_processing']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Image processing algorithm setup

        self.alg = ImageProcessingAlg(self.raw_file, self.ffi_exts,
                                      self.quicklook, self.data_type,
                                      config=self.config, logger=self.logger)

        #Preconditions

        #Postconditions

    #Perform - primitive's action
    def _perform(self) -> None:
        """Primitive action -
        Performs image processing by calling method 'image_processing' from
        ImageProcess.  Returns the bias/dark/background corrected raw data, L0
        object.

        Returns:
            Arguments object(np.ndarray): Level 0 observation data
        """

        #until master file part of data model is fixed

        DEFINED_ACTIONS = ['remove_cosmics']

        if self.correcting_file_or_action not in DEFINED_ACTIONS:
            #until master file part of data model is fixed
            if isinstance(self.correcting_file_or_action, KPF0):
                correcting_file_or_action = self.correcting_file_or_action
            else:
                correcting_file_or_action = KPF0.from_fits(
                    self.correcting_file_or_action
                )
            
            if 'IMTYPE' not in correcting_file_or_action.header['PRIMARY']:
                raise KeyError("IMTYPE not in header of file {}".format(correcting_file_or_action.filename))

            if correcting_file_or_action.header['PRIMARY']['IMTYPE'].lower() == 'ordermask':
                action_type = 'Background_Subtraction'
            else:
                action_type = correcting_file_or_action.header['PRIMARY']['IMTYPE']

        else:
            if self.correcting_file_or_action == 'remove_cosmics':
                action_type = 'Remove_Cosmics'

        if action_type == 'Bias':
            if self.logger:
                self.logger.info(
                    f'Bias Subtraction: subtracting master bias from raw FFI(s)'
                )
            bias_subbed = self.alg.bias_subtraction(correcting_file_or_action)

        if action_type == 'Dark':
            if self.logger:
                self.logger.info(
                    f'Dark Subtraction: subtracting dark frame from raw FFI(s)'
                )
            dark_subbed = self.alg.dark_subtraction(correcting_file_or_action)

        if action_type == 'Flat':
            if self.logger:
                self.logger.info(
                    f'Flat Division: dividing out flat frame {correcting_file_or_action} from raw FFI(s)'
                )
            flat_corrected = self.alg.flat_division(correcting_file_or_action)

        if action_type == 'Remove_Cosmics':
            if self.logger:
                self.logger.info(
                    f'Cosmic ray removal: running astroscrappy on raw FFI(s)'
                )
            cosmicray_subbed = self.alg.cosmic_ray_masking()

        if action_type == 'Background_Subtraction':
            if self.logger:
                self.logger.info(
                    f'Background Subtraction: subtracting background from raw FFI(s) {self.ffi_exts}'
                )
            self.alg.background_subtraction(correcting_file_or_action)
        
        if action_type == 'pixelmask':
            if self.logger:
                self.logger.info(
                    f'Bad pixel masking: masking bad pixels in FFI(s) {self.ffi_exts}'
                )
            self.alg.bad_pixel_mask(correcting_file_or_action)

        return Arguments(self.alg.get())
