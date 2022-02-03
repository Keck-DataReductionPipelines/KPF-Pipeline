"""
    The file contains modules to handle the data model

    Attributes:
        ExtCopy: module to copy the data from one extension to another extension


"""

from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

import numpy as np


class ExtCopy(KPF_Primitive):
    """
    This module copies the data from one extension to another extension

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `ExtCopy` event issued in the recipe:

            - `action.args[0] (kpfpipe.models.KPFDataModel)`: Instance of `KPFDataModel` containing data model object
            - `action.args[1] (string)`: name of the extension to copy the data from
            - `action.args[2] (string)`: name of the extension to copy the data to
            - `action.args['output_file'] (string)`: path of the file to write out the data model object, optional.
            - `action.args['size_as'] (string)`: does the extension copy based on the size of given extension, optional.
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.data_model = action.args[0]
        self.ext_from = action.args[1]
        self.ext_to = action.args[2]
        self.size_as = (action.args['size_as']).upper() if 'size_as' in args_keys else None

        self.output_file = action.args['output_file'] if 'output_file' in args_keys else None

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """

        success = self.ext_from and self.ext_from in self.data_model.__dict__ and \
                  self.ext_to and self.ext_to in self.data_model.__dict__

        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        if self.size_as is not None and hasattr(self.data_model, self.size_as):
            s_y, s_x = np.shape(self.data_model[self.size_as])
            self.data_model[self.ext_to] = self.data_model[self.ext_from][0:s_y, 0:s_x]
        else:
            self.data_model[self.ext_to] = self.data_model[self.ext_from]

        self.data_model.receipt_add_entry('ExtCopy', self.__module__,
                                          f'extension copy from {self.ext_from} to {self.ext_to}', 'PASS')

        if self.output_file is not None:
            self.data_model.to_fits(self.output_file)

        return Arguments(self.data_model)
