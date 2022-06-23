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
from kpfpipe.models.base_model import KPFDataModel
import os
import pandas as pd



class ExtCopy(KPF_Primitive):
    """
    This module copies the data from one extension to another extension of the same data model object or another one.

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `ExtCopy` event issued in the recipe:

            - `action.args[0] (kpfpipe.models.KPFDataModel)`: Instance of `KPFDataModel` containing data model object
                   with the extension data to be copied from.
            - `action.args[1] (string)`: name of the extension to copy the data from.
            - `action.args[2] (string)`: name of the extension to copy the data to.
            - `action.args['to_data_model'] (string)`: the data model destination object, optional.
            - `action.args['output_file'] (string)`: path of the file to write out the data model object, optional.
            - `action.args['size_as'] (string)`: does the extension copy based on the size of given extension, optional.
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.from_data_model = action.args[0]
        self.ext_from = action.args[1]
        self.ext_to = action.args[2]
        self.size_as = (action.args['size_as']).upper() if 'size_as' in args_keys else None
        self.to_data_model = action.args['to_data_model'] if 'to_data_model' in args_keys else self.from_data_model
        self.output_file = action.args['output_file'] if 'output_file' in args_keys else None
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """

        success = self.ext_from and self.ext_from in self.from_data_model.__dict__ and \
                  self.ext_to and self.ext_to in self.to_data_model.__dict__

        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        if self.size_as is not None and hasattr(self.from_data_model, self.size_as):
            s_y, s_x = np.shape(self.from_data_model[self.size_as])
            self.to_data_model[self.ext_to] = self.from_data_model[self.ext_from][0:s_y, 0:s_x]
        else:
            self.to_data_model[self.ext_to] = self.from_data_model[self.ext_from]

        from_file = ''
        to_file = ''
        if isinstance(self.from_data_model, KPFDataModel) and isinstance(self.to_data_model, KPFDataModel):
            from_file = self.from_data_model.filename + ':'
            to_file = self.to_data_model.filename + ':'


        self.to_data_model.receipt_add_entry('ExtCopy', self.__module__,
                            f'extension copy from {from_file}{self.ext_from}  to {to_file}{self.ext_to}', 'PASS')


        if self.logger:
            self.logger.info("ExtCopy: copy from " + from_file + self.ext_from + ' to ' + to_file + self.ext_to)

        if self.output_file is not None:
            self.to_data_model.to_fits(self.output_file)

        return Arguments(self.to_data_model)


class FromCSV(KPF_Primitive):
    """
    This module read the data from csv file and return the data of specified columns.

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `FromCSV` event issued in the recipe:

            - `action.args[0] (str)`: file path.
            - `uescols (list|str)`: list of columns for the columns to be included in the output.
            - `<column_name> (str)`: row selection for the output in case the column value is met as specified.
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.csvfile = action.args[0]

        usecols = action.args['usecols'] if 'usecols' in args_keys else None
        if isinstance(usecols, str):
            self.usecols = [usecols]
        elif isinstance(usecols, int):
            self.usecols = [usecols]
        elif not isinstance(usecols, list):
            self.usecols = None
        else:
            self.usecols = usecols

        filters = {}

        for csvcol in args_keys:
            if csvcol == 'usecols':
                continue
            filters[csvcol] = action.args[csvcol]
        self.filters = filters

        self.logger = None
        if not self.logger:
            self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """

        success = os.path.exists(self.csvfile)

        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        csv_df = pd.read_csv(self.csvfile)
        csv_data = csv_df.values
        csv_cols = csv_df.columns.values.tolist()
        rows, cols = np.shape(csv_data)

        for key in self.filters:
            col_idx = csv_cols.index(key)

            if col_idx >= 0:
                val = self.filters[key]
                sel_rows = np.where(np.array([val in d for d in  csv_data[:, col_idx]]))[0]
                csv_data = csv_data[sel_rows, :]

        col_idx = []
        if self.usecols is not None:
            for ucol in self.usecols:
                idx = csv_cols.index(ucol)
                if idx >= 0:
                    col_idx.append(idx)
            col_sels = np.unique(np.array(col_idx, dtype=int))
        else:
            col_sels = np.arange(0, cols, dtype=int)

        data_sels = csv_data[:, col_sels]

        if self.logger:
            self.logger.info("FromCSV: done")

        return Arguments(data_sels)


class GetHeaderValue(KPF_Primitive):
    """
    This module read the data from csv file and return the data of specified columns.

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `GetHeaderValue` event issued in the recipe:

            - `action.args[0] (kpfpipe.models.level0.KPF0): instance of `KPF0`
            - `action.args(1) (str|list)`: key values
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)

        self.kpfobj = action.args[0]
        self.key_list = [str] if isinstance(action.args[1], str) else action.args[1]

        self.logger = None
        if not self.logger:
            self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """
        success = isinstance(self.key_list, list)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        val = None
        primary_header = self.kpfobj.header['PRIMARY']
        for k in self.key_list:
            if k in primary_header:
                val = primary_header[k]
                break

        if self.logger:
            self.logger.info("GetHeaderValue: done")

        return Arguments(val)


class SelectObs(KPF_Primitive):
    """
    This module read the data from csv file and return the data of specified columns.

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `GetHeaderValue` event issued in the recipe:

            - `action.args[0] (list): list of candidate files
            - `action.args['selection_ref']: csv file path containing objid for selection
            - `<column_name> (str)`: row selection for the output in case the column value is met as specified.
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.candidate_files = action.args[0]
        self.refcsv = action.args['selection_ref'] if 'selection_ref' in args_keys else None

        filters = {}

        for csvcol in args_keys:
            if csvcol == 'selection_ref':
                continue
            filters[csvcol] = action.args[csvcol]
        self.filters = filters
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """
        return True

    def _post_condition(self) -> bool:
        return True

    def _perform(self):

        file_selected = []
        if self.refcsv is not None and os.path.isfile(self.refcsv):
            csv_df = pd.read_csv(self.refcsv)
            csv_data = csv_df.values
            csv_cols = csv_df.columns.values.tolist()

            for key in self.filters:
                col_idx = csv_cols.index(key)

                if col_idx >= 0:
                    val = self.filters[key]
                    sel_rows = np.where(np.array([val in d for d in csv_data[:, col_idx]]))[0]
                    csv_data = csv_data[sel_rows, :]

            col_idx = csv_cols.index('observation_id')
            data_sels = csv_data[:, col_idx]
            for obs_sel in data_sels:
                for file_can in self.candidate_files:
                    if obs_sel in file_can:
                        file_selected.append(file_can)
                        break
        else:
            file_selected = self.candidate_files

        if self.logger:
            self.logger.info("SelectObs: done")
        return Arguments(file_selected)

