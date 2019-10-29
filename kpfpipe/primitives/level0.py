"""
Define primitives that operate on KPF data
"""

import numpy as np

from keckdrpframework.primitives.base_primitive import Base_primitive
from keckdrpframework.models.arguments import Arguments

from kpfpipe.primitives.core import KPF_Primitive
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2


class KPF0_Primitive(KPF_Primitive):
    """
    Base primitive for other KPF0 primitives.
    All KPF0 primitives should inherit from this one.
    
    """
    def __init__(self, action, context):
        Base_primitive.__init__(self, action, context)

        self.level0 = action.args.level0

    def checklevel0(self):
        if ((self.level0 == None) or
                (not self.valid_level0_data())):
            raise (TypeError, "Invalid data")

    def valid_level0_data(self):
        # The absolutely necessary data in a level0 array is in the self.data dictionary.
        # Check that it has not been corrupted
        if type(self.level0.data) is not dict:
            return False
        # Check that it contains some data
        if len(self.level0.data) <= 0:
            return False
        # And that the data is appropriate
        for key in self.level0.data:
            if ((not isinstance(self.level0.data[key], np.ndarray)) or
                    not np.all(np.isfinite(self.level0.data[key]))):
                return False
            # Could check for dimensionality, but we won't do this for flexibility
            # if self.data[key].shape == shape:
            #    return False
        # We will also want to check some other things eventually
        # if self.level0.header is None:
        #    return False
        return True

    def _pre_condition(self):
        if (self.action.args.level0 is None) or (not self.valid_level0_data()):
            self.logger.error("Invalid level 0 data")
            return False
        else:
            return True


class subtract_bias(KPF0_Primitive):

    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)

        self.chips = self.action.args.chips

    def _perform(self):
        self.logger.debug('Entered subtract_bias')
        if self.chips is True:
            self.logger.warning('Chips have not been explicitly set in subtract_bias method')
            self.logger.warning('Setting chips to self.level0.data.keys()')
            self.chips = self.level0.data.keys()
        for chip in self.chips:
            self.logger.debug('Subtracting bias from chip: ' + chip)
            try:
                self.level0.data[chip] -= self.level0.bias[chip]
                rms = np.sqrt(np.mean(np.square(self.level0.data[chip])))
                med = np.median(self.level0.data[chip])
                self.logger.info('After bias subtraction on chip ' + chip + ', RMS = %f, median = %f' % (rms, med))
            except AttributeError:
                self.logger.error('There is no bias for chip ' + chip + ' so bias subtraction failed')
                pass  # maybe we don't want this to crash the program, but we record an ERROR in the log
        self.logger.debug('Finished subtract_bias method')

        self.action.args.level0 = self.level0

        return self.action.args


class divide_flat(KPF0_Primitive):

    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)

        self.chips = self.action.args.chips

    def _perform(self):
        if self.chips is True:
            self.chips = self.level0.data.keys()
        for chip in self.chips:
            try:
                self.level0.data[chip] -= self.level0.flat[chip]
                rms = np.sqrt(np.mean(np.square(self.level0.data[chip])))
                med = np.median(self.level0.data[chip])
                self.logger.info('After dividing flat on chip ' + chip + ', RMS = %f, median = %f' % (rms, med))
            except AttributeError:
                self.logger.error('There is no bias for chip ' + chip + ' so dividing flat failed')
                pass  # maybe we don't want this to crash the program, but we record an ERROR in the log
                      # Perhaps log some additional data if desired

        self.action.args.level0 = self.level0

        return self.action.args


class extract_spectrum(KPF0_Primitive):

    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)

        self.chips = self.action.args.chips
        self.orders = self.action.args.orders
        self.level1 = self.action.args.level1

    def _perform(self):
        if self.chips is True:
            self.chips = self.level0.data.keys()
        for chip in self.chips:
            for i in range(self.level1.Norderlets[chip]):
                # grab some parameter from the config objects embedded in self
                max_extraction_width = self.config.max_extraction_width

                # This is where the extraction algorithm is called. For now we just use np.mean
                self.logger.info('Extracting order {} on the {} chip'.format(i, chip))
                self.level1.orderlets[chip][i].flux = np.mean(self.level0.data[chip], axis=1)
                self.level1.orderlets[chip][i].flux_err = np.mean(self.level0.data[chip], axis=1)
                rms = np.sqrt(np.mean(np.square(self.level0.data[chip])))
                med = np.median(self.level0.data[chip])

        self.action.args.level1 = self.level1

        return self.action.args

