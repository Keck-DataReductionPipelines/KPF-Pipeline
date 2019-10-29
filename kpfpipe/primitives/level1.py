"""
Define primitives that operate on KPF data
"""

import numpy as np

from keckdrpframework.primitives.base_primitive import Base_primitive

from kpfpipe.primitives.core import KPF_Primitive
from kpfpipe.level2 import KPF2


class KPF1_Primitive(KPF_Primitive):
    """
    Base primitive for other KPF1 primitives.
    All KPF1 primitives should inherit from this one.
    
    Args:
        action (keckdrpframework.models.action.Action): Keck DRPF Action object
        context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object

    """
    def __init__(self, action, context):
        Base_primitive.__init__(self, action, context)

        self.level0 = self.action.args.level0
        self.level1 = self.action.args.level1

        # get some config from args
        self.orders = self.context.config.run.orders
        self.chips = self.context.config.instrument.chips
        self.regions = self.context.config.run.regions

        # get some config from file
        self.correction_mask = self.context.config.instrument.correction_mask

    def checklevel1(self):
        if (self.level1 is None) or (not self.valid_level1_data()):
            raise (TypeError, "Invalid data")

    def valid_level1_data(self):
        """
        Confirms that KPF1 object has necessary data/structure to operate on (after checking for its existence)
        The first checks are similar to valid_level0_data(), but we keep them separated because different checks may be
        necessary e.g., checking that we only have 1-d spectra for each chip now, the correct number of orderlets, etc.

        Returns:
            bool
        """

        # The absolutely necessary data in a level1 array is in the self.data dictionary.
        # Check that it has not been corrupted
        if type(self.level1.orderlets) is not dict:
            return False
        # Check that it contains some data
        if len(self.level1.orderlets) <= 0:
            return False
        # And that the data is appropriate
        for key in self.level1.orderlets.keys():
            if ((not isinstance(self.level1.orderlets[key][0].flux, np.ndarray)) or
                    not np.all(np.isfinite(self.level1.orderlets[key][0].flux))):
                return False
            # Could check for dimensionality, but we won't do this for flexibility
            # if self.data[key].shape == shape:
            #    return False
        # We will also want to check some other things eventually
        # if self.level1.header is None:
        #    return False
        return True

    def _pre_condition(self):
        """execute before any _perform method"""
        self.logger.debug('Running level1 primitive %s' % self.action.name)
        # self.logger.debug('Appending method to method list')
        # self.method_list.append(str(level1_method_function.__name__))
        # self.logger.debug('Checking level1 before method')

        if (self.action.args.level1 is None) or (not self.valid_level1_data()):
            self.logger.error("Invalid level 1 data")
            return False
        else:
            return True


class calibrate_wavelengths(KPF1_Primitive):
    """Calibrate wavelengths and append wavelength solution"""
    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    def _perform(self):
        """execute primitive"""
        if self.chips is True:
            self.chips = self.level0.data.keys()
        for chip in self.chips:
            for i in range(self.level1.Norderlets[chip]):
                self.logger.info('Calibrating wavelengths for order {} on the {} chip'.format(i, chip))
                self.level1.orderlets[chip][i].wav = np.mean(self.level0.data[chip], axis=1)


class remove_emission_line_regions(KPF1_Primitive):
    """Remove emission lines"""

    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    def _perform(self):
        """execute primitive"""
        if self.regions is True:
            self.logger.warning('Using default emission line region mask')
            self.regions = {'green': {0: [1, 2, 3], 3: [4, 5]}, 'red': {1: [1, 2, 3], 2: [1, 5]}}
        self.logger.info('removing emission line regions: %s' % str(self.regions))
        # I don't know how this will work so I'm just zero masking some random points
        for key in self.regions.keys():
            for orderlet_index in self.regions[key].keys():
                for wavelength_index in self.regions[key][orderlet_index]:
                    try:
                        self.level1.orderlets[key][orderlet_index].flux[wavelength_index] = 0.
                    except:
                        pass  # log errors, etc.


class remove_solar_regions(KPF1_Primitive):
    """Remove solar regions"""

    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    def _perform(self):
        """execute primitive"""
        if self.regions is True:
            self.logger.warning('Using default emission line region mask')
            self.regions = {'green': {0: [7], 2: [1, 5]}, 'red': {3: [1, 2, 3], 4: [0, 1, 2, 3, 4]}}
        self.logger.info('removing solar line regions: %s' % self.regions)
        # I don't know how this will work so I'm just zero masking some random points
        for key in self.regions.keys():
            for orderlet_index in self.regions[key].keys():
                for wavelength_index in self.regions[key][orderlet_index]:
                    try:
                        self.level1.orderlets[key][orderlet_index].flux[wavelength_index] = 0.
                    except:
                        pass  # log errors, etc.


class correct_telluric_lines(KPF1_Primitive):
    """Remove telluric lines"""

    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    def _perform(self):
        """execute primitive"""
        if self.correction_mask is True:
            self.logger.warning('Using default emission line region mask')
            self.correction_mask = {'green': {0: [[0, 0.1], [1, 0.5], [7, 0.1]], 2: [[1, 0.5]]}}  # no corrections in red
        self.logger.info('correcting tellurics with correction_mask: %s' % self.correction_mask)
        # Here I'm just adding some random values to random positions in random orderlets
        for key in self.correction_mask.keys():
            for orderlet_index in self.correction_mask[key].keys():
                for index_correction in self.correction_mask[key][orderlet_index]:
                    try:
                        self.level1.orderlets[key][orderlet_index].flux[index_correction[0]] += index_correction[1]
                    except:
                        pass  # log errors, etc.


class correct_wavelength_dependent_barycentric_velocity(KPF1_Primitive):
    """Correct for barycentric velocity"""

    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    def _perform(self):
        """execute primitive"""
        for key in self.level1.orderlets.keys():
            for orderlet in self.level1.orderlets[key]:
                for i in range(len(orderlet.flux)):
                    orderlet.flux[i] += 0.02 * i


class calculate_RV_from_spectrum(KPF1_Primitive):
    """Calculate CCF RV"""

    def __init__(self, action, context):
        """
        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        KPF1_Primitive.__init__(self, action, context)

    # Get the RV
    # Again, should we initialize the level2 object in this function, or have it separately
    def _perform(self):
        """execute primitive"""
        self.action.args.level2 = KPF2()
        if self.chips is True:
            self.chips = self.level1.orderlets.keys()
        for chip in self.chips:
            try:
                self.level2.rv[chip] = 1.0
            except AttributeError:
                pass  # log Error
