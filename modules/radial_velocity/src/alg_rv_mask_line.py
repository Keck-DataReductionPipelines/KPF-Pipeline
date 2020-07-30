import numpy as np

LIGHT_SPEED = 299792.458  # light speed in km/s


class RadialVelocityMaskLine:
    """ Mask line calculation for radial velocity analysis.

    This module defines class 'RadialVelocityMaskLine' and methods to produce mask line data for radial velocity
    analysis.

    Attributes:
        mask_line (dict): Instance to store mask information for a series of masks defined by mask file. This is used
            for cross correlation computation in radial velocity analysis, like::

                {
                    'start' : numpy.ndarray          # start points of masks
                    'end' : numpy.ndarray            # end points of masks
                    'center': numpy.ndarray          # center of masks
                    'weight': numpy.ndarray          # weight of masks
                    'bc_corr_start': numpy.ndarray     # adjusted start points of masks
                    'bc_corr_end': numpy.ndarray       # adjusted end points of masks
                }

    """

    def __init__(self):
        self.mask_line = None       # def_mask,

    def get_mask_line(self, mask_path, v_steps, zb_range,  mask_width=0.25, air_to_vacuum=False):
        """ Get mask information.

        Args:
            mask_path (str): Mask file path.
            v_steps (numpy.ndarray): Velocity steps at even interval.
            zb_range (numpy.ndarray): Array containing Barycentric velocity correction minimum and maximum.
            mask_width (float, optional): Mask width (km/s). Defaults to 0.25.
            air_to_vacuum (bool, optional): A flag indicating if converting mask from air to vacuum environment.
                Defaults to False.

        Returns:
            dict: Information of mask. Please refer to `mask_line` in `Attributes` of this class.

        """
        if self.mask_line is None:
            self.mask_line = self.compute_mask_line(mask_path, v_steps, zb_range[0], zb_range[1], mask_width,
                                                    air_to_vacuum)
        return self.mask_line

    @staticmethod
    def compute_mask_line(mask_path, v_steps, bc_corr_min, bc_corr_max, mask_width=0.25, air_to_vacuum=False):
        """ Calculate mask coverage based on the mask center, mask width, and barycentric velocity correction.

        The calculation includes the following steps,

            * collect the mask centers and weights from the mask file.
            * convert mask wavelength from air to vacuum environment if needed.
            * compute the start and end points around each center.
            * adjust start and end points around each center per velocity range and  maximum and minimum barycentric
              velocity correction from a period of time.

        Args:
            mask_path (str): Mask file path.
            v_steps (numpy.ndarray): Velocity steps at even interval.
            bc_corr_min (float): Barycentric velocity correction minimum.
            bc_corr_max (float): Barycentric velocity correction maximum.
            mask_width (float, optional): Mask width. Defaults to 0.25 (Angstrom)
            air_to_vacuum (bool, optional): A flag indicating if converting mask from air to vacuum environment.
                Defaults to False.

        Returns:
            dict: Information of mask. Please refer to `mask_line` in `Attributes` of this class.

        """

        line_center, line_weight = np.loadtxt(mask_path, dtype=float, unpack=True)  # load mask file
        if air_to_vacuum:
            line_center = RadialVelocityMaskLine.air_to_vac(line_center)
        line_mask_width = line_center * (mask_width / LIGHT_SPEED)

        mask_line = {'start': line_center - line_mask_width,
                     'end': line_center + line_mask_width,
                     'center': line_center,
                     'weight': line_weight}

        dummy_start = mask_line['start'] * ((1.0 + (v_steps[0] / LIGHT_SPEED)) / (bc_corr_max + 1.0))
        dummy_end = mask_line['end'] * ((1.0 + (v_steps[-1] / LIGHT_SPEED)) / (bc_corr_min + 1.0))
        mask_line.update({'bc_corr_start': dummy_start, 'bc_corr_end': dummy_end})

        return mask_line

    @staticmethod
    def air_to_vac(wave_air):
        """ Convert mask wavelength from air to vacuum environment.

        Args:
            wave_air (Union[numpy.ndarray, float]): Wavelength(s) in air environment.

        Returns:
            Union[numpy.ndarray, float]: Wavelength(s) in vacuum environment.

        """
        is_array = isinstance(wave_air, np.ndarray)
        if not is_array:
            new_wave_air = np.array([wave_air])
        else:
            new_wave_air = wave_air.copy()

        wave_vac = new_wave_air * 1.0
        g = wave_vac > 2000.0  # only modify above 2000 A(ngstroms)

        if np.any(g):
            for _ in [0, 1]:
                if isinstance(g, np.ndarray):
                    sigma2 = (1e4 / wave_vac[g]) ** 2.0  # Convert to wavenumber squared

                    # Compute conversion factor
                    fact = 1. + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)
                    wave_vac[g] = new_wave_air[g] * fact  # Convert Wavelength

        return wave_vac if is_array else wave_vac[0]

