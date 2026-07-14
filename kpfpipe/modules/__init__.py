"""Transform modules operating on KPF data-model objects (L0, L1, L2).

Re-export each processing module's public class so the API reference documents
them inline by class name (ImageAssembly, ...) rather than under their with-stem
module pages. The ``masters`` subpackage keeps its own page (see
``kpfpipe.modules.masters``).
"""

from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.cross_correlation import CrossCorrelation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.radial_velocity import RadialVelocity
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration

__all__ = [
    "BarycentricCorrection",
    "CalibrationAssociation",
    "CrossCorrelation",
    "ImageAssembly",
    "ImageProcessing",
    "RadialVelocity",
    "SpectralExtraction",
    "WavelengthCalibration",
]
