"""
KPF Image Assembly module.

Processes data from L1 to SL2.
 - extracts 1D spectrum from 2D FFI
"""
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial

from kpfpipe import REPO_ROOT
DEFAULTS = {'extraction_method':'box'}

class SpectralExtraction:
    """
    Single-letter variable names for 2D images in this class follow 
    Horne 1986 Optimal Extraction:
      - D = data
      - S = sky / scattered light
      - F = flat
      - P = profile
      - M = mask
      - W = weight
    """
    def __init__(self, l1_obj, config={}):
        self.l1_obj = l1_obj
        self.CHIPS = ['GREEN', 'RED']

        for k in DEFAULTS.keys():
            self.__setattr__(k, config.get(k,DEFAULTS[k]))


    def _read_order_trace_reference(self, chip):
        if not hasattr(self, 'order_trace'):
            self.order_trace = {}

        filepath = f'{REPO_ROOT}/reference/order_trace_{chip.lower()}.csv'
        with open(filepath, 'r') as f:
            self.order_trace[chip.upper()] = pd.read_csv(f, index_col=0)

        return self.order_trace[chip.upper()]


    def _get_orderlet_pixels(self, chip, fiber, order, return_coords=False):
        """
        Get a rectangular section of pixels containing a single orderlet.

        The section may contain pixels from adjacent orderlets if curvature
        of the target orderlet is sufficiently high. This is expected behavior.
        """
        chip = chip.upper()
        fiber = fiber.upper()

        data_image = self.l1_obj.data[f'{chip}_CCD']
        nrow, ncol = data_image.shape

        try:
            trace = self.order_trace[f'{chip}']
        except (KeyError, AttributeError) as e:
            trace = self._read_order_trace_reference(chip)

        trace = trace[(trace.Fiber == fiber) & (trace.Order == order)].squeeze()

        assert trace.ndim == 1, f"Expected only one trace, got {trace.shape[0]}"

        # track the trace position
        coeffs = np.array(trace[[f'Coeff{i}' for i in range(4)]], dtype=np.float32)

        trace_center = polynomial.polyval(np.arange(ncol), coeffs)
        trace_top    = trace_center + trace.TopEdge
        trace_bottom = trace_center - trace.BottomEdge

        off_detector = (trace_top > nrow-1) | (trace_bottom < 0)

        if np.any(off_detector):
            trace_top[off_detector] = np.minimum(trace_top, nrow-1)[off_detector]
            trace_center[off_detector] = np.minimum(trace_center, nrow-1)[off_detector]
            trace_bottom[off_detector] = np.minimum(trace_bottom, nrow-1)[off_detector]
    
            trace_top[off_detector] = np.maximum(trace_top, 0)[off_detector]
            trace_center[off_detector] = np.maximum(trace_center, 0)[off_detector]
            trace_bottom[off_detector] = np.maximum(trace_bottom, 0)[off_detector]

        box_zeropt = int(np.floor(trace_bottom.min()))
        box_height = int(np.ceil(trace_top.max())) - box_zeropt

        edge_pixel_top = np.array(np.floor(trace_top - box_zeropt), dtype=int)
        edge_pixel_bottom = np.array(np.floor(trace_bottom - box_zeropt), dtype=int)

        # broadcast vectors            
        _row = np.arange(box_height)[:,None]
        _edge_pixel_top = edge_pixel_top[None,:]
        _edge_pixel_bottom = edge_pixel_bottom[None,:]
        _trace_top = trace_top[None,:]
        _trace_bottom = trace_bottom[None,:]

        # make data (D) and weight (w) 2D arrays
        # sets W_ij for pixels fully outside (0) or inside (1) trace
        # sets W_ij for pixels at edge of trace to fractional values
        D = data_image[box_zeropt:box_zeropt + box_height]
        
        W = np.zeros_like(D, dtype=np.float32)
        W[(_row > _edge_pixel_bottom) & (_row < _edge_pixel_top)] = 1

        mask_top = _row == _edge_pixel_top
        frac_top = np.tile((_trace_top - box_zeropt - _edge_pixel_top), (box_height,1))
        W[mask_top] = frac_top[mask_top]

        mask_bot = _row == _edge_pixel_bottom
        frac_bot = np.tile((1 - (_trace_bottom - box_zeropt - _edge_pixel_bottom)), (box_height,1))
        W[mask_bot] = frac_bot[mask_bot]

        if return_coords:
            return D, W, box_zeropt, box_zeropt+box_height
        return D, W
