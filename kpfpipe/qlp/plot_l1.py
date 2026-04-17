"""L1 quicklook plots for assembled KPF 2D frames."""

import os
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt


class PlotL1:
    """
    Quicklook plots for KPF L1 (assembled 2D) data.

    Takes a KPF1 object and generates plots of the assembled detector image.
    Pure visualization — no science computation.

    Args:
        l1_obj: KPF1 data object (post-ImageAssembly).
        output_dir: Directory to save PNG files. None = return Figure only.
    """

    def __init__(self, l1_obj, output_dir=None):
        self.l1 = l1_obj
        self.output_dir = output_dir
        self.obs_id = getattr(l1_obj, 'obs_id', None) or ''
        self.name = ''
        if 'PRIMARY' in l1_obj.headers:
            self.name = l1_obj.headers['PRIMARY'].get('OBJECT', '')
