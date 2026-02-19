from astropy.stats import mad_std
import numpy as np

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.utils import get_datecode, fetch_filepath
from kpfpipe.constants import NROW, NCOL

class BaseMastersModule:
    def __init__(self, obs_ids):
        # TODO: swap obs_ids to list of filenames
        self.obs_ids = obs_ids


    @staticmethod
    def load_frame(obs_id):
        # TODO: add with statement for file handling
        datecode = get_datecode(obs_id)
        filepath = fetch_filepath(obs_id)
        l0_obj = KPF0.from_fits(filepath)

        return l0_obj


    @staticmethod
    def assemble_frame(l0_obj):
        return ImageAssembly(l0_obj).perform()


    def stack_frames(self, sigma_clip=5.0):
        """
        Stacks full frame images and computes clipped mean and variance
          * For N <= 5, statistics are computed directly
          * For N > 5, computation uses streaming Welford's algorithm

          Note: need to cache direct frames as e.g. self.direct_frames to avoid re-reading
          Note: make sure that files are clearing from memory after read (in KPF0?)
        """
        if len(self.obs_ids) <= 5:
            mean, var = self.compute_direct_mean_and_variance(sigma_clip = sigma_clip)
        else:
            mean, var = self.compute_streaming_mean_and_variance(sigma_clip = sigma_clip)

        return mean, var


    def compute_direct_mean_and_variance(self, sigma_clip=5.0, nframe_max = None):
        if nframe_max is None:
            nframe = len(self.obs_ids)
            obs_ids = self.obs_ids
        else:
            nframe = np.min([nframe_max,len(self.obs_ids)])
            obs_ids = self.obs_ids[:nframe]

        data_cube = np.zeros((nframe,NROW,NCOL),dtype=float)
        failure = 0
        
        for i, obs_id in enumerate(obs_ids):
            # TODO: scale by exposure time
            try:
                l0_obj = self.load_frame(obs_id)
                frame = self.assemble_frame(l0_obj)
                data_cube[i] = frame
            except Exception as e:
                logger.warning(f"Skipping {obs_id} in compute_direct_mean_and_variance: {e}")
                data_cube[i,...] = np.nan
                failure += 1
                continue

            if failure / nframe > 0.2:
                raise ValueError(f"more than 20% of frames in stack failed to load")

        if not sigma_clip:
            mean = np.nanmean(data_cube, axis=0)
            var = np.nanvar(data_cube, axis=0)
            return mean, var

        med = np.nanmedian(data_cube, axis=0)
        mad = mad_std(data_cube, axis=0, ignore_nan=True)
        out = np.abs(data_cube - med) / mad > sigma_clip

        clipped_mean = np.nansum(np.where(out, np.nan, data_cube), axis=0) / np.sum(~out, axis=0)
        clipped_var = np.nansum(np.where(out, np.nan, (data_cube - clipped_mean)**2), axis=0) / np.sum(~out, axis=0)

        return clipped_mean, clipped_var


    def compute_streaming_mean_and_variance(self, sigma_clip=5.0):
        """
        Computes mean and variance using Welford's algorithm
        Optimized to reduce RAM usage at the expense of compute speed
        Estimates approximate mean and variance directly to perform outlier rejection
        """
        if sigma_clip:
            approx_mean, approx_var = self.compute_direct_mean_and_variance(sigma_clip = sigma_clip, nframe_max = 5)
            lower = approx_mean - sigma_clip * np.sqrt(approx_var)
            upper = approx_mean + sigma_clip * np.sqrt(approx_var)
        else:
            lower = -np.inf
            upper = np.inf
            
        S = np.zeros((NROW,NCOL), dtype=float)
        S2 = np.zeros((NROW,NCOL), dtype=float)
        count = np.zeros((NROW,NCOL), dtype=int)
        failure = 0

        for i, obs_id in enumerate(self.obs_ids):
            # TODO: scale by exposure time
            try:
                l0_obj = self.load_frame(obs_id)
                frame = self.assemble_frame(l0_obj)
            except Exception as e:
                logger.warning(f"Skipping {obs_id} in compute_streaming_mean_and_variance: {e}")
                failure += 1
                continue

            if failure / len(self.obs_ids) > 0.2:
                raise ValueError(f"more than 20% of frames in stack failed to load")

            mask = (frame >= lower) & (frame <= upper)
            S += frame * mask
            S2 += frame ** 2 * mask
            count += mask.astype(int)
    
        bad = count <= 0.5 * len(self.obs_ids)
        count = np.where(bad, 1, count)

        clipped_mean = np.where(bad, np.nan, S / count)
        clipped_var = np.where(bad, np.nan, S2 / count - clipped_mean ** 2)

        return clipped_mean, clipped_var