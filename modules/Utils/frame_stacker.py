import numpy as np
import numpy.ma as ma

class FrameStacker:

    """
    Description:
        This class stacks frames via averaging after clipping data
        some number sigmas +/- the median on a pixel-by-pixel basis.
        Sigma is a robust estimator of data dispersion along the
        z-axis of the input stack at a given pixel position.

    Arguments:
        frames_data (numpy array): 3-D stack of images.
        n_sigma (float): Number of sigmas for data clipping (default = 2.5).

    Attributes:
        frames_data (numpy array) of image stack.
        n_sigma (float): Number of sigmas for data clipping (default = 2.5).
    """

    __version__ = '1.0.1'
    
    # Pre-computed correction factors for common n_sigma values
    # These eliminate the need for expensive Monte Carlo simulations
    _CORRECTION_FACTORS = {
        1.0: 3.468491,
        1.5: 1.827166,
        2.0: 1.299182,
        2.1: 1.243576,  # Common value in KPF pipeline
        2.5: 1.100510,
        3.0: 1.028345,
        3.5: 1.006433,
        4.0: 1.001323
    }
    
    @classmethod
    def calculate_correction_factor(cls, n_sigma, n_trials=50, n_samples=1000000):
        """
        Calculate accurate correction factor for a given n_sigma using optimized Monte Carlo.
        Use this to populate the _CORRECTION_FACTORS cache with accurate values.
        
        Args:
            n_sigma (float): Number of sigmas for clipping
            n_trials (int): Number of Monte Carlo trials (default: 50 for accuracy)
            n_samples (int): Number of samples per trial (default: 1M)
            
        Returns:
            float: Correction factor
        """
        print(f"Calculating correction factor for n_sigma={n_sigma} using {n_trials} trials...")
        
        # Generate all random samples at once: shape (n_trials, n_samples)
        a = np.random.normal(0.0, 1.0, (n_trials, n_samples))
        
        # Calculate statistics for each trial
        med = np.median(a, axis=1)  # Median for each trial
        p16 = np.percentile(a, 16, axis=1)  # 16th percentile for each trial
        p84 = np.percentile(a, 84, axis=1)  # 84th percentile for each trial
        sigma = 0.5 * (p84 - p16)  # Robust sigma for each trial
        
        # Calculate clipping bounds for each trial
        mdmsg = med - n_sigma * sigma  # Lower bound for each trial
        mdpsg = med + n_sigma * sigma  # Upper bound for each trial
        
        # Create masks for each trial
        b = a < mdmsg[:, np.newaxis]  # Below lower bound
        c = a > mdpsg[:, np.newaxis]  # Above upper bound
        mask = np.any([b, c], axis=0)  # Combined mask
        
        # Calculate variance for each trial
        var_trials = []
        for i in range(n_trials):
            trial_data = a[i, ~mask[i]]  # Data not masked for this trial
            if len(trial_data) > 0:
                var_trials.append(np.var(trial_data))
            else:
                var_trials.append(1.0)  # Fallback if all data is masked
        
        np_var_trials = np.array(var_trials)
        avg_var_trials = np.mean(np_var_trials)
        std_var_trials = np.std(np_var_trials)
        corr_fact = 1.0 / avg_var_trials
        
        print(f"Result: avg_var={avg_var_trials:.6f}, std_var={std_var_trials:.6f}, corr_fact={corr_fact:.6f}")
        return corr_fact

    def __init__(self,frames_data,n_sigma=2.5,logger=None):
        self.frames_data = frames_data
        self.n_sigma = n_sigma
        if logger:
            self.logger = logger
        else:
            self.logger = None

        if self.logger:
            self.logger.info('Started {}'.format(self.__class__.__name__))

    def compute_clip_corr(self):

        """
        Compute a correction factor to properly reinflate the variance after it is
        naturally diminished via data-clipping.  Uses pre-computed correction factors
        for common n_sigma values to avoid expensive Monte Carlo simulations.
        Falls back to Monte Carlo method for uncommon n_sigma values.
        """

        n_sigma = self.n_sigma

        # Debug: Print the actual n_sigma value and available cache keys
        if self.logger:
            self.logger.debug('{}.compute_clip_corr(): n_sigma={}, type={}, cache_keys={}'.\
                format(self.__class__.__name__, n_sigma, type(n_sigma), list(self._CORRECTION_FACTORS.keys())))
        else:
            print('---->{}.compute_clip_corr(): n_sigma={}, type={}, cache_keys={}'.\
                format(self.__class__.__name__, n_sigma, type(n_sigma), list(self._CORRECTION_FACTORS.keys())))

        # Check if we have a pre-computed correction factor
        if n_sigma in self._CORRECTION_FACTORS:
            corr_fact = self._CORRECTION_FACTORS[n_sigma]
            
            if self.logger:
                self.logger.debug('{}.compute_clip_corr(): Using cached correction factor for n_sigma={}: {}'.\
                    format(self.__class__.__name__, n_sigma, corr_fact))
            else:
                print('---->{}.compute_clip_corr(): Using cached correction factor for n_sigma={}: {}'.\
                    format(self.__class__.__name__, n_sigma, corr_fact))
            
            return corr_fact

        # Fall back to Monte Carlo for uncommon n_sigma values
        if self.logger:
            self.logger.debug('{}.compute_clip_corr(): n_sigma={} not in cache, running Monte Carlo simulation'.\
                format(self.__class__.__name__, n_sigma))
        else:
            print('---->{}.compute_clip_corr(): n_sigma={} not in cache, running Monte Carlo simulation'.\
                format(self.__class__.__name__, n_sigma))

        # Optimized Monte Carlo: Run all trials at once (much faster)
        n_trials = 10
        n_samples = 1000000
        
        # Generate all random samples at once: shape (n_trials, n_samples)
        a = np.random.normal(0.0, 1.0, (n_trials, n_samples))
        
        # Calculate statistics for each trial
        med = np.median(a, axis=1)  # Median for each trial
        p16 = np.percentile(a, 16, axis=1)  # 16th percentile for each trial
        p84 = np.percentile(a, 84, axis=1)  # 84th percentile for each trial
        sigma = 0.5 * (p84 - p16)  # Robust sigma for each trial
        
        # Calculate clipping bounds for each trial
        mdmsg = med - n_sigma * sigma  # Lower bound for each trial
        mdpsg = med + n_sigma * sigma  # Upper bound for each trial
        
        # Create masks for each trial
        b = a < mdmsg[:, np.newaxis]  # Below lower bound
        c = a > mdpsg[:, np.newaxis]  # Above upper bound
        mask = np.any([b, c], axis=0)  # Combined mask
        
        # Calculate variance for each trial (avoiding masked arrays for speed)
        var_trials = []
        for i in range(n_trials):
            trial_data = a[i, ~mask[i]]  # Data not masked for this trial
            if len(trial_data) > 0:
                var_trials.append(np.var(trial_data))
            else:
                var_trials.append(1.0)  # Fallback if all data is masked
        
        np_var_trials = np.array(var_trials)
        avg_var_trials = np.mean(np_var_trials)
        std_var_trials = np.std(np_var_trials)
        corr_fact = 1.0 / avg_var_trials

        if self.logger:
            self.logger.debug('{}.compute_clip_corr(): avg_var_trials,std_var_trials,corr_fact = {},{},{}'.\
                format(self.__class__.__name__,avg_var_trials,std_var_trials,corr_fact))
        else:
            print('---->{}.compute_clip_corr(): avg_var_trials,std_var_trials,corr_fact = {},{},{}'.\
                format(self.__class__.__name__,avg_var_trials,std_var_trials,corr_fact))

        return corr_fact

    def compute(self):

        """
        Perform n-sigma data clipping and subsequent stack-averaging,
        using data from class attributes.

        Return the data-clipped-mean image.
        """

        cf = self.compute_clip_corr()

        a = self.frames_data
        n_sigma = self.n_sigma
        frames_data_shape = np.shape(a)

        if self.logger:
            self.logger.debug('{}.compute(): self.n_sigma,frames_data_shape = {},{}'.\
                format(self.__class__.__name__,self.n_sigma,frames_data_shape))
        else:
            print('---->{}.compute(): self.n_sigma,frames_data_shape = {},{}'.\
                format(self.__class__.__name__,self.n_sigma,frames_data_shape))

        # Calculate robust statistics (reverted - separate calls are faster)
        med = np.median(a, axis=0)
        p16 = np.percentile(a, 16, axis=0)
        p84 = np.percentile(a, 84, axis=0)
        sigma = 0.5 * (p84 - p16)
        
        # Calculate clipping bounds
        mdmsg = med - n_sigma * sigma
        mdpsg = med + n_sigma * sigma
        
        # Create mask for clipped data (unchanged)
        mask = (a < mdmsg) | (a > mdpsg)
        
        # OPTIMIZED: Replace masked arrays with vectorized operations
        # This is 5-10x faster than masked arrays for large datasets
        
        # Count valid (non-clipped) pixels per position
        cnt = np.sum(~mask, axis=0).astype(np.float32)
        
        # Avoid division by zero
        cnt = np.maximum(cnt, 1.0)
        
        # Calculate mean using vectorized operations
        # Set clipped values to 0, then sum and divide by count
        a_clipped = np.where(mask, 0.0, a)
        sum_clipped = np.sum(a_clipped, axis=0)
        avg = sum_clipped / cnt
        
        # Calculate variance using vectorized operations
        # Var = E[X^2] - E[X]^2, but we need to account for clipping
        a_squared = np.where(mask, 0.0, a * a)
        sum_squared = np.sum(a_squared, axis=0)
        avg_squared = sum_squared / cnt
        
        # Corrected variance calculation for clipped data
        var = (avg_squared - avg * avg) * cf
        
        # Calculate uncertainty
        unc = np.sqrt(var / cnt)

        if self.logger:
            self.logger.debug('{}.compute(): avg(stack_avg),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))
        else:
            print('---->{}.compute(): avg(stack_avg),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))

        return avg,var,cnt,unc

#
# Method similar to compute() to be called in case of insufficient number of frames in stack,
# in which case the estimator of the expected value will be the stack median.
# The calling program determines whether to call this method.
#

    def compute_stack_median(self):

        """
        Compute median of stack.
        Data dispersion is based on the median absolute deviation.

        Returns the stack median image.
        """

        a = self.frames_data
        frames_data_shape = np.shape(a)

        if self.logger:
            self.logger.debug('{}.compute(): frames_data_shape = {}'.\
                format(self.__class__.__name__,frames_data_shape))
        else:
            print('---->{}.compute(): frames_data_shape = {}'.\
                format(self.__class__.__name__,frames_data_shape))

        med = np.median(a, axis=0)
        mad = np.median(np.absolute(a - med),axis=0)

        var = mad * mad
        cnt = np.full((frames_data_shape[1],frames_data_shape[2]),frames_data_shape[0])
        unc = np.sqrt(var/cnt)

        if self.logger:
            self.logger.debug('{}.compute(): avg(stack_med),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,med.mean(),cnt.mean(),unc.mean()))
        else:
            print('---->{}.compute(): avg(stack_med),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,med.mean(),cnt.mean(),unc.mean()))

        return med,var,cnt,unc
    
