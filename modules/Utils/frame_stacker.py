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
        naturally diminished via data-clipping.  Employ a simple Monte Carlo method
        and standard normal deviates to simulate the data-clipping and obtain the
        correction factor.
        """

        n_sigma = self.n_sigma

        var_trials = []
        for x in range(0,10):
            a = np.random.normal(0.0, 1.0, 1000000)
            med = np.median(a, axis=0)
            p16 = np.percentile(a, 16, axis=0)
            p84 = np.percentile(a, 84, axis=0)
            sigma = 0.5 * (p84 - p16)
            mdmsg = med - n_sigma * sigma
            b = np.less(a,mdmsg)
            mdpsg = med + n_sigma * sigma
            c = np.greater(a,mdpsg)
            mask = np.any([b,c],axis=0)
            mx = ma.masked_array(a, mask)
            var = ma.getdata(mx.var(axis=0))
            var_trials.append(var)

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

        med = np.median(a, axis=0)
        p16 = np.percentile(a, 16, axis=0)
        p84 = np.percentile(a, 84, axis=0)
        sigma = 0.5 * (p84 - p16)
        mdmsg = med - n_sigma * sigma
        b = np.less(a,mdmsg)
        mdpsg = med + n_sigma * sigma
        c = np.greater(a,mdpsg)
        mask = np.any([b,c],axis=0)
        mx = ma.masked_array(a, mask)
        avg = ma.getdata(mx.mean(axis=0))
        var = ma.getdata(mx.var(axis=0)) * cf
        cnt = ma.getdata(ma.count(mx,axis=0))
        unc = np.sqrt(var/cnt)

        if self.logger:
            self.logger.debug('{}.compute(): avg(bias),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))
        else:
            print('---->{}.compute(): avg(bias),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))

        return avg,var,cnt,unc



