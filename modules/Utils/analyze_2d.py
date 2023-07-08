import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import emcee
import corner

class Analyze2D:

    """
    Description:
        This class contains functions to analyze 2D images (storing them
        as attributes) and functions to plot the results.
        Some of the functions need to be filled in

    Arguments:
        2D - a 2D object

    Attributes:
        TBD
    """

    def __init__(self, D2, logger=None):
        self.D2 = D2 # use D2 instead of 2D because variable names can't start with a number
        header = L0['PRIMARY'].header
        self.name = HeaderParse(header).get_name()
        self.ObsID = ''
        if 'OFNAME' in header:
            self.ObsID == header['OFNAME']  # better to use header keywords than pass in ObsID
       
        if logger:
            self.logger = logger
            self.logger.debug('Analyze2D class constructor')
        else:
            self.logger = None
            print('---->Analyze2D class constructor')


    def plot_2D_image(self, chip=None, fig_path=None, show_plot=False):

        """
        Generate a plot of the a 2D image.  

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
            image = self.D2[CHIP + '_CCD'].data
        
        # Generate 2D image
        plt.figure(figsize=(10,8), tight_layout=True)
        #plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
        plt.imshow(image, vmin = np.percentile(image,0.1), 
                          vmax = np.percentile(image,99.9), 
                          interpolation = 'None', 
                          origin = 'lower')
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Column (pixel number)', fontsize=14)
        plt.ylabel('Row (pixel number)', fontsize=14)
        cbar = plt.colorbar(label = 'Counts (e-)')
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        
        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=1200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()


    def plot_bias_histogram(self, chip=None, fig_path=None, show_plot=False):

        """
        Plot a histogram of the counts per pixel in a 2D image.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN_CCD'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED_CCD'
                chip_title = 'Red'

        histmin = -40
        histmax = 40
        flattened = self.D2[CHIP].data.flatten()
        flattened = flattened[(flattened >= histmin) & (flattened <= histmax)]
        
        # Fit a normal distribution to the data
        mu, std = norm.fit(flattened)
        median = np.median(flattened)

        innermin = -15
        innermax = 15
        #flattened_inner = flattened[(flattened >= innermin) & (flattened <= innermax)]
        #mu, std = norm.fit(flattened_inner)
        #median = np.median(flattened_inner)
        

#        # Define the model: sum of two Gaussians
#        def gaussian(x, mu, sigma, amplitude):
#            return amplitude * norm.pdf(x, mu, sigma)
#        
#        def model(params, x):
#            mu1, mu2, sigma1, sigma2, amplitude1, amplitude2 = params
#            return gaussian(x, mu1, sigma1, amplitude1) + gaussian(x, mu2, sigma2, amplitude2)
#        
#        # Define the log-probability function
#        def log_prob(params, x, y):
#            model_y = model(params, x)
#            #sigma = params[2] + params[3]
#            sigma_y = np.sqrt(y+1)
#            return -0.5 * np.sum((y - model_y) ** 2 / sigma_y ** 2 + np.log(sigma_y ** 2))
#        
#        # Run the MCMC estimation
#        ndim = 6  # number of parameters in the model
#        nwalkers = 50  # number of MCMC walkers
#        nburn = 1000  # "burn-in" period to let chains stabilize
#        nsteps = 5000  # number of MCMC steps to take
#        
#        # set up initial guess and run MCMC
#        guess = np.array([0, 0, 3, 10, 100000, 1000]) + 0.1 * np.random.randn(nwalkers, ndim)
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[flattened, model(guess, flattened)])
#        sampler.run_mcmc(guess, nsteps)
#        
#        # Discard burn-in and flatten the samples
#        samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)
#        
#        # Make a corner plot with the posterior distribution
#        fig, ax = plt.subplots(ndim, figsize=(10, 7), tight_layout=True)
#        corner.corner(samples, labels=["mu1", "mu2", "sigma1", "sigma2", "amplitude1", "amplitude2"], truths=guess[0], ax=ax)
#        plt.show()        
        
        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(7,5))
        
        # Create histogram with log scale
        n, bins, patches = plt.hist(flattened, bins=range(histmin, histmax+1), color='gray', log=True)
        
        # Plot the distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, histmax-histmin+1)
        p = norm.pdf(x, mu, std) * len(flattened) * np.diff(bins)[0] # scale the PDF to match the histogram
        plt.plot(x, p, 'r', linewidth=2)
        
        # Add annotations
        textstr = '\n'.join((
            r'$\mu=%.2f$ e-' % (mu, ),
            r'$\sigma=%.2f$ e-' % (std, ),
            r'$\mathrm{median}=%.2f$ e-' % (median, )))
        props = dict(boxstyle='round', facecolor='red', alpha=0.15)
        plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Set up axes
        ax.axvline(x=0, color='blue', linestyle='--')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(histmin, histmax)
        plt.ylim(5*10**-1, 10**7)
        #plt.title(str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Counts (e-)', fontsize=14)
        plt.ylabel('Number of Pixels (log scale)', fontsize=14)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()

        
    def plot_bias_histogram2(self, chip=None, fig_path=None, show_plot=False):

        """
        Plot a histogram of the counts per pixel in a 2D image.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN_CCD'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED_CCD'
                chip_title = 'Red'

        histmin = -50
        histmax = 50
        flattened = self.D2[CHIP].data.flatten()
        flattened = flattened[(flattened >= histmin) & (flattened <= histmax)]
        
        # Fit a normal distribution to the data
        mu, std = norm.fit(flattened)
        median = np.median(flattened)

        innermin = -10
        innermax = 10
        flattened_inner = flattened[(flattened >= innermin) & (flattened <= innermax)]
        mu, std = norm.fit(flattened_inner)
        median = np.median(flattened_inner)
        
        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(7,5))
        
        # Create histogram with log scale
        n, bins, patches = plt.hist(flattened, bins=range(histmin, histmax+1), color='gray', log=True)
        
        # Plot the distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(innermin, innermax, innermax-innermin+1)
        p = norm.pdf(x, mu, std) * len(flattened) * np.diff(bins)[0] # scale the PDF to match the histogram
        plt.plot(x, p, 'r', linewidth=2)
        
        # Add annotations
        textstr = '\n'.join((
            r'$\mu=%.2f$ e-' % (mu, ),
            r'$\sigma=%.2f$ e-' % (std, ),
            r'$\mathrm{median}=%.2f$ e-' % (median, )))
        props = dict(boxstyle='round', facecolor='red', alpha=0.15)
        plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Set up axes
        ax.axvline(x=0, color='blue', linestyle='--')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(histmin, histmax)
        plt.ylim(5*10**-1, 10**7)
        #plt.title(str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Counts (e-)', fontsize=14)
        plt.ylabel('Number of Pixels (log scale)', fontsize=14)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=200, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()

        
    def plot_2D_order_trace_image(self, chip=None, fig_path=None, show_plot=False):

        """
        TO-DO: MOVE THE PLOTTING CODE FROM THE QLP HERE. 
               THIS IS A PLACEHOLDER.
        
        Generate a plot of the stitched L0 image.  
        The image will be divided by 2^16, if appropriate.

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
    
    def plot_2D_image_histogram(self, chip=None, fig_path=None, show_plot=False):

        """
        TO-DO: MOVE THE PLOTTING CODE FROM THE QLP HERE. 
               THIS IS A PLACEHOLDER.
        
        Generate a plot of the stitched L0 image.  
        The image will be divided by 2^16, if appropriate.

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
    
    def plot_2D_column_cut(self, chip=None, fig_path=None, show_plot=False):

        """
        TO-DO: MOVE THE PLOTTING CODE FROM THE QLP HERE. 
               THIS IS A PLACEHOLDER.
        
        Generate a plot of the stitched L0 image.  
        The image will be divided by 2^16, if appropriate.

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
