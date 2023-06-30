import numpy as np
import matplotlib.pyplot as plt

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
        starname = D2['PRIMARY'].header['TARGNAME']
        ObsID = D2['PRIMARY'].header['OFNAME']

        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
            image = D2[CHIP + '_CCD'].data
        
        # Generate 2D image
        plt.figure(figsize=(10,8), tight_layout=True)
        #plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
        plt.imshow(image, vmin = np.percentile(image,1), 
                          vmax = np.percentile(image,99.5), 
                          interpolation = 'None', 
                          origin = 'lower')
        plt.title('2D - ' + chip_title + ' CCD: ' + str(ObsID) + ' - ' + starname, fontsize=14)
        plt.xlabel('x (pixel number)', fontsize=14)
        plt.ylabel('y (pixel number)', fontsize=14)
        cbar = plt.colorbar(label = 'Counts (e-)')
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        
        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=1200, facecolor='w')
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
