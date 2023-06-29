import numpy as np
import matplotlib.pyplot as plt

class Analyze2D:

    """
    Description:
        This class contains functions to analyze 2D images (storing them
        as attributes) and functions to plot the results.
        It is only a placeholder for now.

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
