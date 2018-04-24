"""
Placeholder module
"""

class Trace(object):
    __init__(self, ):
        """
        Fill in
        """
        
    def center(self, row, order, orderlet):
        """Ceter of order
        
        Args:
            row (float): row of the CCD
            order (int): index of the order
            orderlet (int): index of orderlet
            
        Returns:
            float: column of orderlet (sub-pixel)
        """
        
    def width(self, row, order, orderlet):
        """Return the width (subpixel) of the order
        
        Args:
            row (float): row of the CCD
            order (int): index of the order
            orderlet (int): index of orderlet
            
        Returns:
            float: widht of the orderlet (sub-pixel)
            
        """


def make_master_flat(flatfns, masterflatfn):
    """Make master flat field frame
     
    Args:
        flatfns (list of str): paths to flat observations
        masterflatfn (str): output file path to master flat 
        
    Returns:
        None
    """

def make_master_bias(baisfns, masterbaisfn):
    """Make master bias frame
    
    Args:
        baisfns (list): paths to bias observations 
        masterbaisfn (list): utput file path to master bias
    """
    
