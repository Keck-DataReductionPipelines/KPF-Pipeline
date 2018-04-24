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
        
        


def spectral_extraction(sci, bias, flat, trace, badpixel, order, orderlet, config):
    """
    Extract spectrum from a single orderlet
    
    Args:
        sci (array): science frame e.g. kpf0.redspec
        bias (array): bias frame e.g. kpf0.redspec
        flat (array): flat field frame e.g. kpf0.redspec
        badpixel (array): bad pixel mask  
        trace (array): a kpfpipe.Trace object
        order (int): index of order
        orderlet (int): index of orderlet
        config (kpfpipe.config): configuration object

    Returns:
        array: a 1D extracted spectrum        
    """
    
    pass
    
    
    
    
    




