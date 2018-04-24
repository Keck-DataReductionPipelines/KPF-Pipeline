"""
Routines for the KPF pipeline

Will probably organize in the future into seperate modules
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
        config (kpfpipe.config): configuration object with different setting for pipeline

    Returns:
        array: a 1D extracted spectrum        
    """
    
    pass
