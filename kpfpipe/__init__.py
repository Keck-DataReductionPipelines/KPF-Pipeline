import sys
import os 

## This is an ugly fix for local package dependencies 
# --TODO-- find a better way to do this 
sys.path.insert(0, os.path.abspath('../KeckDRPFramework'))

import kpfpipe.cli

__version__ = '2.1.0'
