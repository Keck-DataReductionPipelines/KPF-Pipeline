from keckdrpframework.primitives.base_primitive import BasePrimitive

import sys
class KPFModule_dummy(BasePrimitive):

    def __init__(self, action, context):
        try:
            BasePrimitive.__init__(self, action, context)
        except: 
            print(sys.exc_info())
        print('what??')
    def _perform(self):
        print('============DUMMY=============')
    

