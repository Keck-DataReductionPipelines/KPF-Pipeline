

# External dependencies
from keckdrpframework.primitives.base_primitive import Base_primitive

class TFAMain(Base_primitive):

    def __init__(self, action, context):

        Base_primitive.__init__(self, action, context)
    
    def _pre_condition(self):
        return True

    def _post_condition(self):
        return True

    def _perform(self):
        print('Dummy found! Yay!')