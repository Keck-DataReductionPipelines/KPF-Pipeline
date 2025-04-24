#### WARNING: this hasn't been been verified to work yet!

from modules.Utils.kpf_parse import get_data_products_L0 as get_dp_L0
from modules.Utils.kpf_parse import get_data_products_2D as get_dp_2D
from modules.Utils.kpf_parse import get_data_products_L1 as get_dp_L1
from modules.Utils.kpf_parse import get_data_products_L2 as get_dp_L2
from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext


class GetDataProductsFramework(KPF_Primitive):
    """
    This framework primative implements the get_data_products methods 
    from modules/Utils/kpf_parse.py

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains 
                   positional arguments and keyword arguments passed by the 
                   `get_data_products` event issued in the recipe:

            - `action.args[0] (kpf_object)`: L0/2D/L1/L2 object to be analyzed
            - `action.args[1] (str)`: data level ('L0', '2D', 'L1', 'L2')
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)

    def _pre_condition(self) -> bool:
        success = len(self.action.args) >= 2 and isinstance(self.action.args[1], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        kpf_object = self.action.args[0]
        data_level_str = self.action.args[1] # L0, D2, L1, or 2D

        if data_level_str == 'L0':
            data_products = get_dp_L0(kpf_object)
        elif data_level_str == '2D':
            data_products = get_dp_2D(kpf_object)
        elif data_level_str == 'L1':
            data_products = get_dp_L1(kpf_object)
        elif data_level_str == 'L2':
            data_products = get_dp_L2(kpf_object)
        else:
            data_prodcts = ['None']

        return Arguments(data_products)
