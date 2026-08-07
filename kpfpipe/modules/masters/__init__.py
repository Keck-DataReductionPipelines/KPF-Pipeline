"""Masters modules: bias, dark, flat, order trace, and wls."""

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.modules.masters.bias import Bias
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.modules.masters.flat import Flat
from kpfpipe.modules.masters.order_trace import OrderTrace
from kpfpipe.modules.masters.wls import WLS

__all__ = ["BaseMasterModule", "Bias", "Dark", "Flat", "OrderTrace", "WLS"]
