import pytest
import configparser as cp
from modules.TemplateFit.KPFModule_TFA import TFAMakeTemplate

from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext
fpath = 'modules/TemplateFit/data/HARPS_Barnards_Star_benchmark'

@pytest.fixture(scope='module')
def mod():
    action = Action((None, None, None), None)
    action.args = None
    arg = Arguments()
    arg.tfa_input = fpath


    context = Arguments()
    context.logger = None
    context.config = None
    context.tfa_config = 'modules/TemplateFit/configs/default.cfg'
    context.arg = arg
    context.arg.tfa_out = None

    return TFAMakeTemplate(action, context)


def test_run(mod):
    mod()
    