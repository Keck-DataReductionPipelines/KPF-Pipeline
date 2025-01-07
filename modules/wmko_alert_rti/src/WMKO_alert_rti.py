"""
    The file contains primitive KPFModeExample

    Attributes:
       KPFModeExample: primitive for simmple recipe test


"""

import time
import configparser
import os
import requests

from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

DEFAULT_CFG_PATH = 'modules/wmko_alert_rti/configs/default.cfg'

class SendRTIHttp(KPF_Primitive):
    """
    This module sends the HTTP GET request needed by KOA RTI to archive new data products

    Description:
         - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `SendRTIHttp` event issued in the recipe:
    """

def __init__(self,
                action: Action,
                context: ProcessingContext) -> None:

    # Initialize parent class
    KPF_Primitive.__init__(self, action, context)
    args_keys = [item for item in action.args.iter_kw() if item != "name"]


    # input configuration
    self.config = configparser.ConfigParser()
    try:
        self.config_path = context.config_path['wmko_alert_rti']
    except:
        self.config_path = DEFAULT_CFG_PATH
    self.config.read(self.config_path)
    
    
    if not self.logger:
        self.logger = self.context.logger
    self.logger.info('Loading config form: {}'.format(self.config_path))


def _pre_condition(self) -> bool:
    return True

def _post_condition(self) -> bool:
    return True

def _perform(self):
            
    data_directory = os.path.join(self.config.instrument.cwd,
                                self.config.instrument.output_directory)
    
    self.logger.info(f"Alerting RTI that {self.action.args.name} is ready for ingestion")

    url = self.config.rti.rti_url
    data = {
        'instrument': 'KCWI',
        'koaid': self.action.args.ccddata.header['KOAID'],
        'ingesttype': self.config.rti.rti_ingesttype,
        'datadir': str(data_directory),
        'start': str(self.action.args.ingest_time),
        'reingest': self.config.rti.rti_reingest,
        'testonly': self.config.rti.rti_testonly,
        'dev': self.config.rti.rti_dev
    }
    
    attempts = 0
    limit = self.config.rti.rti_attempts
    while attempts < limit:
        res = self.get_url(url, data)
        if res is None:
            t = self.config.rti.rti_retry_time
            attempts += 1
            self.logger.error(f"Waiting {t} seconds to attempt again... ({attempts}/{limit})")
            time.sleep(t)
        else:
            self.logger.info(f"Post returned status code {res.status_code}")
            return self.action.args
    
    self.logger.error(f"Post attempted {limit} times and got no response.")
    self.logger.error("Aborting.")
    return self.action.args

def get_url(self, url, data):
    try:
        res = requests.get(url, params = data, auth=(
                                                    self.user,
                                                    self.pw
                                                    ))
        self.logger.info(f"Sending {res.request.url}")
    except requests.exceptions.RequestException as e:
        self.logger.error(f"Error caught while posting to {url}:")
        self.logger.error(e)
        return None
    return res