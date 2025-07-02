import time
import configparser
import os
import requests

from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

DEFAULT_CFG_PATH = 'modules/keck_alert_rti/configs/default.cfg'

class SendRTIHttp(KPF_Primitive):
    """
    Description:
        Sends the HTTP GET request needed by KOA RTI to archive new
        data products. This module is intended to be used only at WMKO.
        The call to this module will also fail if no RTI credentials are
        provided in the configuration file.

    Arguments:
        input_file (str): The path to the file to be ingested. This filename
        is used to extract the KOAID.
        output_dir (str): The directory where the file is stored.
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
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Load arguments into atributtes
        self.input_file = self.action.args[0]
        self.output_dir = self.action.args[1]
        self.level = self.action.args[2]


    def _pre_condition(self) -> bool:
        """Precondition for the SendRTIHttp primitive. If the RTI URL, user, and
        password are not provided in the config file, then the precondition fails.
        Additionally, if the level is not L1 or L2, the precondition fails.

        Returns
        -------
        bool
            True if the RTI URL, user, and password are provided in the config file.
        """
        if self.config['RTI']['rti_url'] is None:
            self.logger.error("No RTI URL specified in config file.")
            self.logger.error("Please add RTI URL to config file.")
            return False
        if self.config['RTI']['rti_user'] is None:
            self.logger.error("No RTI user specified in config file.")
            self.logger.error("Please add RTI user and password to config file.")
            return False
        if self.config['RTI']['rti_pass'] is None:
            self.logger.error("No RTI password specified in config file.")
            self.logger.error("Please add RTI user and password to config file.")
            return False

        if self.level not in ['L1', 'L2']:
            self.logger.info(f"Level {self.level} will not trigger RTI ingestion.")
            return False
        return True

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        
        self.logger.info(f"Alerting RTI that {self.input_file} is ready for ingestion")

        url = self.config['RTI']['rti_url']
        koaid = os.path.basename(self.input_file).split('_')[0]
        datadir = self.output_dir + koaid + '/' + self.level # /data/QLP/[UTDATE]/[KOAID]/[LEVEL]/
        self.logger.info("Sending GET request to RTI with the following data:")
        self.logger.info(f"URL: {url}")
        self.logger.info(f"Data Directory: {datadir}")
        self.logger.info(f"KOAID: {koaid}.fits")
        data = {
            'instrument': 'KPF',
            'koaid': koaid + ".fits", # actually takes koaid + fits
            'ingesttype': "lev2" if self.level == "L2" else "lev1", # TODO: make this match level
            'datadir': datadir,
            'start': None,
            'reingest': self.config['RTI']['rti_reingest'],
            'testonly': self.config['RTI']['rti_testonly'],
            'dev': self.config['RTI']['rti_dev']
        }
        
        attempts = 0
        limit = int(self.config['RTI']['rti_attempts'])
        while attempts < limit:
            res = self.get_url(url, data)
            if res is None:
                t = self.config['RTI']['rti_retry_time']
                attempts += 1
                self.logger.error(f"Waiting {t} seconds to attempt again... ({attempts}/{limit})")
                time.sleep(t)
            else:
                self.logger.info(f"GET returned status code {res.status_code}")
                if res.status_code != 200:
                    self.logger.warning(f"Received non-200 status code with body: {res.text}")
                return Arguments([self.input_file, self.output_dir])
        
        self.logger.error(f"Post attempted {limit} times and got no response.")
        return Arguments([self.input_file, self.output_dir])

    def get_url(self, url, data):
        """Sends a get request to the specified URL with the specified data.

        Parameters
        ----------
        url : str
            URL to send the request to.
        data : dict
            dictionary of key/value pairs to send as parameters in the request.

        Returns
        -------
        requests.Response
            Response object from the request, or None if an error occurred.
        """
        try:
            res = requests.get(url, params = data, auth=(
                                                        self.config['RTI']['rti_user'],
                                                        self.config['RTI']['rti_pass']
                                                        ))
            self.logger.info(f"Sending {res.request.url}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error caught while posting to {url}:")
            self.logger.error(e)
            return None
        return res