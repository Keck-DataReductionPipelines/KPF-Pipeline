class ConfigHandler():
    """Config 'PARAM' section handler.

    This module define class 'ConfigHandler' and methods to handle the values defined in the specified
    section of a config context.

    Args:
        config (configparser.ConfigParser): config context.
        section (str): Section name from the config context. Defaults to 'PARAM'.

    Attributes:
        config_param (configparser.SectionProxy): Instance containing the key-value pairs associated with
        `section` of `config`
    """
    def __init__(self, config, section=None, default=None):
        if section is not None and config.has_section(section):
            self.config_param = config[section]
        else:
            self.config_param = default

    def get_section(self):
        return self.config_param

    def get_config_value(self, param: str, default=None):
        """Get defined value from the config file.

        Search the value of the specified property from config section. The default value is returned if no found.

        Args:
            param (str): Name of the parameter to be searched.
            default (str/int/float): Default value for the searched parameter.

        Returns:
            int/float/str: Value for the searched parameter.

        """

        if self.config_param is not None:
            if isinstance(default, int):
                return self.config_param.getint(param, default)
            elif isinstance(default, float):
                return self.config_param.getfloat(param, default)
            else:
                return self.config_param.get(param, default)
        else:
            return default
