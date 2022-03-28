import ast

class ConfigHandler():
    """Config file handler.

    This module defines class 'ConfigHandler' and methods to access the values defined in a specified
    section of a config context.

    Args:
        config (configparser.ConfigParser): config context.
        section (str): Section name from the config context. Defaults to None.
        default (ConfigHandler): An instance of ConfigHandler in case `section` not found. Defaults to None.

    Attributes:
        config_param (configparser.SectionProxy): Instance of dict containing the property-value pairs associated with
        `section` in `config`
    """
    def __init__(self, config, section=None, default=None):
        if config is not None and section is not None and config.has_section(section):
            self.config_param = config[section]
        else:
            self.config_param = default.get_section() if default is not None else None

    def get_section(self):
        return self.config_param

    def get_config_value(self, param: str, default=None):
        """Get defined value from the instance associated section.

        Search the value of the specified property from config section. The default value is returned if no found.

        Args:
            param (str): Name of the property to be searched.
            default (str/int/float): Default value for the searched property.

        Returns:
            str/int/float/: Value for the searched property.

        """

        if self.config_param is not None:
            if isinstance(default, int):
                return self.config_param.getint(param, default)
            elif isinstance(default, float):
                return self.config_param.getfloat(param, default)
            else:
                c_str = self.config_param.get(param, default)

                if c_str and isinstance(c_str, str):
                    b_pairs = [['[', ']']]

                    for p in b_pairs:
                        if c_str[0] == p[0] and c_str[-1] == p[1]:
                            try:
                                c_list = ast.literal_eval(c_str)
                                return c_list
                            except ValueError:
                                return c_str
                return c_str
        else:
            return default
