import os
from configparser import ConfigParser

class AppConfiguration:
    # read config file and extract values
    def get_value(attribute, key):
            dirname = os.path.dirname(__file__)
            try:
                config_path = os.path.join(dirname, "config.ini")
                parser = ConfigParser()
                parser.read(config_path)
                secret= parser.get(attribute, key)
                return secret
            except:
                #logger.error("Error in reading config file")
                return "Error in reading config file"