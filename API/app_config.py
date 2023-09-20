import os
from pathlib import Path
from configparser import ConfigParser

class AppConfiguration:
    def get_value(attribute,key):
        try:
            print(attribute, key)
            path = Path(__file__)
            ROOT_DIR = path.parent.absolute()
            config_path = os.path.join(ROOT_DIR "config.ini")
            parser = ConfigParser()
            parser.read(config_path)
            secret = parser.get(attribute, key)
            return secret
        except:
            print("Error in reading config file")
            return "Failure"