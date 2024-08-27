import json
import os
from pathlib import Path

from Configs.ConfigReader import AppConfiguration


class Sidecar:
    def container():
        try:
            # logger.info("- Invoke: Vault Sidecar Service.")
            ## get all the files names from the config.ini
            files = AppConfiguration.get_value("hc-sidecar", "filename").split(",")
            secret_location = AppConfiguration.get_value(
                "hc-sidecar", "secret_location"
            )
            root_dir_level = AppConfiguration.get_value("hc-sidecar", "root_dir_level")

            ## naviate to the vault folder [1] = parent directory
            directory = Path(__file__).parents[int(root_dir_level)]

            ## get all the files from the vault folder
            data = os.path.join(directory, secret_location)

            ## create a dictionary to store the json files
            json_data = dict()

            ## loop through the files and store them in the dictionary
            for filename in os.listdir(data):
                ## check if the file is in the config.ini
                if filename in files:
                    with open(os.path.join(data, filename)) as f:
                        ## load the json file
                        parsed_json = json.load(f)
                        ## add parent element = filename to json and store the json file in the dictionary
                        json_data[filename] = parsed_json
            # return json_data as string
            return json.dumps(json_data)
        except Exception as e:
            # logger.error("- [error] Failed to get secret through sidecar.")
            # logger.error(e)
            raise Exception("- Failed to get secret through sidecar.")
