import os
import requests
from dotenv import load_dotenv, find_dotenv
from Constants.Enumerations import VaultKVOption
from Vault.Sidecar import Sidecar
from Vault.VaultModel import Vault
from Vault.Authentication import Authentication

load_dotenv(find_dotenv(), override=True)
access_token = None


class Vaultclient(): 
    def get_secret(self,vault_type):
        match Vault.sidecar.lower():
            case 'true':
                return Sidecar.container()
            case 'false':
                return Https.http_calls(vault_type)
            case _:
                #logger.error("- [error] please configure vault settings with option of api or sidecar interface.")
                raise Exception("- Invalid option selected in config.ini {sidecar = false/true}.")

    

class Https(Vault):
    def http_calls(vault_type):

        if not os.getenv('HCP_VAULT_TOKEN'):
            #logger.info("---------------------------------------------------")
            #vault authentication - token retrieval
            Authentication.authenticate()
            #logger.info("---------------------------------------------------")
        else:
            headers_config = {
                "Authorization": f"Bearer {os.environ['HCP_VAULT_TOKEN']}"
            }

            response = requests.get(url=Vault.client_url, headers=headers_config)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                secret_data = response.json()
                print("Secret data:", secret_data)
            else:
                print(f"Failed to read secret: {response.status_code}")
        # set the variables
            

class VaultType:
    def match_type(vault_type):
        try:
            match vault_type:
                case VaultKVOption.sqlsever.value:
                    path = AppConfiguration.get_value("hc-vault", "path")
                
                case _:
                    #logger.error("- [error] please configure vault settings with option of api or sidecar interface.")
                    raise Exception("- Invalid option selected in config.ini {sidecar = false/true}.")
