import json
import os

import requests
from dotenv import find_dotenv, load_dotenv

from Configs.ConfigReader import AppConfiguration
from Constants.Enumerations import VaultKVOption
from Vault.Authentication import Authentication
from Vault.Sidecar import Sidecar
from Vault.VaultModel import Vault

load_dotenv(find_dotenv(), override=True)
access_token = None


class Vaultclient:
    def get_secret(self, vault_type):
        match Vault.sidecar.lower():
            case "true":
                return Sidecar.container()
            case "false":
                return Https.http_calls(vault_type)
            case _:
                # logger.error("- [error] please configure vault settings with option of api or sidecar interface.")
                raise Exception(
                    "- Invalid option selected in config.ini {sidecar = false/true}."
                )


class Https(Vault):
    def http_calls(vault_type) -> str:

        if not os.getenv("HCP_VAULT_TOKEN"):
            # logger.info("---------------------------------------------------")
            # vault authentication - token retrieval
            Authentication.Authenticate()
            # logger.info("---------------------------------------------------")
        else:
            headers_config = {
                "Authorization": f"Bearer {os.environ['HCP_VAULT_TOKEN']}"
            }

            response = requests.get(url=Vault.client_url, headers=headers_config)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                secret_data = response.json()
                os.environ["HCP_VAULT_TOKEN"] = json.loads(secret_data)["data"]["token"]
            else:
                print(f"Failed to read secret: {response.status_code}")

        return VaultType.match_type(vault_type)


class VaultType:
    def match_type(vault_type) -> str:
        try:
            match vault_type:
                case VaultKVOption.OpenAi.value:
                    path = AppConfiguration.get_value("hc-openai-llm", "kv_path")

                case _:
                    # logger.error("- [error] please configure vault settings with option of api or sidecar interface.")
                    raise Exception(
                        "- Invalid option selected in config.ini {sidecar = false/true}."
                    )

            formated_url = Vault.client_url + path + "/open"

            headers_config = {"Authorization": f"Bearer {os.environ['HCP_CLIENT_JWT']}"}

            response = requests.request(
                method="GET", url=formated_url, headers=headers_config
            )

            if response.status_code == 200:
                # logger.info("- Secret retrieved from kv-path [" + path  + "]")
                return response.text
            # else:
            # logger.error("- [error] Failed to retrieve secrets, plesae check kv path: " + path)

        except Exception as e:
            # logger.error("- [error] Failed to get secret." + e)
            raise Exception("- Failed to authenticate with vault." + e)
