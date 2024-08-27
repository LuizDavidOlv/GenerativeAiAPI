import json
import os
import sys

from Configs.ConfigReader import AppConfiguration
from Vault.Client import Vaultclient, VaultKVOption

# import logging
# logger = logging.getLogger()


class Llm:

    @staticmethod
    def get_kv():

        if os.getenv("OPENAI_API_KEY"):
            pass
        else:
            sidecar = AppConfiguration.get_value("hc-sidecar", "container")
            vault = Vaultclient()

            match sidecar.lower():
                case "true":
                    secret = vault.get_secret(VaultKVOption.sidecar.value)
                    open_ai_key = json.loads(secret)["sql-server.json"]["data"]["host"]
                case "false":
                    secrets = vault.get_secret(VaultKVOption.OpenAi.value)

                    for secret in json.loads(secrets)["secrets"]:
                        if secret["name"] == "key":
                            open_ai_key = secret["version"]["value"]

                case _:
                    # logger.error("please configure vault settings with option of api or sidecar interface.")
                    raise Exception(
                        "invalid option selected in config.ini {sidecar = false/true}."
                    )
            try:
                os.environ["OPENAI_API_KEY"] = open_ai_key
            except KeyError:
                # logger.error('[error]: `OPENAI_API_KEY` environment variable required')
                sys.exit(1)
