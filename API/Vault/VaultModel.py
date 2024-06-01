import os
from dotenv import find_dotenv, load_dotenv
from API.Configs.ConfigReader import AppConfiguration

load_dotenv(find_dotenv(), override=True)

class Vault():
    test = os.environ['HCP_VAULT_TOKEN']
    auth_url = os.getenv('HCP_VAULT_TOKEN')
    client_url = os.getenv('HCP_VAULT_TOKEN')
    client_id = os.getenv('HCP_CLIENT_ID')
    client_secret = os.getenv('HCP_CLIENT_SECRET')
    namespace = AppConfiguration.get_value("hc-vault", "namespace")
    sidecar = AppConfiguration.get_value("hc-sidecar", "container")