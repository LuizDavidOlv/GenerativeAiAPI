import os
from dotenv import find_dotenv, load_dotenv
from Configs.ConfigReader import AppConfiguration

load_dotenv(find_dotenv(), override=True)

class Vault():
    auth_url = os.getenv('HCP_AUTH_URL')
    client_url = os.getenv('HCP_CLIENT_URL')
    client_id = os.getenv('HCP_CLIENT_ID')
    client_secret = os.getenv('HCP_CLIENT_SECRET')
    namespace = AppConfiguration.get_value("hc-vault", "namespace")
    sidecar = AppConfiguration.get_value("hc-sidecar", "container")