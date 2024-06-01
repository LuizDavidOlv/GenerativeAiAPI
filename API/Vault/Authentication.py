import json
import os
from dotenv import find_dotenv, load_dotenv
import requests
from Vault.Client import Vault

load_dotenv(find_dotenv(), override=True)

class Authentication(Vault):
    def Authenticate():
        # Define the token URL and headers
        #url_config = "https://auth.idp.hashicorp.com/oauth2/token"
        headersConfig = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Define the data payload
        payload = {
            "client_id": Vault.client_id,
            "client_secret":  Vault.client_secret,
            "grant_type": "client_credentials",
            "audience": "https://api.hashicorp.cloud"
        }

        try:
            # Make the POST request to get the token
            response = requests.post(url=Vault.auth_url, headers=headersConfig, data=payload)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response to set the access token
                os.environ['HCP_CLIENT_JWT'] = json.loads(response.text)["access_token"]
                #logger.info("- Authentication successful.")
            else:
                raise Exception(response.json())
        except Exception as e:
            # logger.error("- [error] Failed to authenticate with vault.")
            # logger.error(e)
            raise Exception("- Failed to authenticate with vault.")