
from Configs.ConfigReader import AppConfiguration
from Vault.Keys.OpenAi import Llm

class Globle:

########################################################################################
## get credentials from vault desire kv and set them as environment variables
########################################################################################

   @staticmethod
   def Settings():

    f = lambda x:  x.lower() == "true"
    
    # Configure logger format and omit uvicorn messages
    #apilogger.configure();

    # get the kv path for openai from the config.ini and check if it is enabled
    if f(AppConfiguration.get_value("hc-openai-llm", "enable")):
        Llm.get_kv()
